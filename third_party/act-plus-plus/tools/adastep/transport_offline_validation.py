"""Transport offline validator (migrated + adapted)
- Copies core analysis used in the paper for Transport task
- Uses local HorizonPredictor API from predictors.adastep
"""
from predictors.adastep.adastep_module import HorizonPredictor
import torch
import numpy as np
import h5py
import os

# Adapted load_trajectory_data for third_party layout

def load_trajectory_data(hdf5_path, trajectory_idx=0):
    with h5py.File(hdf5_path, 'r') as f:
        # best-effort: support both ACT & adastep_extension dataset layouts
        if 'data' in f:
            demos = list(f['data'].keys())
            demo = f[f'data/' + demos[trajectory_idx]]
            if 'obs/robot0_eef_pos' in demo:
                eef_pos = demo['obs/robot0_eef_pos'][()]
                eef_quat = demo['obs/robot0_eef_quat'][()]
                states = np.concatenate([eef_pos, eef_quat], axis=-1)
            elif 'obs/robot0_joint_pos' in demo:
                states = demo['obs/robot0_joint_pos'][()]
            else:
                raise ValueError('no valid state data')
            actions = demo['actions'][()]
            return states, actions, {'length': len(states), 'state_dim': states.shape[1], 'action_dim': actions.shape[1]}
        else:
            raise ValueError('unexpected hdf5 layout')

# The rest of the original script (statistics/plotting) is preserved in adastep_extension; here we keep a smoke-friendly subset

def analyze_trajectory_phases(states, actions):
    eef_positions = states[:, :3]
    velocities = np.linalg.norm(np.diff(eef_positions, axis=0), axis=1)
    velocities = np.concatenate([[0], velocities])
    init_pos = eef_positions[0]
    distances_from_start = np.linalg.norm(eef_positions - init_pos, axis=1)
    phases = []
    for t in range(len(states)):
        velocity = velocities[t]
        distance = distances_from_start[t]
        if distance < 0.05:
            phase = 'reaching'
        elif velocity < 0.01:
            phase = 'grasping'
        else:
            phase = 'transporting'
        phases.append(phase)
    return phases

# smoke-run function
def run_offline_validation(hdf5_path, predictor_path, trajectory_indices=None):
    sample_states, _, _ = load_trajectory_data(hdf5_path, 0)
    state_dim = sample_states.shape[1]
    horizon_predictor = HorizonPredictor(input_dim=state_dim, hidden_dim=256)
    device = 'cpu'
    horizon_predictor = horizon_predictor.to(device)
    if os.path.exists(predictor_path):
        horizon_predictor.load_state_dict(torch.load(predictor_path, map_location=device))
        horizon_predictor.eval()
    else:
        print('predictor not found, running diagnostic with random init')
    if trajectory_indices is None:
        trajectory_indices = [0]
    results = []
    for traj_idx in trajectory_indices:
        states, actions, metadata = load_trajectory_data(hdf5_path, traj_idx)
        phases = analyze_trajectory_phases(states, actions)
        k_values = []
        for t in range(len(states)):
            state_feature = states[t, :7] if states.shape[1] >= 7 else states[t]
            state_tensor = torch.from_numpy(state_feature).float().to(device).unsqueeze(0)
            with torch.no_grad():
                k_pred = horizon_predictor.predict_horizon(state_tensor, 5, 50).item()
            k_values.append(int(k_pred))
        results.append({'trajectory_idx': traj_idx, 'k_values': k_values, 'phases': phases})
    return results
