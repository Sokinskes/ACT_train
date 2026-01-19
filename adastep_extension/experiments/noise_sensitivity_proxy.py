"""
Noise-sensitivity proxy for AdaStep horizon predictor.
This offline proxy perturbs states and measures how predicted horizon k changes.
Produces: experiments/sensitivity_results/noise_sensitivity.csv and .png
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from eval_offline_trajectory import load_test_trajectories
from core.adastep_module import HorizonPredictor

DATA_HDF5 = '../robomimic_data/square/mh/low_dim_v15.hdf5'
OUT_DIR = './sensitivity_results'
LAMBDAS = [0.5, 1.0, 2.0, 3.0]
NOISE_STD = 0.01  # noise applied to qpos (state vector)
DELTA_K = 5      # threshold for 'significant' k-change
DEVICE = 'cpu'

os.makedirs(OUT_DIR, exist_ok=True)

def load_predictor(path, state_dim):
    model = HorizonPredictor(input_dim=state_dim, hidden_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def probe_predictor_fragility(predictor, test_demos, noise_std=NOISE_STD, delta_k=DELTA_K):
    frag_points = 0
    total_points = 0
    frag_trajs = 0
    traj_count = 0
    mean_k_clean_all = []
    mean_k_noisy_all = []

    for demo in test_demos.values():
        states = demo['states']
        traj_frag = False
        k_clean_list = []
        k_noisy_list = []
        for t in range(0, len(states), 5):  # check every 5 steps to approximate decision points
            s = states[t].astype(np.float32)
            s_noisy = s + np.random.normal(0, noise_std, size=s.shape).astype(np.float32)

            with torch.no_grad():
                sc = torch.from_numpy(s).float().unsqueeze(0).to(DEVICE)
                sn = torch.from_numpy(s_noisy).float().unsqueeze(0).to(DEVICE)
                k_c = int(predictor.predict_horizon(sc, k_min=5, k_max=50).item())
                k_n = int(predictor.predict_horizon(sn, k_min=5, k_max=50).item())

            k_clean_list.append(k_c)
            k_noisy_list.append(k_n)
            total_points += 1
            if abs(k_n - k_c) >= delta_k:
                frag_points += 1
                traj_frag = True
        if traj_frag:
            frag_trajs += 1
        traj_count += 1
        mean_k_clean_all.append(np.mean(k_clean_list))
        mean_k_noisy_all.append(np.mean(k_noisy_list))

    return {
        'fragility_point_rate': frag_points / max(1, total_points),
        'fragility_trajectory_rate': frag_trajs / max(1, traj_count),
        'mean_k_clean': float(np.mean(mean_k_clean_all)),
        'mean_k_noisy': float(np.mean(mean_k_noisy_all)),
        'checked_points': int(total_points),
        'checked_trajectories': int(traj_count)
    }


def main():
    # load test demos
    test_demos = load_test_trajectories(DATA_HDF5, train_ratio=0.8, max_test_traj=50)
    sample_state = list(test_demos.values())[0]['states'][0]
    state_dim = len(sample_state)

    rows = []
    for lam in LAMBDAS:
        predictor_path = os.path.join(OUT_DIR, f'lambda_{lam}', 'model.pth')
        if not os.path.exists(predictor_path):
            print(f"predictor not found for lambda={lam}: {predictor_path}")
            continue
        predictor = load_predictor(predictor_path, state_dim)
        res = probe_predictor_fragility(predictor, test_demos)
        res['lambda'] = lam
        rows.append(res)
        print(f"lambda={lam} -> frag_point_rate={res['fragility_point_rate']:.3f}, frag_traj_rate={res['fragility_trajectory_rate']:.3f}, mean_k={res['mean_k_clean']:.2f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'noise_sensitivity_summary.csv')
    df.to_csv(csv_path, index=False)

    # plot
    fig, ax1 = plt.subplots(figsize=(5,3.5))
    ax2 = ax1.twinx()
    ax1.plot(df['lambda'], df['mean_k_clean'], '-o', color='C0', label='Mean k (clean)')
    ax2.plot(df['lambda'], df['fragility_trajectory_rate']*100, '-s', color='C1', label='Fragile trajectories (%)')
    ax1.set_xlabel('Safety coefficient $\\lambda$')
    ax1.set_ylabel('Mean predicted horizon $k$ (clean)', color='C0')
    ax2.set_ylabel('Fragile trajectories (%)', color='C1')
    ax1.set_xticks(df['lambda'])
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, 'noise_sensitivity.png')
    plt.savefig(out_png, dpi=300)
    print('Wrote', csv_path, out_png)


if __name__ == '__main__':
    main()
