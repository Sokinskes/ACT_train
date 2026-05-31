"""
AdaStep-Enhanced Evaluation Script for ACT-Plus-Plus
Demonstrates plug-and-play integration of adaptive action chunking.

Key Changes from Original imitate_episodes.py:
1. Import AdaStepAdapter
2. Initialize adapter with policy reference
3. Replace fixed k with adaptive k_t prediction
"""

import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy import ACTPolicy
from imitate_episodes import make_policy, sample_box_pose, sample_insertion_pose
from sim_env import make_sim_env, BOX_POSE
from predictors.adastep import AdaStepAdapter
from constants import FPS, SIM_TASK_CONFIGS
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=True, help='Task name (e.g., sim_transfer_cube_scripted)')
    parser.add_argument('--ckpt_dir', required=True, help='Policy checkpoint directory')
    parser.add_argument('--ckpt_name', default='policy_last.ckpt', help='Policy checkpoint filename')
    parser.add_argument('--predictor_ckpt', default=None, help='AdaStep predictor checkpoint (optional)')
    parser.add_argument('--num_rollouts', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--k_min', type=int, default=5, help='Minimum horizon')
    parser.add_argument('--k_max', type=int, default=50, help='Maximum horizon')
    parser.add_argument('--use_adastep', action='store_true', help='Enable AdaStep adaptive chunking')
    parser.add_argument('--temporal_agg', action='store_true', help='Use temporal aggregation')
    parser.add_argument('--onscreen_render', action='store_true', help='Render on screen')
    parser.add_argument('--device', default='cuda', help='Device')
    return parser.parse_args()


from einops import rearrange

def get_image(ts, camera_names, rand_crop_resize=False):
    """Extract images from timestep."""
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().unsqueeze(0)
    return curr_image


def eval_adastep(args):
    """
    Evaluation with AdaStep adaptive action chunking.
    """
    # Load task config
    task_config = SIM_TASK_CONFIGS[args.task_name]
    camera_names = task_config['camera_names']
    max_timesteps = task_config['episode_len']
    
    # Load policy
    print("📦 Loading ACT policy...")
    # Read actual policy config used during training
    config_path = os.path.join(args.ckpt_dir, 'config.pkl')
    with open(config_path, 'rb') as f:
        training_config = pickle.load(f)
    policy_config = training_config['policy_config']
    camera_names = training_config['camera_names'] # Make sure it perfectly matches training
    
    policy = make_policy('ACT', policy_config)
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    policy.deserialize(torch.load(ckpt_path))
    policy.to(args.device)
    policy.eval()
    print(f"✓ Loaded policy from {ckpt_path}")
    
    # Load stats
    stats_path = os.path.join(args.ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    # Initialize AdaStep adapter
    adapter = None
    if args.use_adastep:
        print("\n🚀 Initializing AdaStep Adapter...")
        adapter = AdaStepAdapter(
            predictor_ckpt=args.predictor_ckpt,
            policy=policy,
            k_min=args.k_min,
            k_max=args.k_max,
            device=args.device
        )
        print(f"✓ AdaStep enabled (k ∈ [{args.k_min}, {args.k_max}])")
    else:
        print("⚠️  Running with fixed horizon baseline")
    
    # Create environment
    env = make_sim_env(args.task_name)
    query_frequency = policy_config['num_queries']
    if args.temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    
    # Evaluation loop
    episode_returns = []
    highest_rewards = []
    all_stats = []
    
    print(f"\n🔄 Running {args.num_rollouts} evaluation episodes...")
    
    for rollout_id in tqdm(range(args.num_rollouts), desc="Rollouts"):
        if 'sim_transfer_cube' in args.task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in args.task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
            
        ts = env.reset()
        
        if args.temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, 16]).to(args.device)
        
        image_list = []
        qpos_list = []
        rewards = []
        
        if adapter:
            adapter.reset_statistics()
        
        with torch.no_grad():
            for t in range(max_timesteps):
                # Process observation
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(args.device).unsqueeze(0)
                
                if t % query_frequency == 0:
                    curr_image = get_image(ts, camera_names, rand_crop_resize=False)
                    curr_image = curr_image.to(args.device)
                
                # Query policy
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                
                # Adaptive horizon selection
                if args.use_adastep and adapter and t % query_frequency == 0:
                    # AdaStep: predict adaptive horizon
                    k_t = adapter.predict_horizon(qpos, curr_image)
                else:
                    # Fixed baseline: use full query_frequency
                    k_t = query_frequency
                
                # Temporal aggregation
                if args.temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).to(args.device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    # Use predicted horizon
                    raw_action = all_actions[:, min(t % query_frequency, k_t - 1)]
                
                # Post-process and execute
                action = post_process(raw_action.squeeze(0).cpu().numpy())
                ts = env.step(action)
                rewards.append(ts.reward)
                
                if ts.last():
                    break
        
        # Episode statistics
        episode_return = np.sum(rewards)
        highest_reward = np.max(rewards)
        episode_returns.append(episode_return)
        highest_rewards.append(highest_reward)
        
        if adapter:
            stats = adapter.get_statistics()
            all_stats.append(stats)
    
    # Summary
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"Success Rate: {np.mean(np.array(highest_rewards) == env.task.max_reward) * 100:.1f}%")
    print(f"Average Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    
    if args.use_adastep and all_stats:
        # AdaStep-specific metrics
        mean_k = np.mean([s['mean_k'] for s in all_stats if s])
        mean_entropy = np.mean([s['entropy'] for s in all_stats if s])
        mean_reduction = np.mean([s['inference_reduction'] for s in all_stats if s])
        
        print(f"\n📊 AdaStep Statistics:")
        print(f"  Mean Horizon: {mean_k:.2f}")
        print(f"  Entropy: {mean_entropy:.3f}")
        print(f"  Inference Reduction: {mean_reduction:.1f}%")
        print(f"  Speedup: {100 / (100 - mean_reduction):.2f}×")
    
    print("=" * 60)


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("  AdaStep-Enhanced Evaluation")
    print("=" * 60)
    print(f"Task: {args.task_name}")
    print(f"AdaStep: {'Enabled' if args.use_adastep else 'Disabled (Baseline)'}")
    print(f"Rollouts: {args.num_rollouts}")
    
    eval_adastep(args)


if __name__ == '__main__':
    main()
