"""
真实物理仿真评估 - AdaStep核心验证
=================================

目标: 在Robomimic MuJoCo环境中验证AdaStep的真实成功率

方法:
1. 使用专家演示数据的ACT策略 (来自数据集)
2. 使用训练好的HorizonPredictor预测步长k
3. 在仿真环境中闭环执行并统计成功率

关键特性:
- 真实物理仿真 (MuJoCo)
- Robomimic官方环境和成功判定
- 对比AdaStep vs 固定步长Baseline
"""

import torch
import numpy as np
import os
import sys
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
import h5py

# Robomimic imports
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.adastep_module import HorizonPredictor


class ExpertReplayPolicy:
    """
    专家重放策略
    
    直接从数据集中提取专家动作序列,模拟ACT行为:
    - 一次推理预测100步
    - 根据AdaStep的k值执行其中k步
    """
    
    def __init__(self, hdf5_path, device='cuda'):
        self.device = device
        self.hdf5_path = hdf5_path
        
        # 加载所有演示数据
        print(f"📂 加载专家演示: {hdf5_path}")
        with h5py.File(hdf5_path, 'r') as f:
            self.demos = {}
            demo_keys = list(f['data'].keys())
            
            for demo_key in demo_keys:
                demo = f[f'data/{demo_key}']
                
                # 提取动作序列
                actions = demo['actions'][()]
                
                # 提取状态 (用于特征提取)
                if 'obs/robot0_eef_pos' in demo:
                    eef_pos = demo['obs/robot0_eef_pos'][()]
                    eef_quat = demo['obs/robot0_eef_quat'][()]
                    states = np.concatenate([eef_pos, eef_quat], axis=-1)
                else:
                    states = demo['obs/robot0_joint_pos'][()]
                
                self.demos[demo_key] = {
                    'actions': actions,
                    'states': states
                }
        
        print(f"✓ 加载 {len(self.demos)} 条演示轨迹")
        
        # 当前使用的演示索引
        self.current_demo_idx = 0
        self.demo_keys = list(self.demos.keys())
    
    def reset_episode(self):
        """每个episode开始时切换演示"""
        self.current_demo_idx = (self.current_demo_idx + 1) % len(self.demo_keys)
        self.current_step = 0
    
    def get_action_sequence(self, obs, chunk_size=100):
        """
        获取动作序列 (模拟ACT一次推理)
        
        Args:
            obs: 当前观测 (未使用,因为是开环重放)
            chunk_size: 动作块大小 (ACT的num_queries)
        
        Returns:
            actions: [chunk_size, action_dim]
            state_feature: 状态特征 (用于HorizonPredictor)
        """
        demo_key = self.demo_keys[self.current_demo_idx]
        demo = self.demos[demo_key]
        
        # 提取动作
        start_idx = self.current_step
        end_idx = min(start_idx + chunk_size, len(demo['actions']))
        
        actions = demo['actions'][start_idx:end_idx]
        
        # Padding (如果不足chunk_size)
        if len(actions) < chunk_size:
            pad_length = chunk_size - len(actions)
            actions = np.concatenate([
                actions,
                np.zeros((pad_length, actions.shape[1]))
            ], axis=0)
        
        # 提取状态特征
        if start_idx < len(demo['states']):
            state = demo['states'][start_idx]
        else:
            state = demo['states'][-1]
        
        # 转换为Tensor
        actions_tensor = torch.from_numpy(actions).float().to(self.device)
        state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        
        return actions_tensor, state_tensor


def run_simulation_episode(env, policy, horizon_predictor, 
                           k_min=5, k_max=50, max_steps=500,
                           use_adastep=True):
    """
    运行单个仿真episode
    
    Returns:
        success: bool - 是否成功
        num_inferences: int - 推理次数
        total_steps: int - 总步数
        k_values: list - 每次推理的k值
    """
    obs = env.reset()
    policy.reset_episode()
    
    done = False
    success = False
    step = 0
    num_inferences = 0
    k_values = []
    
    action_buffer = []
    
    while step < max_steps and not done:
        # === 需要推理 ===
        if len(action_buffer) == 0:
            # 1. 获取专家动作序列
            action_seq, state_feature = policy.get_action_sequence(obs, chunk_size=100)
            
            # 2. 预测步长k
            if use_adastep:
                with torch.no_grad():
                    k_pred = horizon_predictor.predict_horizon(
                        state_feature, k_min=k_min, k_max=k_max
                    )
                    k = int(k_pred.item())
            else:
                # Baseline: 固定步长
                k = 1  # ACT原始: 每步都推理
            
            k_values.append(k)
            
            # 3. 截取前k步
            action_buffer = action_seq[:k].cpu().numpy()
            
            num_inferences += 1
        
        # === 执行动作 ===
        if len(action_buffer) > 0:
            action = action_buffer[0]
            action_buffer = action_buffer[1:]
        else:
            # 异常保护
            action = np.zeros(env.action_dim)
        
        obs, reward, done, info = env.step(action)
        step += 1
        policy.current_step += 1
        
        # 检查成功
        if info.get('success', False):
            success = True
            break
    
    return {
        'success': success,
        'num_inferences': num_inferences,
        'total_steps': step,
        'k_values': k_values
    }


def evaluate_task(hdf5_path, predictor_path, 
                  num_episodes=50, k_min=5, k_max=50,
                  device='cuda'):
    """
    评估单个任务
    
    Returns:
        results_adastep: AdaStep结果
        results_baseline: Baseline结果
    """
    
    # 1. 加载环境元数据
    print(f"\n{'='*60}")
    print(f"📝 任务: {os.path.basename(hdf5_path)}")
    print(f"{'='*60}")
    
    env_meta = FileUtils.get_env_metadata_from_dataset(hdf5_path)
    
    # 2. 创建环境
    print("🌍 创建仿真环境...")
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False  # 使用低维状态 (更快)
    )
    print(f"✓ 环境: {env.name}")
    
    # 3. 加载策略
    print("🤖 加载专家策略...")
    policy = ExpertReplayPolicy(hdf5_path, device=device)
    
    # 4. 加载HorizonPredictor
    print(f"🧠 加载HorizonPredictor: {predictor_path}")
    
    # 获取状态维度
    sample_state = policy.demos[policy.demo_keys[0]]['states'][0]
    state_dim = len(sample_state)
    
    horizon_predictor = HorizonPredictor(
        input_dim=state_dim,
        hidden_dim=256
    ).to(device)
    
    if os.path.exists(predictor_path):
        horizon_predictor.load_state_dict(torch.load(predictor_path))
        horizon_predictor.eval()
        print(f"✓ 模型加载成功 (state_dim={state_dim})")
    else:
        print(f"⚠️  未找到模型,使用随机初始化")
    
    # 5. 运行AdaStep评估
    print(f"\n🚀 [1/2] 运行AdaStep评估 ({num_episodes} episodes)...")
    results_adastep = {
        'successes': [],
        'inferences': [],
        'steps': [],
        'k_values': []
    }
    
    for ep in tqdm(range(num_episodes), desc="AdaStep"):
        result = run_simulation_episode(
            env, policy, horizon_predictor,
            k_min=k_min, k_max=k_max,
            use_adastep=True
        )
        
        results_adastep['successes'].append(result['success'])
        results_adastep['inferences'].append(result['num_inferences'])
        results_adastep['steps'].append(result['total_steps'])
        results_adastep['k_values'].extend(result['k_values'])
    
    # 6. 运行Baseline评估
    print(f"\n🚀 [2/2] 运行Baseline评估 ({num_episodes} episodes)...")
    results_baseline = {
        'successes': [],
        'inferences': [],
        'steps': []
    }
    
    for ep in tqdm(range(num_episodes), desc="Baseline"):
        result = run_simulation_episode(
            env, policy, horizon_predictor,
            k_min=1, k_max=1,  # 固定k=1
            use_adastep=False
        )
        
        results_baseline['successes'].append(result['success'])
        results_baseline['inferences'].append(result['num_inferences'])
        results_baseline['steps'].append(result['total_steps'])
    
    env.close()
    
    return results_adastep, results_baseline


def print_results(results_adastep, results_baseline, task_name):
    """打印对比结果"""
    
    # AdaStep统计
    success_rate_ada = np.mean(results_adastep['successes']) * 100
    avg_inferences_ada = np.mean(results_adastep['inferences'])
    avg_steps_ada = np.mean(results_adastep['steps'])
    avg_k_ada = np.mean(results_adastep['k_values'])
    
    # Baseline统计
    success_rate_base = np.mean(results_baseline['successes']) * 100
    avg_inferences_base = np.mean(results_baseline['inferences'])
    avg_steps_base = np.mean(results_baseline['steps'])
    
    # 推理节省率
    inference_savings = (1 - avg_inferences_ada / avg_inferences_base) * 100
    
    print(f"\n{'='*60}")
    print(f"📊 实验结果: {task_name}")
    print(f"{'='*60}")
    print(f"\n【AdaStep】")
    print(f"  成功率:       {success_rate_ada:.1f}%")
    print(f"  推理次数:     {avg_inferences_ada:.1f}")
    print(f"  总步数:       {avg_steps_ada:.1f}")
    print(f"  平均步长(k):  {avg_k_ada:.1f}")
    
    print(f"\n【Baseline (k=1)】")
    print(f"  成功率:       {success_rate_base:.1f}%")
    print(f"  推理次数:     {avg_inferences_base:.1f}")
    print(f"  总步数:       {avg_steps_base:.1f}")
    
    print(f"\n【对比】")
    print(f"  成功率差异:   {success_rate_ada - success_rate_base:+.1f}%")
    print(f"  推理节省率:   {inference_savings:.1f}%")
    print(f"{'='*60}\n")
    
    return {
        'task': task_name,
        'adastep_success': success_rate_ada,
        'baseline_success': success_rate_base,
        'avg_k': avg_k_ada,
        'inference_savings': inference_savings
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AdaStep真实仿真评估')
    parser.add_argument('--task', type=str, required=True,
                       choices=['transport', 'can', 'lift', 'square'],
                       help='任务名称')
    parser.add_argument('--num_episodes', type=int, default=50,
                       help='评估episode数量')
    parser.add_argument('--k_min', type=int, default=5,
                       help='最小步长')
    parser.add_argument('--k_max', type=int, default=50,
                       help='最大步长')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 数据路径映射
    data_root = '/home/yhj/桌面/ACT/adastep_extension/robomimic_data'
    
    task_configs = {
        'transport': {
            'hdf5': f'{data_root}/transport/mh/low_dim_v141.hdf5',
            'predictor': 'results_transport_mh/stage2_training/best_predictor.pth'
        },
        'can': {
            'hdf5': f'{data_root}/can/mh/low_dim.hdf5',
            'predictor': 'results_can_mh/stage2_training/best_predictor.pth'
        },
        'lift': {
            'hdf5': f'{data_root}/lift/mh/low_dim_v141.hdf5',
            'predictor': 'results_lift_optimized/stage2_training/best_predictor.pth'
        },
        'square': {
            'hdf5': f'{data_root}/square/mh/low_dim_v141.hdf5',
            'predictor': 'results_square_mh/stage2_training/best_predictor.pth'
        }
    }
    
    config = task_configs[args.task]
    
    # 检查文件
    if not os.path.exists(config['hdf5']):
        print(f"❌ 数据文件不存在: {config['hdf5']}")
        # 尝试查找替代文件
        task_dir = os.path.dirname(config['hdf5'])
        if os.path.exists(task_dir):
            files = [f for f in os.listdir(task_dir) if f.endswith('.hdf5')]
            if files:
                config['hdf5'] = os.path.join(task_dir, files[0])
                print(f"✓ 使用替代文件: {config['hdf5']}")
            else:
                sys.exit(1)
        else:
            sys.exit(1)
    
    if not os.path.exists(config['predictor']):
        print(f"⚠️  预测器不存在: {config['predictor']}")
        print("将使用随机初始化的预测器 (仅用于测试流程)")
    
    # 运行评估
    results_ada, results_base = evaluate_task(
        hdf5_path=config['hdf5'],
        predictor_path=config['predictor'],
        num_episodes=args.num_episodes,
        k_min=args.k_min,
        k_max=args.k_max,
        device=args.device
    )
    
    # 打印结果
    summary = print_results(results_ada, results_base, args.task.upper())
    
    # 保存结果
    output_dir = f"results_{args.task}_simulation"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'simulation_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'summary': summary,
            'adastep': {
                'successes': [bool(s) for s in results_ada['successes']],
                'inferences': [int(i) for i in results_ada['inferences']],
                'steps': [int(s) for s in results_ada['steps']],
                'k_values': [float(k) for k in results_ada['k_values']]
            },
            'baseline': {
                'successes': [bool(s) for s in results_base['successes']],
                'inferences': [int(i) for i in results_base['inferences']],
                'steps': [int(s) for s in results_base['steps']]
            }
        }, f, indent=2)
    
    print(f"✓ 结果已保存: {output_file}")
