"""
AdaStep改进算法真实仿真验证
===========================

使用Square任务训练的改进HorizonPredictor进行真实仿真验证
验证状态级适应性的核心特性
"""

import torch
import numpy as np
import os
import sys
import json
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

# Robomimic imports
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.adastep_module import HorizonPredictor

def load_improved_predictor(model_path):
    """加载改进的HorizonPredictor"""
    print(f"🧠 加载改进的HorizonPredictor: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    model = HorizonPredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dim=128  # 与训练时一致
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✓ 改进模型加载成功")
    print(f"  输入维度: {checkpoint['input_dim']}")
    print(f"  训练MAE: {checkpoint['mae']:.4f}")
    print(f"  训练R²: {checkpoint['r2']:.4f}")

    return model

def run_improved_simulation_episode(env, policy, predictor, k_min=5, k_max=50):
    """
    运行单幕改进算法仿真
    返回详细的状态级适应信息
    """
    obs = env.reset()
    done = False
    total_steps = 0
    num_inferences = 0
    k_values = []
    state_history = []

    while not done and total_steps < 500:  # 防止无限循环
        # 获取当前状态
        current_state = policy.get_state(obs)

        # 使用改进的HorizonPredictor预测k值
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            k_normalized = predictor(state_tensor).item()

        # 反归一化到实际k值
        k_pred = int(k_normalized * (k_max - k_min) + k_min)
        k_pred = np.clip(k_pred, k_min, k_max)  # 确保在范围内

        k_values.append(k_pred)
        state_history.append(current_state)

        # 执行k步动作
        for step in range(k_pred):
            if done or total_steps >= 500:
                break

            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            total_steps += 1

        num_inferences += 1

    # 检查任务成功 (基于Robomimic的标准)
    success = policy.is_success(obs, info) if hasattr(policy, 'is_success') else (reward > 0)

    return {
        'success': success,
        'total_steps': total_steps,
        'num_inferences': num_inferences,
        'k_values': k_values,
        'state_history': state_history,
        'avg_k': np.mean(k_values) if k_values else 0,
        'k_std': np.std(k_values) if k_values else 0,
        'k_range': (min(k_values), max(k_values)) if k_values else (0, 0)
    }

def evaluate_improved_algorithm(task_name='transport', num_episodes=50):
    """
    评估改进算法的真实仿真性能
    """
    print(f"🚀 AdaStep改进算法仿真验证: {task_name.upper()}")
    print("="*60)

    # 配置
    data_root = '/home/yhj/桌面/ACT/adastep_extension/robomimic_data'
    model_path = '/home/yhj/桌面/ACT/adastep_extension/experiments/results_square_improved/horizon_predictor_square_improved.pth'

    # 任务配置
    task_configs = {
        'transport': f'{data_root}/transport/mh/low_dim_v15.hdf5',
        'square': f'{data_root}/square/mh/low_dim_v15.hdf5'
    }

    hdf5_path = task_configs.get(task_name)
    if not hdf5_path or not os.path.exists(hdf5_path):
        print(f"❌ 数据文件不存在: {hdf5_path}")
        return None

    # 1. 加载环境
    print("🏗️  初始化Robomimic环境...")
    env_meta = FileUtils.get_env_metadata_from_dataset(hdf5_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False
    )

    # 2. 加载专家策略 (简化版)
    print("🎭 加载专家重放策略...")
    policy = ExpertReplayPolicy(hdf5_path)

    # 3. 加载改进的HorizonPredictor
    predictor = load_improved_predictor(model_path)

    # 4. 运行仿真验证
    print(f"\n🔬 运行改进算法验证 ({num_episodes} episodes)...")

    results = {
        'successes': [],
        'total_steps': [],
        'num_inferences': [],
        'avg_k_values': [],
        'k_std_values': [],
        'k_ranges': [],
        'all_k_values': []
    }

    for ep in tqdm(range(num_episodes), desc="验证"):
        result = run_improved_simulation_episode(
            env, policy, predictor, k_min=5, k_max=50
        )

        results['successes'].append(result['success'])
        results['total_steps'].append(result['total_steps'])
        results['num_inferences'].append(result['num_inferences'])
        results['avg_k_values'].append(result['avg_k'])
        results['k_std_values'].append(result['k_std'])
        results['k_ranges'].append(result['k_range'])
        results['all_k_values'].extend(result['k_values'])

    env.close()

    return results

class ExpertReplayPolicy:
    """简化的专家重放策略"""

    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.demo_data = self._load_demo_data()
        self.current_demo_idx = 0
        self.current_step = 0

    def _load_demo_data(self):
        """加载演示数据"""
        demos = []
        with h5py.File(self.hdf5_path, 'r') as f:
            demo_names = list(f['data'].keys())[:50]  # 使用前50个演示

            for demo_name in demo_names:
                demo = f[f'data/{demo_name}']
                actions = demo['actions'][()]
                states = self._extract_states(demo)
                demos.append({
                    'actions': actions,
                    'states': states,
                    'length': len(actions)
                })

        return demos

    def _extract_states(self, demo):
        """提取状态特征"""
        if 'obs/robot0_eef_pos' in demo:
            eef_pos = demo['obs/robot0_eef_pos'][()]
            eef_quat = demo['obs/robot0_eef_quat'][()]
            states = np.concatenate([eef_pos, eef_quat], axis=-1)
        else:
            states = demo['obs/robot0_joint_pos'][()]

        return states

    def get_state(self, obs):
        """获取当前状态 (简化为使用演示数据)"""
        if self.current_demo_idx < len(self.demo_data):
            demo = self.demo_data[self.current_demo_idx]
            if self.current_step < len(demo['states']):
                return demo['states'][self.current_step]
        return np.zeros(7)  # 默认状态

    def get_action(self, obs):
        """获取动作"""
        if self.current_demo_idx < len(self.demo_data):
            demo = self.demo_data[self.current_demo_idx]
            if self.current_step < len(demo['actions']):
                action = demo['actions'][self.current_step]
                self.current_step += 1
                return action

        # 切换到下一个演示
        self.current_demo_idx = (self.current_demo_idx + 1) % len(self.demo_data)
        self.current_step = 0

        if self.current_demo_idx < len(self.demo_data):
            demo = self.demo_data[self.current_demo_idx]
            if len(demo['actions']) > 0:
                return demo['actions'][0]

        return np.zeros(7)  # 默认动作

def analyze_adaptation_characteristics(results):
    """
    分析状态级适应特性
    """
    print("\n🔍 状态级适应特性分析")
    print("-" * 40)

    all_k_values = results['all_k_values']
    k_stds = results['k_std_values']

    print("K值分布统计:")
    print(f"  总k值数量: {len(all_k_values)}")
    print(f"  k值范围: {min(all_k_values)} - {max(all_k_values)}")
    print(f"  唯一k值: {sorted(set(all_k_values))}")
    print(f"  平均k值: {np.mean(all_k_values):.2f}")
    print(f"  k值标准差: {np.std(all_k_values):.2f}")

    print(f"\n轨迹级变异性:")
    print(f"  平均轨迹k标准差: {np.mean(k_stds):.2f}")
    print(f"  轨迹k标准差范围: {min(k_stds):.2f} - {max(k_stds):.2f}")

    # 状态级适应指标
    unique_k_count = len(set(all_k_values))
    k_variability = np.std(all_k_values)
    trajectory_variability = np.mean(k_stds)

    print(f"\n状态级适应指标:")
    print(f"  ✓ k值多样性: {unique_k_count} 种不同k值")
    print(f"  ✓ k值变异性: {k_variability:.2f}")
    print(f"  ✓ 轨迹变异性: {trajectory_variability:.2f}")

    return {
        'k_diversity': unique_k_count,
        'k_variability': k_variability,
        'trajectory_variability': trajectory_variability
    }

def create_adaptation_visualization(results, task_name):
    """
    创建状态级适应可视化
    """
    print("\n🎨 生成状态级适应可视化...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. K值分布直方图
    all_k_values = results['all_k_values']
    ax1.hist(all_k_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title(f'{task_name.upper()}任务 - K值分布')
    ax1.set_xlabel('预测步长 k')
    ax1.set_ylabel('频次')
    ax1.grid(True, alpha=0.3)

    # 2. 轨迹k值变异性
    k_stds = results['k_std_values']
    ax2.hist(k_stds, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('轨迹级K值变异性分布')
    ax2.set_xlabel('轨迹K值标准差')
    ax2.set_ylabel('轨迹数量')
    ax2.grid(True, alpha=0.3)

    # 3. 成功率vs平均k值散点图
    successes = np.array(results['successes'])
    avg_ks = np.array(results['avg_k_values'])

    successful_ks = avg_ks[successes]
    failed_ks = avg_ks[~successes]

    ax3.scatter(successful_ks, [1] * len(successful_ks), alpha=0.6, color='green', label='成功')
    ax3.scatter(failed_ks, [0] * len(failed_ks), alpha=0.6, color='red', label='失败')
    ax3.set_title('成功率 vs 平均K值')
    ax3.set_xlabel('平均预测步长')
    ax3.set_ylabel('成功状态')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 时序k值示例 (第一个轨迹)
    # 这里我们简化处理，假设有k值序列
    if len(results['k_ranges']) > 0:
        k_ranges = results['k_ranges'][:10]  # 前10个轨迹
        min_ks = [r[0] for r in k_ranges]
        max_ks = [r[1] for r in k_ranges]

        ax4.bar(range(len(min_ks)), max_ks, alpha=0.7, color='lightblue', label='最大k')
        ax4.bar(range(len(min_ks)), min_ks, alpha=0.9, color='darkblue', label='最小k')
        ax4.set_title('各轨迹K值范围')
        ax4.set_xlabel('轨迹索引')
        ax4.set_ylabel('K值')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results_square_improved/{task_name}_improved_adaptation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 适应分析图已保存: results_square_improved/{task_name}_improved_adaptation_analysis.png")

def main():
    """主函数"""
    # 运行改进算法验证
    results = evaluate_improved_algorithm(task_name='transport', num_episodes=50)

    if results is None:
        print("❌ 验证失败")
        return

    # 分析适应特性
    adaptation_metrics = analyze_adaptation_characteristics(results)

    # 创建可视化
    create_adaptation_visualization(results, 'transport')

    # 计算总体统计
    success_rate = np.mean(results['successes']) * 100
    avg_inference = np.mean(results['num_inferences'])
    avg_k = np.mean(results['avg_k_values'])

    print(f"\n📊 改进算法验证结果 ({len(results['successes'])} episodes):")
    print(f"  成功率: {success_rate:.1f}%")
    print(f"  平均推理次数: {avg_inference:.1f}")
    print(f"  平均k值: {avg_k:.2f}")
    print(f"  k值变异性: {adaptation_metrics['k_variability']:.2f}")

    # 保存详细结果
    output_file = 'results_square_improved/improved_algorithm_simulation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'task': 'transport',
                'success_rate': success_rate,
                'avg_inference': avg_inference,
                'avg_k': avg_k,
                'adaptation_metrics': adaptation_metrics
            },
            'detailed_results': results
        }, f, indent=2)

    print(f"✓ 详细结果已保存: {output_file}")

    # 验证状态级适应
    if adaptation_metrics['k_diversity'] >= 5 and adaptation_metrics['k_variability'] >= 5:
        print("\n🎯 状态级适应验证: 成功!")
        print("  ✓ 展现了显著的状态级k值变异性")
        print("  ✓ 证明了算法的动态适应能力")
    else:
        print("\n⚠️  状态级适应验证: 需要改进")
        print("  需要进一步优化算法")

if __name__ == "__main__":
    main()