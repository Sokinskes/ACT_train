"""
AdaStep 分析与可视化工具

功能:
1. 分析评估轨迹中的步长分布
2. 对比固定步长 vs 自适应步长的性能
3. 可视化步长随时间的变化
4. 分析不同状态下的步长选择
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from pathlib import Path
import pickle


def load_trajectory(hdf5_path):
    """加载轨迹数据"""
    with h5py.File(hdf5_path, 'r') as f:
        data = {
            'qpos': f['/observations/qpos'][()],
            'qvel': f['/observations/qvel'][()],
            'action': f['/action'][()],
        }
        
        # 如果有步长历史
        if 'horizon_history' in f:
            data['horizon_history'] = f['horizon_history'][()]
        else:
            data['horizon_history'] = None
    
    return data


def plot_horizon_over_time(horizon_history, save_path):
    """可视化步长随时间的变化"""
    if horizon_history is None:
        print("⚠️  该轨迹没有步长历史记录（可能是固定步长模式）")
        return
    
    plt.figure(figsize=(12, 6))
    
    # 计算累积时间步
    cumulative_steps = np.cumsum(horizon_history)
    inference_points = cumulative_steps - horizon_history
    
    # 绘制步长曲线
    plt.subplot(2, 1, 1)
    plt.plot(inference_points, horizon_history, 'o-', markersize=4, linewidth=1.5)
    plt.axhline(y=np.mean(horizon_history), color='r', linestyle='--', 
               label=f'平均: {np.mean(horizon_history):.1f}')
    plt.xlabel('时间步 (t)')
    plt.ylabel('预测步长 (k)')
    plt.title('自适应步长随时间变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制步长分布直方图
    plt.subplot(2, 1, 2)
    plt.hist(horizon_history, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(np.mean(horizon_history), color='r', linestyle='--', 
               label=f'均值: {np.mean(horizon_history):.1f}')
    plt.axvline(np.median(horizon_history), color='g', linestyle='--', 
               label=f'中位数: {np.median(horizon_history):.1f}')
    plt.xlabel('步长 (k)')
    plt.ylabel('频次')
    plt.title('步长分布直方图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 步长时序图已保存: {save_path}")


def plot_state_horizon_correlation(qpos, horizon_history, save_path):
    """分析状态与步长的相关性"""
    if horizon_history is None:
        return
    
    # 计算每次推理时的状态
    cumulative_steps = np.cumsum(horizon_history)
    inference_points = cumulative_steps - horizon_history
    
    # 提取推理时刻的状态
    states_at_inference = []
    for t in inference_points:
        if t < len(qpos):
            states_at_inference.append(qpos[int(t)])
    
    states_at_inference = np.array(states_at_inference)
    
    if len(states_at_inference) == 0:
        return
    
    # 计算状态变化率（速度）
    state_velocities = np.linalg.norm(np.diff(states_at_inference, axis=0), axis=1)
    state_velocities = np.concatenate([[0], state_velocities])
    
    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 状态空间的前2个维度 vs 步长
    ax1 = axes[0]
    scatter = ax1.scatter(states_at_inference[:, 0], states_at_inference[:, 1], 
                         c=horizon_history[:len(states_at_inference)], 
                         cmap='RdYlGn', s=50, alpha=0.6)
    ax1.set_xlabel('状态维度 1')
    ax1.set_ylabel('状态维度 2')
    ax1.set_title('状态空间与步长的关系')
    plt.colorbar(scatter, ax=ax1, label='步长 (k)')
    ax1.grid(True, alpha=0.3)
    
    # 状态变化率 vs 步长
    ax2 = axes[1]
    ax2.scatter(state_velocities, horizon_history[:len(states_at_inference)], 
               alpha=0.6, s=30)
    ax2.set_xlabel('状态变化率 (速度)')
    ax2.set_ylabel('预测步长 (k)')
    ax2.set_title('状态变化率与步长的关系')
    ax2.grid(True, alpha=0.3)
    
    # 拟合趋势线
    if len(state_velocities) > 1:
        z = np.polyfit(state_velocities, horizon_history[:len(states_at_inference)], 1)
        p = np.poly1d(z)
        ax2.plot(state_velocities, p(state_velocities), "r--", alpha=0.8, 
                label=f'趋势: k = {z[0]:.2f}v + {z[1]:.2f}')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 状态-步长相关性图已保存: {save_path}")


def compare_performance(adaptive_data, fixed_data_list, save_path):
    """对比自适应步长与固定步长的性能"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 轨迹对比（状态空间）
    ax1 = axes[0, 0]
    ax1.plot(adaptive_data['qpos'][:, 0], adaptive_data['qpos'][:, 1], 
            'b-', linewidth=2, label='AdaStep', alpha=0.7)
    for i, fixed_data in enumerate(fixed_data_list):
        ax1.plot(fixed_data['qpos'][:, 0], fixed_data['qpos'][:, 1], 
                '--', linewidth=1.5, label=f'Fixed-{i}', alpha=0.5)
    ax1.set_xlabel('状态维度 1')
    ax1.set_ylabel('状态维度 2')
    ax1.set_title('轨迹对比（状态空间）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 动作幅度对比
    ax2 = axes[0, 1]
    adaptive_action_norm = np.linalg.norm(adaptive_data['action'], axis=1)
    ax2.plot(adaptive_action_norm, 'b-', linewidth=2, label='AdaStep', alpha=0.7)
    for i, fixed_data in enumerate(fixed_data_list):
        fixed_action_norm = np.linalg.norm(fixed_data['action'], axis=1)
        ax2.plot(fixed_action_norm, '--', linewidth=1.5, label=f'Fixed-{i}', alpha=0.5)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('动作幅度 ||a||')
    ax2.set_title('动作幅度对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 推理频率对比
    ax3 = axes[1, 0]
    if adaptive_data['horizon_history'] is not None:
        num_inferences_adaptive = len(adaptive_data['horizon_history'])
        avg_horizon = np.mean(adaptive_data['horizon_history'])
    else:
        num_inferences_adaptive = len(adaptive_data['qpos'])
        avg_horizon = 1
    
    methods = ['AdaStep']
    inferences = [num_inferences_adaptive]
    colors = ['steelblue']
    
    for i, fixed_data in enumerate(fixed_data_list):
        methods.append(f'Fixed-{i}')
        # 假设固定步长为某个值（需要从数据推断）
        fixed_horizon = len(fixed_data['qpos']) // 10  # 示例
        inferences.append(len(fixed_data['qpos']) // fixed_horizon)
        colors.append('coral')
    
    bars = ax3.bar(methods, inferences, color=colors, alpha=0.7)
    ax3.set_ylabel('推理次数')
    ax3.set_title('推理次数对比（越少越好）')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 4. 步长分布（仅 AdaStep）
    ax4 = axes[1, 1]
    if adaptive_data['horizon_history'] is not None:
        ax4.hist(adaptive_data['horizon_history'], bins=20, 
                alpha=0.7, color='steelblue', edgecolor='black')
        ax4.axvline(avg_horizon, color='r', linestyle='--', 
                   label=f'均值: {avg_horizon:.1f}')
        ax4.set_xlabel('步长 (k)')
        ax4.set_ylabel('频次')
        ax4.set_title('AdaStep 步长分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No AdaStep Data', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 性能对比图已保存: {save_path}")


def analyze_efficiency(horizon_history, episode_len, k_min, k_max):
    """分析效率提升"""
    if horizon_history is None:
        print("⚠️  无步长历史数据")
        return
    
    print("\n" + "="*60)
    print("效率分析报告")
    print("="*60 + "\n")
    
    # 实际推理次数
    actual_inferences = len(horizon_history)
    avg_horizon = np.mean(horizon_history)
    
    # 对比基线
    baseline_min = episode_len // k_min  # 最小步长的推理次数
    baseline_max = episode_len // k_max  # 最大步长的推理次数
    baseline_avg = episode_len // int((k_min + k_max) / 2)  # 平均步长
    
    print(f"📊 推理次数对比:")
    print(f"  固定步长 k={k_min} (高频): {baseline_min} 次")
    print(f"  固定步长 k={k_max} (低频): {baseline_max} 次")
    print(f"  固定步长 k={(k_min+k_max)//2} (中等): {baseline_avg} 次")
    print(f"  自适应步长 (AdaStep): {actual_inferences} 次")
    
    print(f"\n⚡ 效率提升:")
    savings_vs_min = (baseline_min - actual_inferences) / baseline_min * 100
    savings_vs_avg = (baseline_avg - actual_inferences) / baseline_avg * 100
    print(f"  相比固定最小步长: 节省 {savings_vs_min:.1f}%")
    print(f"  相比固定平均步长: 节省 {savings_vs_avg:.1f}%")
    
    print(f"\n📈 步长统计:")
    print(f"  平均步长: {avg_horizon:.2f}")
    print(f"  最小步长: {np.min(horizon_history)}")
    print(f"  最大步长: {np.max(horizon_history)}")
    print(f"  标准差: {np.std(horizon_history):.2f}")
    
    # 步长分布
    unique, counts = np.unique(horizon_history, return_counts=True)
    print(f"\n🎯 步长分布:")
    for k, count in zip(unique, counts):
        percentage = count / len(horizon_history) * 100
        print(f"  k={int(k):2d}: {count:3d} 次 ({percentage:5.1f}%)")
    
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=str, required=True,
                       help='AdaStep轨迹文件路径 (.hdf5)')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='输出目录')
    parser.add_argument('--k_min', type=int, default=5)
    parser.add_argument('--k_max', type=int, default=50)
    parser.add_argument('--compare', type=str, nargs='*',
                       help='用于对比的固定步长轨迹文件')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载主轨迹
    print(f"\n📂 加载轨迹: {args.trajectory}")
    adaptive_data = load_trajectory(args.trajectory)
    
    # 生成基础分析图
    base_name = Path(args.trajectory).stem
    
    # 1. 步长时序图
    plot_horizon_over_time(
        adaptive_data['horizon_history'],
        os.path.join(args.output_dir, f'{base_name}_horizon_time.png')
    )
    
    # 2. 状态-步长相关性
    plot_state_horizon_correlation(
        adaptive_data['qpos'],
        adaptive_data['horizon_history'],
        os.path.join(args.output_dir, f'{base_name}_state_correlation.png')
    )
    
    # 3. 效率分析
    episode_len = len(adaptive_data['qpos'])
    analyze_efficiency(
        adaptive_data['horizon_history'],
        episode_len,
        args.k_min,
        args.k_max
    )
    
    # 4. 性能对比（如果提供了对比文件）
    if args.compare:
        print(f"\n📊 加载对比轨迹...")
        fixed_data_list = []
        for compare_path in args.compare:
            print(f"  - {compare_path}")
            fixed_data_list.append(load_trajectory(compare_path))
        
        compare_performance(
            adaptive_data,
            fixed_data_list,
            os.path.join(args.output_dir, f'{base_name}_comparison.png')
        )
    
    print(f"\n✓ 分析完成！结果保存在: {args.output_dir}\n")


if __name__ == '__main__':
    main()
