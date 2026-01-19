"""
AdaStep状态级自适应离线验证
============================

不依赖MuJoCo仿真环境，直接分析训练好的模型在现有轨迹上的表现

验证要点:
1. k值是否在轨迹的不同阶段正确调整
2. 复杂状态（接触/抓取）是否使用较小的k值
3. 简单状态（移动）是否使用较大的k值
"""

import torch
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adastep_module import HorizonPredictor

# 设置matplotlib
import matplotlib
matplotlib.use('Agg')

def load_trajectory_data(hdf5_path, trajectory_idx=0):
    """
    加载单条轨迹的数据用于分析

    Returns:
        states: [T, state_dim] - 状态序列
        actions: [T, action_dim] - 动作序列
        metadata: 轨迹元数据
    """
    with h5py.File(hdf5_path, 'r') as f:
        demos = list(f['data'].keys())
        if trajectory_idx >= len(demos):
            return None, None, None

        demo_name = demos[trajectory_idx]
        demo = f[f'data/{demo_name}']

        # 读取状态
        if 'obs/robot0_eef_pos' in demo:
            eef_pos = demo['obs/robot0_eef_pos'][()]
            eef_quat = demo['obs/robot0_eef_quat'][()]
            states = np.concatenate([eef_pos, eef_quat], axis=-1)
        elif 'obs/robot0_joint_pos' in demo:
            states = demo['obs/robot0_joint_pos'][()]
        else:
            raise ValueError("未找到有效的状态数据！")

        # 读取动作
        actions = demo['actions'][()]

        metadata = {
            'length': len(states),
            'state_dim': states.shape[1],
            'action_dim': actions.shape[1],
            'demo_name': demo_name
        }

        return states, actions, metadata

def analyze_trajectory_phases(states, actions):
    """
    分析轨迹的不同阶段

    使用启发式规则识别:
    - Reaching: 末端执行器远离物体
    - Grasping: 末端执行器接近物体
    - Transporting: 物体被移动

    Returns:
        phases: [T] - 每个时间步的阶段标签
    """
    phases = []

    # 简单的启发式: 基于末端执行器位置的变化
    eef_positions = states[:, :3]  # 假设前3维是位置

    # 计算移动速度
    velocities = np.linalg.norm(np.diff(eef_positions, axis=0), axis=1)
    velocities = np.concatenate([[0], velocities])  # 填充第一个时间步

    # 计算与初始位置的距离
    init_pos = eef_positions[0]
    distances_from_start = np.linalg.norm(eef_positions - init_pos, axis=1)

    for t in range(len(states)):
        velocity = velocities[t]
        distance = distances_from_start[t]

        if distance < 0.05:  # 接近起始位置
            phase = 'reaching'
        elif velocity < 0.01:  # 移动缓慢，可能在抓取
            phase = 'grasping'
        else:  # 移动中
            phase = 'transporting'

        phases.append(phase)

    return phases

def run_offline_validation(hdf5_path, predictor_path, trajectory_indices=None):
    """
    运行离线验证

    Args:
        hdf5_path: 数据文件路径
        predictor_path: 预测器模型路径
        trajectory_indices: 要分析的轨迹索引列表
    """

    print(f"\n{'='*80}")
    print(f"🚀 AdaStep状态级自适应离线验证 - Transport任务")
    print(f"{'='*80}")

    # 1. 加载HorizonPredictor
    print(f"🧠 加载HorizonPredictor: {predictor_path}")

    # 获取状态维度
    sample_states, _, _ = load_trajectory_data(hdf5_path, 0)
    if sample_states is None:
        print("❌ 无法加载轨迹数据")
        return None

    state_dim = sample_states.shape[1]
    print(f"✓ 状态维度: {state_dim}")

    horizon_predictor = HorizonPredictor(
        input_dim=state_dim,
        hidden_dim=256
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    horizon_predictor = horizon_predictor.to(device)

    if os.path.exists(predictor_path):
        checkpoint = torch.load(predictor_path, map_location=device)
        horizon_predictor.load_state_dict(checkpoint)
        horizon_predictor.eval()
        print("✓ 模型加载成功")
    else:
        print("❌ 模型文件不存在")
        return None

    # 2. 分析多个轨迹
    if trajectory_indices is None:
        trajectory_indices = [0, 1, 2, 3, 4]  # 分析前5条轨迹

    all_results = []

    for traj_idx in trajectory_indices:
        print(f"\n--- 分析轨迹 {traj_idx} ---")

        # 加载轨迹数据
        states, actions, metadata = load_trajectory_data(hdf5_path, traj_idx)
        if states is None:
            print(f"⚠️  跳过轨迹 {traj_idx} (无法加载)")
            continue

        print(f"✓ 轨迹长度: {metadata['length']} 步")

        # 分析轨迹阶段
        phases = analyze_trajectory_phases(states, actions)

        # 预测每个时间步的k值
        k_values = []
        phase_k_mapping = {'reaching': [], 'grasping': [], 'transporting': []}

        for t in range(len(states)):
            # 提取状态特征 (末端执行器位姿)
            if states.shape[1] >= 7:  # eef_pos + eef_quat
                state_feature = states[t, :7]
            else:
                state_feature = states[t]

            # 转换为tensor
            state_tensor = torch.from_numpy(state_feature).float().to(device).unsqueeze(0)

            # 预测k值
            with torch.no_grad():
                k_pred = horizon_predictor(state_tensor)
                k_normalized = torch.sigmoid(k_pred).item()
                k_value = int(5 + k_normalized * (50 - 5))  # 映射到[5,50]

            k_values.append(k_value)
            phase_k_mapping[phases[t]].append(k_value)

        # 计算阶段统计
        phase_stats = {}
        for phase, k_vals in phase_k_mapping.items():
            if k_vals:
                phase_stats[phase] = {
                    'mean': np.mean(k_vals),
                    'std': np.std(k_vals),
                    'count': len(k_vals),
                    'min': np.min(k_vals),
                    'max': np.max(k_vals)
                }
            else:
                phase_stats[phase] = {'mean': 0, 'std': 0, 'count': 0, 'min': 0, 'max': 0}

        # 整体统计
        k_values = np.array(k_values)
        trajectory_result = {
            'trajectory_idx': traj_idx,
            'length': len(k_values),
            'k_values': k_values.tolist(),
            'phases': phases,
            'phase_stats': phase_stats,
            'k_stats': {
                'mean': np.mean(k_values),
                'std': np.std(k_values),
                'unique_count': len(np.unique(k_values)),
                'min': np.min(k_values),
                'max': np.max(k_values)
            }
        }

        all_results.append(trajectory_result)

        print(f"  k值统计: 平均={trajectory_result['k_stats']['mean']:.1f}, "
              f"标准差={trajectory_result['k_stats']['std']:.1f}, "
              f"种类数={trajectory_result['k_stats']['unique_count']}")
        print(f"  阶段k值:")
        for phase, stats in phase_stats.items():
            if stats['count'] > 0:
                print(f"    {phase.capitalize()}: k={stats['mean']:.1f} ± {stats['std']:.1f} "
                      f"({stats['count']}步)")

    # 3. 整体分析
    print(f"\n{'='*80}")
    print("📊 整体验证结果分析")
    print(f"{'='*80}")

    # 聚合所有轨迹的阶段统计
    aggregated_phase_stats = {'reaching': [], 'grasping': [], 'transporting': []}

    for result in all_results:
        for phase, stats in result['phase_stats'].items():
            if stats['count'] > 0:
                aggregated_phase_stats[phase].append(stats['mean'])

    # 计算平均阶段k值
    avg_phase_k = {}
    for phase, means in aggregated_phase_stats.items():
        if means:
            avg_phase_k[phase] = {
                'mean': np.mean(means),
                'std': np.std(means),
                'trajectories': len(means)
            }
        else:
            avg_phase_k[phase] = {'mean': 0, 'std': 0, 'trajectories': 0}

    print("各阶段平均k值 (跨轨迹):")
    for phase, stats in avg_phase_k.items():
        if stats['trajectories'] > 0:
            print(f"  {phase.capitalize()}: k={stats['mean']:.1f} ± {stats['std']:.1f} "
                  f"({stats['trajectories']}条轨迹)")

    # 4. 验证状态级自适应
    validation_results = validate_adaptation(avg_phase_k, all_results)

    # 5. 生成可视化
    output_dir = Path("offline_validation_results")
    output_dir.mkdir(exist_ok=True)

    create_validation_plots(all_results, avg_phase_k, output_dir)

    # 保存结果
    results_summary = {
        'task': 'transport',
        'trajectories_analyzed': len(all_results),
        'phase_analysis': avg_phase_k,
        'validation': {k: bool(v) for k, v in validation_results.items()},  # 转换为bool类型
        'individual_results': all_results
    }

    output_file = output_dir / "transport_offline_validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # 自定义JSON编码器处理numpy类型
        def numpy_encoder(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=numpy_encoder)

    print(f"\n✓ 结果已保存: {output_file}")
    print(f"✓ 可视化图表已生成: {output_dir}")

    return results_summary

def validate_adaptation(avg_phase_k, all_results):
    """
    验证状态级自适应是否正确工作
    """
    validation = {}

    # 1. 检查grasping阶段是否有最低的k值
    grasping_k = avg_phase_k['grasping']['mean']
    reaching_k = avg_phase_k['reaching']['mean']
    transporting_k = avg_phase_k['transporting']['mean']

    validation['grasping_has_lowest_k'] = (
        grasping_k < reaching_k and grasping_k < transporting_k
    )

    # 2. 检查k值差异是否显著
    other_k_values = [reaching_k, transporting_k]
    other_k_values = [k for k in other_k_values if k > 0]
    if other_k_values:
        min_other_k = min(other_k_values)
        validation['significant_k_difference'] = (min_other_k - grasping_k) >= 5
    else:
        validation['significant_k_difference'] = False

    # 3. 检查整体k值多样性
    all_k_values = []
    for result in all_results:
        all_k_values.extend(result['k_values'])

    k_std = np.std(all_k_values)
    validation['sufficient_k_variability'] = k_std >= 5

    unique_k = len(set(all_k_values))
    validation['multiple_k_values'] = unique_k >= 3

    # 4. 检查时序稳定性 (k值变化不应该太频繁)
    total_changes = 0
    total_steps = 0
    for result in all_results:
        k_values = result['k_values']
        changes = np.sum(np.diff(k_values) != 0)
        total_changes += changes
        total_steps += len(k_values)

    change_rate = total_changes / total_steps if total_steps > 0 else 0
    validation['temporal_stability'] = change_rate <= 0.2  # 变化率不超过20%

    print("\n🔍 状态级自适应验证:")
    print(f"  ✅ Grasping阶段k值最低: {validation['grasping_has_lowest_k']}")
    print(f"  ✅ k值差异显著: {validation['significant_k_difference']}")
    print(f"  ✅ k值变化充分: {validation['sufficient_k_variability']}")
    print(f"  ✅ 使用多种k值: {validation['multiple_k_values']}")
    print(f"  ✅ 时序稳定性: {validation['temporal_stability']} (变化率: {change_rate:.1%})")

    # 计算自适应得分
    adaptation_score = sum(validation.values()) / len(validation) * 100
    validation['adaptation_score'] = adaptation_score
    print(f"  🎯 自适应得分: {adaptation_score:.1f}%")

    return validation

def create_validation_plots(all_results, avg_phase_k, output_dir):
    """
    生成验证结果的可视化图表
    """
    # 1. k值时序变化图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 子图1: 多个轨迹的k值变化
    ax1 = axes[0, 0]
    for i, result in enumerate(all_results[:3]):  # 只显示前3条
        k_values = result['k_values']
        steps = range(len(k_values))
        ax1.plot(steps, k_values, label=f'轨迹 {result["trajectory_idx"]}', alpha=0.7)

    ax1.set_title('k值时序变化 (前3条轨迹)')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('预测步长 k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 阶段k值分布
    ax2 = axes[0, 1]
    phases = list(avg_phase_k.keys())
    means = [avg_phase_k[p]['mean'] for p in phases]
    stds = [avg_phase_k[p]['std'] for p in phases]

    bars = ax2.bar(phases, means, yerr=stds, capsize=5,
                   color=['lightblue', 'lightcoral', 'lightgreen'])
    ax2.set_title('各阶段平均k值')
    ax2.set_ylabel('预测步长 k')
    ax2.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mean:.1f}', ha='center', va='bottom')

    # 子图3: k值分布直方图
    ax3 = axes[1, 0]
    all_k_values = []
    for result in all_results:
        all_k_values.extend(result['k_values'])

    ax3.hist(all_k_values, bins=15, alpha=0.7, edgecolor='black')
    ax3.set_title('k值分布直方图')
    ax3.set_xlabel('预测步长 k')
    ax3.set_ylabel('频次')
    ax3.grid(True, alpha=0.3)

    # 子图4: 轨迹k值统计对比
    ax4 = axes[1, 1]
    traj_indices = [r['trajectory_idx'] for r in all_results]
    traj_means = [r['k_stats']['mean'] for r in all_results]
    traj_stds = [r['k_stats']['std'] for r in all_results]

    ax4.bar(traj_indices, traj_means, yerr=traj_stds, capsize=5, alpha=0.7)
    ax4.set_title('各轨迹k值统计')
    ax4.set_xlabel('轨迹索引')
    ax4.set_ylabel('平均k值')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'transport_offline_validation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 可视化图表已保存: {output_dir / 'transport_offline_validation_plots.png'}")


if __name__ == "__main__":
    # 配置
    data_path = "/home/yhj/桌面/ACT/adastep_extension/robomimic_data/transport/mh/low_dim_v15.hdf5"
    predictor_path = "/home/yhj/桌面/ACT/adastep_extension/experiments/results_transport_mh/stage2_training/best_predictor.pth"

    # 检查文件存在
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        sys.exit(1)

    if not os.path.exists(predictor_path):
        print(f"❌ 预测器模型不存在: {predictor_path}")
        sys.exit(1)

    # 运行验证
    results = run_offline_validation(
        hdf5_path=data_path,
        predictor_path=predictor_path,
        trajectory_indices=[0, 1, 2]  # 分析前3条轨迹
    )

    if results:
        print(f"\n🎉 离线验证完成！")
        print(f"📊 关键结果:")
        print(f"  分析轨迹数: {results['trajectories_analyzed']}")
        print(f"  自适应得分: {results['validation']['adaptation_score']:.1f}%")

        phase_analysis = results['phase_analysis']
        if phase_analysis['grasping']['mean'] > 0:
            print(f"  k值范围: Grasping={phase_analysis['grasping']['mean']:.1f}, "
                  f"Reaching={phase_analysis['reaching']['mean']:.1f}, "
                  f"Transporting={phase_analysis['transporting']['mean']:.1f}")

        # 判断是否通过验证
        if results['validation']['adaptation_score'] >= 75:
            print(f"\n✅ 状态级自适应验证通过！算法工作正常。")
        else:
            print(f"\n⚠️  需要进一步优化算法。")
    else:
        print("❌ 验证失败")