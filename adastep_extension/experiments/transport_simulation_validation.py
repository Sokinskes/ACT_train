"""
AdaStep状态级自适应仿真验证
============================

目标: 验证新算法是否能在Transport任务中正确实现状态级自适应

关键验证点:
1. k值是否在"抓取瞬间"自动降低 (保守模式)
2. k值是否在"搬运瞬间"自动升高 (激进模式)
3. 整体成功率是否维持

方法:
- 使用训练好的HorizonPredictor
- 在Robomimic Transport环境中运行
- 记录每个时间步的k值变化和任务状态
"""

import torch
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

# Robomimic imports
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adastep_module import HorizonPredictor

# 设置matplotlib
import matplotlib
matplotlib.use('Agg')

class TransportStateAnalyzer:
    """
    Transport任务状态分析器

    分析当前状态属于哪个阶段:
    - Reaching: 接近物体阶段 (应该用较大k)
    - Grasping: 抓取阶段 (应该用较小k)
    - Transporting: 搬运阶段 (应该用较大k)
    """

    def __init__(self):
        # Transport任务的关键位置 (基于经验值)
        self.object_init_pos = np.array([0.0, 0.0, 0.0])  # 物体初始位置
        self.goal_pos = np.array([0.3, 0.0, 0.0])         # 目标位置

    def analyze_state(self, obs):
        """
        分析当前状态的任务阶段

        Args:
            obs: Robomimic观测字典

        Returns:
            phase: 'reaching', 'grasping', 'transporting'
            confidence: 置信度 (0-1)
        """
        # 提取关键信息
        eef_pos = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']

        # 物体位置 (如果可用)
        if 'object' in obs:
            object_pos = obs['object']
        else:
            # 使用默认位置
            object_pos = self.object_init_pos

        # 计算距离
        eef_to_object = np.linalg.norm(eef_pos - object_pos)
        object_to_goal = np.linalg.norm(object_pos - self.goal_pos)

        # 简单的启发式规则
        if eef_to_object < 0.05:  # 接近物体
            return 'grasping', 0.8
        elif object_to_goal < 0.1:  # 物体接近目标
            return 'transporting', 0.7
        else:
            return 'reaching', 0.6


def run_transport_validation(hdf5_path, predictor_path, num_episodes=10, device='cuda'):
    """
    运行Transport任务验证

    Args:
        hdf5_path: 数据文件路径
        predictor_path: 预测器模型路径
        num_episodes: 测试episode数量
        device: 计算设备

    Returns:
        results: 验证结果字典
    """

    print(f"\n{'='*80}")
    print(f"🚀 AdaStep状态级自适应仿真验证 - Transport任务")
    print(f"{'='*80}")

    # 1. 加载环境
    print("🌍 创建Transport仿真环境...")
    env_meta = FileUtils.get_env_metadata_from_dataset(hdf5_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False
    )
    print(f"✓ 环境: {env.name}")

    # 2. 加载HorizonPredictor
    print(f"🧠 加载HorizonPredictor: {predictor_path}")

    # 获取状态维度 (使用末端执行器状态)
    state_dim = 7  # eef_pos (3) + eef_quat (4)

    horizon_predictor = HorizonPredictor(
        input_dim=state_dim,
        hidden_dim=256
    ).to(device)

    if os.path.exists(predictor_path):
        checkpoint = torch.load(predictor_path, map_location=device)
        horizon_predictor.load_state_dict(checkpoint)
        horizon_predictor.eval()
        print("✓ 模型加载成功")
    else:
        print("❌ 模型文件不存在")
        return None

    # 3. 创建状态分析器
    state_analyzer = TransportStateAnalyzer()

    # 4. 运行验证
    print(f"\n🎯 运行验证 ({num_episodes} episodes)...")

    all_results = []

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1}/{num_episodes} ---")

        # 重置环境
        obs = env.reset()
        done = False
        step = 0
        max_steps = 400

        # 记录数据
        episode_data = {
            'k_values': [],
            'eef_positions': [],
            'task_phases': [],
            'success': False,
            'total_steps': 0
        }

        while step < max_steps and not done:
            # 提取状态特征 (末端执行器位姿)
            eef_pos = obs['robot0_eef_pos']
            eef_quat = obs['robot0_eef_quat']
            state_feature = np.concatenate([eef_pos, eef_quat])

            # 转换为tensor
            state_tensor = torch.from_numpy(state_feature).float().to(device).unsqueeze(0)

            # 预测k值
            with torch.no_grad():
                k_pred = horizon_predictor(state_tensor)
                k_normalized = torch.sigmoid(k_pred).item()
                k_value = int(5 + k_normalized * (50 - 5))  # 映射到[5,50]

            # 分析任务阶段
            phase, confidence = state_analyzer.analyze_state(obs)

            # 记录数据
            episode_data['k_values'].append(k_value)
            episode_data['eef_positions'].append(eef_pos.copy())
            episode_data['task_phases'].append(phase)

            # 执行动作 (使用专家动作或简单策略)
            # 这里使用简单的启发式策略来完成transport任务
            action = compute_transport_action(obs, phase)
            obs, reward, done, info = env.step(action)

            step += 1

            # 检查成功
            if info.get('success', False):
                episode_data['success'] = True
                break

        episode_data['total_steps'] = step
        all_results.append(episode_data)

        print(f"  结果: {'✅ 成功' if episode_data['success'] else '❌ 失败'} "
              f"({step}步, 平均k={np.mean(episode_data['k_values']):.1f})")

    env.close()

    # 5. 分析结果
    print(f"\n{'='*80}")
    print("📊 验证结果分析")
    print(f"{'='*80}")

    # 整体统计
    success_rate = np.mean([r['success'] for r in all_results]) * 100
    avg_k_values = [np.mean(r['k_values']) for r in all_results]
    avg_steps = [r['total_steps'] for r in all_results]

    print(f"整体统计:")
    print(f"  成功率: {success_rate:.1f}%")
    print(f"  平均k值: {np.mean(avg_k_values):.1f} ± {np.std(avg_k_values):.1f}")
    print(f"  平均步数: {np.mean(avg_steps):.1f} ± {np.std(avg_steps):.1f}")

    # 阶段分析
    print(f"\n阶段k值分析:")
    phase_k_stats = analyze_phase_k_values(all_results)
    for phase, stats in phase_k_stats.items():
        print(f"  {phase.capitalize()}: k={stats['mean']:.1f} ± {stats['std']:.1f} "
              f"({stats['count']}个样本)")

    # 验证状态级自适应
    validation_results = validate_state_adaptation(all_results, phase_k_stats)

    # 6. 生成可视化
    output_dir = Path("simulation_validation_results")
    output_dir.mkdir(exist_ok=True)

    create_validation_plots(all_results, phase_k_stats, output_dir)

    # 7. 保存结果
    results_summary = {
        'task': 'transport',
        'success_rate': success_rate,
        'avg_k': np.mean(avg_k_values),
        'phase_analysis': phase_k_stats,
        'validation': validation_results,
        'episodes': len(all_results)
    }

    output_file = output_dir / "transport_validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存: {output_file}")
    print(f"✓ 可视化图表已生成: {output_dir}")

    return results_summary


def compute_transport_action(obs, phase):
    """
    计算Transport任务的动作 (简化的启发式策略)

    Args:
        obs: 当前观测
        phase: 任务阶段

    Returns:
        action: 动作向量
    """
    eef_pos = obs['robot0_eef_pos']

    if phase == 'reaching':
        # 向物体移动
        target_pos = np.array([0.0, 0.0, 0.05])  # 物体上方
        pos_error = target_pos - eef_pos
        action_pos = pos_error * 2.0  # 位置控制增益

        # 保持抓取器张开
        action_gripper = np.array([1.0])  # 张开

    elif phase == 'grasping':
        # 下降并抓取
        target_pos = np.array([0.0, 0.0, 0.02])  # 物体位置
        pos_error = target_pos - eef_pos
        action_pos = pos_error * 3.0

        # 闭合抓取器
        action_gripper = np.array([-1.0])  # 闭合

    else:  # transporting
        # 向目标移动
        target_pos = np.array([0.3, 0.0, 0.05])  # 目标上方
        pos_error = target_pos - eef_pos
        action_pos = pos_error * 2.0

        # 保持抓取器闭合
        action_gripper = np.array([-1.0])

    # 组合动作 (位置控制 + 抓取器)
    action = np.concatenate([action_pos, action_gripper])

    # 限制动作范围
    action = np.clip(action, -1.0, 1.0)

    return action


def analyze_phase_k_values(all_results):
    """
    分析不同阶段的k值统计

    Returns:
        phase_stats: {phase: {'mean': float, 'std': float, 'count': int}}
    """
    phase_k_values = {'reaching': [], 'grasping': [], 'transporting': []}

    for episode in all_results:
        for k_val, phase in zip(episode['k_values'], episode['task_phases']):
            phase_k_values[phase].append(k_val)

    phase_stats = {}
    for phase, k_vals in phase_k_values.items():
        if k_vals:
            phase_stats[phase] = {
                'mean': np.mean(k_vals),
                'std': np.std(k_vals),
                'count': len(k_vals)
            }
        else:
            phase_stats[phase] = {'mean': 0, 'std': 0, 'count': 0}

    return phase_stats


def validate_state_adaptation(all_results, phase_stats):
    """
    验证状态级自适应是否正确工作

    检查:
    1. Grasping阶段的k值是否显著低于其他阶段
    2. k值变化是否合理
    """
    validation = {}

    # 1. 阶段间k值差异
    grasping_k = phase_stats['grasping']['mean']
    reaching_k = phase_stats['reaching']['mean']
    transporting_k = phase_stats['transporting']['mean']

    # Grasping应该有最低的k值
    validation['grasping_has_lowest_k'] = (
        grasping_k < reaching_k and grasping_k < transporting_k
    )

    # k值差异显著性 (至少相差5)
    validation['significant_k_difference'] = (
        min(reaching_k, transporting_k) - grasping_k >= 5
    )

    # 2. 整体k值分布
    all_k_values = []
    for episode in all_results:
        all_k_values.extend(episode['k_values'])

    k_std = np.std(all_k_values)
    validation['sufficient_k_variability'] = k_std >= 5  # 标准差至少5

    unique_k = len(set(all_k_values))
    validation['multiple_k_values'] = unique_k >= 3  # 至少3种不同k值

    print(f"\n🔍 状态级自适应验证:")
    print(f"  ✅ Grasping阶段k值最低: {validation['grasping_has_lowest_k']}")
    print(f"  ✅ k值差异显著: {validation['significant_k_difference']}")
    print(f"  ✅ k值变化充分: {validation['sufficient_k_variability']}")
    print(f"  ✅ 使用多种k值: {validation['multiple_k_values']}")

    # 计算自适应得分
    adaptation_score = sum(validation.values()) / len(validation) * 100
    validation['adaptation_score'] = adaptation_score
    print(f"  🎯 自适应得分: {adaptation_score:.1f}%")

    return validation


def create_validation_plots(all_results, phase_stats, output_dir):
    """
    生成验证结果的可视化图表
    """
    # 1. k值时序变化图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 子图1: 多个episode的k值变化
    ax1 = axes[0, 0]
    for i, episode in enumerate(all_results[:5]):  # 只显示前5个
        k_values = episode['k_values']
        steps = range(len(k_values))
        ax1.plot(steps, k_values, label=f'Episode {i+1}', alpha=0.7)

    ax1.set_title('k值时序变化 (前5个Episodes)')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('预测步长 k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 阶段k值分布
    ax2 = axes[0, 1]
    phases = list(phase_stats.keys())
    means = [phase_stats[p]['mean'] for p in phases]
    stds = [phase_stats[p]['std'] for p in phases]

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
    for episode in all_results:
        all_k_values.extend(episode['k_values'])

    ax3.hist(all_k_values, bins=20, alpha=0.7, edgecolor='black')
    ax3.set_title('k值分布直方图')
    ax3.set_xlabel('预测步长 k')
    ax3.set_ylabel('频次')
    ax3.grid(True, alpha=0.3)

    # 子图4: 成功率和k值关系
    ax4 = axes[1, 1]
    success_episodes = [ep for ep in all_results if ep['success']]
    failed_episodes = [ep for ep in all_results if not ep['success']]

    success_k = [np.mean(ep['k_values']) for ep in success_episodes]
    failed_k = [np.mean(ep['k_values']) for ep in failed_episodes]

    ax4.scatter(success_k, [1] * len(success_k), color='green',
                label='成功', s=50, alpha=0.7)
    ax4.scatter(failed_k, [0] * len(failed_k), color='red',
                label='失败', s=50, alpha=0.7)

    ax4.set_title('成功率 vs 平均k值')
    ax4.set_xlabel('平均预测步长 k')
    ax4.set_ylabel('成功 (1) / 失败 (0)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['失败', '成功'])

    plt.tight_layout()
    plt.savefig(output_dir / 'transport_validation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 可视化图表已保存: {output_dir / 'transport_validation_plots.png'}")


if __name__ == "__main__":
    # 配置
    data_path = "/home/yhj/桌面/ACT/adastep_extension/robomimic_data/transport/mh/low_dim_v15.hdf5"
    predictor_path = "/home/yhj/桌面/ACT/adastep_extension/experiments/results_transport_mh/stage2_training/best_predictor.pth"

    # 检查文件存在
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        # 尝试查找替代文件
        transport_dir = "/home/yhj/桌面/ACT/adastep_extension/robomimic_data/transport"
        if os.path.exists(transport_dir):
            files = [f for f in os.listdir(transport_dir) if f.endswith('.hdf5')]
            if files:
                data_path = os.path.join(transport_dir, files[0])
                print(f"✓ 使用替代文件: {data_path}")
            else:
                print("❌ 未找到Transport数据文件")
                sys.exit(1)

    if not os.path.exists(predictor_path):
        print(f"❌ 预测器模型不存在: {predictor_path}")
        sys.exit(1)

    # 运行验证
    results = run_transport_validation(
        hdf5_path=data_path,
        predictor_path=predictor_path,
        num_episodes=10,  # 先用少量episode测试
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if results:
        print(f"\n🎉 验证完成！")
        print(f"📊 关键结果:")
        print(f"  成功率: {results['success_rate']:.1f}%")
        print(f"  自适应得分: {results['validation']['adaptation_score']:.1f}%")
        print(f"  k值范围: {results['phase_analysis']['grasping']['mean']:.1f} - "
              f"{max([results['phase_analysis'][p]['mean'] for p in results['phase_analysis']]):.1f}")

        # 判断是否通过验证
        if (results['validation']['adaptation_score'] >= 75 and
            results['success_rate'] >= 50):
            print(f"\n✅ 状态级自适应验证通过！算法工作正常。")
        else:
            print(f"\n⚠️  需要进一步优化算法。")
    else:
        print("❌ 验证失败")