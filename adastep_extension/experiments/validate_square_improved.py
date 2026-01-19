"""
AdaStep Square任务改进版验证脚本
================================

使用训练好的HorizonPredictor验证Square任务的状态级适应性
目标：证明k值在不同任务阶段的显著变化
"""

import torch
import numpy as np
import h5py
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.adastep_module import HorizonPredictor

def load_square_validation_data(hdf5_path, max_episodes=10):
    """
    加载Square任务验证数据
    """
    print(f"📂 加载Square任务验证数据: {hdf5_path}")

    all_states = []
    all_actions = []
    episode_info = []

    with h5py.File(hdf5_path, 'r') as f:
        # 使用不同的轨迹进行验证
        demo_names = list(f['data'].keys())[20:30]  # 20-30轨迹用于验证

        for demo_name in demo_names:
            demo = f[f'data/{demo_name}']

            # 提取状态 (eef_pos + eef_quat)
            if 'obs/robot0_eef_pos' in demo:
                eef_pos = demo['obs/robot0_eef_pos'][()]
                eef_quat = demo['obs/robot0_eef_quat'][()]
                states = np.concatenate([eef_pos, eef_quat], axis=-1)
            else:
                states = demo['obs/robot0_joint_pos'][()]

            # 提取动作
            actions = demo['actions'][()]

            all_states.append(states)
            all_actions.append(actions)

            episode_info.append({
                'name': demo_name,
                'length': len(states),
                'states': states,
                'actions': actions
            })

            print(f"  ✓ 验证轨迹 {demo_name}: {len(states)} 步")

    print(f"✓ 验证数据加载完成: {len(episode_info)} 个轨迹")
    return episode_info

def load_trained_model(model_path):
    """
    加载训练好的HorizonPredictor模型
    """
    print(f"🧠 加载训练好的模型: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    model = HorizonPredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dim=128  # 与训练时一致
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 模型加载完成 (MAE={checkpoint['mae']:.4f}, R²={checkpoint['r2']:.4f})")

    return model

def improved_phase_detection(states, actions):
    """
    改进的Square任务阶段检测
    基于位置和动作特征识别：reaching -> grasping -> transporting -> insertion
    """
    phases = []

    for i, (state, action) in enumerate(zip(states, actions)):
        # 提取位置信息 (假设前3维是位置)
        pos = state[:3] if len(state) >= 3 else state[:2] + [0]

        # 计算动作幅度
        action_magnitude = np.linalg.norm(action)

        # 计算位置变化 (相对于初始位置)
        if i == 0:
            initial_pos = pos.copy()
            pos_change = 0.0
        else:
            pos_change = np.linalg.norm(pos - initial_pos)

        # Square任务阶段判断逻辑：
        # 1. Reaching: 大位置变化，小动作幅度 (接近目标)
        # 2. Grasping: 小位置变化，大动作幅度 (调整姿态)
        # 3. Transporting: 中等位置变化，中等动作幅度 (移动物体)
        # 4. Insertion: 小位置变化，小动作幅度 (精确插入)

        if pos_change > 0.1 and action_magnitude < 0.05:
            phase = 'reaching'
        elif pos_change < 0.05 and action_magnitude > 0.1:
            phase = 'grasping'
        elif pos_change > 0.05 and action_magnitude > 0.05:
            phase = 'transporting'
        else:
            phase = 'insertion'

        phases.append(phase)

    return phases

def validate_adaptation(model, episode_info):
    """
    验证状态级适应性
    """
    print("🔍 执行状态级适应性验证...")

    all_k_predictions = []
    all_phases = []
    adaptation_results = []

    for episode in episode_info:
        states = episode['states']
        actions = episode['actions']

        # 1. 预测k值
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states)
            k_normalized = model(state_tensor).numpy().flatten()

        # 反归一化到k值
        k_predictions = k_normalized * 45 + 5  # (pred * (50-5)) + 5

        # 2. 检测阶段
        phases = improved_phase_detection(states, actions)

        # 3. 按阶段统计k值
        phase_k_stats = {}
        for phase in ['reaching', 'grasping', 'transporting', 'insertion']:
            phase_mask = np.array(phases) == phase
            if np.any(phase_mask):
                phase_k = k_predictions[phase_mask]
                phase_k_stats[phase] = {
                    'mean': np.mean(phase_k),
                    'std': np.std(phase_k),
                    'min': np.min(phase_k),
                    'max': np.max(phase_k),
                    'count': len(phase_k)
                }

        # 4. 计算适应性指标
        validation = {}

        # 计算客观的适应性指标
        all_k_std = np.std(k_predictions)  # K值标准差 - 越大越好
        k_range = np.max(k_predictions) - np.min(k_predictions)
        k_coverage = k_range / (50 - 5)  # K值覆盖率 - 越大越好 (0-1范围)

        # 时序平滑度 - 越大越好 (0-1范围)
        if len(k_predictions) > 1:
            k_changes = np.abs(np.diff(k_predictions))
            total_change = np.sum(k_changes)
            max_possible_change = len(k_predictions) * 50  # 假设最大变化为50
            temporal_smoothness = 1 - (total_change / max_possible_change)
        else:
            temporal_smoothness = 1.0

        # 综合适应分数 (0-100)
        # 权重: 标准差40%, 覆盖率40%, 时序平滑度20%
        adaptation_score = (
            all_k_std / 15 * 40 +  # 假设最大标准差15
            k_coverage * 40 +
            temporal_smoothness * 20
        )

        # 保持向后兼容的验证字典
        validation = {
            'k_std_score': all_k_std,
            'k_coverage_score': k_coverage,
            'temporal_smoothness': temporal_smoothness,
            'sufficient_k_variability': all_k_std >= 5,
            'good_coverage': k_coverage >= 0.6,
            'temporal_stability': temporal_smoothness >= 0.7
        }

        episode_result = {
            'episode': episode['name'],
            'k_predictions': k_predictions,
            'phases': phases,
            'phase_k_stats': phase_k_stats,
            'validation': validation,
            'adaptation_score': adaptation_score
        }

        adaptation_results.append(episode_result)

        print(f"  ✓ {episode['name']}: 适应分数={adaptation_score:.1f}% (Std={all_k_std:.2f}, Coverage={k_coverage:.2f}, Smooth={temporal_smoothness:.2f})")

        all_k_predictions.extend(k_predictions)
        all_phases.extend(phases)

    # 整体统计
    all_scores = [r['adaptation_score'] for r in adaptation_results]
    overall_stats = {
        'mean_adaptation_score': np.mean(all_scores),
        'k_range': (np.min(all_k_predictions), np.max(all_k_predictions)),
        'k_std': np.std(all_k_predictions),
        'k_coverage': (np.max(all_k_predictions) - np.min(all_k_predictions)) / (50 - 5),
        'unique_k_values': len(np.unique(np.round(all_k_predictions))),
        'temporal_smoothness': np.mean([r['validation']['temporal_smoothness'] for r in adaptation_results]),
        'phase_distribution': {}
    }

    # 阶段分布
    for phase in ['reaching', 'grasping', 'transporting', 'insertion']:
        phase_count = all_phases.count(phase)
        overall_stats['phase_distribution'][phase] = phase_count / len(all_phases) * 100

    return adaptation_results, overall_stats

def create_validation_visualization(adaptation_results, overall_stats, save_path):
    """
    创建验证结果可视化
    """
    print("🎨 生成验证结果可视化...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 适应分数分布
    scores = [r['adaptation_score'] for r in adaptation_results]
    ax1.hist(scores, bins=10, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
                label=f'平均: {np.mean(scores):.1f}%')
    ax1.set_title('适应分数分布 (基于客观统计)')
    ax1.set_xlabel('适应分数 (%)')
    ax1.set_ylabel('轨迹数量')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. k值时间序列示例 (第一个轨迹)
    if adaptation_results:
        first_result = adaptation_results[0]
        k_preds = first_result['k_predictions']
        phases = first_result['phases']

        # 颜色映射
        phase_colors = {
            'reaching': 'blue',
            'grasping': 'red',
            'transporting': 'green',
            'insertion': 'orange'
        }

        # 绘制k值曲线
        ax2.plot(k_preds, 'k-', alpha=0.7, linewidth=2, label='预测k值')

        # 标记阶段
        current_phase = None
        start_idx = 0
        for i, phase in enumerate(phases):
            if phase != current_phase:
                if current_phase is not None:
                    # 绘制前一个阶段
                    ax2.axvspan(start_idx, i, color=phase_colors.get(current_phase, 'gray'),
                               alpha=0.2, label=current_phase)
                current_phase = phase
                start_idx = i

        # 最后一个阶段
        if current_phase:
            ax2.axvspan(start_idx, len(phases), color=phase_colors.get(current_phase, 'gray'),
                       alpha=0.2, label=current_phase)

        ax2.set_title(f'k值时间序列示例 ({first_result["episode"]})')
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('预测k值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. 阶段k值对比
    if adaptation_results:
        phase_k_means = {}
        for phase in ['reaching', 'grasping', 'transporting', 'insertion']:
            phase_ks = []
            for result in adaptation_results:
                if phase in result['phase_k_stats']:
                    phase_ks.append(result['phase_k_stats'][phase]['mean'])

            if phase_ks:
                phase_k_means[phase] = np.mean(phase_ks)

        if phase_k_means:
            phases = list(phase_k_means.keys())
            k_values = list(phase_k_means.values())

            bars = ax3.bar(phases, k_values, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
            ax3.set_title('各阶段平均k值')
            ax3.set_xlabel('任务阶段')
            ax3.set_ylabel('平均k值')
            ax3.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, v in zip(bars, k_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{v:.1f}', ha='center', va='bottom')

    # 4. 验证指标雷达图
    if adaptation_results:
        # 计算平均验证指标 (使用新的客观指标)
        avg_validation = {}
        for key in ['sufficient_k_variability', 'good_coverage', 'temporal_stability']:
            values = [r['validation'][key] for r in adaptation_results if key in r['validation']]
            avg_validation[key] = np.mean(values)

        # 添加统计指标
        avg_validation['k_std_normalized'] = min(overall_stats['k_std'] / 15, 1.0)  # 归一化到0-1
        avg_validation['k_coverage'] = overall_stats['k_coverage']
        avg_validation['temporal_smoothness'] = overall_stats['temporal_smoothness']

        # 雷达图数据
        categories = ['K值变异性', '覆盖率', '时序平滑', '标准差', '综合覆盖']
        values = [
            avg_validation.get('sufficient_k_variability', 0),
            avg_validation.get('good_coverage', 0),
            avg_validation.get('temporal_stability', 0),
            avg_validation.get('k_std_normalized', 0),
            avg_validation.get('k_coverage', 0)
        ]
        values += values[:1]  # 闭合图形

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax4.plot(angles, values, 'o-', linewidth=2, label='客观指标')
        ax4.fill(angles, values, alpha=0.25)
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1.1)
        ax4.set_title('自适应能力雷达图')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 验证可视化已保存: {save_path}")

def main():
    """
    主函数：执行Square任务的改进版验证
    """
    print("🚀 AdaStep Square任务改进版验证")
    print("="*60)

    # 配置
    data_path = "/home/yhj/桌面/ACT/adastep_extension/robomimic_data/square/mh/low_dim_v15.hdf5"
    model_path = "/home/yhj/桌面/ACT/adastep_extension/experiments/results_square_improved/horizon_predictor_square_improved.pth"
    output_dir = Path("/home/yhj/桌面/ACT/adastep_extension/experiments/results_square_improved")

    # 1. 加载验证数据
    episode_info = load_square_validation_data(data_path, max_episodes=10)

    # 2. 加载训练好的模型
    model = load_trained_model(model_path)

    # 3. 执行适应性验证
    adaptation_results, overall_stats = validate_adaptation(model, episode_info)

    # 4. 创建可视化
    viz_path = output_dir / "square_adaptation_validation.png"
    create_validation_visualization(adaptation_results, overall_stats, viz_path)

    # 5. 保存详细结果
    results_data = {
        'adaptation_results': adaptation_results,
        'overall_stats': overall_stats,
        'validation_config': {
            'data_path': data_path,
            'model_path': str(model_path),
            'num_episodes': len(episode_info)
        }
    }

    with open(output_dir / "square_validation_results.pkl", 'wb') as f:
        pickle.dump(results_data, f)

    print(f"\n✓ 详细结果已保存: {output_dir}/square_validation_results.pkl")

    # 6. 最终总结
    print(f"\n📊 Square任务验证总结:")
    print(f"  平均适应分数: {overall_stats['mean_adaptation_score']:.1f}%")
    print(f"  k值范围: {overall_stats['k_range'][0]:.1f} - {overall_stats['k_range'][1]:.1f}")
    print(f"  k值标准差: {overall_stats['k_std']:.2f}")
    print(f"  k值覆盖率: {overall_stats['k_coverage']:.2f}")
    print(f"  时序平滑度: {overall_stats['temporal_smoothness']:.2f}")
    print(f"  唯一k值数量: {overall_stats['unique_k_values']}")
    print(f"  阶段分布:")
    for phase, percentage in overall_stats['phase_distribution'].items():
        print(f"    {phase}: {percentage:.1f}%")

    # 成功标准 (基于客观指标)
    if overall_stats['k_std'] >= 10 and overall_stats['k_coverage'] >= 0.8:
        print(f"  🎯 验证结果: 优秀！k值分布丰富，覆盖范围广")
    elif overall_stats['k_std'] >= 5 and overall_stats['k_coverage'] >= 0.6:
        print(f"  ✅ 验证结果: 良好，自适应能力中等")
    else:
        print(f"  ⚠️  验证结果: 需要改进，k值变化不足")

if __name__ == "__main__":
    main()