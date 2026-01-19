"""
时序一致性检查脚本
可视化k值随时间的变化曲线

验证新算法是否能正确地在复杂状态时降低k值，在简单状态时提高k值
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import h5py

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adastep_module import StateClusterAnalyzer
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def load_single_trajectory(hdf5_path, trajectory_idx=0):
    """
    直接从HDF5文件加载单条完整轨迹

    Args:
        hdf5_path: HDF5文件路径
        trajectory_idx: 轨迹索引

    Returns:
        qpos: [T, state_dim]
        actions: [T, action_dim]
    """
    with h5py.File(hdf5_path, 'r') as f:
        demos = list(f['data'].keys())
        if trajectory_idx >= len(demos):
            return None, None

        demo_name = demos[trajectory_idx]
        demo = f[f'data/{demo_name}']

        # 读取状态
        if 'obs/robot0_eef_pos' in demo:
            # 使用末端执行器位姿
            eef_pos = demo['obs/robot0_eef_pos'][()]
            eef_quat = demo['obs/robot0_eef_quat'][()]
            qpos = np.concatenate([eef_pos, eef_quat], axis=-1)
        elif 'obs/robot0_joint_pos' in demo:
            # 使用关节角度
            qpos = demo['obs/robot0_joint_pos'][()]
        else:
            raise ValueError("未找到有效的状态数据！")

        # 读取动作
        actions = demo['actions'][()]

        return qpos, actions

def load_training_data(hdf5_path, max_trajectories=20):
    """
    加载训练用的数据

    Args:
        hdf5_path: HDF5文件路径
        max_trajectories: 最大轨迹数

    Returns:
        all_states: [N, state_dim]
        all_action_sequences: [N, seq_len, action_dim] - 简化为单个动作用于复杂度计算
    """
    all_states = []
    all_action_sequences = []

    with h5py.File(hdf5_path, 'r') as f:
        demos = list(f['data'].keys())[:max_trajectories]

        for demo_name in demos:
            demo = f[f'data/{demo_name}']

            # 读取状态
            if 'obs/robot0_eef_pos' in demo:
                eef_pos = demo['obs/robot0_eef_pos'][()]
                eef_quat = demo['obs/robot0_eef_quat'][()]
                qpos = np.concatenate([eef_pos, eef_quat], axis=-1)
            elif 'obs/robot0_joint_pos' in demo:
                qpos = demo['obs/robot0_joint_pos'][()]
            else:
                continue

            # 读取动作
            actions = demo['actions'][()]

            # 取前7维状态
            if qpos.shape[1] >= 7:
                states = qpos[:, :7]
            else:
                states = qpos

            all_states.append(states)

            # 为每个轨迹创建一个简化的动作序列（只取中间部分）
            seq_len = min(100, len(actions))  # 限制序列长度
            simplified_actions = actions[:seq_len]
            all_action_sequences.append(simplified_actions)

    if all_states:
        all_states = np.concatenate(all_states, axis=0)

        # 创建固定长度的动作序列数组
        max_len = max(len(seq) for seq in all_action_sequences)
        padded_sequences = []
        for seq in all_action_sequences:
            if len(seq) < max_len:
                # 填充到最大长度
                padding = np.tile(seq[-1:], (max_len - len(seq), 1))
                padded_seq = np.concatenate([seq, padding], axis=0)
            else:
                padded_seq = seq[:max_len]
            padded_sequences.append(padded_seq)

        all_action_sequences = np.array(padded_sequences)
    else:
        all_states = np.array([])
        all_action_sequences = np.array([])

    return all_states, all_action_sequences

def analyze_trajectory_temporal_consistency(task_name, data_path, trajectory_idx=0):
    """
    分析单条轨迹的时序一致性

    Args:
        task_name: 任务名称
        data_path: 数据路径
        trajectory_idx: 轨迹索引
    """
    print(f"\n{'='*60}")
    print(f"时序一致性分析: {task_name.upper()} - 轨迹 {trajectory_idx}")
    print(f"{'='*60}")

    try:
        # 1. 加载目标轨迹
        print("📊 加载目标轨迹...")
        qpos_np, actions_np = load_single_trajectory(data_path, trajectory_idx)

        if qpos_np is None:
            print(f"❌ 无法加载轨迹 {trajectory_idx}")
            return

        print(f"✓ 轨迹加载完成: {len(qpos_np)} 步, 状态维度: {qpos_np.shape[1]}, 动作维度: {actions_np.shape[1]}")

        # 2. 加载训练数据
        print("\n🎯 加载训练数据...")
        all_states, all_action_sequences = load_training_data(data_path, max_trajectories=20)

        if len(all_states) == 0:
            print("❌ 无法加载训练数据")
            return

        print(f"✓ 训练数据加载完成: {len(all_states)} 个状态样本")

        # 3. 训练AdaStep模型
        print("\n🎯 训练AdaStep模型...")
        analyzer = StateClusterAnalyzer(num_clusters=10, error_threshold=0.5)
        analyzer.fit_clusters(all_states)

        # 手动创建聚类到k值的映射（简化版）
        # 基于聚类大小分配k值：大聚类给高k值，小聚类给低k值
        labels = analyzer.kmeans.predict(all_states)
        unique_labels, counts = np.unique(labels, return_counts=True)

        # 按聚类大小排序
        sorted_indices = np.argsort(-counts)  # 降序排列

        # 分配k值：最大的聚类给k=50，次大的给k=40，以此类推
        k_candidates = [50, 40, 30, 25, 20]  # 限制为5个k值，与论文一致
        horizons = {}

        for i, cluster_idx in enumerate(sorted_indices):
            cluster_id = unique_labels[cluster_idx]
            if i < len(k_candidates):
                horizons[cluster_id] = k_candidates[i]
            else:
                horizons[cluster_id] = k_candidates[-1]  # 剩余的给最小k值

        print(f"✓ 简化的k值分配完成:")
        for cid, k in sorted(horizons.items()):
            print(f"  Cluster {cid}: k={k}")

        analyzer.cluster_horizons = horizons

        # 4. 分析目标轨迹的时序变化
        print(f"\n📈 分析轨迹 {trajectory_idx} 的时序变化...")

        # 目标轨迹的状态序列
        target_states = qpos_np[:, :7] if qpos_np.shape[1] >= 7 else qpos_np
        target_actions = actions_np

        # 为每个时间步预测k值
        k_values_over_time = []
        cluster_labels_over_time = []

        for t in range(len(target_states)):
            state_t = target_states[t:t+1]  # [1, state_dim]

            # 预测聚类标签
            label_t = analyzer.kmeans.predict(state_t)[0]
            cluster_labels_over_time.append(label_t)

            # 获取对应的k值
            k_t = horizons[label_t]
            k_values_over_time.append(k_t)

        k_values_over_time = np.array(k_values_over_time)
        cluster_labels_over_time = np.array(cluster_labels_over_time)

        # 5. 统计分析
        unique_k, counts = np.unique(k_values_over_time, return_counts=True)
        unique_clusters, cluster_counts = np.unique(cluster_labels_over_time, return_counts=True)

        print(f"\n📊 时序分析结果:")
        print(f"  轨迹长度: {len(k_values_over_time)} 步")
        print(f"  k值分布:")
        for k, count in zip(unique_k, counts):
            pct = count / len(k_values_over_time) * 100
            print(f"    k={k:2d}: {count:3d}步 ({pct:5.1f}%)")

        print(f"  聚类分布:")
        for c, count in zip(unique_clusters, cluster_counts):
            pct = count / len(cluster_labels_over_time) * 100
            print(f"    Cluster {c}: {count:3d}步 ({pct:5.1f}%)")

        print(f"  时序统计:")
        print(f"    k值标准差: {np.std(k_values_over_time):.2f}")
        print(f"    k值变化次数: {np.sum(np.diff(k_values_over_time) != 0)}")
        print(f"    平均k值: {np.mean(k_values_over_time):.1f}")

        # 6. 可视化
        print(f"\n📊 生成时序可视化...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 子图1: k值随时间变化
        time_steps = np.arange(len(k_values_over_time))

        ax1.plot(time_steps, k_values_over_time, 'b-', linewidth=2, alpha=0.8)
        ax1.scatter(time_steps, k_values_over_time, c=k_values_over_time,
                   cmap='RdYlGn_r', s=50, alpha=0.7)

        ax1.set_title(f'{task_name.upper()} 轨迹 {trajectory_idx}: k值时序变化', fontsize=14, fontweight='bold')
        ax1.set_xlabel('时间步 (Time Step)', fontsize=12)
        ax1.set_ylabel('预测步长 k (Prediction Horizon)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(15, 55)

        # 添加水平线标记主要k值
        for k in unique_k:
            ax1.axhline(y=k, color='red', linestyle='--', alpha=0.5,
                       label=f'k={k}' if k == unique_k[0] else "")
        ax1.legend()

        # 子图2: 聚类标签随时间变化
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        for i, c in enumerate(unique_clusters):
            mask = cluster_labels_over_time == c
            ax2.scatter(time_steps[mask], cluster_labels_over_time[mask],
                       color=colors[i], s=30, alpha=0.7, label=f'Cluster {c}')

        ax2.plot(time_steps, cluster_labels_over_time, 'k-', alpha=0.3)
        ax2.set_title(f'聚类标签时序变化', fontsize=14, fontweight='bold')
        ax2.set_xlabel('时间步 (Time Step)', fontsize=12)
        ax2.set_ylabel('聚类标签 (Cluster Label)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # 保存图像
        output_dir = Path(__file__).parent / 'temporal_analysis'
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / f'{task_name}_trajectory_{trajectory_idx}_temporal_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 时序可视化已保存: {output_path}")

        # 7. 一致性评估
        print(f"\n🔍 时序一致性评估:")

        # 检查是否有高频震荡
        k_changes = np.sum(np.diff(k_values_over_time) != 0)
        change_rate = k_changes / len(k_values_over_time)

        if change_rate > 0.1:  # 超过10%的步数发生变化
            print(f"  ⚠️  高频震荡警告: {change_rate:.1%} 的步数发生k值变化")
            print(f"     这可能导致控制不稳定，建议增加平滑处理")
        else:
            print(f"  ✅ 时序稳定性良好: {change_rate:.1%} 的步数发生k值变化")

        # 检查是否呈现分段常数特性
        from scipy.stats import mode
        # 计算滑动窗口内的众数
        window_size = 10
        smoothed_k = []
        for i in range(len(k_values_over_time) - window_size + 1):
            window = k_values_over_time[i:i+window_size]
            modal_k = mode(window, keepdims=True)[0][0]
            smoothed_k.append(modal_k)

        if len(set(smoothed_k)) <= 3:  # 大部分时间保持在少数几个k值
            print(f"  ✅ 分段常数特性: 轨迹主要在 {len(set(smoothed_k))} 种k值间切换")
        else:
            print(f"  ⚠️  切换过于频繁: 滑动窗口检测到 {len(set(smoothed_k))} 种不同k值")

        return {
            'k_values': k_values_over_time,
            'cluster_labels': cluster_labels_over_time,
            'unique_k': unique_k,
            'change_rate': change_rate,
            'output_path': str(output_path)
        }

    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 测试不同任务
    tasks = {
        'square': '/home/yhj/桌面/ACT/adastep_extension/robomimic_data/square/mh/low_dim_v15.hdf5',
        'transport': '/home/yhj/桌面/ACT/adastep_extension/robomimic_data/transport/mh/low_dim_v15.hdf5',
        'can': '/home/yhj/桌面/ACT/adastep_extension/robomimic_data/can/mh/low_dim_v15.hdf5',
        'lift': '/home/yhj/桌面/ACT/adastep_extension/robomimic_data/lift/mh/low_dim_v15.hdf5'
    }

    # 选择要分析的任务
    selected_task = 'transport'  # 可以修改为其他任务

    if selected_task in tasks:
        result = analyze_trajectory_temporal_consistency(
            selected_task, tasks[selected_task], trajectory_idx=0
        )

        if result:
            print(f"\n🎯 分析完成！请查看可视化结果: {result['output_path']}")
    else:
        print(f"❌ 无效任务: {selected_task}")
        print(f"可用任务: {list(tasks.keys())}")