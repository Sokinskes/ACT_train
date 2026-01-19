"""
AdaStep Square任务基准版标签生成 (旧算法)
=========================================

使用旧的线性映射方法：
1. K-Means聚类 (K=3)
2. 固定阈值 (error_threshold = 0.15)
3. 线性映射分配k值
"""

import torch
import numpy as np
import h5py
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.adastep_module import StateClusterAnalyzer

def load_square_data(hdf5_path, max_episodes=20):
    """
    加载Square任务数据
    """
    print(f"📂 加载Square任务数据: {hdf5_path}")

    all_states = []
    all_actions = []

    with h5py.File(hdf5_path, 'r') as f:
        demo_names = list(f['data'].keys())[:max_episodes]

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

            print(f"  ✓ 轨迹 {demo_name}: {len(states)} 步, 状态维度: {states.shape[1]}")

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    print(f"✓ 总数据: {states.shape[0]} 个状态点")
    return states, actions, demo_names

def baseline_clustering_analysis(states, actions, num_clusters=3, error_threshold=0.15):
    """
    基准版聚类分析：使用旧的线性映射方法
    """
    print(f"📊 执行基准版聚类分析 (K-Means K={num_clusters})...")

    # 1. K-Means聚类 (K=3，旧算法)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(states)

    print(f"✓ K-Means聚类完成，各类样本数:")
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} 样本")

    # 2. 计算每个Cluster的复杂度指标 (使用旧方法)
    cluster_complexities = {}

    for cluster_id in range(num_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_states = states[cluster_mask]
        cluster_actions = actions[cluster_mask] if len(actions) == len(states) else actions[:len(cluster_states)]

        if len(cluster_states) == 0:
            cluster_complexities[cluster_id] = 0.0
            continue

        # 计算状态变化率 (旧的复杂度度量)
        if len(cluster_states) > 1:
            state_diffs = np.linalg.norm(np.diff(cluster_states, axis=0), axis=1)
            avg_state_change = np.mean(state_diffs)
        else:
            avg_state_change = 0.0

        # 计算动作变化率
        if len(cluster_actions) > 1:
            action_diffs = np.linalg.norm(np.diff(cluster_actions, axis=0), axis=1)
            avg_action_change = np.mean(action_diffs)
        else:
            avg_action_change = 0.0

        # 复杂度 = 状态变化 + 动作变化
        complexity = avg_state_change + avg_action_change
        cluster_complexities[cluster_id] = complexity

        print(f"  Cluster {cluster_id}: 复杂度={complexity:.6f}")

    # 3. 线性映射分配k值 (旧方法)
    complexities = list(cluster_complexities.values())

    # 归一化复杂度
    if max(complexities) > min(complexities):
        normalized_complexities = (np.array(complexities) - min(complexities)) / (max(complexities) - min(complexities))
    else:
        normalized_complexities = np.zeros(len(complexities))

    # 线性映射：复杂度越高 -> k值越小
    k_min, k_max = 5, 50
    cluster_horizons = {}
    for i, cluster_id in enumerate(range(num_clusters)):
        complexity = normalized_complexities[i]

        # 反比例映射：complexity=0 -> k_max, complexity=1 -> k_min
        k_value = k_max - complexity * (k_max - k_min)
        k_value = int(round(k_value))

        # 确保在范围内
        k_value = max(k_min, min(k_max, k_value))

        cluster_horizons[cluster_id] = k_value

    print(f"✓ 基准版分析完成！各聚类步长分配:")
    for cid, k in cluster_horizons.items():
        complexity = cluster_complexities[cid]
        print(f"  Cluster {cid}: k={k} (复杂度={complexity:.6f})")

    # 统计k值分布
    k_values = list(cluster_horizons.values())
    print(f"  k值分布: min={min(k_values)}, max={max(k_values)}, unique={len(set(k_values))}")

    return kmeans, cluster_horizons, cluster_complexities

def create_tsne_visualization(states, cluster_labels, cluster_horizons, save_path):
    """
    创建t-SNE可视化：检查聚类质量和k值分配合理性
    """
    print("🎨 生成t-SNE可视化...")

    # 标准化特征
    scaler = StandardScaler()
    states_scaled = scaler.fit_transform(states)

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    states_2d = tsne.fit_transform(states_scaled[:5000])  # 采样前5000个点

    # 创建颜色映射 (k值从低到高: 红色->蓝色)
    k_values = np.array([cluster_horizons[label] for label in cluster_labels[:5000]])
    norm_k = (k_values - k_values.min()) / (k_values.max() - k_values.min())

    # 绘图
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(states_2d[:, 0], states_2d[:, 1],
                         c=norm_k, cmap='RdYlBu_r', alpha=0.6, s=20)

    plt.colorbar(scatter, label='预测步长 k')
    plt.title('Square任务基准版状态聚类与k值分配 (t-SNE可视化)')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')

    # 添加图例
    k_min, k_max = k_values.min(), k_values.max()
    plt.text(0.02, 0.98, f'k值范围: {k_min}-{k_max}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ t-SNE可视化已保存: {save_path}")

def generate_baseline_labels(states, kmeans, cluster_horizons):
    """
    生成基准版的训练标签
    """
    print("🏷️ 生成基准版的训练标签...")

    # 预测聚类标签
    labels = kmeans.predict(states)

    # 获取对应的k值
    k_values = np.array([cluster_horizons[label] for label in labels])

    # 归一化到[0,1]
    k_min, k_max = 5, 50
    normalized_labels = (k_values - k_min) / (k_max - k_min)
    normalized_labels = normalized_labels.reshape(-1, 1).astype(np.float32)

    print(f"✓ 标签统计:")
    print(f"  原始k值范围: {k_values.min()} - {k_values.max()}")
    print(f"  归一化标签范围: {normalized_labels.min():.3f} - {normalized_labels.max():.3f}")
    print(f"  唯一k值数量: {len(np.unique(k_values))}")

    # 统计k值分布
    unique_k, counts = np.unique(k_values, return_counts=True)
    for k, count in zip(unique_k, counts):
        percentage = count / len(k_values) * 100
        print(f"  k={k}: {count} 样本 ({percentage:.1f}%)")

    return normalized_labels

def main():
    """
    主函数：执行Square任务基准版标签生成
    """
    print("🚀 AdaStep Square任务基准版标签生成 (旧算法)")
    print("="*60)

    # 配置
    data_path = "/home/yhj/桌面/ACT/adastep_extension/robomimic_data/square/mh/low_dim_v15.hdf5"
    output_dir = Path("/home/yhj/桌面/ACT/adastep_extension/experiments/results_square_baseline")
    output_dir.mkdir(exist_ok=True)

    # 1. 加载数据
    states, actions, demo_names = load_square_data(data_path, max_episodes=20)

    # 2. 执行基准版聚类分析 (K=3, 固定阈值)
    kmeans, cluster_horizons, complexities = baseline_clustering_analysis(
        states, actions, num_clusters=3, error_threshold=0.15
    )

    # 3. 生成t-SNE可视化
    tsne_path = output_dir / "square_baseline_tsne_visualization.png"
    create_tsne_visualization(states, kmeans.labels_, cluster_horizons, tsne_path)

    # 4. 生成基准版标签
    labels = generate_baseline_labels(states, kmeans, cluster_horizons)

    # 5. 保存结果
    # 保存标签
    np.save(output_dir / "horizon_labels_baseline.npy", labels)

    # 保存聚类分析器
    analyzer_data = {
        'kmeans': kmeans,
        'cluster_horizons': cluster_horizons,
        'complexities': complexities,
        'num_clusters': 3,
        'error_threshold': 0.15
    }

    with open(output_dir / "cluster_analyzer_baseline.pkl", 'wb') as f:
        pickle.dump(analyzer_data, f)

    print(f"\n✓ 所有结果已保存到: {output_dir}")
    print(f"  - 标签文件: horizon_labels_baseline.npy")
    print(f"  - 分析器文件: cluster_analyzer_baseline.pkl")
    print(f"  - 可视化文件: square_baseline_tsne_visualization.png")

    # 6. 总结
    k_values = list(cluster_horizons.values())
    unique_k_count = len(set(k_values))

    print(f"\n📊 基准版质量检查:")
    print(f"  📊 k值唯一数量: {unique_k_count} 种")
    print(f"  📊 k值分布: {sorted(k_values)}")

    if unique_k_count <= 2:
        print(f"  ✅ 符合预期：旧算法k值分布单一，缺乏自适应性")
    else:
        print(f"  ⚠️  不符合预期：旧算法k值过于多样")

if __name__ == "__main__":
    main()