"""
AdaStep Square任务改进版标签生成
================================

使用真正的Pareto Analysis而不是线性映射：
1. K-Means聚类 (K=10)
2. 计算每个Cluster的开环误差增长率
3. 根据误差增长率分配k值
4. 添加t-SNE可视化debug
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

# 现在可以导入core模块
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

def calculate_open_loop_error(actions, k):
    """
    计算开环误差 (Open-loop Error)
    使用k步预测的线性外推与真实轨迹的误差
    """
    if k >= len(actions):
        return 0.0

    # 使用前k步动作
    action_chunk = actions[:k]

    # 线性外推：用前k步的平均速度预测下一步
    if k >= 2:
        velocities = np.diff(action_chunk, axis=0)
        avg_velocity = np.mean(velocities, axis=0)

        # 预测下一步动作
        predicted_next = action_chunk[-1] + avg_velocity

        # 计算与真实值的误差
        if k < len(actions):
            true_next = actions[k]
            error = np.linalg.norm(predicted_next - true_next)
        else:
            error = 0.0
    else:
        # k太小，无法计算速度
        error = np.linalg.norm(action_chunk[-1])  # 使用动作幅度作为误差

    return error

def improved_pareto_analysis(states, actions, k_min=5, k_max=50, k_candidates=None):
    """
    改进的Pareto分析：基于开环误差增长率分配k值
    """
    if k_candidates is None:
        k_candidates = np.arange(k_min, k_max + 1, 5)  # [5,10,15,...,50]

    print(f"📈 执行改进Pareto分析 (K-Means K=10)...")

    # 1. K-Means聚类 (K=10，增加粒度)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(states)

    print(f"✓ K-Means聚类完成，各类样本数:")
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} 样本")

    # 2. 计算每个Cluster的误差增长特性
    cluster_error_profiles = {}

    for cluster_id in range(10):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            cluster_error_profiles[cluster_id] = {'errors': [], 'growth_rate': 0.0}
            continue

        # 采样 (避免计算量过大)
        sample_size = min(100, len(cluster_indices))
        sampled_indices = np.random.choice(cluster_indices, sample_size, replace=False)

        errors_by_k = {}

        for k in k_candidates:
            cluster_errors = []

            for idx in sampled_indices:
                # 获取该状态点对应的动作序列 (前后各k步)
                start_idx = max(0, idx - k//2)
                end_idx = min(len(actions), idx + k//2 + 1)

                if end_idx - start_idx >= k:
                    action_seq = actions[start_idx:end_idx]
                    error = calculate_open_loop_error(action_seq, k)
                    cluster_errors.append(error)

            if cluster_errors:
                errors_by_k[k] = np.mean(cluster_errors)
            else:
                errors_by_k[k] = 0.0

        # 计算误差增长率 (Lipschitz常数)
        k_values = np.array(list(errors_by_k.keys()))
        error_values = np.array(list(errors_by_k.values()))

        if len(k_values) >= 2:
            # 线性拟合：error = a * k + b
            coeffs = np.polyfit(k_values, error_values, 1)
            growth_rate = coeffs[0]  # 斜率a
        else:
            growth_rate = 0.0

        cluster_error_profiles[cluster_id] = {
            'errors': errors_by_k,
            'growth_rate': growth_rate,
            'sample_count': len(sampled_indices)
        }

        print(f"  Cluster {cluster_id}: 增长率={growth_rate:.6f}, 样本数={len(sampled_indices)}")

    # 3. 根据增长率分配k值
    growth_rates = [profile['growth_rate'] for profile in cluster_error_profiles.values()]

    # 归一化增长率到[0,1]
    if max(growth_rates) > min(growth_rates):
        normalized_rates = (np.array(growth_rates) - min(growth_rates)) / (max(growth_rates) - min(growth_rates))
    else:
        normalized_rates = np.zeros(len(growth_rates))

    # 分配k值：增长率越高(误差增加越快) -> k值越小
    cluster_horizons = {}
    for i, cluster_id in enumerate(range(10)):
        rate = normalized_rates[i]

        # 反比例映射：rate=0 -> k_max, rate=1 -> k_min
        k_value = k_max - rate * (k_max - k_min)

        # 对齐到候选k值
        k_value = min(k_candidates, key=lambda x: abs(x - k_value))

        cluster_horizons[cluster_id] = int(k_value)

    print(f"✓ Pareto分析完成！各聚类最优步长:")
    sorted_clusters = sorted(cluster_horizons.items(), key=lambda x: x[1])
    for cid, k in sorted_clusters:
        growth = cluster_error_profiles[cid]['growth_rate']
        print(f"  Cluster {cid}: k={k:2d} (增长率={growth:.6f})")

    # 统计k值分布
    k_values = list(cluster_horizons.values())
    print(f"  k值分布: min={min(k_values)}, max={max(k_values)}, unique={len(set(k_values))}")

    return kmeans, cluster_horizons, cluster_error_profiles

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
    plt.title('Square任务状态聚类与k值分配 (t-SNE可视化)')
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

def generate_improved_labels(states, kmeans, cluster_horizons):
    """
    生成改进的训练标签
    """
    print("🏷️ 生成改进的训练标签...")

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
    主函数：执行完整的Square任务标签生成流程
    """
    print("🚀 AdaStep Square任务改进版标签生成")
    print("="*60)

    # 配置
    data_path = "/home/yhj/桌面/ACT/adastep_extension/robomimic_data/square/mh/low_dim_v15.hdf5"
    output_dir = Path("/home/yhj/桌面/ACT/adastep_extension/experiments/results_square_improved")
    output_dir.mkdir(exist_ok=True)

    # 1. 加载数据
    states, actions, demo_names = load_square_data(data_path, max_episodes=20)

    # 2. 执行改进的Pareto分析
    kmeans, cluster_horizons, error_profiles = improved_pareto_analysis(states, actions)

    # 3. 生成t-SNE可视化
    tsne_path = output_dir / "square_tsne_visualization.png"
    create_tsne_visualization(states, kmeans.labels_, cluster_horizons, tsne_path)

    # 4. 生成改进的标签
    labels = generate_improved_labels(states, kmeans, cluster_horizons)

    # 5. 保存结果
    # 保存标签
    np.save(output_dir / "horizon_labels_improved.npy", labels)

    # 保存聚类分析器
    analyzer_data = {
        'kmeans': kmeans,
        'cluster_horizons': cluster_horizons,
        'error_profiles': error_profiles,
        'num_clusters': 10,
        'error_threshold': 0.1  # 保持一致
    }

    with open(output_dir / "cluster_analyzer_improved.pkl", 'wb') as f:
        pickle.dump(analyzer_data, f)

    print(f"\n✓ 所有结果已保存到: {output_dir}")
    print(f"  - 标签文件: horizon_labels_improved.npy")
    print(f"  - 分析器文件: cluster_analyzer_improved.pkl")
    print(f"  - 可视化文件: square_tsne_visualization.png")

    # 6. 总结
    k_values = list(cluster_horizons.values())
    has_small_k = any(k <= 10 for k in k_values)
    has_large_k = any(k >= 40 for k in k_values)

    print(f"\n📊 质量检查:")
    print(f"  ✅ 包含小k值 (≤10): {has_small_k}")
    print(f"  ✅ 包含大k值 (≥40): {has_large_k}")
    print(f"  ✅ k值多样性: {len(set(k_values))} 种不同k值")

    if has_small_k and has_large_k:
        print(f"  🎯 标签质量: 优秀 - 具备显著的状态级差异")
    else:
        print(f"  ⚠️  标签质量: 需要检查 - k值分布不够极化")

if __name__ == "__main__":
    main()