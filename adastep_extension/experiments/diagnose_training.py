"""
AdaStep训练过程诊断脚本
========================

检查为什么所有训练标签都是1.0
"""

import torch
import numpy as np
import pickle
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

def diagnose_clustering_issue():
    """
    诊断聚类和标签生成的问题
    """
    print("🔍 AdaStep训练过程诊断")
    print("="*50)

    # 1. 检查聚类分析器
    print("\n1. 检查聚类分析器...")
    with open('/home/yhj/桌面/ACT/adastep_extension/experiments/results_transport_mh/stage1_clustering/cluster_analyzer.pkl', 'rb') as f:
        analyzer_data = pickle.load(f)

    print(f"✓ 聚类数: {analyzer_data['num_clusters']}")
    print(f"✓ 误差阈值: {analyzer_data['error_threshold']}")
    print(f"✓ 聚类k值分配: {analyzer_data['cluster_horizons']}")

    # 2. 检查原始数据
    print("\n2. 检查原始数据...")
    data_path = "/home/yhj/桌面/ACT/adastep_extension/robomimic_data/transport/mh/low_dim_v15.hdf5"

    with h5py.File(data_path, 'r') as f:
        demo_names = list(f['data'].keys())[:3]  # 检查前3条轨迹

        for demo_name in demo_names:
            demo = f[f'data/{demo_name}']
            actions = demo['actions'][()]
            print(f"  轨迹 {demo_name}: {len(actions)} 步, 动作维度: {actions.shape[1]}")

            # 计算动作变化
            action_diff = np.linalg.norm(np.diff(actions, axis=0), axis=1)
            print(f"    平均动作变化: {action_diff.mean():.6f}")
            print(f"    最大动作变化: {action_diff.max():.6f}")

    # 3. 手动计算复杂度
    print("\n3. 手动复杂度计算测试...")

    # 加载一个简单的轨迹片段
    with h5py.File(data_path, 'r') as f:
        demo = f[f'data/{demo_names[0]}']
        actions = demo['actions'][()][:50]  # 前50步

    # 测试不同k值的复杂度
    k_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    complexities = []

    for k in k_values:
        if k >= len(actions):
            continue

        # 计算线性偏离度
        action_chunk = actions[:k]
        start = action_chunk[0]
        end = action_chunk[-1]

        # 线性插值
        linear_traj = np.linspace(start, end, k)

        # 计算偏离
        deviations = np.linalg.norm(action_chunk - linear_traj, axis=1)
        avg_deviation = np.mean(deviations)

        # 归一化
        action_magnitude = np.linalg.norm(action_chunk, axis=1).mean() + 1e-6
        normalized_deviation = avg_deviation / action_magnitude

        complexities.append(normalized_deviation)
        print(f"  k={k:2d}: 复杂度={normalized_deviation:.6f}")

    # 4. 分析复杂度分布
    complexities = np.array(complexities)
    print("\n复杂度统计:")
    print(f"  范围: {complexities.min():.6f} - {complexities.max():.6f}")
    print(f"  均值: {complexities.mean():.6f}")
    print(f"  标准差: {complexities.std():.6f}")

    # 5. 检查阈值设置
    print("\n4. 阈值分析...")
    threshold = analyzer_data['error_threshold']
    print(f"  当前阈值: {threshold}")

    # 基于当前复杂度计算合适的阈值
    if len(complexities) > 0:
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            val = np.percentile(complexities, p)
            print(f"  {p}百分位数: {val:.6f}")

    # 6. 建议
    print("\n5. 诊断结论和建议...")
    all_k_50 = all(v == 50 for v in analyzer_data['cluster_horizons'].values())

    if all_k_50:
        print("❌ 问题: 所有聚类都被分配了最大k值(50)")
        print("💡 可能原因:")
        print("   1. Transport任务相对简单，所有状态都被认为是低复杂度")
        print("   2. 复杂度计算可能有问题")
        print("   3. 阈值设置可能过高")
        print("   4. 聚类算法没有找到有意义的聚类")

        print("🔧 建议解决方案:")
        print("   1. 降低error_threshold (当前0.4)，尝试0.1或0.2")
        print("   2. 增加聚类数 (当前3)，尝试5或更多")
        print("   3. 检查复杂度计算逻辑")
        print("   4. 考虑使用不同的复杂度度量方法")
    else:
        print("✅ 聚类分配看起来正常")
        print("⚠️  但所有标签都是1.0，说明k值范围映射有问题")

if __name__ == "__main__":
    diagnose_clustering_issue()