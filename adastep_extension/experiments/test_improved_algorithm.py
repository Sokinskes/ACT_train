"""
测试改进的AdaStep算法
验证状态级自适应(而非任务级)
"""
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adastep_module import StateClusterAnalyzer

def generate_synthetic_trajectories(num_samples=1000, seq_len=100, action_dim=7):
    """
    生成合成轨迹数据
    
    包含三种模式:
    1. 线性运动(简单) - 可用大k
    2. 曲线运动(中等) - 用中等k
    3. 复杂抖动(困难) - 需小k
    """
    states = []
    actions = []
    
    for i in range(num_samples):
        # 随机选择轨迹类型
        traj_type = i % 3
        
        # 生成状态(随机)
        state = np.random.randn(action_dim)
        states.append(state)
        
        # 生成动作序列
        if traj_type == 0:  # 线性运动
            start = np.random.randn(action_dim) * 0.1
            end = start + np.random.randn(action_dim) * 2.0
            traj = np.linspace(start, end, seq_len)
            
        elif traj_type == 1:  # 曲线运动(正弦)
            t = np.linspace(0, 2*np.pi, seq_len)
            amplitude = np.random.rand(action_dim) * 0.5
            frequency = np.random.rand(action_dim) * 2 + 1
            traj = np.sin(t[:, None] * frequency) * amplitude
            
        else:  # 复杂抖动
            traj = np.cumsum(np.random.randn(seq_len, action_dim) * 0.3, axis=0)
        
        actions.append(traj)
    
    return np.array(states), np.array(actions)


def main():
    print("=" * 70)
    print("测试改进的AdaStep算法")
    print("=" * 70)
    
    # 生成测试数据
    print("\n📊 生成测试数据...")
    states, actions = generate_synthetic_trajectories(num_samples=1000, seq_len=100)
    print(f"  states shape: {states.shape}")
    print(f"  actions shape: {actions.shape}")
    
    # 测试旧算法(K=3, threshold=0.15)
    print("\n" + "=" * 70)
    print("【旧算法】K=3, threshold=0.15")
    print("=" * 70)
    
    old_analyzer = StateClusterAnalyzer(num_clusters=3, error_threshold=0.15)
    old_analyzer.fit_clusters(states)
    old_horizons = old_analyzer.pareto_analysis(
        states, actions, k_min=5, k_max=50, sample_size=100
    )
    
    # 统计k值分布
    old_labels = old_analyzer.kmeans.predict(states)
    old_k_values = np.array([old_horizons[l] for l in old_labels])
    unique_k, counts = np.unique(old_k_values, return_counts=True)
    
    print(f"\n📈 旧算法k值分布:")
    for k, count in zip(unique_k, counts):
        percentage = count / len(old_k_values) * 100
        print(f"  k={k:2d}: {count:4d}个样本 ({percentage:5.1f}%)")
    print(f"  k值种类数: {len(unique_k)}")
    print(f"  k值标准差: {np.std(old_k_values):.2f}")
    
    # 测试新算法(K=10, threshold=0.5作为百分位数)
    print("\n" + "=" * 70)
    print("【新算法】K=10, threshold=0.5 (50%分位数)")
    print("=" * 70)
    
    new_analyzer = StateClusterAnalyzer(num_clusters=10, error_threshold=0.5)
    new_analyzer.fit_clusters(states)
    new_horizons = new_analyzer.pareto_analysis(
        states, actions, k_min=5, k_max=50, sample_size=100
    )
    
    # 统计k值分布
    new_labels = new_analyzer.kmeans.predict(states)
    new_k_values = np.array([new_horizons[l] for l in new_labels])
    unique_k, counts = np.unique(new_k_values, return_counts=True)
    
    print(f"\n📈 新算法k值分布:")
    for k, count in zip(unique_k, counts):
        percentage = count / len(new_k_values) * 100
        print(f"  k={k:2d}: {count:4d}个样本 ({percentage:5.1f}%)")
    print(f"  k值种类数: {len(unique_k)}")
    print(f"  k值标准差: {np.std(new_k_values):.2f}")
    
    # 对比结论
    print("\n" + "=" * 70)
    print("对比结论")
    print("=" * 70)
    
    old_diversity = len(np.unique(old_k_values))
    new_diversity = len(np.unique(new_k_values))
    
    print(f"  k值多样性: 旧算法 {old_diversity}种 → 新算法 {new_diversity}种")
    print(f"  k值标准差: 旧算法 {np.std(old_k_values):.2f} → 新算法 {np.std(new_k_values):.2f}")
    
    if new_diversity > old_diversity and np.std(new_k_values) > np.std(old_k_values):
        print("\n✅ 改进成功！新算法实现了状态级自适应")
    else:
        print("\n⚠️  改进不明显,需要进一步调整参数")


if __name__ == "__main__":
    main()
