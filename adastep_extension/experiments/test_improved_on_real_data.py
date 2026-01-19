"""
用改进算法重新分析现有数据
验证状态级自适应效果
"""
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adastep_module import StateClusterAnalyzer
from data.robomimic_loader import RobomimicSquareDataset
from torch.utils.data import DataLoader

def reanalyze_with_improved_algorithm(task_name, data_path):
    """
    用改进算法重新分析现有数据
    """
    print(f"\n{'='*60}")
    print(f"重新分析任务: {task_name.upper()}")
    print(f"数据路径: {data_path}")
    print(f"{'='*60}")

    try:
        # 加载数据
        print("📊 加载数据...")
        dataset = RobomimicSquareDataset(
            hdf5_path=data_path,
            max_episodes=50,  # 限制为50条轨迹以加快速度
            camera_names=['agentview_image'],
            chunk_size=100
        )

        # 提取状态和动作序列
        states = []
        action_sequences = []

        for i in range(len(dataset)):
            sample = dataset[i]
            if sample is not None:
                # 从元组中提取数据: (images, qpos, actions, is_pad)
                images, qpos, actions, is_pad = sample
                
                # 提取状态 (qpos)
                state = qpos.numpy()[:7] if hasattr(qpos, 'numpy') else qpos[:7]
                states.append(state)
                
                # 提取动作序列 (取前100步)
                action_seq = actions.numpy()[:100] if hasattr(actions, 'numpy') else actions[:100]
                action_sequences.append(action_seq)

        states = np.array(states)
        action_sequences = np.array(action_sequences)

        print(f"✓ 数据加载完成: {len(states)}个状态样本")

        # 使用改进算法
        print("\n🎯 使用改进算法重新分析...")
        analyzer = StateClusterAnalyzer(num_clusters=10, error_threshold=0.5)

        # 聚类
        analyzer.fit_clusters(states)

        # 帕累托分析
        horizons = analyzer.pareto_analysis(
            states, action_sequences, k_min=5, k_max=50, sample_size=100
        )

        # 分析结果
        labels = analyzer.kmeans.predict(states)
        k_values = np.array([horizons[l] for l in labels])

        unique_k, counts = np.unique(k_values, return_counts=True)

        print(f"\n📈 改进算法结果:")
        print(f"  聚类数: {analyzer.num_clusters}")
        print(f"  动态阈值: {analyzer.error_threshold} (50%分位数)")
        print(f"  k值分布:")
        for k, count in zip(unique_k, counts):
            pct = count / len(k_values) * 100
            bar = '█' * int(pct / 5)  # 每5%一个方块
            print(f"    k={k:2d}: {count:4d}样本 ({pct:5.1f}%) {bar}")

        print(f"\n  📊 关键指标:")
        print(f"    k值种类数: {len(unique_k)} (旧算法: 1)")
        print(f"    k值标准差: {np.std(k_values):.2f} (旧算法: 0.00)")
        print(f"    k值范围: [{int(k_values.min())}, {int(k_values.max())}]")

        # 评估改进效果
        diversity_improvement = len(unique_k) - 1  # 相对于旧算法的1个k值
        std_improvement = np.std(k_values)  # 旧算法标准差为0

        print(f"\n  🎯 改进效果:")
        print(f"    多样性提升: +{diversity_improvement}种k值")
        print(f"    区分度提升: +{std_improvement:.2f}标准差")

        if len(unique_k) >= 3 and np.std(k_values) > 5:
            print(f"    ✅ 状态级自适应: 成功实现!")
        else:
            print(f"    ⚠️  改进有限: 仍需优化")

        return {
            'task': task_name,
            'old_diversity': 1,
            'new_diversity': len(unique_k),
            'old_std': 0.0,
            'new_std': float(np.std(k_values)),
            'k_distribution': dict(zip(unique_k.tolist(), counts.tolist())),
            'success': len(unique_k) >= 3 and np.std(k_values) > 5
        }

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

def main():
    print("🔬 AdaStep算法改进验证")
    print("对比: 旧算法 vs 改进算法")
    print("="*60)

    # 数据路径
    data_paths = {
        'square': '/home/yhj/桌面/ACT/adastep_extension/robomimic_data/square/mh/low_dim_v15.hdf5',
        'transport': '/home/yhj/桌面/ACT/adastep_extension/robomimic_data/transport/mh/low_dim_v15.hdf5',
        'can': '/home/yhj/桌面/ACT/adastep_extension/robomimic_data/can/mh/low_dim_v15.hdf5',
        'lift': '/home/yhj/桌面/ACT/adastep_extension/robomimic_data/lift/mh/low_dim_v15.hdf5'
    }

    results = []

    for task, data_path in data_paths.items():
        if os.path.exists(data_path):
            result = reanalyze_with_improved_algorithm(task, data_path)
            if result:
                results.append(result)
        else:
            print(f"⚠️  数据文件不存在: {data_path}")

    # 汇总对比
    print(f"\n{'='*80}")
    print("算法改进效果汇总")
    print(f"{'='*80}")

    print(f"\n{'任务':<10} {'旧多样性':>8} {'新多样性':>8} {'提升':>6} {'旧标准差':>8} {'新标准差':>8} {'状态级自适应':<12}")
    print("-"*80)

    successful_tasks = 0
    for r in results:
        diversity_gain = r['new_diversity'] - r['old_diversity']
        std_gain = r['new_std'] - r['old_std']
        adaptive_status = "✅ 是" if r['success'] else "❌ 否"

        print(f"{r['task']:<10} {r['old_diversity']:>8} {r['new_diversity']:>8} {diversity_gain:>+6} {r['old_std']:>8.2f} {r['new_std']:>8.2f} {adaptive_status:<12}")

        if r['success']:
            successful_tasks += 1

    print("-"*80)
    print(f"成功实现状态级自适应的任务数: {successful_tasks}/{len(results)}")

    if successful_tasks == len(results):
        print("\n🎉 结论: 改进算法在所有任务上都成功实现了状态级自适应！")
        print("   建议: 可以进行完整训练以获得最终模型。")
    elif successful_tasks > 0:
        print(f"\n⚠️  结论: {successful_tasks}个任务成功实现状态级自适应")
        print("   建议: 可以针对性优化未成功的任务，或直接进行训练。")
    else:
        print("\n❌ 结论: 改进算法效果不明显")
        print("   建议: 需要进一步调试算法参数。")

if __name__ == "__main__":
    main()