"""
快速测试AdaStep扩展模块

不需要数据集，快速验证安装正确性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

print("="*70)
print("AdaStep Extension 模块测试")
print("="*70)
print()

# 测试1: 核心模块
print("✓ 测试1: 导入核心模块...")
try:
    from core.adastep_module import (
        HorizonPredictor,
        StateClusterAnalyzer,
        AdaptiveHorizonLoss
    )
    print("  ✓ HorizonPredictor")
    print("  ✓ StateClusterAnalyzer")
    print("  ✓ AdaptiveHorizonLoss")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

# 测试2: HorizonPredictor
print("\n✓ 测试2: HorizonPredictor 功能...")
try:
    predictor = HorizonPredictor(input_dim=7, hidden_dim=128)
    dummy_input = torch.randn(4, 7)
    output = predictor(dummy_input)
    horizons = predictor.predict_horizon(dummy_input, k_min=5, k_max=50)
    
    assert output.shape == (4, 1), f"输出形状错误: {output.shape}"
    assert horizons.shape == (4,), f"步长形状错误: {horizons.shape}"
    assert torch.all((horizons >= 5) & (horizons <= 50)), "步长超出范围"
    
    print(f"  ✓ 预测步长示例: {horizons.tolist()}")
    print(f"  ✓ 参数量: {sum(p.numel() for p in predictor.parameters()):,}")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

# 测试3: StateClusterAnalyzer
print("\n✓ 测试3: StateClusterAnalyzer 功能...")
try:
    analyzer = StateClusterAnalyzer(num_clusters=3, error_threshold=0.02)
    
    # 生成模拟数据
    dummy_states = np.random.randn(200, 7)
    analyzer.fit_clusters(dummy_states)
    
    # 模拟帕累托分析
    dummy_actions = np.random.randn(200, 7)
    dummy_sequences = np.random.randn(200, 100, 7)
    cluster_horizons = analyzer.pareto_analysis(
        dummy_states, dummy_sequences, k_min=5, k_max=50
    )
    
    # 生成标签
    labels = analyzer.get_labels(dummy_states, k_min=5, k_max=50)
    
    print(f"  ✓ 聚类中心: {analyzer.kmeans.cluster_centers_.shape}")
    print(f"  ✓ 各聚类最优步长: {cluster_horizons}")
    print(f"  ✓ 标签范围: [{labels.min():.2f}, {labels.max():.2f}]")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 数据加载器
print("\n✓ 测试4: 导入数据加载器...")
try:
    from data.robomimic_loader import RobomimicSquareDataset, download_robomimic_dataset
    print("  ✓ RobomimicSquareDataset")
    print("  ✓ download_robomimic_dataset")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

# 测试5: 验证器
print("\n✓ 测试5: 导入离线验证器...")
try:
    from validation.offline_validator import OfflineValidator
    print("  ✓ OfflineValidator")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

# 测试6: AdaptiveHorizonLoss
print("\n✓ 测试6: AdaptiveHorizonLoss 功能...")
try:
    loss_fn = AdaptiveHorizonLoss(kl_weight=10.0, horizon_weight=1.0)
    
    action_pred = torch.randn(4, 100, 7)
    action_gt = torch.randn(4, 100, 7)
    is_pad = torch.zeros(4, 100).bool()
    kl_loss = torch.tensor(0.5)
    horizon_pred = torch.randn(4, 1)
    horizon_gt = torch.randn(4, 1)
    
    loss_dict = loss_fn(action_pred, action_gt, is_pad, kl_loss, horizon_pred, horizon_gt)
    
    assert 'loss' in loss_dict, "缺少总损失"
    assert 'horizon' in loss_dict, "缺少步长损失"
    
    print(f"  ✓ 总损失: {loss_dict['loss'].item():.4f}")
    print(f"  ✓ 步长损失: {loss_dict['horizon'].item():.4f}")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ 所有模块测试通过！")
print("="*70)
print("\n下一步:")
print("  1. 下载Robomimic数据集")
print("  2. 运行完整实验:")
print("     cd experiments")
print("     python run_full_experiment.py --data_path <path_to_square_ph.hdf5>")
print()
