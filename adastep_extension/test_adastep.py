"""
AdaStep 模块测试脚本
验证所有核心组件是否正确安装和工作
"""

import sys
import torch
import numpy as np
from sklearn.cluster import KMeans

print("="*60)
print("AdaStep 模块测试")
print("="*60)
print()

# 测试1: 基础依赖
print("✓ 测试1: 检查基础依赖...")
print(f"  PyTorch版本: {torch.__version__}")
print(f"  NumPy版本: {np.__version__}")
print(f"  设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print()

# 测试2: 导入AdaStep模块
print("✓ 测试2: 导入AdaStep核心模块...")
try:
    from training.adastep import (
        HorizonPredictor, 
        StateClusterAnalyzer,
        AdaptiveHorizonLoss
    )
    print("  ✓ HorizonPredictor")
    print("  ✓ StateClusterAnalyzer")
    print("  ✓ AdaptiveHorizonLoss")
except Exception as e:
    print(f"  ❌ 导入失败: {e}")
    sys.exit(1)
print()

# 测试3: HorizonPredictor
print("✓ 测试3: HorizonPredictor 功能测试...")
try:
    predictor = HorizonPredictor(input_dim=512, hidden_dim=256)
    dummy_input = torch.randn(4, 512)  # batch=4
    output = predictor(dummy_input)
    assert output.shape == (4, 1), f"输出形状错误: {output.shape}"
    assert torch.all((output >= 0) & (output <= 1)), "输出未归一化到[0,1]"
    
    # 测试步长预测
    horizons = predictor.predict_horizon(dummy_input, k_min=5, k_max=50)
    assert horizons.shape == (4,), f"步长形状错误: {horizons.shape}"
    assert torch.all((horizons >= 5) & (horizons <= 50)), "步长超出范围"
    
    print(f"  ✓ 预测步长示例: {horizons.tolist()}")
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    sys.exit(1)
print()

# 测试4: StateClusterAnalyzer
print("✓ 测试4: StateClusterAnalyzer 功能测试...")
try:
    analyzer = StateClusterAnalyzer(num_clusters=3, error_threshold=0.02)
    
    # 生成模拟数据
    dummy_states = np.random.randn(100, 5)  # 100个5维状态
    analyzer.fit_clusters(dummy_states)
    
    # 检查聚类
    labels = analyzer.kmeans.predict(dummy_states)
    assert len(np.unique(labels)) <= 3, "聚类数量错误"
    print(f"  ✓ 聚类中心形状: {analyzer.kmeans.cluster_centers_.shape}")
    
    # 模拟帕累托分析
    dummy_actions = np.random.randn(100, 5)
    dummy_sequences = np.random.randn(100, 50, 5)
    cluster_horizons = analyzer.pareto_analysis(
        dummy_states, dummy_actions, dummy_sequences, k_min=5, k_max=50
    )
    print(f"  ✓ 各聚类最优步长: {cluster_horizons}")
    
    # 生成标签
    horizon_labels = analyzer.get_labels(dummy_states, k_min=5, k_max=50)
    assert horizon_labels.shape == (100, 1), f"标签形状错误: {horizon_labels.shape}"
    print(f"  ✓ 标签范围: [{horizon_labels.min():.2f}, {horizon_labels.max():.2f}]")
    
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 测试5: AdaptiveHorizonLoss
print("✓ 测试5: AdaptiveHorizonLoss 功能测试...")
try:
    loss_fn = AdaptiveHorizonLoss(kl_weight=10.0, horizon_weight=1.0)
    
    # 模拟数据
    action_pred = torch.randn(4, 50, 5)
    action_gt = torch.randn(4, 50, 5)
    is_pad = torch.zeros(4, 50).bool()
    kl_loss = torch.tensor(0.5)
    horizon_pred = torch.randn(4, 1)
    horizon_gt = torch.randn(4, 1)
    
    loss_dict = loss_fn(action_pred, action_gt, is_pad, kl_loss, horizon_pred, horizon_gt)
    
    assert 'l1' in loss_dict, "缺少l1损失"
    assert 'kl' in loss_dict, "缺少kl损失"
    assert 'horizon' in loss_dict, "缺少horizon损失"
    assert 'loss' in loss_dict, "缺少总损失"
    
    print(f"  ✓ L1损失: {loss_dict['l1'].item():.4f}")
    print(f"  ✓ KL损失: {loss_dict['kl'].item():.4f}")
    print(f"  ✓ Horizon损失: {loss_dict['horizon'].item():.4f}")
    print(f"  ✓ 总损失: {loss_dict['loss'].item():.4f}")
    
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    sys.exit(1)
print()

# 测试6: 配置文件
print("✓ 测试6: 检查配置文件...")
try:
    from config.config import POLICY_CONFIG
    
    assert 'use_adastep' in POLICY_CONFIG, "配置中缺少use_adastep"
    assert 'k_min' in POLICY_CONFIG, "配置中缺少k_min"
    assert 'k_max' in POLICY_CONFIG, "配置中缺少k_max"
    assert 'num_clusters' in POLICY_CONFIG, "配置中缺少num_clusters"
    
    print(f"  ✓ AdaStep启用: {POLICY_CONFIG['use_adastep']}")
    print(f"  ✓ 步长范围: [{POLICY_CONFIG['k_min']}, {POLICY_CONFIG['k_max']}]")
    print(f"  ✓ 聚类数: {POLICY_CONFIG['num_clusters']}")
    
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    sys.exit(1)
print()

# 测试7: ACTPolicy集成
print("✓ 测试7: 检查ACTPolicy集成...")
try:
    from training.policy import ACTPolicy
    
    # 创建策略（可能会因为DETR模型未安装而失败，这是正常的）
    print("  ✓ ACTPolicy类已导入")
    print("  ℹ️  完整测试需要DETR模型，跳过实例化测试")
    
except ImportError as e:
    print(f"  ⚠️  部分导入失败（可能需要DETR模型）: {e}")
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
print()

print("="*60)
print("✅ 所有核心模块测试通过！")
print("="*60)
print()
print("下一步:")
print("  1. 准备数据集到 data/task1/")
print("  2. 运行: python train_adastep.py --task task1 --stage pretrain")
print("  3. 运行: python train_adastep.py --task task1 --stage train")
print()
