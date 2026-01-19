"""
AdaStep Square任务改进版训练脚本
================================

使用Pareto分析生成的改进标签训练HorizonPredictor
目标：实现真正的状态级适应性
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.adastep_module import HorizonPredictor

class HorizonDataset(Dataset):
    """
    地平线预测数据集
    """
    def __init__(self, states, labels):
        self.states = torch.FloatTensor(states)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]

def load_square_data_and_labels():
    """
    加载Square任务数据和改进标签
    """
    print("📂 加载Square任务数据和改进标签...")

    # 加载数据
    data_path = "/home/yhj/桌面/ACT/adastep_extension/robomimic_data/square/mh/low_dim_v15.hdf5"
    labels_path = "/home/yhj/桌面/ACT/adastep_extension/experiments/results_square_improved/horizon_labels_improved.npy"

    # 加载状态数据
    import h5py
    states_list = []
    with h5py.File(data_path, 'r') as f:
        demo_names = list(f['data'].keys())[:20]  # 使用20个轨迹

        for demo_name in demo_names:
            demo = f[f'data/{demo_name}']

            # 提取状态 (eef_pos + eef_quat)
            if 'obs/robot0_eef_pos' in demo:
                eef_pos = demo['obs/robot0_eef_pos'][()]
                eef_quat = demo['obs/robot0_eef_quat'][()]
                states = np.concatenate([eef_pos, eef_quat], axis=-1)
            else:
                states = demo['obs/robot0_joint_pos'][()]

            states_list.append(states)

    states = np.concatenate(states_list, axis=0)

    # 加载标签
    labels = np.load(labels_path)

    print(f"✓ 数据加载完成:")
    print(f"  状态维度: {states.shape}")
    print(f"  标签维度: {labels.shape}")
    print(f"  标签范围: {labels.min():.3f} - {labels.max():.3f}")

    return states, labels

def train_horizon_predictor(states, labels, epochs=100, batch_size=64, lr=1e-3):
    """
    训练HorizonPredictor模型
    """
    print(f"🚀 开始训练HorizonPredictor (epochs={epochs}, batch_size={batch_size})...")

    # 创建数据集
    dataset = HorizonDataset(states, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    input_dim = states.shape[1]
    model = HorizonPredictor(input_dim=input_dim, hidden_dim=128)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练历史
    train_losses = []
    val_losses = []

    # 训练循环
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for batch_states, batch_labels in dataloader:
            # 前向传播
            predictions = model(batch_states)
            loss = criterion(predictions, batch_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count
        train_losses.append(avg_epoch_loss)

        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {avg_epoch_loss:.6f}")

    print(f"✓ 训练完成！最终损失: {train_losses[-1]:.6f}")

    return model, train_losses

def evaluate_model(model, states, labels):
    """
    评估模型性能
    """
    print("📊 评估模型性能...")

    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(states))
        predictions = predictions.numpy()

    # 计算指标
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)

    # 反归一化预测值 (0-1 -> 5-50)
    pred_k_values = predictions * 45 + 5  # (pred * (50-5)) + 5
    true_k_values = labels * 45 + 5

    print(f"✓ 评估结果:")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  预测k值范围: {pred_k_values.min():.1f} - {pred_k_values.max():.1f}")
    print(f"  真实k值范围: {true_k_values.min():.1f} - {true_k_values.max():.1f}")

    # 统计预测k值的分布
    unique_pred_k = np.unique(np.round(pred_k_values).astype(int))
    print(f"  预测的唯一k值: {sorted(unique_pred_k)}")

    return predictions, mae, r2

def create_training_visualization(train_losses, predictions, labels, save_path):
    """
    创建训练可视化
    """
    print("🎨 生成训练可视化...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 训练损失曲线
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_title('训练损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True, alpha=0.3)

    # 2. 预测vs真实值散点图
    ax2.scatter(labels, predictions, alpha=0.6, s=20, c='red')
    ax2.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', linewidth=2)
    ax2.set_title('预测值 vs 真实值')
    ax2.set_xlabel('真实标签')
    ax2.set_ylabel('预测标签')
    ax2.grid(True, alpha=0.3)

    # 3. 预测误差分布
    errors = predictions - labels
    ax3.hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_title('预测误差分布')
    ax3.set_xlabel('误差')
    ax3.set_ylabel('频次')
    ax3.grid(True, alpha=0.3)

    # 4. k值预测分布
    pred_k = predictions * 45 + 5
    true_k = labels * 45 + 5

    ax4.hist(pred_k, bins=20, alpha=0.7, label='预测', color='blue', edgecolor='black')
    ax4.hist(true_k, bins=20, alpha=0.7, label='真实', color='orange', edgecolor='black')
    ax4.set_title('k值分布对比')
    ax4.set_xlabel('k值')
    ax4.set_ylabel('频次')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 训练可视化已保存: {save_path}")

def main():
    """
    主函数：训练Square任务的改进HorizonPredictor
    """
    print("🚀 AdaStep Square任务改进版训练")
    print("="*60)

    # 配置
    output_dir = Path("/home/yhj/桌面/ACT/adastep_extension/experiments/results_square_improved")
    output_dir.mkdir(exist_ok=True)

    # 1. 加载数据和标签
    states, labels = load_square_data_and_labels()

    # 2. 训练模型
    model, train_losses = train_horizon_predictor(states, labels, epochs=100, batch_size=64, lr=1e-3)

    # 3. 评估模型
    predictions, mae, r2 = evaluate_model(model, states, labels)

    # 4. 创建可视化
    viz_path = output_dir / "square_training_visualization.png"
    create_training_visualization(train_losses, predictions, labels, viz_path)

    # 5. 保存模型
    model_path = output_dir / "horizon_predictor_square_improved.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': states.shape[1],
        'mae': mae,
        'r2': r2,
        'train_losses': train_losses
    }, model_path)

    print(f"\n✓ 模型已保存: {model_path}")
    print(f"✓ 所有结果已保存到: {output_dir}")

    # 6. 总结
    print(f"\n📊 训练总结:")
    print(f"  最终训练损失: {train_losses[-1]:.6f}")
    print(f"  验证MAE: {mae:.4f}")
    print(f"  验证R²: {r2:.4f}")

    if r2 > 0.8:
        print(f"  🎯 模型质量: 优秀")
    elif r2 > 0.6:
        print(f"  ✅ 模型质量: 良好")
    else:
        print(f"  ⚠️  模型质量: 需要改进")

if __name__ == "__main__":
    main()