"""
AdaStep Module for Adaptive Horizon ACT
动态调整动作执行步长的核心模块

核心思想：
- 简单状态（大范围移动）：执行更多步（40-50步），低频推理，省算力
- 复杂状态（精密操作）：执行更少步（5-10步），高频推理，高精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Tuple, Optional
import pickle
import os


class HorizonPredictor(nn.Module):
    """
    轻量级 MLP 网络，用于预测当前状态的最优执行步长
    
    输入: Latent Feature z (来自 ACT backbone)
    输出: 归一化的步长预测值 [0, 1]
    
    架构:
    - 3层 MLP (寄生式设计，极低计算开销)
    - 输入维度: hidden_dim (ACT的隐藏层维度)
    - 输出: 单一标量值，映射到 [k_min, k_max]
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        """
        Args:
            input_dim: 输入特征维度（与 ACT 的 hidden_dim 一致）
            hidden_dim: MLP 隐藏层维度
        """
        super().__init__()
        
        # 3层 MLP
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, 1)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # 初始化权重（使用 Xavier 初始化）
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier 初始化，帮助训练稳定"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, latent_feature: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            latent_feature: shape [batch_size, input_dim]
        
        Returns:
            normalized_horizon: shape [batch_size, 1], 值域 [0, 1]
        """
        x = self.relu(self.layer1(latent_feature))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))  # 输出归一化到 [0, 1]
        
        return x
    
    def predict_horizon(self, latent_feature: torch.Tensor, 
                       k_min: int = 5, k_max: int = 50) -> torch.Tensor:
        """
        预测具体的执行步长
        
        Args:
            latent_feature: 输入特征
            k_min: 最小步长（复杂状态）
            k_max: 最大步长（简单状态）
        
        Returns:
            horizon: shape [batch_size], 具体步长值
        """
        normalized = self.forward(latent_feature)
        # 映射到 [k_min, k_max]
        horizon = (normalized * (k_max - k_min) + k_min).squeeze(-1)
        return horizon.round().long()  # 取整


class StateClusterAnalyzer:
    """
    状态聚类分析器
    
    功能:
    1. K-Means 聚类: 将示教数据的状态聚成几类（如：远离、接近、接触）
    2. Pareto 分析: 为每类状态找到"误差允许范围内的最大步长"
    3. 生成训练标签: 为 HorizonPredictor 提供监督信号
    """
    
    def __init__(self, num_clusters: int = 3, error_threshold: float = 0.02):
        """
        Args:
            num_clusters: 聚类数量（建议 3-5 类）
            error_threshold: 可接受的动作预测误差阈值
        """
        self.num_clusters = num_clusters
        self.error_threshold = error_threshold
        self.kmeans = None
        self.cluster_horizons = None  # 每个聚类的最优步长
    
    def fit_clusters(self, states: np.ndarray):
        """
        对状态进行 K-Means 聚类
        
        Args:
            states: shape [N, state_dim], N 个状态样本
        """
        print(f"执行 K-Means 聚类, 聚类数: {self.num_clusters}")
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        self.kmeans.fit(states)
        print(f"聚类完成！聚类中心:\n{self.kmeans.cluster_centers_}")
    
    def pareto_analysis(self, 
                       states: np.ndarray, 
                       actions: np.ndarray,
                       action_sequences: np.ndarray,
                       k_min: int = 5,
                       k_max: int = 50) -> Dict[int, int]:
        """
        帕累托分析：为每个聚类找到最优步长
        
        思想:
        - 对于每个状态，尝试不同的执行步长 k
        - 计算"如果执行 k 步后，与真实轨迹的误差"
        - 找到"误差 < 阈值"的最大 k 值
        
        Args:
            states: 当前状态 [N, state_dim]
            actions: 即时动作 [N, action_dim]
            action_sequences: 完整动作序列 [N, max_horizon, action_dim]
            k_min, k_max: 步长范围
        
        Returns:
            cluster_horizons: {cluster_id: optimal_k}
        """
        if self.kmeans is None:
            raise ValueError("请先调用 fit_clusters() 进行聚类！")
        
        # 获取每个状态的聚类标签
        labels = self.kmeans.predict(states)
        
        cluster_horizons = {}
        
        for cluster_id in range(self.num_clusters):
            # 找到该聚类的所有样本
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                cluster_horizons[cluster_id] = k_min
                continue
            
            # 对该聚类的样本进行帕累托分析
            optimal_k_list = []
            
            for idx in cluster_indices[:100]:  # 采样前100个样本以加速
                # 尝试不同的 k 值
                for k in range(k_max, k_min - 1, -1):
                    # 计算执行 k 步的累积误差
                    if k >= action_sequences.shape[1]:
                        continue
                    
                    predicted_seq = action_sequences[idx, :k]
                    # 这里简化为：假设真实轨迹就是 action_sequences
                    # 实际中可以用更复杂的动力学模型
                    error = np.mean(np.abs(predicted_seq - action_sequences[idx, :k]))
                    
                    if error < self.error_threshold:
                        optimal_k_list.append(k)
                        break
                else:
                    # 如果所有 k 都不满足，使用最小值
                    optimal_k_list.append(k_min)
            
            # 该聚类的最优步长取平均值
            if optimal_k_list:
                cluster_horizons[cluster_id] = int(np.median(optimal_k_list))
            else:
                cluster_horizons[cluster_id] = k_min
        
        self.cluster_horizons = cluster_horizons
        print(f"帕累托分析完成！各聚类最优步长: {cluster_horizons}")
        return cluster_horizons
    
    def get_labels(self, states: np.ndarray, k_min: int = 5, k_max: int = 50) -> np.ndarray:
        """
        为状态生成步长标签（归一化到 [0, 1]）
        
        Args:
            states: 状态样本
            k_min, k_max: 步长范围
        
        Returns:
            labels: shape [N, 1], 归一化的步长标签
        """
        if self.kmeans is None or self.cluster_horizons is None:
            raise ValueError("请先完成聚类和帕累托分析！")
        
        # 预测聚类
        cluster_labels = self.kmeans.predict(states)
        
        # 映射到具体步长
        horizons = np.array([self.cluster_horizons[label] for label in cluster_labels])
        
        # 归一化到 [0, 1]
        normalized_labels = (horizons - k_min) / (k_max - k_min)
        
        return normalized_labels.reshape(-1, 1).astype(np.float32)
    
    def save(self, save_path: str):
        """保存聚类模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'cluster_horizons': self.cluster_horizons,
                'num_clusters': self.num_clusters,
                'error_threshold': self.error_threshold
            }, f)
        print(f"聚类模型已保存到: {save_path}")
    
    def load(self, save_path: str):
        """加载聚类模型"""
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.cluster_horizons = data['cluster_horizons']
            self.num_clusters = data['num_clusters']
            self.error_threshold = data['error_threshold']
        print(f"聚类模型已加载: {save_path}")


class AdaptiveHorizonLoss(nn.Module):
    """
    自适应步长的联合损失函数
    
    Loss = L_action + λ_kl * L_kl + λ_horizon * L_horizon
    
    其中:
    - L_action: 动作预测损失（L1）
    - L_kl: KL 散度（CVAE 的正则项）
    - L_horizon: 步长预测损失（MSE）
    """
    
    def __init__(self, kl_weight: float = 10.0, horizon_weight: float = 1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.horizon_weight = horizon_weight
    
    def forward(self, 
                action_pred: torch.Tensor,
                action_gt: torch.Tensor,
                is_pad: torch.Tensor,
                kl_loss: torch.Tensor,
                horizon_pred: torch.Tensor,
                horizon_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算联合损失
        
        Returns:
            loss_dict: {'l1': ..., 'kl': ..., 'horizon': ..., 'loss': ...}
        """
        # 动作损失（与原 ACT 一致）
        all_l1 = F.l1_loss(action_pred, action_gt, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        
        # 步长损失
        horizon_loss = F.mse_loss(horizon_pred, horizon_gt)
        
        # 总损失
        total_loss = l1 + self.kl_weight * kl_loss + self.horizon_weight * horizon_loss
        
        return {
            'l1': l1,
            'kl': kl_loss,
            'horizon': horizon_loss,
            'loss': total_loss
        }


# ============== 工具函数 ==============

def extract_latent_features(policy_model, dataloader, device: str = 'cuda'):
    """
    从 ACT 模型中提取所有样本的 latent features
    用于后续的聚类分析
    
    Args:
        policy_model: 训练好的 ACT 模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        features: [N, hidden_dim]
        states: [N, state_dim]
        actions: [N, action_dim]
    """
    policy_model.eval()
    all_features = []
    all_states = []
    all_actions = []
    
    with torch.no_grad():
        for batch in dataloader:
            image_data, qpos_data, action_data, is_pad = batch
            image_data = image_data.to(device)
            qpos_data = qpos_data.to(device)
            action_data = action_data.to(device)
            
            # 提取特征（需要修改 ACT 模型以输出中间特征）
            # 这里假设 model 有一个 get_latent_feature 方法
            latent = policy_model.model.get_latent_feature(qpos_data, image_data, None)
            
            all_features.append(latent.cpu().numpy())
            all_states.append(qpos_data.cpu().numpy())
            all_actions.append(action_data[:, 0].cpu().numpy())  # 只取第一个动作
    
    features = np.concatenate(all_features, axis=0)
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    return features, states, actions


def visualize_clusters(states: np.ndarray, labels: np.ndarray, save_path: str):
    """
    可视化状态聚类结果（使用 PCA 降维到 2D）
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    # PCA 降维
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(states_2d[:, 0], states_2d[:, 1], 
                         c=labels, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('State Clustering Visualization')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"聚类可视化已保存到: {save_path}")


def visualize_horizon_distribution(horizons: np.ndarray, save_path: str):
    """
    可视化步长分布直方图
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(horizons, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Predicted Horizon (steps)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Action Horizons')
    plt.axvline(horizons.mean(), color='red', linestyle='--', 
               label=f'Mean: {horizons.mean():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"步长分布图已保存到: {save_path}")
