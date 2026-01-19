"""
AdaStep 核心模块 - 独立扩展版本
不修改原始ACT代码，可独立使用

核心组件:
1. HorizonPredictor - 步长预测器（3层MLP）
2. StateClusterAnalyzer - 状态聚类分析器
3. AdaptiveHorizonLoss - 联合损失函数
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
    轻量级步长预测器 - 3层MLP
    
    输入: ACT的Latent Feature (通常从CVAE的mu或backbone feature提取)
    输出: 归一化的步长预测值 [0, 1]，映射到 [k_min, k_max]
    
    设计理念:
    - 寄生式设计：不增加主干网络负担
    - 极轻量：参数量 < 1% of ACT
    - 可解释：通过聚类标签监督学习
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        # 3层MLP架构
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Xavier初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            latent: [B, input_dim] - ACT的latent feature
        
        Returns:
            normalized_k: [B, 1] - 归一化步长，值域[0, 1]
        """
        x = self.relu(self.fc1(latent))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
    def predict_horizon(self, latent: torch.Tensor, 
                       k_min: int = 5, k_max: int = 50) -> torch.Tensor:
        """
        预测具体步长值
        
        Args:
            latent: 输入特征
            k_min: 最小步长（复杂状态，如插孔）
            k_max: 最大步长（简单状态，如移动）
        
        Returns:
            k: [B] - 整数步长
        """
        normalized = self.forward(latent)
        k = normalized * (k_max - k_min) + k_min
        return k.squeeze(-1).round().long()


class StateClusterAnalyzer:
    """
    状态聚类分析器
    
    核心思想（来自AdaStep论文）:
    1. K-Means聚类：将状态分为"简单/中等/复杂"几类
    2. 帕累托分析：为每类找到"误差允许下的最大步长"
    3. 生成标签：为MLP提供监督信号
    
    物理意义:
    - Cluster 0 (远离目标): k=45-50, 大步快走
    - Cluster 1 (接近目标): k=15-25, 小心翼翼
    - Cluster 2 (接触/插入): k=5-10, 精雕细琢
    """
    
    def __init__(self, num_clusters: int = 3, error_threshold: float = 0.02):
        self.num_clusters = num_clusters
        self.error_threshold = error_threshold
        self.kmeans = None
        self.cluster_horizons = None
    
    def fit_clusters(self, states: np.ndarray):
        """
        执行K-Means聚类
        
        Args:
            states: [N, state_dim] - 所有状态样本
        """
        print(f"🎯 执行K-Means聚类 (K={self.num_clusters})...")
        self.kmeans = KMeans(
            n_clusters=self.num_clusters, 
            random_state=42, 
            n_init=10
        )
        self.kmeans.fit(states)
        
        labels = self.kmeans.labels_
        print(f"✓ 聚类完成！各类样本数:")
        for i in range(self.num_clusters):
            count = np.sum(labels == i)
            print(f"  Cluster {i}: {count} 样本")
    
    def pareto_analysis(self, 
                       states: np.ndarray,
                       action_sequences: np.ndarray,
                       k_min: int = 5,
                       k_max: int = 50,
                       sample_size: int = 200) -> Dict[int, int]:
        """
        帕累托分析 - 核心算法
        
        思想:
        对于每个状态，尝试不同的k值，计算"执行k步后的累积误差"
        找到"误差 < threshold 的最大k值"
        
        Args:
            states: 状态样本
            action_sequences: [N, seq_len, action_dim] - 完整动作序列
            k_min, k_max: 步长范围
            sample_size: 每类采样数量（加速计算）
        
        Returns:
            cluster_horizons: {cluster_id: optimal_k}
        """
        if self.kmeans is None:
            raise ValueError("请先调用 fit_clusters()！")
        
        print(f"📈 执行帕累托分析...")
        labels = self.kmeans.predict(states)
        cluster_horizons = {}
        
        for cluster_id in range(self.num_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                cluster_horizons[cluster_id] = k_min
                continue
            
            # 采样加速
            if len(cluster_indices) > sample_size:
                cluster_indices = np.random.choice(
                    cluster_indices, sample_size, replace=False
                )
            
            optimal_ks = []
            
            for idx in cluster_indices:
                seq_len = min(k_max, action_sequences.shape[1])
                
                # 尝试不同的k值（从k_min到k_max）
                best_k = k_min
                min_relative_error = float('inf')
                
                for k in range(k_min, min(seq_len, k_max) + 1, 5):  # 步长5加速
                    if k > action_sequences.shape[1]:
                        continue
                    
                    # 计算执行k步的相对误差
                    action_chunk = action_sequences[idx, :k]
                    
                    # 使用动作的累积变化量作为复杂度指标
                    action_changes = np.diff(action_chunk, axis=0)
                    complexity = np.linalg.norm(action_changes, axis=1).mean()
                    
                    # 归一化复杂度（避免数值过大）
                    action_magnitude = np.linalg.norm(action_chunk, axis=1).mean() + 1e-6
                    relative_complexity = complexity / action_magnitude
                    
                    # 如果复杂度低于阈值，说明可以使用更大的k
                    if relative_complexity < self.error_threshold:
                        best_k = k
                        min_relative_error = relative_complexity
                
                optimal_ks.append(best_k)
            
            # 使用中位数（比平均数更鲁棒）
            cluster_horizons[cluster_id] = int(np.median(optimal_ks)) if optimal_ks else k_min
        
        print(f"✓ 帕累托分析完成！各聚类最优步长:")
        for cid, k in cluster_horizons.items():
            print(f"  Cluster {cid}: k={k}")
        
        self.cluster_horizons = cluster_horizons
        return cluster_horizons
    
    def get_labels(self, states: np.ndarray, 
                   k_min: int = 5, k_max: int = 50) -> np.ndarray:
        """
        为状态生成归一化的步长标签 [0, 1]
        
        用于训练HorizonPredictor
        """
        if self.kmeans is None or self.cluster_horizons is None:
            raise ValueError("请先完成聚类和帕累托分析！")
        
        labels = self.kmeans.predict(states)
        horizons = np.array([self.cluster_horizons[l] for l in labels])
        
        # 归一化到[0, 1]
        normalized = (horizons - k_min) / (k_max - k_min)
        return normalized.reshape(-1, 1).astype(np.float32)
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'cluster_horizons': self.cluster_horizons,
                'num_clusters': self.num_clusters,
                'error_threshold': self.error_threshold
            }, f)
        print(f"✓ 模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.cluster_horizons = data['cluster_horizons']
            self.num_clusters = data['num_clusters']
            self.error_threshold = data['error_threshold']
        print(f"✓ 模型已加载: {path}")


class AdaptiveHorizonLoss(nn.Module):
    """
    联合损失函数
    
    Loss = L_action + λ_kl * L_kl + λ_horizon * L_horizon
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
        """
        # 动作L1损失
        all_l1 = F.l1_loss(action_pred, action_gt, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        
        # 步长MSE损失
        horizon_loss = F.mse_loss(horizon_pred, horizon_gt)
        
        # 总损失
        total_loss = l1 + self.kl_weight * kl_loss + self.horizon_weight * horizon_loss
        
        return {
            'l1': l1,
            'kl': kl_loss,
            'horizon': horizon_loss,
            'loss': total_loss
        }
