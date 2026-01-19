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
    状态聚类分析器 - 改进版(状态级自适应)
    
    核心思想（改进后）:
    1. K-Means聚类：将状态分为10个细粒度类别
    2. 线性偏离度量：计算轨迹的非线性程度
    3. 动态阈值：基于数据分布的百分位数
    4. 分层分配：让不同聚类获得差异化的k值
    
    改进点:
    - 聚类数量: 3 → 10 (提升状态级细粒度)
    - 复杂度度量: 动作变化率 → 线性偏离误差
    - 阈值策略: 固定值 → 动态百分位数
    """
    
    def __init__(self, num_clusters: int = 10, error_threshold: float = 0.5):
        self.num_clusters = num_clusters
        self.error_threshold = error_threshold  # 现在作为百分位数(0-1)
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
    
    def calculate_linearity_deviation(self, actions: np.ndarray, k: int) -> float:
        """
        计算k步内轨迹的线性偏离程度
        
        原理:
        - 如果轨迹是直线运动(简单),线性插值误差接近0,可用大k
        - 如果轨迹是曲线运动(复杂),线性插值误差大,需用小k
        
        Args:
            actions: [seq_len, action_dim] 动作序列
            k: 步长
        
        Returns:
            deviation: 平均偏离度(归一化)
        """
        if k >= len(actions) or k < 2:
            return 0.0
        
        # 取前k步
        action_chunk = actions[:k]
        start = action_chunk[0]
        end = action_chunk[-1]
        
        # 线性插值轨迹
        linear_traj = np.linspace(start, end, k)
        
        # 计算真实轨迹与线性插值的偏离
        deviations = np.linalg.norm(action_chunk - linear_traj, axis=1)
        avg_deviation = np.mean(deviations)
        
        # 归一化: 除以动作幅度,避免绝对值依赖
        action_magnitude = np.linalg.norm(action_chunk, axis=1).mean() + 1e-6
        normalized_deviation = avg_deviation / action_magnitude
        
        return normalized_deviation
    
    def pareto_analysis(self, 
                       states: np.ndarray,
                       action_sequences: np.ndarray,
                       k_min: int = 5,
                       k_max: int = 50,
                       sample_size: int = 200,
                       lambda_param: float = 1.0) -> Dict[int, int]:
        """
        帕累托分析 - 改进版(状态级自适应)
        
        改进思想:
        1. 使用"线性偏离度"度量复杂度(而非简单的动作变化率)
        2. 动态阈值: 基于数据分布的百分位数,而非固定值
        3. 分层分配: 让不同聚类获得差异化的k值
        
        Args:
            states: 状态样本
            action_sequences: [N, seq_len, action_dim] - 完整动作序列
            k_min, k_max: 步长范围
            sample_size: 每类采样数量
        
        Returns:
            cluster_horizons: {cluster_id: optimal_k}
        """
        if self.kmeans is None:
            raise ValueError("请先调用 fit_clusters()！")
        
        print(f"📈 执行帕累托分析(改进版)...")
        labels = self.kmeans.predict(states)
        
        # 第一阶段: 收集所有聚类的复杂度统计
        cluster_complexities = {}  # {cluster_id: avg_complexity}
        
        for cluster_id in range(self.num_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                cluster_complexities[cluster_id] = 0.0
                continue
            
            # 采样
            if len(cluster_indices) > sample_size:
                cluster_indices = np.random.choice(
                    cluster_indices, sample_size, replace=False
                )
            
            # 计算该聚类的平均复杂度(使用中等k值评估)
            k_probe = (k_min + k_max) // 2  # 用中间k值探测
            complexities = []
            
            for idx in cluster_indices:
                if k_probe < action_sequences.shape[1]:
                    deviation = self.calculate_linearity_deviation(
                        action_sequences[idx], k_probe
                    )
                    complexities.append(deviation)
            
            cluster_complexities[cluster_id] = np.mean(complexities) if complexities else 0.0
        
        # 第二阶段: 动态阈值(使用百分位数)
        all_complexities = list(cluster_complexities.values())
        all_complexities = [c for c in all_complexities if c > 0]  # 过滤空聚类
        if len(all_complexities) == 0:
            # 降级方案: 全部使用中间值
            dynamic_threshold = 0.1
        else:
            # 使用self.error_threshold作为百分位数(0.5 = 中位数)
            percentile = int(self.error_threshold * 100)
            base_threshold = np.percentile(all_complexities, percentile)
            dynamic_threshold = base_threshold * lambda_param
        
        print(f"  动态阈值: {dynamic_threshold:.4f} (基准{percentile}%分位数 * lambda {lambda_param})")
        print(f"  动态阈值: {dynamic_threshold:.4f} (基于{percentile}%分位数)")
        
        # 第三阶段: 根据复杂度分配k值
        cluster_horizons = {}
        k_candidates = np.arange(k_min, k_max + 1, 5)  # [5,10,15,...,50]
        
        for cluster_id in range(self.num_clusters):
            complexity = cluster_complexities[cluster_id]
            
            if complexity == 0:
                cluster_horizons[cluster_id] = k_min
                continue
            
            # 复杂度越高,k值越小(反比关系)
            # 使用线性映射: complexity_ratio ∈ [0,1] → k ∈ [k_min, k_max]
            if complexity > dynamic_threshold:
                # 高复杂度 → 小k (在k_min到中间值之间)
                complexity_ratio = min(complexity / dynamic_threshold - 1, 1.0)
                assigned_k = k_min + int((1 - complexity_ratio) * (k_max - k_min) / 2)
            else:
                # 低复杂度 → 大k (在中间值到k_max之间)
                complexity_ratio = complexity / dynamic_threshold
                assigned_k = k_min + int((1 + complexity_ratio) * (k_max - k_min) / 2)
            
            # 对齐到候选k值
            assigned_k = min(k_candidates, key=lambda x: abs(x - assigned_k))
            cluster_horizons[cluster_id] = int(assigned_k)
        
        print(f"✓ 帕累托分析完成！各聚类最优步长:")
        sorted_clusters = sorted(cluster_horizons.items(), key=lambda x: x[1])
        for cid, k in sorted_clusters:
            complexity = cluster_complexities[cid]
            print(f"  Cluster {cid}: k={k:2d} (复杂度={complexity:.4f})")
        
        # 统计k值分布
        k_values = list(cluster_horizons.values())
        print(f"  k值分布: min={min(k_values)}, max={max(k_values)}, mean={np.mean(k_values):.1f}")
        
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
