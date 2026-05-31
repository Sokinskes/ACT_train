"""
Lightweight Horizon Predictor - 3-Layer MLP
Predicts the optimal action chunk size k based on visual state embeddings.
"""

import torch
import torch.nn as nn


class HorizonPredictor(nn.Module):
    """
    Lightweight MLP for predicting adaptive action horizons.
    
    Design Philosophy:
    - Parasitic: Shares visual encoder with ACT policy (zero backbone overhead)
    - Ultra-lightweight: <1% parameters of ACT backbone
    - Fast: <1ms inference latency on Jetson Orin Nano
    
    Architecture:
        Input (512-dim latent) → FC(512→256) → ReLU → FC(256→128) → ReLU → FC(128→1) → Sigmoid
    
    Output:
        Normalized horizon ∈ [0, 1], maps to discrete k ∈ [k_min, k_max]
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output ∈ [0, 1]
        )
        
        # Xavier initialization for stable training
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, input_dim] - Visual embeddings from ACT encoder
        
        Returns:
            normalized_k: [B, 1] - Normalized horizon ∈ [0, 1]
        """
        return self.network(latent)
    
    def predict_horizon(
        self, 
        latent: torch.Tensor, 
        k_min: int = 5, 
        k_max: int = 50
    ) -> torch.Tensor:
        """
        Predict discrete horizon values.
        
        Args:
            latent: Input feature embeddings
            k_min: Minimum horizon (for precision-critical phases like insertion)
            k_max: Maximum horizon (for stable free-space motion)
        
        Returns:
            k: [B] - Integer horizon values
        """
        normalized = self.forward(latent)
        k = normalized * (k_max - k_min) + k_min
        return k.squeeze(-1).round().long()
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
