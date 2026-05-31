"""
AdaStep: Pareto-Optimal Adaptive Action Chunking for Efficient Robot Learning

This module provides a plug-and-play adaptive horizon mechanism for ACT/Diffusion policies.
Designed to work seamlessly with act-plus-plus without modifying the core policy code.

Usage:
    from predictors.adastep import AdaStepAdapter
    
    adapter = AdaStepAdapter(
        predictor_ckpt='path/to/horizon_predictor.pth',
        policy=act_policy,
        k_min=5,
        k_max=50
    )
    
    # Inside your evaluation loop:
    k_t = adapter.predict_horizon(qpos, curr_image)
    raw_action = all_actions[:, :k_t]  # Adaptive truncation
"""

from .adapter import AdaStepAdapter
from .predictor import HorizonPredictor
from .analyzer import StateClusterAnalyzer

__all__ = ['AdaStepAdapter', 'HorizonPredictor', 'StateClusterAnalyzer']
__version__ = '1.0.0'
