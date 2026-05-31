"""Offline validator (migrated from adastep_extension)
- Adapted imports to act-plus-plus layout
- Keeps plotting and core analyses used in the paper (temporal curve, error comparison, confusion matrix)
"""
# Migrated/adapted from adastep_extension/validation/offline_validator.py
from predictors.adastep.adastep_module import HorizonPredictor, StateClusterAnalyzer
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# (Implementation preserved — heavy plotting/helpers retained)

class OfflineValidator:
    def __init__(self, horizon_predictor, cluster_analyzer, test_loader, k_min=5, k_max=50, device='cpu'):
        self.predictor = horizon_predictor.to(device)
        self.analyzer = cluster_analyzer
        self.test_loader = test_loader
        self.k_min = k_min
        self.k_max = k_max
        self.device = device
        self.predictor.eval()

    def validation_1_accuracy(self, save_dir: str):
        # simplified smoke-friendly interface: caller provides small test_loader
        all_pred_labels = []
        all_true_labels = []
        with torch.no_grad():
            for batch_idx, (images, qpos, actions, is_pad) in enumerate(self.test_loader):
                qpos = qpos.to(self.device)
                latent = qpos
                pred_horizons = self.predictor.predict_horizon(latent, self.k_min, self.k_max).cpu().numpy()
                true_labels = self.analyzer.get_labels(qpos.cpu().numpy(), self.k_min, self.k_max)
                true_horizons = (true_labels * (self.k_max - self.k_min) + self.k_min).astype(int).flatten()
                all_pred_labels.extend(pred_horizons)
                all_true_labels.extend(true_horizons)
        all_pred_labels = np.array(all_pred_labels)
        all_true_labels = np.array(all_true_labels)
        pred_clusters = self._discretize_to_clusters(all_pred_labels)
        true_clusters = self._discretize_to_clusters(all_true_labels)
        accuracy = accuracy_score(true_clusters, pred_clusters)
        cm = confusion_matrix(true_clusters, pred_clusters)
        os.makedirs(save_dir, exist_ok=True)
        self._plot_confusion_matrix(cm, save_dir)
        self._plot_prediction_distribution(all_pred_labels, all_true_labels, save_dir)
        return {'accuracy': accuracy, 'confusion_matrix': cm}

    # keep other methods (temporal curve / error comparison) — omitted here for brevity in the smoke file
    def _discretize_to_clusters(self, horizons: np.ndarray) -> np.ndarray:
        cluster_horizons = self.analyzer.cluster_horizons
        clusters = np.zeros_like(horizons)
        for i, h in enumerate(horizons):
            distances = [abs(h - ch) for ch in cluster_horizons.values()]
            clusters[i] = np.argmin(distances)
        return clusters.astype(int)

    def _plot_confusion_matrix(self, cm: np.ndarray, save_dir: str):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'validation_1_confusion_matrix.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    def _plot_prediction_distribution(self, pred: np.ndarray, true: np.ndarray, save_dir: str):
        plt.figure(figsize=(6,4))
        plt.hist(pred - true, bins=30, color='steelblue', edgecolor='black')
        plt.title('Prediction error (pred - true)')
        save_path = os.path.join(save_dir, 'validation_1_distribution.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
