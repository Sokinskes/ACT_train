"""
Real A/B Test: Baseline vs Improved AdaStep
============================================================

Objective comparison using academic metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import pandas as pd
from scipy.stats import entropy

def load_algorithm_results():
    print("📂 Loading algorithm results...")

    with open('results_square_baseline/cluster_analyzer_baseline.pkl', 'rb') as f:
        baseline_analyzer = pickle.load(f)
    baseline_labels = np.load('results_square_baseline/horizon_labels_baseline.npy')

    with open('results_square_improved/cluster_analyzer_improved.pkl', 'rb') as f:
        improved_analyzer = pickle.load(f)
    improved_labels = np.load('results_square_improved/horizon_labels_improved.npy')

    baseline_labels_denorm = baseline_labels.flatten() * 45 + 5
    improved_labels_denorm = improved_labels * 45 + 5

    return baseline_labels_denorm, improved_labels_denorm

def calculate_academic_metrics(k_values, algorithm_name):
    metrics = {}
    metrics['mean_k'] = np.mean(k_values)
    metrics['std_k'] = np.std(k_values)
    metrics['min_k'] = np.min(k_values)
    metrics['max_k'] = np.max(k_values)
    metrics['unique_k_count'] = len(np.unique(k_values))

    metrics['inference_saving'] = (1 - 1/metrics['mean_k']) * 100
    metrics['speedup'] = metrics['mean_k']

    hist, _ = np.histogram(k_values, bins=np.arange(5, 51, 1), density=True)
    hist = hist[hist > 0]
    metrics['entropy'] = entropy(hist) if len(hist) > 1 else 0

    theoretical_range = 50 - 5
    actual_range = metrics['max_k'] - metrics['min_k']
    metrics['coverage_ratio'] = actual_range / theoretical_range
    metrics['distribution_diversity'] = metrics['std_k'] * np.log(metrics['unique_k_count'] + 1)

    print(f"\n📊 {algorithm_name} Metrics:")
    print(f"  Mean k: {metrics['mean_k']:.2f}")
    print(f"  Saving: {metrics['inference_saving']:.1f}%")
    print(f"  Entropy: {metrics['entropy']:.3f}")
    return metrics

def create_comparison_visualization(k_old, k_new, metrics_old, metrics_new):
    print("\n🎨 Creating comparison visualization...")

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. K Distribution
    sns.histplot(k_old, bins=15, alpha=0.7, color='gray', label='Baseline (K=3)',
                ax=ax1, kde=True, stat='density')
    sns.histplot(k_new, bins=25, alpha=0.7, color='red', label='AdaStep (K=10)',
                ax=ax1, kde=True, stat='density')

    ax1.axvline(metrics_old['mean_k'], color='gray', linestyle='--', linewidth=2,
                label=f'Baseline Mean: {metrics_old["mean_k"]:.1f}')
    ax1.axvline(metrics_new['mean_k'], color='red', linestyle='--', linewidth=2,
                label=f'AdaStep Mean: {metrics_new["mean_k"]:.1f}')

    ax1.set_title('Horizon Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Horizon k')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Efficiency
    efficiency_metrics = ['Inference Saving', 'Speedup']
    old_eff = [metrics_old['inference_saving'], metrics_old['speedup']]
    new_eff = [metrics_new['inference_saving'], metrics_new['speedup']]

    x = np.arange(len(efficiency_metrics))
    width = 0.35

    bars1 = ax2.bar(x - width/2, old_eff, width, label='Baseline', alpha=0.7, color='gray')
    bars2 = ax2.bar(x + width/2, new_eff, width, label='AdaStep', alpha=0.7, color='red')

    ax2.set_title('Efficiency Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels(efficiency_metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for bars, values in [(bars1, old_eff), (bars2, new_eff)]:
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)

    # 3. Adaptability
    adaptability_metrics = ['Std Dev k', 'Entropy', 'Coverage', 'Diversity']
    old_adapt = [metrics_old['std_k'], metrics_old['entropy'],
                 metrics_old['coverage_ratio'], metrics_old['distribution_diversity']]
    new_adapt = [metrics_new['std_k'], metrics_new['entropy'],
                 metrics_new['coverage_ratio'], metrics_new['distribution_diversity']]

    x = np.arange(len(adaptability_metrics))
    width = 0.35

    bars1 = ax3.bar(x - width/2, old_adapt, width, label='Baseline', alpha=0.7, color='gray')
    bars2 = ax3.bar(x + width/2, new_adapt, width, label='AdaStep', alpha=0.7, color='red')

    ax3.set_title('Adaptability Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Value')
    ax3.set_xticks(x)
    ax3.set_xticklabels(adaptability_metrics, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    for bars, values in [(bars1, old_adapt), (bars2, new_adapt)]:
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    # 4. Radar Chart
    categories = ['Efficiency', 'Adaptability', 'Coverage', 'Diversity']
    old_radar = [
        metrics_old['inference_saving'] / 100,
        metrics_old['std_k'] / 15,
        metrics_old['coverage_ratio'],
        min(metrics_old['distribution_diversity'] / 50, 1.0)
    ]
    new_radar = [
        metrics_new['inference_saving'] / 100,
        metrics_new['std_k'] / 15,
        metrics_new['coverage_ratio'],
        min(metrics_new['distribution_diversity'] / 50, 1.0)
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    old_radar += old_radar[:1]
    new_radar += new_radar[:1]

    ax4.plot(angles, old_radar, 'o-', linewidth=2, label='Baseline', color='gray')
    ax4.fill(angles, old_radar, alpha=0.25, color='gray')
    ax4.plot(angles, new_radar, 'o-', linewidth=2, label='AdaStep', color='red')
    ax4.fill(angles, new_radar, alpha=0.25, color='red')

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1.1)
    ax4.set_title('Performance Radar Chart', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('real_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: real_algorithm_comparison.png")

def main():
    print("🚀 AdaStep Real A/B Test Comparison")
    print("="*60)
    k_old, k_new = load_algorithm_results()
    metrics_old = calculate_academic_metrics(k_old, "Baseline")
    metrics_new = calculate_academic_metrics(k_new, "AdaStep")
    create_comparison_visualization(k_old, k_new, metrics_old, metrics_new)
    print("\nDone.")

if __name__ == "__main__":
    main()
