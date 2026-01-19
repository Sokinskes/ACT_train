"""
四任务最终对比图表生成
用于论文投稿
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 实验数据
tasks = ['Square', 'Lift', 'Can', 'Transport']
avg_lengths = [218.5, 75.8, 143.8, 701.9]
avg_horizons = [5.0, 25.75, 42.91, 48.95]
inference_savings = [0.0, 80.58, 88.35, 89.79]
accuracies = [100.0, 88.89, 100.0, 100.0]

# 聚类分布
cluster_data = {
    'Square': [5, 5, 5],
    'Lift': [20, 35, 50],
    'Can': [50, 50, 50],
    'Transport': [50, 50, 50]
}


def plot_main_comparison():
    """图1: 主要性能对比（2x2子图）"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 推理节省对比
    ax1 = axes[0, 0]
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
    bars = ax1.bar(tasks, inference_savings, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Inference Saving (%)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Inference Count Reduction', fontsize=15, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, inference_savings):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # 子图2: 平均预测步长
    ax2 = axes[0, 1]
    bars = ax2.bar(tasks, avg_horizons, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Average Predicted Horizon k', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Average Predicted Horizon', fontsize=15, fontweight='bold')
    ax2.set_ylim(0, 55)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, avg_horizons):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'k={val:.1f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # 子图3: 轨迹长度对比
    ax3 = axes[1, 0]
    bars = ax3.bar(tasks, avg_lengths, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Average Trajectory Length (steps)', fontsize=14, fontweight='bold')
    ax3.set_title('(c) Task Trajectory Lengths', fontsize=15, fontweight='bold')
    ax3.set_ylim(0, 750)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, avg_lengths):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{val:.0f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # 子图4: 准确率对比
    ax4 = axes[1, 1]
    bars = ax4.bar(tasks, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Prediction Accuracy (%)', fontsize=14, fontweight='bold')
    ax4.set_title('(d) Prediction Accuracy', fontsize=15, fontweight='bold')
    ax4.set_ylim(80, 105)
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('final_four_task_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: final_four_task_comparison.png")
    plt.close()


def plot_cluster_distribution():
    """图2: 聚类步长分布"""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    colors_palette = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (task, ax) in enumerate(zip(tasks, axes)):
        horizons = cluster_data[task]
        clusters = ['C0', 'C1', 'C2']
        
        bars = ax.bar(clusters, horizons, color=colors_palette, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Optimal Horizon k', fontsize=12, fontweight='bold')
        ax.set_title(f'{task}\n({["Conservative", "Adaptive", "Aggressive", "Aggressive"][idx]})',
                    fontsize=13, fontweight='bold')
        ax.set_ylim(0, 55)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, horizons):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'k={val}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('final_cluster_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: final_cluster_distribution.png")
    plt.close()


def plot_efficiency_vs_complexity():
    """图3: 效率提升 vs 任务复杂度"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 任务复杂度评分（手动定义）
    complexity_scores = [100, 60, 20, 10]  # Square最复杂，Transport最简单
    
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
    sizes = np.array(avg_lengths) / 5  # 点的大小代表轨迹长度
    
    scatter = ax.scatter(complexity_scores, inference_savings, s=sizes, 
                        c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    # 添加标签
    for i, task in enumerate(tasks):
        ax.annotate(f'{task}\n({avg_lengths[i]:.0f} steps)',
                   xy=(complexity_scores[i], inference_savings[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))
    
    ax.set_xlabel('Task Complexity Score (0=Simple, 100=Complex)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inference Saving (%)', fontsize=14, fontweight='bold')
    ax.set_title('Task Complexity vs Optimization Potential', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 95)
    
    # 添加趋势线
    z = np.polyfit(complexity_scores, inference_savings, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(0, 100, 100)
    ax.plot(x_smooth, p(x_smooth), 'r--', alpha=0.5, linewidth=2, label='Trend')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('final_complexity_vs_efficiency.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: final_complexity_vs_efficiency.png")
    plt.close()


def plot_absolute_savings():
    """图4: 绝对推理次数节省"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 计算绝对节省
    baseline_inferences = [avg_lengths[i] / 5 for i in range(len(tasks))]
    adastep_inferences = [avg_lengths[i] / avg_horizons[i] for i in range(len(tasks))]
    absolute_savings = [baseline_inferences[i] - adastep_inferences[i] for i in range(len(tasks))]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    colors_baseline = ['#95a5a6'] * 4
    colors_adastep = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
    
    bars1 = ax.bar(x - width/2, baseline_inferences, width, label='Baseline (k=5)',
                   color=colors_baseline, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, adastep_inferences, width, label='AdaStep (Adaptive)',
                   color=colors_adastep, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Inference Count per Trajectory', fontsize=14, fontweight='bold')
    ax.set_title('Absolute Inference Count Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=13)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加节省次数标注
    for i, (b1, b2, saving) in enumerate(zip(bars1, bars2, absolute_savings)):
        mid_x = x[i]
        mid_y = max(baseline_inferences[i], adastep_inferences[i]) + 5
        ax.annotate(f'↓{saving:.0f} inferences\n({inference_savings[i]:.1f}%)',
                   xy=(mid_x, mid_y), ha='center', fontsize=11, fontweight='bold',
                   color='red')
    
    plt.tight_layout()
    plt.savefig('final_absolute_savings.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: final_absolute_savings.png")
    plt.close()


def generate_all_final_plots():
    """生成所有最终图表"""
    print("\n" + "="*70)
    print("AdaStep 四任务最终图表生成")
    print("="*70)
    print()
    
    plot_main_comparison()
    plot_cluster_distribution()
    plot_efficiency_vs_complexity()
    plot_absolute_savings()
    
    print()
    print("="*70)
    print("✅ 所有最终图表生成完成！")
    print("="*70)
    print()
    print("生成的文件（用于论文）:")
    print("  1. final_four_task_comparison.png - 四任务性能对比（2x2）")
    print("  2. final_cluster_distribution.png - 聚类步长分布")
    print("  3. final_complexity_vs_efficiency.png - 复杂度vs效率散点图")
    print("  4. final_absolute_savings.png - 绝对推理次数对比")
    print()
    print("建议用途:")
    print("  - 图1: 论文主图（展示全部性能指标）")
    print("  - 图2: 展示AdaStep的自适应能力")
    print("  - 图3: 讨论章节（复杂度与优化空间的关系）")
    print("  - 图4: 突出实际部署价值（Transport节省127次！）")
    print()


if __name__ == '__main__':
    generate_all_final_plots()
