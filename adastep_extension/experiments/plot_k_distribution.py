"""
绘制k值分布图 - 用于论文Figure
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置论文风格
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

# 加载数据
results_dir = 'offline_evaluation_results'

tasks = ['transport', 'can', 'lift', 'square']
task_names = ['Transport', 'Can', 'Lift', 'Square']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

data = {}
for task in tasks:
    file_path = os.path.join(results_dir, f'{task}_detailed.json')
    if os.path.exists(file_path):
        with open(file_path) as f:
            data[task] = json.load(f)
    else:
        print(f"⚠️  文件不存在: {file_path}")

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for idx, (task, name, color) in enumerate(zip(tasks, task_names, colors)):
    if task not in data:
        continue
    
    k_values = data[task]['adastep']['k_values']
    
    # 绘制直方图
    axes[idx].hist(k_values, bins=np.arange(0, 55, 5), 
                   edgecolor='black', color=color, alpha=0.7)
    
    # 统计
    mean_k = np.mean(k_values)
    min_k = np.min(k_values)
    max_k = np.max(k_values)
    
    # 标题和标签
    axes[idx].set_title(f'{name} (k={mean_k:.1f}, range={min_k}-{max_k})', 
                       fontweight='bold')
    axes[idx].set_xlabel('Predicted Horizon $k$')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
    
    # 设置x轴范围
    axes[idx].set_xlim(0, 55)

plt.tight_layout()

# 保存
output_path = 'k_distribution.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved: {output_path}")

# 同时保存PNG版本
output_png = 'k_distribution.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved: {output_png}")

plt.show()

# 打印统计摘要
print("\n" + "="*60)
print("k值统计摘要")
print("="*60)
for task, name in zip(tasks, task_names):
    if task in data:
        k_values = data[task]['adastep']['k_values']
        print(f"{name:12s}: mean={np.mean(k_values):5.2f}, "
              f"std={np.std(k_values):5.2f}, "
              f"range=[{np.min(k_values)}, {np.max(k_values)}]")
