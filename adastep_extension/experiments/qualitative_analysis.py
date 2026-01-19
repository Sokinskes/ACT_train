"""
AdaStep Qualitative Analysis Visualization
=======================================

Creates intuitive charts to demonstrate AdaStep's mechanism:
1. k-value timeline
2. Trajectory visualization (color-coded by k)
3. State complexity vs k
4. Decision pattern comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import pickle
import json

def load_trajectory_data():
    """
    Load trajectory data for qualitative analysis
    """
    print("📂 Loading trajectory data...")

    trajectories = {}

    # Try to load AdaStep trajectory
    try:
        with open('trajectory_data_adastep.pkl', 'rb') as f:
            trajectories['adastep'] = pickle.load(f)
        print("✓ Loaded AdaStep trajectory data")
    except FileNotFoundError:
        print("⚠ AdaStep data not found, using mock data")
        trajectories['adastep'] = generate_mock_trajectory('adastep')

    # Try to load fixed k trajectories
    for k in [10, 25, 40]:
        try:
            with open(f'trajectory_data_k{k}.pkl', 'rb') as f:
                trajectories[f'fixed_k{k}'] = pickle.load(f)
            print(f"✓ Loaded fixed k={k} data")
        except FileNotFoundError:
            print(f"⚠ Fixed k={k} data not found, using mock data")
            trajectories[f'fixed_k{k}'] = generate_mock_trajectory(f'fixed_k{k}')

    return trajectories

def generate_mock_trajectory(strategy):
    """
    Generate mock trajectory for demonstration
    """
    np.random.seed(42)

    # Mock Square task trajectory (approx 200 steps)
    n_steps = 200

    # State info
    states = np.random.randn(n_steps, 6)

    # Generate k values based on strategy
    if strategy == 'adastep':
        complexity = np.linalg.norm(states, axis=1)
        k_values = np.clip(50 - complexity * 2 + np.random.normal(0, 5, n_steps), 5, 50)
    elif strategy.startswith('fixed_k'):
        k_val = int(strategy[7:])
        k_values = np.full(n_steps, k_val)
    else:
        k_values = np.full(n_steps, 25)

    # Trajectory positions
    positions = np.cumsum(np.random.randn(n_steps, 2) * 0.1, axis=0)
    positions = np.clip(positions, -1, 1)

    return {
        'states': states,
        'k_values': k_values,
        'positions': positions,
        'time_steps': np.arange(n_steps)
    }

def create_k_value_timeline(trajectories):
    """
    Create k-value timeline visualization
    """
    print("\n🎨 Creating k-value timeline visualization...")

    # Use default font to avoid missing font issues
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. AdaStep k-value timeline
    traj = trajectories['adastep']
    time_steps = traj['time_steps']
    k_values = traj['k_values']

    ax1.plot(time_steps, k_values, 'r-', linewidth=2, alpha=0.8, label='AdaStep')
    ax1.fill_between(time_steps, k_values, alpha=0.3, color='red')
    ax1.axhline(y=np.mean(k_values), color='red', linestyle='--', linewidth=2,
                label=f'Mean k: {np.mean(k_values):.1f}')
    ax1.set_title('AdaStep: Adaptive Horizon over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Predicted Horizon k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 55)

    # 2. Fixed k comparison
    colors = ['blue', 'green', 'orange']
    strategies = ['fixed_k10', 'fixed_k25', 'fixed_k40']

    for i, strategy in enumerate(strategies):
        if strategy in trajectories:
            traj = trajectories[strategy]
            k_val = traj['k_values'][0]
            ax2.plot(traj['time_steps'], traj['k_values'], color=colors[i], linewidth=2,
                    label=f'Fixed k={k_val}', alpha=0.8)
            ax2.axhline(y=k_val, color=colors[i], linestyle='--', linewidth=1, alpha=0.7)

    ax2.set_title('Fixed-k Baselines: Constant Horizon', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Predicted Horizon k')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 55)

    # 3. Distribution comparison
    k_data = []
    labels = []

    for strategy, traj in trajectories.items():
        k_data.append(traj['k_values'])
        if strategy == 'adastep':
            labels.append('AdaStep')
        elif strategy.startswith('fixed_k'):
            k_val = strategy[7:]
            labels.append(f'Fixed k={k_val}')

    ax3.hist(k_data, bins=15, alpha=0.7, label=labels, density=True)
    ax3.set_title('Horizon Distribution Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted Horizon k')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Rate of change analysis
    adastep_traj = trajectories['adastep']
    k_changes = np.abs(np.diff(adastep_traj['k_values']))

    ax4.plot(adastep_traj['time_steps'][1:], k_changes, 'purple', linewidth=2, alpha=0.8)
    ax4.fill_between(adastep_traj['time_steps'][1:], k_changes, alpha=0.3, color='purple')
    ax4.axhline(y=np.mean(k_changes), color='purple', linestyle='--', linewidth=2,
                label=f'Mean Rate: {np.mean(k_changes):.2f}')
    ax4.set_title('AdaStep: Horizon Change Rate (Adaptability)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Change Magnitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('k_value_timeline_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Saved: k_value_timeline_analysis.png")

def create_trajectory_visualization(trajectories):
    """
    Create trajectory visualization (color-coded by k)
    """
    print("\n🎨 Creating trajectory visualization...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Colormap
    cmap = LinearSegmentedColormap.from_list("k_cmap", ["blue", "cyan", "yellow", "red"])

    # 1. AdaStep trajectory
    traj = trajectories['adastep']
    positions = traj['positions']
    k_values = traj['k_values']

    k_norm = (k_values - 5) / (50 - 5)
    
    scatter = ax1.scatter(positions[:, 0], positions[:, 1], c=k_values, cmap=cmap,
                         s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax1.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)

    ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='*',
               edgecolors='black', linewidth=2, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200, marker='X',
               edgecolors='black', linewidth=2, label='Goal')

    ax1.set_title('AdaStep Trajectory (Color-coded by Horizon)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)

    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Horizon k')

    # 2. Fixed k=10
    traj = trajectories.get('fixed_k10', trajectories['adastep'])
    positions = traj['positions']
    k_val = traj['k_values'][0]

    ax2.scatter(positions[:, 0], positions[:, 1], c='blue', s=50, alpha=0.8,
               edgecolors='black', linewidth=0.5, label=f'k={k_val}')
    ax2.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax2.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='*',
               edgecolors='black', linewidth=2, label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200, marker='X',
               edgecolors='black', linewidth=2, label='Goal')

    ax2.set_title(f'Fixed k={k_val} Trajectory', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)

    # 3. Fixed k=40
    traj = trajectories.get('fixed_k40', trajectories['adastep'])
    positions = traj['positions']
    k_val = traj['k_values'][0]

    ax3.scatter(positions[:, 0], positions[:, 1], c='red', s=50, alpha=0.8,
               edgecolors='black', linewidth=0.5, label=f'k={k_val}')
    ax3.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax3.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='*',
               edgecolors='black', linewidth=2, label='Start')
    ax3.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200, marker='X',
               edgecolors='black', linewidth=2, label='Goal')

    ax3.set_title(f'Fixed k={k_val} Trajectory', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)

    # 4. Statistics
    strategies = list(trajectories.keys())
    mean_ks = [np.mean(trajectories[s]['k_values']) for s in strategies]
    std_ks = [np.std(trajectories[s]['k_values']) for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax4.bar(x - width/2, mean_ks, width, label='Mean k', alpha=0.8, color='lightblue')
    bars2 = ax4.bar(x + width/2, std_ks, width, label='Std Dev k', alpha=0.8, color='lightcoral')

    ax4.set_title('Horizon Statistics by Strategy', fontsize=14, fontweight='bold')
    ax4.set_ylabel('k Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.replace('_', '\n') for s in strategies])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    for bars, values in [(bars1, mean_ks), (bars2, std_ks)]:
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('trajectory_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Saved: trajectory_visualization.png")

def create_state_complexity_analysis(trajectories):
    """
    Create state complexity vs k visualization
    """
    print("\n🎨 Creating state complexity analysis...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Complexity vs k (AdaStep)
    traj = trajectories['adastep']
    states = traj['states']
    k_values = traj['k_values']

    complexities = np.linalg.norm(states, axis=1)

    ax1.scatter(complexities, k_values, s=50, alpha=0.7, c='red', edgecolors='black')
    ax1.set_title('AdaStep: State Complexity vs Horizon', fontsize=14, fontweight='bold')
    ax1.set_xlabel('State Complexity (L2 Norm)')
    ax1.set_ylabel('Predicted Horizon k')
    ax1.grid(True, alpha=0.3)

    z = np.polyfit(complexities, k_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(complexities), max(complexities), 100)
    ax1.plot(x_trend, p(x_trend), 'k--', linewidth=2, label=f'Trend Slope={z[0]:.2f}')
    ax1.legend()

    # 2. Complexity Distribution
    ax2.hist(complexities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=np.mean(complexities), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(complexities):.2f}')
    ax2.set_title('State Complexity Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('State Complexity')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Decision Pattern
    strategies = ['adastep', 'fixed_k10', 'fixed_k25', 'fixed_k40']
    colors = ['red', 'blue', 'green', 'orange']

    for i, strategy in enumerate(strategies):
        if strategy in trajectories:
            traj = trajectories[strategy]
            complexities = np.linalg.norm(traj['states'], axis=1)
            k_vals = traj['k_values']

            ax3.scatter(complexities, k_vals, s=30, alpha=0.6, color=colors[i],
                       label=strategy.replace('_', ' '), edgecolors='none')

    ax3.set_title('Complexity-Horizon Decision Patterns', fontsize=14, fontweight='bold')
    ax3.set_xlabel('State Complexity')
    ax3.set_ylabel('Predicted Horizon k')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Metrics
    adastep_traj = trajectories['adastep']
    complexities = np.linalg.norm(adastep_traj['states'], axis=1)
    k_vals = adastep_traj['k_values']

    correlation = np.corrcoef(complexities, k_vals)[0, 1]
    
    complexity_changes = np.abs(np.diff(complexities))
    k_changes = np.abs(np.diff(k_vals))
    adaptation_score = np.corrcoef(complexity_changes, k_changes)[0, 1]

    ax4.text(0.1, 0.8, f'Correlation: {correlation:.3f}', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.7, f'Adaptation Score: {adaptation_score:.3f}', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f'Mean k: {np.mean(k_vals):.1f}', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.5, f'Std Dev k: {np.std(k_vals):.1f}', fontsize=12, transform=ax4.transAxes)

    ax4.set_title('AdaStep Efficiency Metrics', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig('state_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Saved: state_complexity_analysis.png")

def main():
    print("🎨 AdaStep Qualitative Analysis")
    print("="*50)
    trajectories = load_trajectory_data()
    create_k_value_timeline(trajectories)
    create_trajectory_visualization(trajectories)
    create_state_complexity_analysis(trajectories)
    print("\n" + "="*50)
    print("Done.")

if __name__ == "__main__":
    main()
