import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_pareto_frontier():
    # Data from SUCCESS_RATE_VALIDATION_REPORT.md
    # Fixed-k baseline
    # k_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    efficiency = [80.0, 90.0, 93.3, 95.0, 96.0, 96.7, 97.1, 97.5, 97.8, 98.0]
    success_rate = [86.5, 82.2, 85.0, 84.4, 84.1, 82.6, 82.1, 84.6, 80.7, 81.1]

    # AdaStep
    adastep_eff = 95.8
    adastep_succ = 87.5

    # Filter/Sort for smooth curve (Pareto Frontier usually convex/concave)
    # Ideally, for the baseline frontier, we want the max success for a given efficiency?
    # Or just plot the curve as is (showing it fluctuates). 
    # The user said "connect ... into a smooth curve".
    # Let's sort by efficiency.
    sorted_indices = np.argsort(efficiency)
    eff_sorted = np.array(efficiency)[sorted_indices]
    succ_sorted = np.array(success_rate)[sorted_indices]

    # Smooth curve
    X_Y_Spline = make_interp_spline(eff_sorted, succ_sorted)
    X_ = np.linspace(eff_sorted.min(), eff_sorted.max(), 500)
    Y_ = X_Y_Spline(X_)

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # Plot Baseline Curve
    plt.plot(X_, Y_, color='gray', linestyle='-', linewidth=2, label='Fixed-Step Baseline Frontier')
    plt.scatter(eff_sorted, succ_sorted, color='gray', alpha=0.6, s=50)

    # Plot AdaStep
    plt.scatter(adastep_eff, adastep_succ, color='red', marker='*', s=300, label='AdaStep (Ours)', zorder=10)

    # Vertical Line (Iso-Efficiency)
    # Find closest point on baseline or just draw strictly vertical to y-axis of the baseline point at analogous efficiency
    # k=25 is at 96.0 efficiency. AdaStep is 95.8. Close enough.
    # Baseline success at 96.0 is 84.1.
    
    plt.plot([adastep_eff, adastep_eff], [84.1, adastep_succ], color='red', linestyle='--', linewidth=1.5)
    plt.text(adastep_eff + 0.5, (84.1 + adastep_succ)/2, '+3.4% Success Rate', color='red', fontweight='bold', ha='left')

    # Horizontal Line (Higher Efficiency)
    # Finding x where baseline success is ~87.5. 
    # Actually k=5 is 86.5. AdaStep is higher than the max baseline! 
    # So a horizontal line goes all the way to the left?
    # Let's interpret "Higher Efficiency" relative to a baseline of similar success.
    # Since AdaStep beats all baselines in success, this line would be long.
    # The user suggested "horizontal line ... to Baseline curve ... annotate 'Higher Efficiency'".
    # Let's point it towards the general direction of the "frontier".
    # Or maybe compare to k=5 (86.5%) which is the closest competitor in success.
    # k=5 has eff 80.0. AdaStep has 95.8.
    plt.plot([80.0, adastep_eff], [86.5, 86.5], color='blue', linestyle='--', linewidth=1.5) # Comparison with best baseline
    plt.text((80.0 + adastep_eff)/2, 86.5 + 0.5, 'Higher Efficiency (+15.8%)', color='blue', fontweight='bold', ha='center')

    # Highlight k=25 point specifically since it's the comparison point
    k25_idx = 4 # 5, 10, 15, 20, 25 (index 4)
    plt.scatter(efficiency[k25_idx], success_rate[k25_idx], color='black', s=100, zorder=5)
    plt.annotate('Fixed k=25\n(Iso-Efficiency)', 
                 xy=(efficiency[k25_idx], success_rate[k25_idx]), 
                 xytext=(efficiency[k25_idx]+1, success_rate[k25_idx]-2),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel('Inference Efficiency (Saving %)', fontsize=12)
    plt.ylabel('Task Success Rate (%)', fontsize=12)
    plt.title('Pareto Frontier: Efficiency vs. Success Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('pareto_frontier_comparison.png', dpi=300)
    print("✓ Saved pareto_frontier_comparison.png")

if __name__ == "__main__":
    plot_pareto_frontier()
