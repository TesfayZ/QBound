"""Generate comparison plot from experimental results."""
import numpy as np
import matplotlib.pyplot as plt

# Load results
data = np.load('results_20251024_132000.npz')
qbound_rewards = data['qclip_rewards']
baseline_rewards = data['baseline_rewards']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Learning curve (smoothed)
window = 50
qbound_smooth = np.convolve(qbound_rewards, np.ones(window)/window, mode='valid')
baseline_smooth = np.convolve(baseline_rewards, np.ones(window)/window, mode='valid')

axes[0].plot(qbound_smooth, label='QBound', linewidth=2, color='#2E86AB')
axes[0].plot(baseline_smooth, label='Baseline', linewidth=2, color='#A23B72')
axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('Success Rate (smoothed)', fontsize=12)
axes[0].set_title('GridWorld 10x10: Learning Curves', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% target')

# Plot 2: Cumulative reward
qbound_cumsum = np.cumsum(qbound_rewards)
baseline_cumsum = np.cumsum(baseline_rewards)

axes[1].plot(qbound_cumsum, label='QBound', linewidth=2, color='#2E86AB')
axes[1].plot(baseline_cumsum, label='Baseline', linewidth=2, color='#A23B72')
axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel('Cumulative Reward', fontsize=12)
axes[1].set_title('Sample Efficiency: Cumulative Reward', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

# Add annotations
qbound_80 = 206
baseline_80 = 352
axes[0].axvline(x=qbound_80, color='#2E86AB', linestyle=':', alpha=0.7)
axes[0].axvline(x=baseline_80, color='#A23B72', linestyle=':', alpha=0.7)
axes[0].text(qbound_80+10, 0.5, f'QBound:\n{qbound_80} eps', fontsize=9, color='#2E86AB')
axes[0].text(baseline_80+10, 0.3, f'Baseline:\n{baseline_80} eps', fontsize=9, color='#A23B72')

plt.tight_layout()
plt.savefig('QBound/gridworld_results.png', dpi=300, bbox_inches='tight')
plt.savefig('QBound/gridworld_results.pdf', bbox_inches='tight')
print("Plot saved to QBound/gridworld_results.png and .pdf")

# Print summary
print("\nGridWorld Experimental Results:")
print("="*50)
print(f"Episodes to 80% success:")
print(f"  QBound: {qbound_80}")
print(f"  Baseline: {baseline_80}")
print(f"  Improvement: {((baseline_80-qbound_80)/baseline_80)*100:.1f}%")
print(f"  Speedup: {baseline_80/qbound_80:.2f}x")
print(f"\nTotal reward:")
print(f"  QBound: {np.sum(qbound_rewards):.0f}")
print(f"  Baseline: {np.sum(baseline_rewards):.0f}")
print(f"  Improvement: {((np.sum(qbound_rewards)/np.sum(baseline_rewards)-1)*100):.1f}%")
