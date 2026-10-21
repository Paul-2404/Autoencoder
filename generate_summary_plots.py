import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('results/summary_plots', exist_ok=True)
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# 1. Latent Dimension
df_latent = pd.read_csv('results/metrics/latent_dimension_results.csv')
fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=200)
color = '#2563EB'
ax1.set_xlabel('Latent Space Dimension (Neurons)', fontsize=12, fontweight='bold', labelpad=8)
ax1.set_ylabel('Test MSE (Lower is better)', color=color, fontsize=12, fontweight='bold')
line1 = ax1.plot(df_latent['latent_size'], df_latent['test_mse'], marker='o', color=color, linewidth=2.5, markersize=8, label='Test MSE')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
color2 = '#0D9488'
ax2.set_ylabel('Compression Ratio (x:1)', color=color2, fontsize=12, fontweight='bold')
line2 = ax2.plot(df_latent['latent_size'], df_latent['compression_ratio'], marker='s', color=color2, linewidth=2.5, linestyle='--', markersize=8, label='Compression Ratio')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.grid(False)

plt.title('Latent Bottleneck: Test MSE vs Compression Ratio', fontsize=14, fontweight='bold', pad=12)
fig.tight_layout()
plt.savefig('results/summary_plots/latent_tradeoff.png')
plt.close()

# 2. Architecture Comparison
df_arch = pd.read_csv('results/metrics/architecture_results.csv')
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
bars = ax.bar(df_arch['experiment'].str.replace('arch_', '').str.capitalize(), df_arch['test_mse'], color=['#94A3B8', '#3B82F6', '#10B981', '#6366F1'], width=0.55, edgecolor='#1E293B', linewidth=1)
ax.set_ylabel('Test MSE', fontsize=12, fontweight='bold')
ax.set_title('Architecture Depth vs Reconstruction Error (MSE)', fontsize=14, fontweight='bold', pad=12)
ax.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0003, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.tight_layout()
plt.savefig('results/summary_plots/arch_comparison.png')
plt.close()

# 3. Activations
df_act = pd.read_csv('results/metrics/activation_results.csv')
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
colors = ['#F59E0B', '#EF4444', '#10B981']
bars = ax.bar(df_act['activation'].str.upper(), df_act['test_mse'], color=colors, width=0.5, edgecolor='#1E293B', linewidth=1)
ax.set_ylabel('Test MSE', fontsize=12, fontweight='bold')
ax.set_title('Hidden Layer Activation Function Comparison', fontsize=14, fontweight='bold', pad=12)
ax.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0004, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.tight_layout()
plt.savefig('results/summary_plots/activation_comparison.png')
plt.close()

# 4. Optimizers
df_opt = pd.read_csv('results/metrics/optimizer_results.csv')
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
bars = ax.bar(df_opt['optimizer'].str.upper(), df_opt['test_mse'], color=['#3B82F6', '#EF4444'], width=0.45, edgecolor='#1E293B', linewidth=1)
ax.set_ylabel('Test MSE', fontsize=12, fontweight='bold')
ax.set_title('Optimizer Performance: Adam vs SGD', fontsize=14, fontweight='bold', pad=12)
ax.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.tight_layout()
plt.savefig('results/summary_plots/optimizer_comparison.png')
plt.close()

# 5. Regularization
df_reg = pd.read_csv('results/metrics/regularization_results.csv')
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
labels = ['None', 'L2 (0.001)', 'L2 (0.01)', 'L1 (0.001)']
bars = ax.bar(labels, df_reg['test_mse'], color=['#10B981', '#F59E0B', '#EF4444', '#EC4899'], width=0.55, edgecolor='#1E293B', linewidth=1)
ax.set_ylabel('Test MSE', fontsize=12, fontweight='bold')
ax.set_title('Regularization Penalty Impact on Reconstruction MSE', fontsize=14, fontweight='bold', pad=12)
ax.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0015, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.tight_layout()
plt.savefig('results/summary_plots/regularization_comparison.png')
plt.close()

# 6. Batch Size Tradeoff
df_batch = pd.read_csv('results/metrics/batch_size_results.csv')
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
bars = ax.bar([str(b) for b in df_batch['batch_size']], df_batch['test_mse'], color='#6366F1', width=0.5, edgecolor='#1E293B', linewidth=1)
ax.set_xlabel('Mini-Batch Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Test MSE', fontsize=12, fontweight='bold')
ax.set_title('Mini-Batch Size vs Generalization Error (MSE)', fontsize=14, fontweight='bold', pad=12)
ax.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0003, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.tight_layout()
plt.savefig('results/summary_plots/batch_size_comparison.png')
plt.close()

# 7. Autoencoder Architecture Diagram
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(11.5, 3.8), dpi=300)
ax.set_facecolor('#F8FAFC')
fig.patch.set_facecolor('#F8FAFC')

layers = [
    {"name": "Input Layer\n(Image Pixels)", "dim": "784-D\n(28x28)", "h": 2.8, "w": 0.45, "x": 0.8, "color": "#3B82F6", "border": "#1D4ED8"},
    {"name": "Encoder 1", "dim": "128 Units", "h": 2.1, "w": 0.4, "x": 2.4, "color": "#60A5FA", "border": "#2563EB"},
    {"name": "Encoder 2", "dim": "64 Units", "h": 1.5, "w": 0.4, "x": 3.9, "color": "#93C5FD", "border": "#3B82F6"},
    {"name": "BOTTLENECK\n(Latent Code Z)", "dim": "32 Units\n(24.5x)", "h": 0.9, "w": 0.48, "x": 5.5, "color": "#F59E0B", "border": "#B45309"},
    {"name": "Decoder 1", "dim": "64 Units", "h": 1.5, "w": 0.4, "x": 7.1, "color": "#A7F3D0", "border": "#059669"},
    {"name": "Decoder 2", "dim": "128 Units", "h": 2.1, "w": 0.4, "x": 8.6, "color": "#6EE7B7", "border": "#047857"},
    {"name": "Output Layer\n(Reconstruction)", "dim": "784-D\n(28x28)", "h": 2.8, "w": 0.45, "x": 10.2, "color": "#10B981", "border": "#065F46"}
]

for i in range(len(layers) - 1):
    l1 = layers[i]
    l2 = layers[i + 1]
    x1 = l1["x"] + l1["w"] / 2
    x2 = l2["x"] - l2["w"] / 2
    y1_top = 1.9 + l1["h"] / 2
    y1_bot = 1.9 - l1["h"] / 2
    y2_top = 1.9 + l2["h"] / 2
    y2_bot = 1.9 - l2["h"] / 2
    
    fill_col = "#E2E8F0"
    trapezoid = patches.Polygon(
        [[x1, y1_top], [x2, y2_top], [x2, y2_bot], [x1, y1_bot]],
        closed=True, facecolor=fill_col, edgecolor='#CBD5E1', linewidth=1, alpha=0.55, zorder=1
    )
    ax.add_patch(trapezoid)
    ax.annotate('', xy=((x1 + x2)/2 + 0.12, 1.9), xytext=((x1 + x2)/2 - 0.12, 1.9),
                arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.5, mutation_scale=10), zorder=2)

for l in layers:
    y_bottom = 1.9 - l["h"] / 2
    rect = patches.FancyBboxPatch(
        (l["x"] - l["w"]/2, y_bottom), l["w"], l["h"],
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=l["color"], edgecolor=l["border"], linewidth=2, zorder=3
    )
    ax.add_patch(rect)
    ax.text(l["x"], y_bottom + l["h"]/2, l["dim"], ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#0F172A', zorder=4)
    ax.text(l["x"], y_bottom - 0.28, l["name"], ha='center', va='top',
            fontsize=8, fontweight='bold', color='#1E293B', zorder=4)

ax.annotate('', xy=(0.55, 3.5), xytext=(4.75, 3.5),
            arrowprops=dict(arrowstyle="<->", color='#2563EB', lw=2))
ax.text(2.65, 3.58, "ENCODER (Compresses Data)", ha='center', va='bottom',
        fontsize=10.5, fontweight='bold', color='#1D4ED8')

b_box = patches.FancyBboxPatch(
    (5.05, 0.45), 0.9, 3.15,
    boxstyle="round,pad=0.08,rounding_size=0.12",
    facecolor='none', edgecolor='#D97706', linewidth=2.2, linestyle='--', zorder=5
)
ax.add_patch(b_box)
ax.text(5.5, 3.68, "BOTTLENECK SIZE", ha='center', va='bottom',
        fontsize=10.5, fontweight='bold', color='#B45309')
ax.text(5.5, 0.18, "Forces feature discovery\nby squeezing capacity", ha='center', va='top',
        fontsize=7.5, color='#B45309', fontweight='bold')

ax.annotate('', xy=(6.25, 3.5), xytext=(10.45, 3.5),
            arrowprops=dict(arrowstyle="<->", color='#059669', lw=2))
ax.text(8.35, 3.58, "DECODER (Reconstructs Data)", ha='center', va='bottom',
        fontsize=10.5, fontweight='bold', color='#047857')

ax.annotate('', xy=(0.8, -0.65), xytext=(10.2, -0.65),
            arrowprops=dict(arrowstyle="<->", color='#475569', lw=2.5))
ax.text(5.5, -0.60, "NETWORK DEPTH (Number of Layers in Encoder & Decoder)", ha='center', va='bottom',
        fontsize=9.5, fontweight='bold', color='#334155')

ax.annotate('LAYER WIDTH\n(# of neurons in layer)',
            xy=(2.4, 3.0), xytext=(1.4, 3.3),
            arrowprops=dict(arrowstyle="->", color='#2563EB', lw=1.5, connectionstyle="arc3,rad=0.2"),
            fontsize=8, fontweight='bold', color='#1D4ED8', ha='center')

ax.annotate('RECONSTRUCTION ERROR (MSE Loss: compare Input x with Output x̂)',
            xy=(1.0, 0.2), xytext=(10.0, 0.2),
            arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.8, linestyle=':', connectionstyle="arc3,rad=-0.15"),
            fontsize=8.5, fontweight='bold', color='#DC2626', ha='center')

ax.set_xlim(0.0, 11.0)
ax.set_ylim(-0.95, 4.05)
ax.axis('off')

plt.tight_layout()
plt.savefig('results/summary_plots/autoencoder_architecture.png', bbox_inches='tight', dpi=300)
plt.close()

print('All summary plots created successfully!')
