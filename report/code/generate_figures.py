"""
Generate 6 publication-quality figures for MLLM hallucination report.
Style: top conference / Nature-style — clean, minimal, no titles on figures.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Style: Nature / NeurIPS clean
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'axes.unicode_minus': False,
    'legend.frameon': False,
    'legend.fontsize': 7,
})

FIG_DIR = 'report/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Colors — Low-saturation Nature-style palette
C = ['#8CBAD6', '#D4B680', '#97C1A1', '#C99494']  # models: soft blue, sand, sage, rose
C2 = ['#7FAFCC', '#CC8B8B']  # Direct (soft blue), CoT (soft rose)
PIE = ['#CC8B8B', '#E8C88A', '#B5CC8A', '#8CBAD6']  # Faith., Fact., Logical, None

MODELS = ['GPT-5.4-mini', 'Gemini 2.5 Flash', 'Qwen3.5-35B-A3B', 'Qwen3-VL-235B-A22B']
MODELS_SHORT = ['GPT-5.4-mini', 'Gemini 2.5\nFlash', 'Qwen3.5\n35B-A3B', 'Qwen3-VL\n235B-A22B']

# ========== Data ==========
pope_splits = ['Random', 'Popular', 'Adversarial']
pope_hr = {
    'GPT-5.4-mini':        [0.70, 5.40, 7.80],
    'Gemini 2.5 Flash':    [0.80, 5.70, 5.10],
    'Qwen3.5-35B-A3B':     [1.10, 6.20, 8.90],
    'Qwen3-VL-235B-A22B':  [0.30, 2.40, 4.80],
}

mv_direct_hr = [45.1, 34.3, 13.3, 16.3]
mv_cot_hr = [34.4, 27.6, 15.5, 24.0]

mv_types_short = ['Faith.', 'Fact.', 'Logical', 'None']
mv_type_data = {
    'GPT-5.4-mini':        [189, 14, 248, 549],
    'Gemini 2.5 Flash':    [278, 13, 52, 657],
    'Qwen3.5-35B-A3B':     [98, 5, 30, 867],
    'Qwen3-VL-235B-A22B':  [113, 6, 44, 837],
}

vqa_type_data = {
    'GPT-5.4-mini':        [145, 1, 4, 301],
    'Gemini 2.5 Flash':    [296, 2, 1, 152],
    'Qwen3.5-35B-A3B':     [171, 4, 2, 274],
    'Qwen3-VL-235B-A22B':  [257, 5, 1, 188],
}

radar_dims = ['POPE\nRandom', 'POPE\nPopular', 'POPE\nAdvers.',
              'MathVista\nDirect', 'MathVista\nCoT', 'VQA-RAD']
radar_data = {
    'GPT-5.4-mini':        [99.30, 94.60, 92.20, 54.90, 65.60, 66.74],
    'Gemini 2.5 Flash':    [99.20, 94.30, 94.90, 65.70, 72.40, 33.70],
    'Qwen3.5-35B-A3B':     [98.90, 93.80, 91.10, 86.70, 84.50, 60.75],
    'Qwen3-VL-235B-A22B':  [99.70, 97.60, 95.20, 83.70, 76.00, 41.69],
}

thresh_models = ['gpt-5.4-mini', 'gemini-2.5-flash', 'Qwen3.5-35B-A3B', 'Qwen3-VL-235B-A22B-Instruct']
threshold_data = json.load(open('results/errors_analysis/threshold_sensitivity.json'))


def savefig(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  → {name}.pdf')


# ════════════════════════════════════════════
# Fig 1: POPE HR — Grouped bar (NMI pastel)
# ════════════════════════════════════════════
print('[1/6] POPE hallucination rate...')

# NMI pastel palette for four models
model_colors = ['#484878', '#7884B4', '#B4C0E4', '#E4E4F0']
# Hatching by split for grayscale differentiate
split_hatches = ['', '///', 'xxx']
split_alphas = [1.0, 0.85, 0.70]

fig, ax = plt.subplots(figsize=(5.5, 3.5))
x = np.arange(3)
n_models = 4
w = 0.20

for i, m in enumerate(MODELS):
    vals = pope_hr[m]
    offset = (i - (n_models - 1) / 2) * w
    bars = ax.bar(x + offset, vals, w,
                  color=model_colors[i],
                  label=m,
                  linewidth=0.4,
                  edgecolor='white')

    # Value labels on top of each bar
    for j, v in enumerate(vals):
        ax.text(x[j] + offset, v + 0.2, f'{v:.1f}',
                ha='center', va='bottom', fontsize=6, fontweight='bold',
                color='#333333')

ax.set_xticks(x)
ax.set_xticklabels(pope_splits)
ax.set_ylabel('Hallucination Rate (%)')
ax.set_ylim(0, 11.5)
ax.legend(loc='upper left', fontsize=7, frameon=False)
plt.tight_layout()
savefig(fig, 'fig1_pope_hr_bar')


# ════════════════════════════════════════════
# Fig 2: MathVista Direct vs CoT — refined
# ════════════════════════════════════════════
print('[2/6] MathVista Direct vs CoT...')

# Unified NMI palette: two shades of the same hue (baseline vs. treatment)
direct_color = '#484878'   # deep navy — Direct (baseline)
cot_color    = '#B4C0E4'   # light blue-purple — CoT (treatment)

fig, ax = plt.subplots(figsize=(5.5, 3.5))
x = np.arange(4)
w = 0.32

ax.bar(x - w / 2, mv_direct_hr, w, label='Direct',
       color=direct_color, linewidth=0.4, edgecolor='white')
ax.bar(x + w / 2, mv_cot_hr, w, label='CoT',
       color=cot_color, linewidth=0.4, edgecolor='white')

# Value labels on top of each bar (consistent with Fig 1)
for i, v in enumerate(mv_direct_hr):
    ax.text(i - w / 2, v + 0.7, f'{v:.1f}',
            ha='center', va='bottom', fontsize=6.5, fontweight='bold',
            color='#333333')
for i, v in enumerate(mv_cot_hr):
    ax.text(i + w / 2, v + 0.7, f'{v:.1f}',
            ha='center', va='bottom', fontsize=6.5, fontweight='bold',
            color='#333333')

# Arrow connectors + Δ labels above bars
for i in range(4):
    d = mv_cot_hr[i] - mv_direct_hr[i]
    improve = d < 0
    arrow_color = '#2E8B57' if improve else '#CC3333'

    # Arrow from Direct bar-top → CoT bar-top (direction = change direction)
    ax.annotate('',
                xy=(i + w / 2, mv_cot_hr[i]),
                xytext=(i - w / 2, mv_direct_hr[i]),
                arrowprops=dict(arrowstyle='->', color=arrow_color,
                                lw=1.1, alpha=0.75,
                                shrinkA=3, shrinkB=3))

    # Δ label centered above the higher bar
    top_y = max(mv_direct_hr[i], mv_cot_hr[i])
    txt = f'$\\Delta$={d:+.1f}'
    ax.text(i, top_y + 4.2, txt, ha='center', va='bottom',
            fontsize=7, color=arrow_color, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(MODELS_SHORT, fontsize=6.5)
ax.set_ylabel('Hallucination Rate (%)')
ax.legend(loc='upper right', fontsize=7.5)
ax.set_ylim(0, 56)
plt.tight_layout()
savefig(fig, 'fig2_mathvista_direct_cot_bar')


# ════════════════════════════════════════════
# Fig 3: Threshold sensitivity — line
# ════════════════════════════════════════════
print('[3/6] Threshold sensitivity...')

# NMI palette — same family as Fig 1/2 for cross-figure consistency
line_colors = ['#484878', '#7884B4', '#B4C0E4', '#888888']  # last → neutral gray for the lightest tier (contrast)
line_styles = ['-', '--', '-.', ':']
line_markers = ['o', 's', '^', 'D']

fig, ax = plt.subplots(figsize=(5.5, 3.5))
T = list(range(7))

# Collect end-of-line y-values for direct labeling
hr_curves = {}
for i, m in enumerate(thresh_models):
    key = f'{m}@mathvista'
    hr = [threshold_data[key]['hr_curve'][str(t)] * 100 for t in T]
    hr_curves[i] = hr
    ax.plot(T, hr, color=line_colors[i], lw=1.6, ls=line_styles[i],
            marker=line_markers[i], ms=4.5, mec='white', mew=0.5,
            label=MODELS[i])

# Highlight Judge threshold T=3
ax.axvline(x=3, color='#CC3333', ls='--', alpha=0.45, lw=0.9, zorder=0)
ax.text(3.08, 102, 'Judge threshold $T=3$',
        fontsize=6.5, color='#CC3333', ha='left', va='top',
        fontweight='bold', alpha=0.85)

# Mark T=3 data points with subtle ring highlight
for i in range(4):
    ax.scatter([3], [hr_curves[i][3]], s=55, facecolors='none',
               edgecolors=line_colors[i], lw=1.0, zorder=5)

ax.set_xlabel(r'Threshold $T$  (score $< T \Rightarrow$ hallucination)')
ax.set_ylabel('Hallucination Rate (%)')
ax.set_xticks(T)
ax.set_ylim(0, 108)
ax.set_xlim(-0.2, 6.3)
ax.legend(loc='upper left', fontsize=7, ncol=1,
          handlelength=2.2, borderaxespad=0.4)
plt.tight_layout()
savefig(fig, 'fig3_threshold_sensitivity_line')


# ════════════════════════════════════════════
# Helper: hallucination type composition (used by Fig 4 & Fig 6)
# ════════════════════════════════════════════
def hallucination_composition_bar(type_data, fig_name):
    """100% horizontal stacked bar over hallucinated samples only (drop 'None').
    Rows = models; right-side annotation shows n (total hallucinated)."""
    fig, ax = plt.subplots(figsize=(6.5, 2.6))

    bar_colors = PIE[:3]                  # Faith. (rose), Fact. (sand), Logical (sage)
    type_labels = ['Faithfulness', 'Factuality', 'Logical']
    text_colors = ['white', '#333333', '#333333']  # readable on each segment

    y_positions = np.arange(len(MODELS))[::-1]

    for y_idx, m in enumerate(MODELS):
        counts = type_data[m][:3]         # drop 'None'
        total = sum(counts)
        if total == 0:
            continue
        pcts = [c / total * 100 for c in counts]
        left = 0
        for i, pct in enumerate(pcts):
            ax.barh(y_positions[y_idx], pct, left=left,
                    color=bar_colors[i], edgecolor='white', linewidth=0.8,
                    height=0.62)
            if pct >= 8:
                ax.text(left + pct / 2, y_positions[y_idx],
                        f'{pct:.1f}%', ha='center', va='center',
                        fontsize=7, color=text_colors[i], fontweight='bold')
            left += pct
        ax.text(102, y_positions[y_idx], f'$n$={total}',
                ha='left', va='center', fontsize=7, color='#555')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(MODELS, fontsize=7.5)
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0', '25', '50', '75', '100'], fontsize=7)
    ax.set_xlabel('Composition over hallucinated samples (%)')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=bar_colors[i])
                      for i in range(3)]
    ax.legend(legend_patches, type_labels,
              loc='upper center', bbox_to_anchor=(0.5, 1.18),
              ncol=3, fontsize=7.5, frameon=False)

    plt.tight_layout()
    savefig(fig, fig_name)


# ════════════════════════════════════════════
# Fig 4: MathVista hallucination type composition
# ════════════════════════════════════════════
print('[4/6] MathVista hallucination type composition...')
hallucination_composition_bar(mv_type_data, 'fig4_mathvista_composition')


# ════════════════════════════════════════════
# Fig 5: Radar — refined (NMI palette, no fill, vertex markers)
# ════════════════════════════════════════════
print('[5/6] Radar chart...')

radar_colors = ['#484878', '#7884B4', '#888888', '#B4C0E4']
radar_ls = ['-', '--', '-.', ':']
radar_markers = ['o', 's', '^', 'D']

N = len(radar_dims)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Refined grid
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels([], fontsize=6)              # suppress default radial labels
ax.yaxis.set_tick_params(pad=-2, labelsize=6.5)
ax.grid(True, color='#dddddd', lw=0.4, zorder=0)
ax.spines['polar'].set_visible(False)

# Display % only at outer ring for selected angles (0°, 90°, 180°, 270°)
for val, label in [(20, '20%'), (40, '40%'), (60, '60%'), (80, '80%')]:
    ax.text(np.pi / 2, val, label, ha='center', va='bottom',
            fontsize=6, color='#999999')

# Axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_dims, fontsize=7)

# Plot each model
for i, m in enumerate(MODELS):
    vals = radar_data[m] + radar_data[m][:1]
    ax.plot(angles, vals, color=radar_colors[i], lw=1.6,
            ls=radar_ls[i], label=m,
            marker=radar_markers[i], ms=5, mec='white', mew=0.6)

# Legend — placed inside top-right empty sector
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.08),
          fontsize=7.5, frameon=False, handlelength=2.5)

plt.tight_layout()
savefig(fig, 'fig5_radar_comparison')



print('\n✅ All 5 figures saved to report/figures/')
