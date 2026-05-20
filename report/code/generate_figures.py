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
# Fig 1: POPE HR — Grouped bar
# ════════════════════════════════════════════
print('[1/6] POPE hallucination rate...')
fig, ax = plt.subplots(figsize=(5.5, 3.5))
x = np.arange(3)
w = 0.18
for i, m in enumerate(MODELS):
    vals = pope_hr[m]
    ax.bar(x + i * w - 1.5 * w, vals, w, label=m, color=C[i], linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels(pope_splits)
ax.set_ylabel('Hallucination Rate (%)')
ax.legend(ncol=4, fontsize=6.5, columnspacing=1, handlelength=1)
ax.set_ylim(0, 11)
plt.tight_layout()
savefig(fig, 'fig1_pope_hr_bar')


# ════════════════════════════════════════════
# Fig 2: MathVista Direct vs CoT
# ════════════════════════════════════════════
print('[2/6] MathVista Direct vs CoT...')
fig, ax = plt.subplots(figsize=(5.5, 3.5))
x = np.arange(4)
w = 0.30
ax.bar(x - w / 2, mv_direct_hr, w, label='Direct', color=C2[0], linewidth=0.3)
ax.bar(x + w / 2, mv_cot_hr, w, label='CoT', color=C2[1], linewidth=0.3)

for i in range(4):
    d = mv_cot_hr[i] - mv_direct_hr[i]
    txt = f'$\Delta$={d:+.1f}'
    clr = '#2E8B57' if d < 0 else '#CC3333'
    ax.text(i + w / 2, mv_cot_hr[i] + 1.2, txt, ha='center', va='bottom',
            fontsize=6.5, color=clr, fontweight='bold')
    ax.plot([i - w / 2, i + w / 2], [mv_direct_hr[i], mv_cot_hr[i]],
            color='gray', lw=0.5, ls='--', alpha=0.4)

ax.set_xticks(x)
ax.set_xticklabels(MODELS_SHORT, fontsize=6.5)
ax.set_ylabel('Hallucination Rate (%)')
ax.legend(fontsize=8)
ax.set_ylim(0, 53)
plt.tight_layout()
savefig(fig, 'fig2_mathvista_direct_cot_bar')


# ════════════════════════════════════════════
# Fig 3: Threshold sensitivity — line
# ════════════════════════════════════════════
print('[3/6] Threshold sensitivity...')
fig, ax = plt.subplots(figsize=(5, 3.5))
T = list(range(7))
ls = ['-', '--', '-.', ':']

for i, m in enumerate(thresh_models):
    key = f'{m}@mathvista'
    hr = [threshold_data[key]['hr_curve'][str(t)] * 100 for t in T]
    ax.plot(T, hr, label=MODELS[i], color=C[i], lw=1.8, ls=ls[i], marker='o', ms=3.5)

ax.axvline(x=3, color='gray', ls='--', alpha=0.4, lw=0.8)
ax.axhline(y=50, color='gray', ls=':', alpha=0.3, lw=0.6)
ax.set_xlabel('Threshold $T$  (score $< T$ → hallucination)')
ax.set_ylabel('Hallucination Rate (%)')
ax.set_xticks(T)
ax.legend(fontsize=7)
ax.set_ylim(0, 105)
plt.tight_layout()
savefig(fig, 'fig3_threshold_sensitivity_line')


# ════════════════════════════════════════════
# Fig 4: MathVista pie 2×2
# ════════════════════════════════════════════
print('[4/6] MathVista pie chart...')
fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
explode = (0, 0, 0, 0.04)
for idx, (m, ax_i) in enumerate(zip(MODELS, axes)):
    vals = mv_type_data[m]
    def pct_formatter(pct):
        return f'{pct:.1f}%' if pct > 5 else ''
    wedges, texts, autotexts = ax_i.pie(vals, colors=PIE, explode=explode, startangle=90,
                                         autopct=pct_formatter, pctdistance=0.72)
    for at in autotexts:
        at.set_fontsize(6)
        at.set_fontweight('bold')
    ax_i.set_title(m, fontsize=7, pad=2)
fig.legend(wedges, mv_types_short, loc='lower center', ncol=4, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.02))
plt.subplots_adjust(bottom=0.12, wspace=0.1)
savefig(fig, 'fig4_mathvista_pie')


# ════════════════════════════════════════════
# Fig 5: Radar
# ════════════════════════════════════════════
print('[5/6] Radar chart...')
N = len(radar_dims)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

radar_ls = ['-', '--', '-.', ':']
for i, m in enumerate(MODELS):
    vals = radar_data[m] + radar_data[m][:1]
    ax.fill(angles, vals, alpha=0.05, color=C[i])
    ax.plot(angles, vals, color=C[i], lw=1.5, ls=radar_ls[i], label=m)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_dims, fontsize=7)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=6)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)
plt.tight_layout()
savefig(fig, 'fig5_radar_comparison')


# ════════════════════════════════════════════
# Fig 6: VQA-RAD pie 2×2
# ════════════════════════════════════════════
print('[6/6] VQA-RAD pie chart...')
fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
for idx, (m, ax_i) in enumerate(zip(MODELS, axes)):
    vals = vqa_type_data[m]
    def pct_formatter(pct):
        return f'{pct:.1f}%' if pct > 5 else ''
    wedges, texts, autotexts = ax_i.pie(vals, colors=PIE, explode=explode, startangle=90,
                                         autopct=pct_formatter, pctdistance=0.72)
    for at in autotexts:
        at.set_fontsize(6)
        at.set_fontweight('bold')
    ax_i.set_title(m, fontsize=7, pad=2)
fig.legend(wedges, mv_types_short, loc='lower center', ncol=4, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.02))
plt.subplots_adjust(bottom=0.12, wspace=0.1)
savefig(fig, 'fig6_vqarad_pie')

print('\n✅ All 6 figures saved to report/figures/')
