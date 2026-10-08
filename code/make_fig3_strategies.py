"""Figure 3: exploration strategies compared on the same campaigns, top-20 metric only.

    random / AL (sigma) / greedy (mu) / EI (expected improvement) / UCB (mu + sigma) /
    GAL (the unsampled Hamming-1 neighbours of measured high-expression sequences first)

One panel per feature set, one row. Thick lines pool the three learners, which are drawn
thin behind; end labels give identity independently of colour.

120 random initial spacers then two rounds of 10; 20 paired campaigns per cell sharing
seeds across every arm.

No panel titles, no figure title and no notes: the panels carry their data and a legend,
the numbers live in the caption.

Writes fig3_strategies.png and fig3_strategies.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parent
DATA = R.parent / 'data'
D = pd.read_csv(DATA / 'strategies_fig3.csv')

FS = 17
INK, INK2 = '#0b0b0b', '#52514e'
ARMS = [('guided', 'GAL — guided', '#eb6834', '-', 3.2, 'o', 6.2),
        ('greedy', 'greedy $\\mu$', '#2a78d6', '-', 2.4, 's', 5.6),
        ('EI', 'EI', '#1baf7a', '-', 2.4, 'v', 5.6),
        ('UCB', 'UCB $\\mu+\\sigma$', '#8a5cd6', '-', 2.4, 'P', 5.6),
        ('AL', 'AL — $\\sigma$', '#7d7c78', (0, (5, 2.6)), 2.3, '^', 5.6),
        ('random', 'random', '#c2c1bc', (0, (1.4, 2.2)), 2.2, 'D', 5.0)]

# Panels 1 and 2 come from the six-arm comparison on the 421-member library (20 paired
# campaigns). Panel 3 is the Nucleotide Transformer v2 embedding, which exists for only
# 417 of the 421 and was run in a separate campaign with the two arms that matter here,
# AL and the guided rule, over 75 paired campaigns. The arm sets therefore differ across
# panels, and an arm absent from a panel's data is simply not drawn.
D_ALL = pd.read_csv(DATA / 'curves_3features_nt.csv')
FEATS = [(D, 'kmer', 'k-mer composition · 91'),
         (D, 'physico', 'physicochemical descriptors · 5'),
         (D_ALL, 'nt', 'Nucleotide Transformer v2 · 512')]
MODELS = ['XGB', 'GPR', 'DNN']
STAGES = [0, 1, 2]
XT = ['initial\n(120)', 'round 1\n(+10)', 'round 2\n(+10)']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'font.size': FS, 'axes.titlesize': FS + 1,
    'axes.labelsize': FS + 1, 'savefig.dpi': 400, 'figure.dpi': 130,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#dcdad4', 'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': FS - 1,
    'ytick.labelsize': FS - 1, 'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': '#eceae5', 'grid.linewidth': 1.0,
})

fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.4), sharey=True,
                         layout='constrained')
rows = []
for c, (SRC, fkey, flabel) in enumerate(FEATS):
    ax = axes[c]
    S = SRC[SRC.features == fkey]
    ends = {}
    for arm, lab, col, ls, lw, mk, ms in ARMS:
        if not (S.arm == arm).any():
            continue
        for m in MODELS:
            g = S[(S.arm == arm) & (S.learner == m)].groupby('stage').topk
            ax.plot(STAGES, g.mean().reindex(STAGES), color=col, lw=0.9, ls=ls,
                    alpha=.28, zorder=2)
        g = S[S.arm == arm].groupby('stage').topk
        mu = g.mean().reindex(STAGES).to_numpy(float)
        n = g.size()
        se = (g.std(ddof=1) / np.sqrt(n)).reindex(STAGES).to_numpy(float)
        ax.errorbar(STAGES, mu, yerr=1.96 * se, color=col, lw=lw, ls=ls, marker=mk,
                    ms=ms, mfc='white' if arm in ('AL', 'random') else col,
                    mec=col, mew=1.5, capsize=0, elinewidth=0.9,
                    zorder=6 if arm == 'guided' else 5)
        ends[arm] = (mu[-1], col, lab)
        for st, v in zip(STAGES, mu):
            rows.append(dict(feature=fkey, metric='topk', arm=arm, stage=int(st),
                             value=float(v)))
    order = sorted(ends, key=lambda a: -ends[a][0])
    for rank, a in enumerate(order):
        v, col, lab = ends[a]
        ax.annotate(lab, (STAGES[-1], v), xytext=(8, [10, 3, -4, -11, 6, -18][rank]),
                    textcoords='offset points', va='center', ha='left',
                    fontsize=FS - 4, color=col, fontweight='bold')
    if c == 0:
        ax.set_ylabel('Top-20 found  (of 20)')
    ax.set_xticks(STAGES); ax.set_xticklabels(XT)
    ax.set_xlabel('Measured library')
    ax.set_xlim(-.15, 2.65)
    ax.grid(axis='y'); ax.set_axisbelow(True)

legend = [Line2D([], [], color=col, lw=lw, ls=ls, marker=mk, ms=ms,
                 mfc='white' if arm in ('AL', 'random') else col, mec=col, mew=1.5,
                 label=lab) for arm, lab, col, ls, lw, mk, ms in ARMS]
legend += [Line2D([], [], color='#8a8880', lw=0.9, alpha=.5,
                  label='individual learners')]
fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1.07), ncol=7,
           frameon=False, fontsize=FS - 1, handletextpad=.5, columnspacing=1.6)

fig.savefig(R / 'fig3_strategies.png', bbox_inches='tight', pad_inches=.3)
pd.DataFrame(rows).to_csv(DATA / 'fig3_strategies.csv', index=False)
print('wrote fig3_strategies.png')

from scipy.stats import ttest_rel                                 # noqa: E402
print('\nround 2, mean paired difference GAL minus each arm:')
E_ALL = D_ALL[D_ALL.stage == 2]
for SRC, fkey, _ in FEATS:
    E = E_ALL if SRC is D_ALL else D[D.stage == 2]
    for arm, lab, *_ in ARMS:
        if arm == 'guided' or not (E.arm == arm).any():
            continue
        a = E[(E.features == fkey) & (E.arm == 'guided')].groupby(
            ['block', 'trial']).topk.mean()
        b = E[(E.features == fkey) & (E.arm == arm)].groupby(
            ['block', 'trial']).topk.mean()
        d = (a - b).to_numpy()
        t, p = ttest_rel(a, b)
        print(f'  {fkey:8s} vs {arm:7s} {d.mean():+.3f}   t={t:5.1f}  p={p:.1e}'
              f'   wins {int((d > 0).sum())}/{len(d)}')
