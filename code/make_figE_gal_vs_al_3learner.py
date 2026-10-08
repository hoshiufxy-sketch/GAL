"""Figure E: learning curves by feature set, GAL against AL, three learners.

Panels 1 and 2 are the original runs, unchanged: k-mer composition (91 descriptors) and
five physicochemical descriptors, both on the 421-member library. Panel 3 is the
Nucleotide Transformer v2 embedding (512 descriptors), which exists for only 417 of the
421 and is therefore run on that 417 pool.

The label of the middle panel is the one thing changed from the earlier version. It read
"ViennaRNA physics - 5", but the five columns are
[Total_MW, Total_Stacking, GC_Front, GC_Back, DG_total] - molecular weight, a stacking
term and two GC counts are not ViennaRNA quantities, and only DG_total is.

Because panel 3 is on 417 and panels 1-2 on 421, the three panels are not on one pool
and their levels should not be compared across panels. Each panel is internally paired:
75 campaigns, seeds shared across arms and learners.

Writes figE_gal_vs_al_3learner.{png,svg,pdf} and .csv
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
OLD = pd.read_csv(DATA / 'curves_3learner.csv')
NEW = pd.read_csv(DATA / 'curves_3features_nt.csv')
for d in (OLD, NEW):
    d['cum'] = d.n_hits / d.n_measured

FS = 17
INK, INK2 = '#0b0b0b', '#52514e'
MCOL = {'XGB': '#2a78d6', 'GPR': '#eb6834', 'DNN': '#1baf7a'}
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'font.size': FS, 'axes.titlesize': FS + 2,
    'axes.labelsize': FS + 1, 'savefig.dpi': 400, 'figure.dpi': 130,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#dcdad4', 'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': FS - 1,
    'ytick.labelsize': FS - 1, 'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': '#eceae5', 'grid.linewidth': 1.0,
})

PANELS = [(OLD, 'kmer', 'A', 421),
          (OLD, 'physico', 'B', 421),
          (NEW, 'nt', 'C', 417)]
MODELS = ['XGB', 'GPR', 'DNN']
STAGES = [0, 1, 2]
XT = ['initial\n(120)', 'round 1\n(130)', 'round 2\n(140)']

fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.6), sharey=True,
                         layout='constrained')
rows = []

for c, (D, feat, title, pool) in enumerate(PANELS):
    ax = axes[c]
    ax.axhline(.20, color='#a8a6a0', ls=(0, (1, 2.2)), lw=1.7, zorder=1)
    for m in MODELS:
        col = MCOL[m]
        for arm, solid, alpha, lw in [('guided', True, 1.0, 3.1),
                                      ('AL', False, .50, 2.5)]:
            S = D[(D.features == feat) & (D.learner == m) & (D.arm == arm)]
            g = S.groupby('stage').cum
            mu = g.mean().reindex(STAGES).to_numpy(float)
            n = g.size()
            se = (g.std(ddof=1) / np.sqrt(n)).reindex(STAGES).to_numpy(float)
            ax.errorbar(STAGES, mu, yerr=1.96 * se, color=col, alpha=alpha, lw=lw,
                        ls='-' if solid else (0, (4.5, 2.6)),
                        capsize=3.6, capthick=1.5, elinewidth=1.5,
                        zorder=4 if solid else 3)
            for st, v in zip(STAGES, mu):
                rows.append(dict(feature=feat, learner=m, arm=arm, stage=st,
                                 pool=pool, mean=v))
    if c == 0:
        ax.set_ylabel('Cumulative hit rate')
    ax.set_xticks(STAGES); ax.set_xticklabels(XT)
    ax.set_xlabel('Measured library')
    ax.set_xlim(-.14, 2.20); ax.set_ylim(.178, .243)
    ax.set_yticks([.18, .19, .20, .21, .22, .23, .24])
    ax.grid(axis='y'); ax.set_axisbelow(True)

legend = [Line2D([], [], color=MCOL[m], lw=3.1, label=m) for m in MODELS]
legend += [Line2D([], [], color='#7a7a76', lw=3.1, label='GAL — guided'),
           Line2D([], [], color='#7a7a76', lw=2.5, ls=(0, (4.5, 2.6)), alpha=.55,
                  label='AL — uncertainty $\\sigma$')]
fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1.13), ncol=5,
           frameon=False, fontsize=FS, handletextpad=.7, columnspacing=2.8)

fig.suptitle('Guided acquisition raises the cumulative hit rate for every learner; '
             'uncertainty sampling does not', y=1.21, fontsize=FS + 7, color=INK)

for ext in ['png', 'svg', 'pdf']:
    fig.savefig(R / f'figE_gal_vs_al_3learner.{ext}', bbox_inches='tight',
                pad_inches=.3)
pd.DataFrame(rows).to_csv(R / 'figE_gal_vs_al_3learner.csv', index=False)
print('wrote figE_gal_vs_al_3learner.png')
for D, feat, title, pool in PANELS:
    g = D[D.arm == 'guided'].pivot_table(index=['features', 'learner'], columns='stage',
                                         values='cum', aggfunc='mean')
    a = D[D.arm == 'AL'].pivot_table(index=['features', 'learner'], columns='stage',
                                     values='cum', aggfunc='mean')
    print(f'{title:34s} pool {pool}   GAL-AL round2: '
          + '  '.join(f'{m} {g.loc[(feat, m), 2] - a.loc[(feat, m), 2]:+.4f}'
                      for m in MODELS))
