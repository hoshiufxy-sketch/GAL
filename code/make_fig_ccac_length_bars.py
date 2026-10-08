"""Bar chart: the CCAC-anchored family by spacer length, 6 / 7 / 8 nt, n = 4 / 6 / 6.

Sixteen designed spacers that all carry CCAC at the 5' end, so the motif and its
position are held fixed and only the spacer length varies.

Bars are the mean of the per-sequence means; error bars are one standard deviation
across the sequences in that group, not the replicate error, and the individual
sequences are drawn over the bars so n and the spread are both visible. Six of the
sixteen were measured in six replicates and ten in three.

PNG only.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
DATA = R.parent / 'data'
c = pd.read_csv(DATA / 'ccac28_extracted.csv')
c['L'] = c.spacer.str.len()
c['v'] = c['mean'] / 1e6
a = c[c.spacer.str.startswith('CCAC')]

FS = 15
INK, INK2 = '#0b0b0b', '#52514e'
ACC, ACC_D = '#eb6834', '#b8431c'
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': FS,
                     'axes.titlesize': FS + 2, 'axes.labelsize': FS + 1,
                     'savefig.dpi': 400, 'figure.dpi': 130,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.edgecolor': '#dcdad4', 'axes.labelcolor': INK2,
                     'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'xtick.labelsize': FS, 'ytick.labelsize': FS - 1,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'grid.color': '#eceae5', 'grid.linewidth': 1.0})

fig, ax = plt.subplots(figsize=(8.6, 7.6))
rng = np.random.default_rng(7)
LENS = [6, 7, 8]
means, sds, ns = [], [], []
for i, L in enumerate(LENS):
    s = a[a.L == L].v
    means.append(s.mean()); sds.append(s.std(ddof=1)); ns.append(len(s))
    ax.bar(i, means[-1], width=.62, color=ACC, alpha=.30, edgecolor=ACC, lw=2.0,
           zorder=2)
    ax.errorbar(i, means[-1], yerr=sds[-1], color=ACC_D, lw=2.4, capsize=9,
                capthick=2.4, zorder=5)
    ax.scatter(np.full(len(s), i) + rng.uniform(-.13, .13, len(s)), s, s=62,
               color=ACC_D, edgecolors='white', lw=1.1, zorder=6)

ax.axhline(1.1436, color='#a8a6a0', ls=(0, (5, 3)), lw=1.8, zorder=1)
ax.annotate('library Q80 = 1.144', (2.42, 1.1436), xytext=(0, 7),
            textcoords='offset points', ha='right', fontsize=FS - 3, color='#8a8880')

for i, (m, sd, n) in enumerate(zip(means, sds, ns)):
    ax.annotate(f'{m:.3f}', (i, m + sd), xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=FS + 1, color=INK, fontweight='bold')

p_kw = kruskal(*[a[a.L == L].v.values for L in LENS])[1]
p67 = mannwhitneyu(a[a.L == 6].v, a[a.L == 7].v, alternative='two-sided').pvalue
p68 = mannwhitneyu(a[a.L == 6].v, a[a.L == 8].v, alternative='two-sided').pvalue
p78 = mannwhitneyu(a[a.L == 7].v, a[a.L == 8].v, alternative='two-sided').pvalue

ybar = 1.94
ax.plot([0, 0, 2, 2], [ybar, ybar + .035, ybar + .035, ybar], color=INK, lw=1.6)
ax.annotate(f'Kruskal-Wallis  $p$ = {p_kw:.2f}   —   no significant difference '
            f'between lengths',
            (.5, ybar + .045), xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=FS, color=ACC_D, fontweight='bold')
ax.annotate(f'pairwise Mann-Whitney:      6 vs 7  $p$ = {p67:.2f}       '
            f'6 vs 8  $p$ = {p68:.2f}       '
            f'7 vs 8  $p$ = {p78:.2f}',
            (.5, ybar + .045), xytext=(0, 34), textcoords='offset points',
            ha='center', fontsize=FS - 3, color=INK2)

ax.set_xticks(range(3))
ax.set_xticklabels(['6 nt', '7 nt', '8 nt'])
ax.set_xlim(-.55, 2.55); ax.set_ylim(0, 2.30)
ax.set_ylabel('Expression  (fluorescence/OD$_{600}$, $10^6$)')
ax.set_title('CCAC anchored at the 5\' end: no length effect', loc='left', color=INK,
             pad=14)
ax.grid(axis='y'); ax.set_axisbelow(True)

fig.text(.5, -.035,
         'Sixteen designed CCAC spacers that all carry the motif at the 5\' end, so only the spacer length varies. Bars are the mean of the per-sequence means;\n'
         'error bars are one standard deviation across the sequences in the group, not the replicate error, and each point is one sequence. Six were measured in\n'
         'six replicates and ten in three. The group means span 12% and are not separable (Kruskal-Wallis p = 0.30); the widest pairwise contrast, 6 vs 8 nt, is\n'
         'p = 0.067 with four and six sequences respectively.',
         ha='center', va='top', fontsize=FS - 3, color=INK2, linespacing=1.7)

fig.savefig(R / 'fig_ccac_length_bars.png', bbox_inches='tight', pad_inches=.3)
print('wrote fig_ccac_length_bars.png')
for L, m, sd, n in zip(LENS, means, sds, ns):
    print(f'  {L}nt n={n}  mean={m:.4f}  SD={sd:.4f}')
print('KW p', round(p_kw, 4), ' 6v7', round(p67, 4), ' 6v8', round(p68, 4),
      ' 7v8', round(p78, 4))
