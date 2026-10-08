"""Bar chart: CCAC start position, in the same form as the length chart.

All twenty-eight designed CCAC spacers, grouped by the index at which the motif begins.
Position 0 means CCAC occupies the first four bases; position 2 and 3 mean it has been
shifted inward.

Bars are the mean of the per-sequence means; error bars are one standard deviation
across the sequences in the group, not the replicate error, and each sequence is drawn
as a point. The position-3 group carries two very low sequences, which is why its
standard deviation is three times the others'.

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
c['v'] = c['mean'] / 1e6
c['pos'] = c.spacer.str.find('CCAC')

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
rng = np.random.default_rng(11)
POS = [0, 2, 3]
means, sds, ns = [], [], []
for i, p_ in enumerate(POS):
    s = c[c.pos == p_].v
    means.append(s.mean()); sds.append(s.std(ddof=1)); ns.append(len(s))
    ax.bar(i, means[-1], width=.62, color=ACC, alpha=.30, edgecolor=ACC, lw=2.0,
           zorder=2)
    ax.errorbar(i, means[-1], yerr=sds[-1], color=ACC_D, lw=2.4, capsize=9,
                capthick=2.4, zorder=5)
    ax.scatter(np.full(len(s), i) + rng.uniform(-.13, .13, len(s)), s, s=62,
               color=ACC_D, edgecolors='white', lw=1.1, zorder=6)

ax.axhline(1.1436, color='#a8a6a0', ls=(0, (5, 3)), lw=1.8, zorder=1)
ax.annotate('library Q80 = 1.144', (-.34, 1.1436), xytext=(0, 7),
            textcoords='offset points', ha='left', fontsize=FS - 3, color='#8a8880')

for i, (m, sd, n) in enumerate(zip(means, sds, ns)):
    ax.annotate(f'{m:.3f}', (i, m + sd), xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=FS + 1, color=INK, fontweight='bold')

p_kw = kruskal(*[c[c.pos == p_].v.values for p_ in POS])[1]
p02 = mannwhitneyu(c[c.pos == 0].v, c[c.pos == 2].v, alternative='two-sided').pvalue
p03 = mannwhitneyu(c[c.pos == 0].v, c[c.pos == 3].v, alternative='two-sided').pvalue
p23 = mannwhitneyu(c[c.pos == 2].v, c[c.pos == 3].v, alternative='two-sided').pvalue

ybar = 1.94
ax.plot([0, 0, 2, 2], [ybar, ybar + .035, ybar + .035, ybar], color=INK, lw=1.6)
ax.annotate(f'Kruskal-Wallis  $p$ = {p_kw:.4f}   —   position matters',
            (.5, ybar + .045), xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=FS, color=ACC_D, fontweight='bold')
ax.annotate(f'pairwise Mann-Whitney:      0 vs 2  $p$ = {p02:.4f}       '
            f'0 vs 3  $p$ = {p03:.4f}       '
            f'2 vs 3  $p$ = {p23:.2f}',
            (.5, ybar + .045), xytext=(0, 34), textcoords='offset points',
            ha='center', fontsize=FS - 3, color=INK2)

ax.set_xticks(range(3))
ax.set_xticklabels(["5' end\n(position 0)", 'position 2', 'position 3'])
ax.set_xlim(-.55, 2.55); ax.set_ylim(0, 2.30)
ax.set_ylabel('Expression  (fluorescence/OD$_{600}$, $10^6$)')
ax.set_title('Moving CCAC off the 5\' end costs expression', loc='left', color=INK,
             pad=14)
ax.grid(axis='y'); ax.set_axisbelow(True)

fig.text(.5, -.035,
         'All twenty-eight designed CCAC spacers, three or six replicates each. Position is the index at which CCAC begins, so position 0 means the motif occupies the first\n'
         'four bases. Bars are the mean of the per-sequence means and the error bars are one standard deviation across the sequences in the group, not the replicate error;\n'
         'each point is one sequence. The position-3 group holds two very low sequences (0.17 and 0.35), which is why its standard deviation is roughly three times the\n'
         'others. Reading the same data as a rank correlation against position gives rho = -0.82, p = 1.1e-07.',
         ha='center', va='top', fontsize=FS - 3, color=INK2, linespacing=1.7)

fig.savefig(R / 'fig_ccac_position_bars.png', bbox_inches='tight', pad_inches=.3)
print('wrote fig_ccac_position_bars.png')
for p_, m, sd, n in zip(POS, means, sds, ns):
    print(f'  pos {p_} n={n}  mean={m:.4f}  SD={sd:.4f}')
print('KW p', f'{p_kw:.4f}', ' 0v2', f'{p02:.4f}', ' 0v3', f'{p03:.4f}',
      ' 2v3', f'{p23:.4f}')
