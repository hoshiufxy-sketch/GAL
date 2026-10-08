"""Bar chart: CCAC start position, on the same data and the same bins as the motif figure.

Panel C of the motif figure pools the library's CCAC-containing sequences with the
twenty-eight designed ones, de-duplicates by spacer and computes a high-value rate per
position. This chart uses that identical grouping - same pooling, same de-duplication,
same five position bins, same position labels - but plots mean expression with a
standard deviation instead of a rate, so the two can be read side by side.

Position 1 means CCAC occupies the first four bases of the spacer. Position 5 has only
two sequences and is shown for completeness.

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
LIBP = DATA / 'sequence_means.csv'
DESP = DATA / 'ccac28_extracted.csv'

lib = pd.read_csv(LIBP)
lib['spacer'] = lib.spacer.str.strip().str.upper()
Q80 = float(np.quantile(lib.norm, 0.80))
des = pd.read_csv(DESP)
des['spacer'] = des.spacer.str.upper()
r1 = lib[lib.spacer.str.contains('CCAC')][['spacer', 'norm']].rename(
    columns={'norm': 'val'})
r2 = des[des.spacer.str.contains('CCAC')][['spacer', 'mean']].rename(
    columns={'mean': 'val'})
M = pd.concat([r1, r2]).groupby('spacer', as_index=False).val.mean()
M['high'] = M.val >= Q80
M['pos'] = M.spacer.str.find('CCAC')

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
                     'xtick.labelsize': FS - 2, 'ytick.labelsize': FS - 1,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'grid.color': '#eceae5', 'grid.linewidth': 1.0})

fig, ax = plt.subplots(figsize=(10.4, 7.6))
rng = np.random.default_rng(11)
POS = sorted(M.pos.unique())
means, sds, ns, rates = [], [], [], []
for i, p_ in enumerate(POS):
    s = M[M.pos == p_]
    means.append(s.val.mean() / 1e6)
    sds.append(s.val.std(ddof=1) / 1e6)
    ns.append(len(s)); rates.append(s.high.mean())
    ax.bar(i, means[-1], width=.62, color=ACC, alpha=.30, edgecolor=ACC, lw=2.0,
           zorder=2)
    ax.errorbar(i, means[-1], yerr=sds[-1], color=ACC_D, lw=2.4, capsize=9,
                capthick=2.4, zorder=5)
    ax.scatter(np.full(len(s), i) + rng.uniform(-.15, .15, len(s)),
               s.val.to_numpy(float) / 1e6, s=58, color=ACC_D,
               edgecolors='white', lw=1.1, zorder=6)

ax.axhline(Q80 / 1e6, color='#a8a6a0', ls=(0, (5, 3)), lw=1.8, zorder=1)
ax.annotate(f'library Q80 = {Q80/1e6:.3f}', (4.50, Q80 / 1e6), xytext=(0, 7),
            textcoords='offset points', ha='right', fontsize=FS - 3, color='#8a8880')
for i, (m, sd) in enumerate(zip(means, sds)):
    ax.annotate(f'{m:.3f}', (i, m + sd), xytext=(0, 11), textcoords='offset points',
                ha='center', fontsize=FS, color=INK, fontweight='bold')
    ax.annotate(f'{rates[i]:.0%}', (i, 0.03), xycoords=('data', 'axes fraction'),
                ha='center', fontsize=FS - 2, color=ACC_D, fontweight='bold')

p_kw = kruskal(*[M[M.pos == p_].val.values for p_ in POS])[1]
p13 = mannwhitneyu(M[M.pos == 0].val, M[M.pos == 2].val,
                   alternative='two-sided').pvalue
p14 = mannwhitneyu(M[M.pos == 0].val, M[M.pos == 3].val,
                   alternative='two-sided').pvalue
p15 = mannwhitneyu(M[M.pos == 0].val, M[M.pos == 4].val,
                   alternative='two-sided').pvalue
p12 = mannwhitneyu(M[M.pos == 0].val, M[M.pos == 1].val,
                   alternative='two-sided').pvalue

ybar = 2.00
ax.plot([0, 0, 4, 4], [ybar, ybar + .03, ybar + .03, ybar], color=INK, lw=1.6)
ax.annotate(f'Kruskal-Wallis  $p$ = {p_kw:.4f}   —   position matters',
            (.5, ybar + .04), xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=FS, color=ACC_D, fontweight='bold')
ax.annotate(f'position 1 vs 2  $p$ = {p12:.3f}      vs 3  $p$ = {p13:.3f}      '
            f'vs 4  $p$ = {p14:.4f}      vs 5  $p$ = {p15:.3f}',
            (.5, ybar + .04), xytext=(0, 32), textcoords='offset points',
            ha='center', fontsize=FS - 3, color=INK2)

ax.set_xticks(range(len(POS)))
ax.set_xticklabels([f'position {int(p)+1}' for p in POS])
ax.set_xlim(-.55, 4.55); ax.set_ylim(0, 2.42)
ax.set_ylabel('Expression  (fluorescence/OD$_{600}$, $10^6$)')
ax.set_title('Where CCAC sits in the spacer', loc='left', color=INK, pad=14)
ax.grid(axis='y'); ax.set_axisbelow(True)

fig.text(.5, -.035,
         'Same grouping as panel C of the motif figure: the library\'s CCAC-containing sequences pooled with the twenty-eight designed ones, de-duplicated by spacer, five\n'
         'position bins. Bars are the mean of the per-sequence means, error bars one standard deviation across sequences, and each point is one sequence. Position 5 holds two\n'
         'sequences only. Unlike the length comparison this is strongly significant, but it is not a clean dose-response: position 1 is far above the rest while positions 2, 3,\n'
         '4 and 5 are not ordered among themselves, and only position 1 clears the base rate of 20%.',
         ha='center', va='top', fontsize=FS - 3, color=INK2, linespacing=1.7)

fig.savefig(R / 'fig_ccac_position_bars_unified.png', bbox_inches='tight',
            pad_inches=.3)
print('wrote fig_ccac_position_bars_unified.png')
for p_, m, sd, n, r in zip(POS, means, sds, ns, rates):
    print(f'  position {int(p_)+1}  n={n:3d}  mean={m:.4f}  SD={sd:.4f}  '
          f'high={r:.0%}')
print('KW p', f'{p_kw:.4f}')
