"""The highest value each approach reaches, as a bar chart.

    ML          the machine-learning model's highest prediction
    AL          traditional active learning's highest value
    GAL R1      guided Round 1's highest value
    GAL R2      guided Round 2's highest value

The dotted line is the 421-member library's 80th percentile, 1,143,620, which is the
high-value threshold used everywhere else in this project.

PNG only.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent

FS = 22
INK, INK2, INK3 = '#0b0b0b', '#52514e', '#8a8984'
Q80 = 1_143_620
BARS = [('Model prediction', 871_462, '#93A0AC'),
        ('AL', 746_801, '#69737D'),
        ('GAL round 1', 1_716_200, '#C4805F'),
        ('GAL round 2', 1_793_400, '#8E4630')]

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': FS,
                     'axes.titlesize': FS + 4, 'axes.labelsize': FS + 3,
                     'savefig.dpi': 400, 'figure.dpi': 130,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.edgecolor': '#dcdad4', 'axes.labelcolor': INK2,
                     'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'xtick.labelsize': FS, 'ytick.labelsize': FS - 1,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'grid.color': '#eceae5', 'grid.linewidth': 1.1})

fig, ax = plt.subplots(figsize=(12.2, 8.6), layout='constrained')
x = np.arange(len(BARS))
vals = [b[1] for b in BARS]
ax.bar(x, [v / 1e6 for v in vals], width=.34,
       color=[b[2] for b in BARS], edgecolor='none', zorder=3)

for xi, v in zip(x, vals):
    ax.annotate(f'{v:,}', (xi, v / 1e6), xytext=(0, 9), textcoords='offset points',
                ha='center', fontsize=FS - 1, color=INK, fontweight='bold')

ax.axhline(Q80 / 1e6, color='#b9b6b0', ls=(0, (1, 2.4)), lw=1.5, zorder=4)
ax.annotate(f'high-expression threshold  {Q80:,}', (-.34, Q80 / 1e6), xytext=(0, 9),
            textcoords='offset points', ha='left', va='bottom', fontsize=FS - 5,
            color=INK2, zorder=6)

ax.set_xticks(x); ax.set_xticklabels([b[0] for b in BARS], fontsize=FS)
ax.set_xlim(-.62, 3.62)
ax.set_ylim(0, 2.02); ax.set_yticks([0, .5, 1.0, 1.5, 2.0])
ax.set_ylabel('fluorescence / OD$_{600}$  ($10^6$)')
ax.grid(axis='y'); ax.set_axisbelow(True)

fig.text(.5, -.03,
         'The first bar is the measured expression of the sequence with the highest predicted value; the remaining three are the highest values reached by\n'
         f'each acquisition arm. Dotted line, the 421-member library\'s 80th percentile ({Q80:,}), the high-expression threshold.',
         ha='center', va='top', fontsize=FS - 5, color=INK3, linespacing=1.8)

fig.savefig(R / 'fig_best_value_comparison.png', bbox_inches='tight', pad_inches=.3)
print('wrote fig_best_value_comparison.png')
for name, v, _ in BARS:
    print(f'  {name:16s} {v:>12,}   {v/Q80:.2f} x the threshold')
