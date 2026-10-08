"""Three learners on two metrics, as bars.

Panel A  R-squared, in-sample against grouped out-of-fold, for XGBoost ensemble,
         Gaussian process and neural-network ensemble on the deployed 128 descriptors.
Panel B  the same three learners' out-of-fold rank correlation.

Zero on panel A is the constant predictor. Grouped folds put every one-base adjacency
component entirely on one side, so the out-of-fold number is what the model does on
sequences related to nothing it has seen.

Axes carry the bare quantity names (the target definition lives in the footnote) and
every glyph is set as large as the layout allows.

PNG only.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
DATA = R.parent / 'data'
D = pd.read_csv(DATA / 'fit_quality_3models.csv')

FS = 24
INK, INK2 = '#0b0b0b', '#52514e'
C_IN, C_OUT = '#c9c8c2', '#eb6834'
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': FS,
                     'axes.titlesize': FS + 5, 'axes.labelsize': FS + 5,
                     'savefig.dpi': 400, 'figure.dpi': 130,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.edgecolor': '#dcdad4', 'axes.labelcolor': INK2,
                     'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'xtick.labelsize': FS + 1, 'ytick.labelsize': FS,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'grid.color': '#eceae5', 'grid.linewidth': 1.2})

M = D.model.tolist()
x = np.arange(len(M))
fig, axes = plt.subplots(1, 2, figsize=(15.5, 9.0), layout='constrained')

ax = axes[0]
w = .34
ax.bar(x - w / 2, D.train_r2, width=w, color=C_IN, edgecolor='white', lw=1.8,
       zorder=3, label='fitted in-sample')
ax.bar(x + w / 2, D.grouped_r2, width=w, color=C_OUT, edgecolor='white', lw=1.8,
       zorder=3, label='out-of-fold, grouped')
for i in range(len(M)):
    ax.annotate(f'{D.train_r2[i]:.2f}', (x[i] - w / 2, D.train_r2[i]),
                xytext=(0, 8), textcoords='offset points', ha='center',
                fontsize=FS - 1, color=INK2)
    ax.annotate(f'{D.grouped_r2[i]:.3f}', (x[i] + w / 2, D.grouped_r2[i]),
                xytext=(0, 8), textcoords='offset points', ha='center',
                fontsize=FS - 1, color='#b8431c', fontweight='bold')
ax.axhline(0, color=INK, lw=1.8)
ax.annotate('predicting the mean', (len(M) - .52, 0), xytext=(0, 9),
            textcoords='offset points', ha='right', fontsize=FS - 3, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(M, fontsize=FS + 2)
ax.set_ylim(-.03, .80); ax.set_yticks([0, .2, .4, .6])
ax.set_ylabel('$R^2$')
ax.set_title('A   Variance explained', loc='left', color=INK, pad=14)
ax.legend(frameon=False, fontsize=FS - 2, loc='upper right', handlelength=1.3,
          labelspacing=.35)
ax.grid(axis='y'); ax.set_axisbelow(True)

ax = axes[1]
ax.bar(x, D.grouped_rho, width=.5, color=C_OUT, edgecolor='white', lw=1.8, zorder=3)
for i in range(len(M)):
    ax.annotate(f'{D.grouped_rho[i]:.3f}', (x[i], D.grouped_rho[i]),
                xytext=(0, 9), textcoords='offset points', ha='center',
                fontsize=FS - 1, color='#b8431c', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(M, fontsize=FS + 2)
ax.set_ylim(0, .44); ax.set_yticks([0, .1, .2, .3, .4])
ax.set_ylabel('Spearman $\\rho$')
ax.set_title('B   Ranking quality', loc='left', color=INK, pad=14)
ax.grid(axis='y'); ax.set_axisbelow(True)

fig.suptitle('Three different learners converge on the same weak fit', y=1.075,
             fontsize=FS + 10, color=INK)

fig.savefig(R / 'fig_fit_quality_bars.png', bbox_inches='tight', pad_inches=.35)
print('wrote fig_fit_quality_bars.png')
print(D.round(3).to_string(index=False))
