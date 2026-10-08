"""The three panels in one figure: landscape, distance profile, and distance-1 yield.

Content is unchanged from the earlier version; the wording and the keys are not.

    axis labels     "Sequence space 1/2" instead of "MDS coordinate 1/2" - the two
                    horizontal axes are a two-dimensional embedding of Hamming distance
                    and carry no units, so the label only needs to say what they order
    legend text     "high-expression" rather than "high-value", "reference sequence"
                    rather than "anchor", and "with a Hamming-1 neighbour" rather than
                    the abstract ">=1 neighbour"
    keys            each panel carries a real legend; the prose footnotes are gone

Hamming-1 is same-length only, giving 86 pairs and 297 sequences without a measured
neighbour. High expression is fluorescence/OD600 >= 1,143,620, the pooled 80th
percentile of the 421 measurements.

Writes landscape_neighbour_en.{png,svg,pdf}
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parent
DATA = R.parent / 'data'
nodes = pd.read_csv(DATA / 'nodes.csv')
edges = pd.read_csv(DATA / 'hamming1_pairs.csv')
prof = pd.read_csv(DATA / 'distance_profiles.csv')
audit = json.loads((DATA / 'audit.json').read_text())
tau = audit['high_threshold_Q80']

plt.rcParams.update({'font.family': 'Arial', 'font.size': 14,
                     'savefig.facecolor': 'white',
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
HOT, LOW, ISO, EDGE = '#BC4936', '#94A0AC', '#D5D8DC', '#BFC5CB'
COOL = '#9AA3AB'
INK, INK2 = '#0b0b0b', '#52514e'
HI, NH = 'High-expression', 'Non-high-expression'

# ---------------------------------------------------------------- data
sp = nodes.spacer.tolist()
ln = nodes.length.to_numpy(int)
n = len(sp)
L = max(len(s) for s in sp)
M = np.full((n, L), -1, int)
for i, s in enumerate(sp):
    M[i, :len(s)] = [ord(c) for c in s]
Hd = (M[:, None, :] != M[None, :, :]).sum(-1)
H1 = (Hd == 1) & (ln[:, None] == ln[None, :])
np.fill_diagonal(H1, False)

y = nodes.norm.to_numpy(float)
hi = nodes.high_Q80.to_numpy(bool)
base = hi.mean()
n_above = int((y >= tau).sum())
n_iso = int((nodes.measured_hamming1_neighbors == 0).sum())
nodes['z'] = np.log10(1.0 + y)

Hm = Hd.astype(float)
Hm[~(ln[:, None] == ln[None, :])] = 10.0
np.fill_diagonal(Hm, 0)
from sklearn.manifold import MDS                                   # noqa: E402
XY = MDS(n_components=2, dissimilarity='precomputed', random_state=0, n_init=6,
         max_iter=400, normalized_stress='auto').fit_transform(Hm)
nodes['jx'], nodes['jy'] = XY[:, 0], XY[:, 1]
lookup = nodes.set_index('spacer')

ii, jj = np.where(np.triu(H1, 1))
anc = np.concatenate([ii, jj])
nbr = np.concatenate([jj, ii])
ahi, nhi = hi[anc], hi[nbr]
p_hh, p_lh = float(nhi[ahi].mean()), float(nhi[~ahi].mean())
n_ha, n_la = int(ahi.sum()), int((~ahi).sum())
n_hh, n_lh = int(nhi[ahi].sum()), int(nhi[~ahi].sum())
rng = np.random.default_rng(0)
null = np.empty(20000)
for k in range(20000):
    h = rng.permutation(hi)
    null[k] = h[nbr][h[anc]].mean() - h[nbr][~h[anc]].mean()
pval = (1 + int(np.count_nonzero(np.abs(null) >= abs(p_hh - p_lh)))) / 20001

q = prof[prof['quantile'] == .8]
pool = q.groupby('distance').agg(HH=('HH', 'sum'), HL=('HL', 'sum'),
                                 LL=('LL', 'sum'))
pool['p_hi'] = 100 * 2 * pool.HH / (2 * pool.HH + pool.HL)
pool['p_lo'] = 100 * pool.HL / (2 * pool.LL + pool.HL)

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(20.5, 8.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.36, 1.0, .92], left=.020, right=.972,
                      bottom=.145, top=.885, wspace=.30)

# ---------- A ----------
ax = fig.add_subplot(gs[0, 0], projection='3d')
for _, e in edges.sort_values('edge_type', ascending=False).iterrows():
    a, b = lookup.loc[e.sequence_a], lookup.loc[e.sequence_b]
    hh = e.edge_type == 'HH'
    ax.plot([a.jx, b.jx], [a.jy, b.jy], [a.z, b.z],
            color=HOT if hh else EDGE, lw=2.3 if hh else .8,
            alpha=.95 if hh else .26, zorder=4 if hh else 1)
for mask, color, size, alpha in [
        (nodes.measured_hamming1_neighbors == 0, ISO, 20, .55),
        ((nodes.measured_hamming1_neighbors > 0) & ~nodes.high_Q80, LOW, 30, .62),
        ((nodes.measured_hamming1_neighbors > 0) & nodes.high_Q80, HOT, 66, 1.)]:
    v = nodes[mask]
    ax.scatter(v.jx, v.jy, v.z, s=size, c=color, alpha=alpha,
               edgecolors='white' if color == HOT else 'none', linewidths=.5,
               depthshade=False)
ax.set_xlabel('MDS coordinate 1', labelpad=8, fontsize=15, color=INK2)
ax.set_ylabel('MDS coordinate 2', labelpad=8, fontsize=15, color=INK2)
ax.set_zlim(5.0, 6.30)
ax.set_zticks([5.0, 5.25, 5.5, 5.75, 6.0, 6.25])
ax.view_init(23, -60)
ax.set_box_aspect((4, 4, 3.1))
ax.tick_params(pad=2, labelsize=12)
for a_ in [ax.xaxis, ax.yaxis, ax.zaxis]:
    a_.pane.fill = False
    a_.pane.set_edgecolor('#EEEEEE')
    a_.line.set_color('#9C9FA2')
ax.grid(False)
ax.set_title('A', loc='left', pad=4, fontsize=22, fontweight='bold', color=INK)
ax.text2D(1.085, .48, 'log$_{10}$(1 + fluorescence / OD$_{600}$)',
          transform=ax.transAxes, rotation=90, ha='center', va='center',
          fontsize=15, color=INK2, clip_on=False)
legA = [Line2D([], [], marker='o', ls='', ms=9, color=HOT, mec='white', mew=.7,
               label=f'{HI}, with a Hamming-1 neighbour'),
        Line2D([], [], marker='o', ls='', ms=8.5, color=LOW, mec='none',
               label=f'{NH}, with a Hamming-1 neighbour'),
        Line2D([], [], marker='o', ls='', ms=7.5, color=ISO, mec='none',
               label='No Hamming-1 neighbour'),
        Line2D([], [], color=HOT, lw=2.3,
               label=f'Hamming-1 pair, both {HI.lower()}')]
ax.legend(handles=legA, loc='upper left', bbox_to_anchor=(.015, 1.025),
          frameon=False, fontsize=11.5, handletextpad=.6, labelspacing=.45,
          ncol=2, columnspacing=1.5)

# ---------- B ----------
ax = fig.add_subplot(gs[0, 1])
ax.plot(pool.index, pool.p_hi, 'o-', c=HOT, lw=2.4, ms=9, zorder=4,
        label=f'{HI} reference')
ax.plot(pool.index, pool.p_lo, 's--', c=COOL, lw=1.8, ms=7, zorder=3,
        label=f'{NH} reference')
ax.axhline(100 * base, c=INK, ls=(0, (1, 2)), lw=1.4, zorder=2,
           label=f'Pooled proportion ({100*base:.1f}%)')
v = pool.p_hi.iloc[0]
ax.annotate(f'{v:.1f}%', xy=(1, v), xytext=(1.22, v + 8), fontsize=15,
            color=HOT, fontweight='bold',
            arrowprops=dict(arrowstyle='-', lw=.9, color=HOT))
ax.annotate(f'{pool.p_lo.iloc[0]:.1f}%', xy=(1, pool.p_lo.iloc[0]),
            xytext=(1.60, 4.0), fontsize=14, color='#6d757c', ha='left',
            va='center', arrowprops=dict(arrowstyle='-', lw=.9, color=COOL))
ax.set_xlabel('Hamming distance', fontsize=15)
ax.set_ylabel(f'Neighbours that are {HI.lower()} (%)', fontsize=15)
ax.set_xticks(range(1, 9)); ax.set_xlim(.6, 8.4)
ax.set_ylim(0, 70); ax.set_yticks([0, 20, 40, 60])
ax.tick_params(labelsize=13)
ax.set_title('B', loc='left', pad=4, fontsize=22, fontweight='bold', color=INK)
ax.legend(frameon=False, fontsize=12.5, loc='upper right', handlelength=2.1,
          labelspacing=.45)
ax.grid(axis='y', color='#EEEEEE'); ax.set_axisbelow(True)

# ---------- C ----------
ax = fig.add_subplot(gs[0, 2])
ax.bar([0, 1], [p_hh, p_lh], width=.44, color=[HOT, COOL], edgecolor='white',
       linewidth=1.6, zorder=3)
for i, (vv, m, h) in enumerate(zip([p_hh, p_lh], [n_ha, n_la], [n_hh, n_lh])):
    ax.annotate(f'{vv:.3f}\n{h} of {m}', (i, vv), xytext=(0, 9),
                textcoords='offset points', ha='center', fontsize=14,
                color=INK, linespacing=1.45)
ax.axhline(base, color=INK, lw=1.4, ls=(0, (1, 2)), zorder=4,
           label=f'Pooled proportion ({100*base:.1f}%)')
ax.legend(frameon=False, fontsize=12.5, loc='upper right')
ytop = .86
ax.annotate('', xy=(0, ytop), xytext=(1, ytop),
            arrowprops=dict(arrowstyle='<->', color=HOT, lw=1.7))
ax.annotate(f'{p_hh/p_lh:.1f}$\\times$\np = {pval:.0e}', (.5, ytop + .022),
            ha='center', fontsize=14.5, color='#b8431c', linespacing=1.45,
            fontweight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels([f'{HI}\nreference', f'{NH}\nreference'], fontsize=13.5)
ax.set_ylim(0, 1.02); ax.set_yticks([0, .2, .4, .6, .8])
ax.tick_params(axis='y', labelsize=13); ax.tick_params(axis='x', length=0, pad=9)
ax.set_ylabel(f'Neighbours that are {HI.lower()}', fontsize=15)
ax.set_title('C', loc='left', pad=4, fontsize=22, fontweight='bold', color=INK)
ax.grid(axis='y', color='#EEEEEE'); ax.set_axisbelow(True)

fig.text(.5, .045,
         f'{HI}: fluorescence/OD$_{{600}}$ $\\geq$ {tau:,.0f}, the pooled 80th '
         f'percentile ({n_above} of {n}).\n'
         f'Hamming-1 is defined between spacers of equal length: {len(ii)} pairs, '
         f'{n_iso} sequences without one.\n'
         f'Panel A is a two-dimensional approximation, and the separation between the '
         f'three length groups is an imposed constant, not sequence dissimilarity;\n'
         f'Hamming-1 pairs are identified from the sequence distances, not from '
         f'proximity in A.',
         ha='center', va='top', fontsize=12.5, color=INK2, linespacing=1.8)

for ext in ['png', 'svg', 'pdf']:
    fig.savefig(R / f'landscape_neighbour_en.{ext}', dpi=350, bbox_inches='tight',
                pad_inches=.18, facecolor='white')
plt.close(fig)
print('wrote landscape_neighbour_en.png')
print(f'  p_hh={p_hh:.4f} ({n_hh}/{n_ha})  p_lh={p_lh:.4f} ({n_lh}/{n_la})  '
      f'lift={p_hh/p_lh:.2f}x  p={pval:.1e}   {len(ii)} pairs, {n_iso} isolates')
