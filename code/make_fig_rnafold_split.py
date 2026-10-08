"""RNAfold profiles, one figure per spacer.

The same content as the combined three-column figure, split so each construct gets its
own file:

    fig_rnafold_wt.png        spacer TATACC
    fig_rnafold_ccacaga.png   spacer CCACAGA
    fig_rnafold_ccacaag.png   spacer CCACAAG

Upper panel: mountain plot - the height at a position is the number of base pairs that
enclose it, drawn for the minimum-free-energy structure, the centroid structure and the
partition-function ensemble. Lower panel: positional entropy.

Construct as entered on the RNAfold web server: the 75-nt T7-leader template ending at
the Shine-Dalgarno core, then the spacer, no downstream coding sequence.

PNG only.
"""
from pathlib import Path

import numpy as np
import RNA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parent
TPL = ('GACTCACTATAGGGGAATTGTGAGCGGATAACAATTCCCCTCTAGAAATAATTTTGTTTAACTTTAAGAAGGAGA')
SD = 'AAGAAGGAGA'

FS = 14
INK, INK2 = '#0b0b0b', '#52514e'
RED, BLUE, GREEN, ORANGE = '#BC4936', '#2a78d6', '#1baf7a', '#eb6834'
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': FS,
                     'axes.titlesize': FS + 2, 'axes.labelsize': FS,
                     'savefig.dpi': 400, 'figure.dpi': 130,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.edgecolor': '#dcdad4', 'axes.labelcolor': INK2,
                     'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'xtick.labelsize': FS - 1, 'ytick.labelsize': FS - 1,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'grid.color': '#eceae5', 'grid.linewidth': 1.0})


def mountain(ss):
    m = np.zeros(len(ss))
    st = []
    for i, ch in enumerate(ss):
        if ch == '(':
            st.append(i)
        elif ch == ')':
            if st:
                a = st.pop()
                m[a:i + 1] += 1
    return m


def analyse(spacer):
    s = (TPL + spacer).replace('T', 'U')
    fc = RNA.fold_compound(s)
    ss, mfe = fc.mfe()
    fc.pf()
    ed = fc.mean_bp_distance()
    cs, cd = fc.centroid()
    bppm = np.array(fc.bpp())
    ens = np.zeros(len(s))
    for a in range(1, len(s) + 1):
        for b in range(a + 1, len(s) + 1):
            p = bppm[a, b]
            if p > 0:
                ens[a - 1:b] += p
    pl = fc.plist_from_probs(0.0001)
    pp = np.zeros(len(s))
    for ep in pl:
        pp[ep.i - 1] += ep.p
        pp[ep.j - 1] += ep.p
    ent = -(pp * np.log(np.clip(pp, 1e-9, 1))
            + (1 - pp) * np.log(np.clip(1 - pp, 1e-9, 1)))
    sd0 = s.find(SD)
    return dict(s=s, ss=ss, mfe=mfe, ed=ed, mount=mountain(ss), mount_c=mountain(cs),
                ens=ens, ent=ent, sd0=sd0, sd1=sd0 + len(SD),
                unpaired=float(1 - pp[sd0:sd0 + len(SD)].mean()))


SPACERS = [('Wild type', 'TATACC', 'wt'), ('CCACAAA', 'CCACAAA', 'ccacaaa'),
           ('CCACAGA', 'CCACAGA', 'ccacaga'), ('CCACAAG', 'CCACAAG', 'ccacaag')]

for nm, sp, tag in SPACERS:
    r = analyse(sp)
    n = len(r['s'])
    x = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 8.4), sharex=True,
                             layout='constrained')
    for row, ax in enumerate(axes):
        ax.axvspan(r['sd0'] - .5, r['sd1'] - .5, color=GREEN, alpha=.13, lw=0,
                   zorder=0)
        ax.set_xlim(-1, n + 1)
        ax.grid(axis='y'); ax.set_axisbelow(True)
        if row == 1:
            ax.annotate('SD core', ((r['sd0'] + r['sd1']) / 2 - .5, .96),
                        xycoords=('data', 'axes fraction'), ha='center', va='top',
                        fontsize=FS - 3, color='#0f7a55', fontweight='bold')

    ax = axes[0]
    ax.plot(x, r['ens'], color=GREEN, lw=2.2, zorder=3)
    ax.plot(x, r['mount_c'], color=BLUE, lw=1.8, ls=(0, (5, 2.4)), zorder=4)
    ax.plot(x, r['mount'], color=RED, lw=2.0, zorder=5)
    ax.set_ylim(-.5, max(r['mount'].max(), r['ens'].max()) + 9.0)
    ax.set_ylabel('Base-pair height')
    ax.annotate(f'spacer {sp}   ·   MFE = {r["mfe"]:.2f} kcal/mol\n'
                f'ensemble diversity = {r["ed"]:.1f}   ·   '
                f'SD mean unpaired = {r["unpaired"]:.3f}',
                (.008, .995), xycoords='axes fraction', va='top', fontsize=FS - 3,
                color=INK2, linespacing=1.6,
                bbox=dict(fc='white', ec='none', pad=2.5))

    ax = axes[1]
    ax.fill_between(x, r['ent'], color=ORANGE, alpha=.35, lw=0, zorder=2)
    ax.plot(x, r['ent'], color='#b8431c', lw=1.6, zorder=3)
    ax.set_ylim(0, .78)
    ax.set_ylabel('Positional entropy')
    ax.set_xlabel('Position in the transcript (nt)')

    axes[0].set_title(f'{nm}', loc='left', color=INK, pad=10)
    axes[0].legend(handles=[Line2D([], [], color=RED, lw=2.0, label='MFE'),
                            Line2D([], [], color=GREEN, lw=2.2,
                                   label='partition-function ensemble'),
                            Line2D([], [], color=BLUE, lw=1.8, ls=(0, (5, 2.4)),
                                   label='centroid')],
                    frameon=False, fontsize=FS - 4, loc='upper left',
                    bbox_to_anchor=(-.01, .855), handletextpad=.6)
    fig.text(.5, -.02,
             'ViennaRNA 2.7.2, default parameters, on the\n'
             'construct entered on the RNAfold web server: the\n'
             '75-nt T7-leader template ending at the\n'
             'Shine-Dalgarno core, then the spacer, with no\n'
             'downstream coding sequence. The green band is\n'
             'the SD core.',
             ha='center', va='top', fontsize=FS - 4, color=INK2, linespacing=1.7)
    out = R / f'fig_rnafold_{tag}.png'
    fig.savefig(out, bbox_inches='tight', pad_inches=.3)
    plt.close(fig)
    print(f'  {nm:10s} -> {out.name}   MFE={r["mfe"]:6.2f}  '
          f'div={r["ed"]:5.2f}  SD unpaired={r["unpaired"]:.3f}')
