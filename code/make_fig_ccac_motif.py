"""The boundary of the CCAC high-value region.

Four panels, English, emphasis colour (one accent against greys).

    A  where the measured sequences sit. The library as a grey backdrop on the
       expression axis, the CCAC[AT][AT] class in the accent, the rest of the CCAC
       family in dark grey, against the analysis threshold.
    B  the two bases after CCAC, as a grid: measured high-value rate per pair.
    C  where CCAC sits in the spacer, as a rate per position.
    D  the model's own view of the whole 336-member 5'-CCAC sub-space, ranked, so
       the motif it converges on without being told is visible.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data'
LIBP = DATA / 'sequence_means.csv'          # local copy, so the folder is self-contained
DESP = DATA / 'ccac28_extracted.csv'

SURFACE, INK, INK2 = '#ffffff', '#0b0b0b', '#52514e'
ACCENT, ACCENT_D = '#eb6834', '#b8431c'
GREY_1, GREY_2, GREY_3 = '#8a8984', '#c9c8c2', '#e6e5e0'
BLUE_RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95']
BASES = list('ACGT')


def main():
    sys.path.insert(0, str(ROOT))   # ensemble_learner.py, run.py and gal/ live here
    from ensemble_learner import BootstrapLearner, XGB_PARAMS
    from run import features
    from xgboost import XGBRegressor
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.unicode_minus': False, 'font.size': 9.5, 'axes.titlesize': 10.5,
        'axes.labelsize': 9.5, 'axes.linewidth': .6,
        'savefig.dpi': 400, 'figure.dpi': 150,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'axes.edgecolor': GREY_2, 'axes.labelcolor': INK2,
        'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'xtick.labelsize': 8.6, 'ytick.labelsize': 8.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': GREY_3, 'grid.alpha': .9, 'grid.linewidth': .5,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 2.4, 'ytick.major.size': 2.4,
    })
    cmap = LinearSegmentedColormap.from_list('b', BLUE_RAMP)

    lib = pd.read_csv(LIBP)
    lib['spacer'] = lib.spacer.str.strip().str.upper()
    y = lib.norm.to_numpy(float)
    Q80 = float(np.quantile(y, .80))
    des = pd.read_csv(DESP)
    des['spacer'] = des.spacer.str.upper()
    r1 = lib[lib.spacer.str.contains('CCAC')][['spacer', 'norm']].rename(
        columns={'norm': 'val'})
    r2 = des[des.spacer.str.contains('CCAC')][['spacer', 'mean']].rename(
        columns={'mean': 'val'})
    M = pd.concat([r1, r2]).groupby('spacer', as_index=False).val.mean()
    M['high'] = M.val >= Q80
    M['pos'] = M.spacer.str.find('CCAC')
    M6 = M[M.spacer.str.startswith('CCAC') & (M.spacer.str.len() >= 6)].copy()
    M6['b5'], M6['b6'] = M6.spacer.str[4], M6.spacer.str[5]
    M6['cls'] = np.where(M6.b5.isin(list('AT')) & M6.b6.isin(list('AT')),
                         'CCAC[AT][AT]', 'other CCAC')

    # The position panel C is duplicated by the standalone position charts
    # (fig_ccac_position_bars_unified uses the same pooling and the same five bins),
    # so it can be dropped here without losing anything.
    DROP_C = True
    STEM = 'fig_ccac_motif_3panel' if DROP_C else 'fig_ccac_motif'

    fig = plt.figure(figsize=(12.0, 4.5) if DROP_C else (15.2, 4.5),
                     layout='constrained')
    gs = (fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.15], wspace=.30)
          if DROP_C else
          fig.add_gridspec(1, 4, width_ratios=[1.35, 1.0, .78, 1.15], wspace=.30))

    # ---------- A distribution ----------
    ax = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(0)
    jit = lambda k: (rng.random(k) - .5) * .30
    ax.scatter(y / 1e6, 2 + jit(len(y)), s=5, color=GREY_3, alpha=.9,
               edgecolors='none', zorder=1)
    ax.annotate('library\n(n = 421)', (0.16, 2.0), fontsize=8.2, color=GREY_1,
                va='center', ha='left', linespacing=1.4, zorder=6)
    for cls, yy, col, mk in [('other CCAC', 1, GREY_1, 'o'),
                             ('CCAC[AT][AT]', 0, ACCENT, 'o')]:
        s = M6[M6.cls == cls]
        ax.scatter(s.val / 1e6, yy + jit(len(s)), s=46, color=col, marker=mk,
                   edgecolors=SURFACE, linewidths=.9, zorder=5)
        ax.annotate(f'{cls}\n{int(s.high.sum())}/{len(s)} above',
                    (2.05, yy), fontsize=8.2, color=col, va='center', ha='left',
                    linespacing=1.4, zorder=6)
    ax.axvline(Q80 / 1e6, color=INK, lw=1.3, ls=(0, (4, 2.5)), zorder=4)
    ax.annotate('threshold', (Q80 / 1e6, 2.62), xytext=(4, 0),
                textcoords='offset points', fontsize=8.2, color=INK)
    ax.set(xlabel='Measured normalised fluorescence (a.u., millions)',
           ylim=(2.75, -.75), xlim=(0, 2.95), yticks=[])
    ax.grid(axis='x')

    # ---------- B grid ----------
    ax = fig.add_subplot(gs[0, 1])
    G = np.full((4, 4), np.nan)
    N = np.zeros((4, 4), int)
    for i, b5 in enumerate(BASES):
        for j, b6 in enumerate(BASES):
            s = M6[(M6.b5 == b5) & (M6.b6 == b6)]
            if len(s):
                G[i, j] = s.high.mean()
                N[i, j] = len(s)
    im = ax.imshow(G, cmap=cmap, vmin=0, vmax=1, origin='upper')
    for i in range(4):
        for j in range(4):
            if N[i, j]:
                ax.text(j, i, f'{G[i,j]:.0%}\n{N[i,j]}', ha='center', va='center',
                        fontsize=9, color=SURFACE if G[i, j] > .55 else INK,
                        linespacing=1.25)
            else:
                ax.text(j, i, '·', ha='center', va='center', fontsize=10,
                        color=GREY_2)
    ax.set(xticks=range(4), xticklabels=BASES, yticks=range(4), yticklabels=BASES,
           xlabel='base at position 6', ylabel='base at position 5')
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2.2)
    ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=ax, shrink=.74, pad=.04)
    cb.set_label('measured high-value rate', fontsize=8)
    cb.ax.tick_params(labelsize=7.4)
    cb.outline.set_visible(False)

    # ---------- C position ----------
    if not DROP_C:
        ax = fig.add_subplot(gs[0, 2])
        P = M.groupby('pos').agg(n=('high', 'size'), rate=('high', 'mean')).reset_index()
        P = P[P.n >= 2]
        yy = np.arange(len(P))[::-1]
        for i, r in P.reset_index().iterrows():
            yb = yy[i]
            c = ACCENT if r['pos'] == 0 else GREY_1
            ax.plot([0, r.rate], [yb, yb], color=c, lw=2.2, solid_capstyle='round',
                    zorder=2)
            ax.scatter([r.rate], [yb], s=64, color=c, zorder=4, edgecolors=SURFACE,
                       linewidths=1.0)
            ax.annotate(f"{r.rate:.0%}  (n={int(r['n'])})", (r.rate, yb),
                        xytext=(8, 0), textcoords='offset points', va='center',
                        fontsize=8.2, color=ACCENT_D if r['pos'] == 0 else INK2)
        ax.axvline(0.20, color=INK, lw=1.1, ls=(0, (1, 2)), zorder=1)
        ax.annotate('chance', (0.20, -.62), xytext=(3, 0), textcoords='offset points',
                    fontsize=7.8, color=INK2)
        ax.set(yticks=yy, yticklabels=[f"position {int(p)+1}" for p in P['pos']],
               xlabel='High-value rate', xlim=(0, 1.22), ylim=(-1.0, len(P) - .35))
        ax.tick_params(axis='y', labelsize=8.4)
        ax.grid(axis='x')

    # ---------- D model over the sub-space ----------
    ax = fig.add_subplot(gs[0, 2 if DROP_C else 3])
    X = features(lib)
    lr = BootstrapLearner(
        lambda s: XGBRegressor(random_state=s, n_jobs=1, verbosity=0, **XGB_PARAMS),
        999)
    lr.fit(X, y)
    sub = ['CCAC' + ''.join(t) for L in (6, 7, 8)
           for t in itertools.product('ACGT', repeat=L - 4)]
    S = pd.DataFrame({'spacer': sub})
    mu, sd = lr.predict(features(S))
    S['mu'] = mu
    S['cls'] = np.where(S.spacer.str[4].isin(list('AT'))
                        & S.spacer.str[5].isin(list('AT')),
                        'CCAC[AT][AT]', 'other CCAC')
    for cls, col, z in [('other CCAC', GREY_1, 2), ('CCAC[AT][AT]', ACCENT, 3)]:
        s = S[S.cls == cls]
        ax.scatter(s.mu, (rng.random(len(s)) - .5) * .5 + (0 if cls.startswith('CCAC[')
                                                           else 1),
                   s=13, color=col, alpha=.6, edgecolors='none', zorder=z)
    ax.annotate('CCAC[AT][AT]\n(n = 168)', (13.10, 0), fontsize=8.2, color=ACCENT_D,
                va='center', ha='left', linespacing=1.4)
    ax.annotate('other CCAC\n(n = 168)', (13.10, 1), fontsize=8.2, color=GREY_1,
                va='center', ha='left', linespacing=1.4)
    top = S.sort_values('mu', ascending=False).head(12)
    cut = float(top.mu.min())
    ax.axvspan(cut, 14.20, color=ACCENT, alpha=.10, lw=0, zorder=1)
    ax.axvline(cut, color=ACCENT_D, lw=1.0, ls=(0, (4, 2.5)), zorder=2)
    ax.annotate('top 12 by $\\mu$:\nall CCAC[AT][AT]', (cut, 1.46), xytext=(5, 0),
                textcoords='offset points', fontsize=8.0, color=ACCENT_D,
                linespacing=1.4, va='top')
    ax.annotate('no member of the\nother class reaches here', (cut, -1.06),
                xytext=(5, 0), textcoords='offset points', fontsize=7.8,
                color=GREY_1, linespacing=1.4, va='center')
    ax.set(xlabel='Predicted mean  $\\mu$', xlim=(13.05, 14.20), yticks=[],
           ylim=(-1.30, 1.62))
    ax.grid(axis='x')

    fig.savefig(ROOT / f'{STEM}.png', facecolor=SURFACE)
    print(f'wrote {STEM}.png')
    print(f'  CCAC[AT][AT]: {int(M6[M6.cls=="CCAC[AT][AT]"].high.sum())}/'
          f'{len(M6[M6.cls=="CCAC[AT][AT]"])} high')
    print(f'  other CCAC:   {int(M6[M6.cls=="other CCAC"].high.sum())}/'
          f'{len(M6[M6.cls=="other CCAC"])} high')
    print(f'  model top-12 all [AT][AT]: {bool((top.cls=="CCAC[AT][AT]").all())}')


if __name__ == '__main__':
    main()
