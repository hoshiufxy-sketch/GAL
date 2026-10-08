"""Exploration strategies compared end to end.

    random    a random permutation of the pool
    AL        sigma, model uncertainty
    greedy    mu, predicted mean
    EI        expected improvement over the best value measured so far
    UCB       mu + sigma, coefficient 1, not tuned
    GAL       the unsampled same-length Hamming-1 neighbours of measured sequences
              above the campaign's frozen initial-Q80, ordered by mu, then the rest of
              the batch by mu

Two outcome metrics, because they answer slightly different questions: how many of the
library's true top-20 have been measured, and the highest value reached. Both are
offline; neither set of labels is ever visible to the policy.

120 random initial spacers then two rounds of 10, three feature sets, three learners,
20 paired campaigns per cell with seeds shared across every arm.

A note on the ordering inside the neighbourhood. Under this schedule the whole
neighbourhood fits inside a batch of 10 — about 5 candidates in round one and none
after — so ordering it by mu, by sigma or at random gives the identical campaign. The
ranker only becomes a choice when the neighbourhood is larger than the batch, and in a
separate held-out test it is mu that wins there (pairwise concordance 0.616 against 0.5).

Writes strategies_fig3.csv.
"""
import os
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_k] = '1'

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data'
B_ = ROOT.parent
sys.path.insert(0, str(ROOT))     # ensemble_learner.py, run.py and gal/ live here
from ensemble_learner import make_learner                        # noqa: E402
from run import features                                         # noqa: E402

SOURCE = DATA / 'sequence_means.csv'          # local copy, so the folder is self-contained
ARMS = ['random', 'AL', 'greedy', 'EI', 'UCB', 'guided']
FEATURES = [('kmer', list(range(0, 91))),
            ('physico', list(range(123, 128))),
            ('all', list(range(128)))]
LEARNERS = [('XGB', 'xgb_ens'), ('GPR', 'gp'), ('DNN', 'dnn_ens')]
N_INIT, BATCH, ROUNDS, TAU_Q, TOPK = 120, 10, 2, 0.80, 20
BLOCKS = [11100010, 11100020]
PER_BLOCK = 10


def hamming_matrix(spacers):
    L = max(len(s) for s in spacers)
    M = np.full((len(spacers), L), -1, int)
    for i, s in enumerate(spacers):
        M[i, :len(s)] = [ord(c) for c in s]
    return (M[:, None, :] != M[None, :, :]).sum(-1)


def main():
    warnings.filterwarnings('ignore')
    d = pd.read_csv(SOURCE)
    d.spacer = d.spacer.str.strip().str.upper().str.replace('U', 'T')
    y = d.norm.to_numpy(float)
    X_all = features(d)
    sp = d.spacer.tolist()
    H = hamming_matrix(sp)
    LN = np.array([len(x) for x in sp])
    n = len(y)
    top = set(np.argsort(-y)[:TOPK].tolist())
    print(f'n={n}  library max={y.max():,.0f}', flush=True)

    rows, t0 = [], time.monotonic()
    for fname, cols in FEATURES:
        Xf = X_all[:, cols]
        for lname, fam in LEARNERS:
            for block in BLOCKS:
                for t in range(PER_BLOCK):
                    seed = block + t
                    init = np.sort(np.random.default_rng(seed).choice(
                        n, N_INIT, replace=False))
                    tau = float(np.quantile(y[init], TAU_Q))
                    for arm in ARMS:
                        known = init.copy()
                        lr = make_learner(fam, (seed * 100 + 90) % 2147483647)
                        lr.fit(Xf[known], y[known])

                        def rec(stage):
                            rows.append(dict(
                                features=fname, learner=lname, arm=arm, block=block,
                                trial=t, stage=stage, n_measured=len(known),
                                topk=len(set(known.tolist()) & top),
                                best=float(y[known].max())))

                        rec(0)
                        for stage in range(1, ROUNDS + 1):
                            cand = np.setdiff1d(np.arange(n), known)
                            mu, sd = lr.predict(Xf[cand])
                            if arm == 'random':
                                idx = np.random.default_rng(
                                    seed * 31 + stage).permutation(len(cand))[:BATCH]
                            elif arm == 'AL':
                                idx = np.argsort(-sd)[:BATCH]
                            elif arm == 'greedy':
                                idx = np.argsort(-mu)[:BATCH]
                            elif arm == 'UCB':
                                idx = np.argsort(-(mu + sd))[:BATCH]
                            elif arm == 'EI':
                                fbest = np.log1p(y[known].max())
                                s = np.maximum(sd, 1e-9)
                                z = (mu - fbest) / s
                                ei = (mu - fbest) * norm.cdf(z) + s * norm.pdf(z)
                                idx = np.argsort(-ei)[:BATCH]
                            else:
                                hv = known[y[known] >= tau]
                                # strict: same length only. hamming_matrix right-pads,
                                # so without this mask a 3'-terminal indel would count
                                # as distance 1.
                                sl = LN[cand][:, None] == LN[hv][None, :]
                                near = (((H[np.ix_(cand, hv)] == 1) & sl).any(axis=1)
                                        if len(hv) else np.zeros(len(cand), bool))
                                m = np.flatnonzero(near)
                                m = m[np.argsort(-mu[m])]
                                l = m[:BATCH]
                                rest = [j for j in np.argsort(-mu)
                                        if j not in set(l)]
                                idx = np.concatenate(
                                    [l, np.array(rest[:BATCH - len(l)], int)])
                            known = np.sort(np.concatenate([known, cand[idx]]))
                            rec(stage)
                            lr = make_learner(
                                fam, (seed * 100 + 90 + stage) % 2147483647)
                            lr.fit(Xf[known], y[known])
            print(f'  {fname}/{lname}  [{time.monotonic()-t0:.0f}s]', flush=True)

    T = pd.DataFrame(rows)
    T.to_csv(DATA / 'strategies_fig3.csv', index=False)
    print(f'\nwrote strategies_fig3.csv  {T.shape}  {time.monotonic()-t0:.0f}s\n')
    rg = np.random.default_rng(5)
    for metric, scale in [('topk', 1), ('best', 1e6)]:
        print(f'--- {metric} at round 2 ---')
        p = T[T.stage == ROUNDS].pivot_table(index=['features', 'learner'],
                                             columns='arm', values=metric)
        print((p[ARMS] / scale).round(4).to_string())
        w = T[T.stage == ROUNDS].pivot_table(
            index=['features', 'learner', 'block', 'trial'], columns='arm',
            values=metric)
        for ctrl in ['AL', 'greedy', 'EI', 'UCB', 'random']:
            dd = (w['guided'] - w[ctrl]).to_numpy(float)
            null = (rg.choice([-1, 1], (50000, len(dd))) * dd).mean(1)
            pv = (1 + np.count_nonzero(np.abs(null) >= abs(dd.mean()) - 1e-12)) / 50001
            up = int((p['guided'] > p[ctrl]).sum())
            print(f'  GAL-{ctrl:7s} = {dd.mean()/scale:+.4g} (p={pv:.5f})  '
                  f'GAL higher in {up}/9')
        print()


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
