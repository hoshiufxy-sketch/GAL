"""Single definition of every learner used in the active-learning experiments.

Every script must import mu/sigma from here. Duplicating the ensemble logic per
script caused the sigma used by the AL arm to mean different things in different
files (spread across the trees of one forest, versus spread across independently
resampled models), which the protocol described only one of.

Contract for all learners:

    fit(X_known, y_known) -> None
        X_known, y_known are the spacers measured SO FAR. No unmeasured label
        ever reaches this call.

    predict(X_pool) -> (mu, sigma)
        both in the learner's own target scale; ranking is invariant to the
        monotone target transform, and the acquisition functions receive
        quantities from the same call so their scales always agree.

Target scale: log1p for every learner except Ridge on raw, which is retained
because it is the historical comparison and is explicitly labelled.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

N_MEMBERS = 16
N_PCA = 20
N_TREES = 48
# tuned on the development initialisation block, optimising the deployed ensemble
XGB_PARAMS = dict(n_estimators=100, learning_rate=0.1, max_depth=3,
                  min_child_weight=1, subsample=0.6, colsample_bytree=0.3,
                  reg_lambda=0.1, reg_alpha=1.0)


class BootstrapLearner:
    """Bagged learner: members resample the measured set, spread gives sigma."""

    def __init__(self, make_member, seed, n_members=N_MEMBERS, use_log=True):
        self.make_member = make_member
        self.seed = seed
        self.n_members = n_members
        self.use_log = use_log

    def fit(self, X_known, y_known, sample_weight=None):
        # sample_weight is None for uniform training, which is what every arm did
        # until the denoising experiments. Weights are resampled with the same
        # bootstrap indices so the member spread keeps its meaning.
        self.target_ = np.log1p(y_known) if self.use_log else y_known.copy()
        w = None if sample_weight is None else np.asarray(sample_weight, float)
        rng = np.random.default_rng(self.seed)
        self.models_ = []
        for m in range(self.n_members):
            b = rng.integers(0, len(self.target_), len(self.target_))
            model = self.make_member(self.seed + m)
            if w is None:
                model.fit(X_known[b], self.target_[b])
            else:
                model.fit(X_known[b], self.target_[b], sample_weight=w[b])
            self.models_.append(model)
        return self

    def predict(self, X_pool):
        preds = np.asarray([m.predict(X_pool) for m in self.models_])
        return preds.mean(axis=0), preds.std(axis=0, ddof=1)


class GaussianProcessLearner:
    """Posterior mean and posterior standard deviation; input PCA inside."""

    def __init__(self, seed, n_pca=N_PCA):
        self.seed = seed
        self.n_pca = n_pca

    def fit(self, X_known, y_known, sample_weight=None):
        # GaussianProcessRegressor.fit takes no sample_weight, so weighting enters
        # through one weighted resample. The unweighted path is untouched.
        if sample_weight is None:
            Xf, yf = X_known, np.log1p(y_known)
        else:
            w = np.asarray(sample_weight, float)
            b = np.random.default_rng(self.seed).choice(
                len(w), len(w), replace=True, p=w / w.sum())
            Xf, yf = X_known[b], np.log1p(y_known)[b]
        self.pipe_ = Pipeline([
            ('scale', StandardScaler()),
            ('pca', PCA(n_components=min(self.n_pca, Xf.shape[1]), svd_solver='full')),
            ('gp', GaussianProcessRegressor(
                kernel=Matern(length_scale=10, nu=2.5) + WhiteKernel(0.3),
                optimizer=None, normalize_y=True))])
        self.pipe_.fit(Xf, yf)
        return self

    def predict(self, X_pool):
        mu, sd = self.pipe_.predict(X_pool, return_std=True)
        return mu, sd


class DNNExtendedLearner:
    """Networks on scaled inputs and log1p targets, same member count as the rest."""

    def __init__(self, seed, n_members=N_MEMBERS, n_pca=N_PCA):
        self.seed = seed
        self.n_members = n_members
        self.n_pca = n_pca

    def fit(self, X_known, y_known, sample_weight=None):
        target = np.log1p(y_known)
        rng = np.random.default_rng(self.seed)
        # Unweighted members fit on everything, as they always have. MLPRegressor
        # takes no sample_weight, so the weighted path enters through the resample.
        w = None if sample_weight is None else np.asarray(sample_weight, float)
        p = None if w is None else w / w.sum()
        self.models_ = []
        for m in range(self.n_members):
            net = Pipeline([
                ('scale', StandardScaler()),
                ('pca', PCA(n_components=min(self.n_pca, X_known.shape[1]),
                            svd_solver='full')),
                ('model', TransformedTargetRegressor(
                    regressor=MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1.,
                                           early_stopping=True, validation_fraction=.15,
                                           max_iter=800, random_state=self.seed + m),
                    transformer=StandardScaler()))])
            if p is None:
                net.fit(X_known, target)
            else:
                b = rng.choice(len(target), len(target), replace=True, p=p)
                net.fit(X_known[b], target[b])
            self.models_.append(net)
        return self

    def predict(self, X_pool):
        preds = np.asarray([m.predict(X_pool) for m in self.models_])
        return preds.mean(axis=0), preds.std(axis=0, ddof=1)


def make_learner(family, seed):
    """The single place that maps a family name to a learner."""
    if family == 'xgb_ens':
        return BootstrapLearner(
            lambda s: XGBRegressor(random_state=s, n_jobs=1, verbosity=0, **XGB_PARAMS),
            seed)
    if family == 'rf':
        return BootstrapLearner(
            lambda s: RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=4,
                                            min_samples_split=2, max_features=0.7,
                                            random_state=s, n_jobs=1),
            seed)
    if family == 'gp':
        return GaussianProcessLearner(seed)
    if family == 'dnn_ens':
        return DNNExtendedLearner(seed)
    if family == 'ridge_raw':
        return BootstrapLearner(lambda s: Ridge(alpha=300.), seed, use_log=False)
    if family == 'ridge_log':
        return BootstrapLearner(lambda s: Ridge(alpha=300.), seed, use_log=True)
    raise ValueError(f'unknown family: {family}')


FAMILIES = ['xgb_ens', 'gp', 'dnn_ens', 'rf']
