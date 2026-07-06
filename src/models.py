"""Model factories for Task A (W->l nu signal vs background classification).

Two complementary baselines are provided:

* :func:`train_xgb`     -- gradient-boosted decision trees (XGBoost), the workhorse
  for tabular HEP problems. Handles non-linear feature interactions out of the box.
* :func:`train_logreg`  -- a StandardScaler + LogisticRegression pipeline. A simple,
  interpretable linear baseline; comparing it against the BDT quantifies how much the
  ML model buys us over a linear cut-based approach.

Both are trained with per-event ``sample_weight`` so that the physics event weights
(cross-section * lumi / sum-of-weights * scale factors, see :mod:`src.weights`) are
respected during the fit -- exactly as they are in the downstream weighted evaluation.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

__all__ = ["train_xgb", "train_logreg"]


# Sensible defaults for a ~few-thousand-event tabular HEP classification. Shallow
# trees + many estimators + a low learning rate is the standard robust starting point.
_XGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}


def train_xgb(
    X_train,
    y_train,
    w_train,
    params: Optional[dict[str, Any]] = None,
) -> XGBClassifier:
    """Train a weighted XGBoost binary classifier for signal vs background.

    Parameters
    ----------
    X_train : array-like or pandas.DataFrame, shape (n_samples, n_features)
        Feature matrix (the columns produced by :func:`src.features.build_features`).
    y_train : array-like, shape (n_samples,)
        Binary labels (1 = W->l nu signal, 0 = background).
    w_train : array-like, shape (n_samples,)
        Per-event physics weights, passed through as ``sample_weight``.
    params : dict, optional
        Overrides/extends the default XGBoost hyper-parameters. Keys given here take
        precedence over the defaults (e.g. ``{"max_depth": 5, "n_estimators": 500}``).

    Returns
    -------
    xgboost.XGBClassifier
        The fitted classifier. Call ``.predict_proba(X)[:, 1]`` for signal scores.
    """
    hyper = dict(_XGB_DEFAULTS)
    if params:
        hyper.update(params)

    clf = XGBClassifier(**hyper)
    clf.fit(X_train, y_train, sample_weight=np.asarray(w_train, dtype=float))
    return clf


def train_logreg(X_train, y_train, w_train) -> Pipeline:
    """Train a weighted logistic-regression baseline (scaled features).

    Features are standardized first because logistic regression is sensitive to their
    scale (our inputs mix GeV-scale pt/met with O(1) isolation fractions and jet counts).

    Parameters
    ----------
    X_train : array-like or pandas.DataFrame, shape (n_samples, n_features)
        Feature matrix.
    y_train : array-like, shape (n_samples,)
        Binary labels (1 = signal, 0 = background).
    w_train : array-like, shape (n_samples,)
        Per-event physics weights, routed to the logistic-regression step as
        ``sample_weight``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted ``Pipeline(StandardScaler, LogisticRegression)``. The classifier step is
        named ``"logisticregression"`` (via :func:`make_pipeline`), so the fit routes the
        weights through ``logisticregression__sample_weight``.
    """
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    )
    pipe.fit(
        X_train,
        y_train,
        logisticregression__sample_weight=np.asarray(w_train, dtype=float),
    )
    return pipe
