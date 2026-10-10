"""Step 3 - Candidate models, calibration and forward-chaining model selection.

Design choices (justified in the report and model card):
* Baseline = draft position alone. It is what the market already 'knows', so the model must beat it.
* Logistic regression = transparent linear benchmark.
* Random forest and gradient boosting = tree ensembles that capture non-linear, interacting signals
  (e.g. a young, low-minute player with elite per-possession production) and work with exact SHAP.
* Calibration: Platt (sigmoid) or isotonic mapping fitted on out-of-fold training predictions, with
  a single base model kept for explanation (CalibratedClassifierCV with ensemble=False). Calibrated
  probabilities let the decision threshold come straight from the agreed cost ratio: 1 / (1 + ratio).
* Selection: every choice is made on the 2011-14 validation classes. The 2015-18 test classes are
  scored once, at the end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_FEATURES = ["draft_pick", "undrafted"]


def candidate_models(seed: int = 42) -> dict:
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    return {
        "Logistic regression": Pipeline([("impute", imp()), ("scale", StandardScaler()),
                                         ("model", LogisticRegression(C=0.3, max_iter=5000))]),
        "Random forest": Pipeline([("impute", imp()),
                                   ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                                                    max_features="sqrt", random_state=seed, n_jobs=-1))]),
        "Gradient boosting": Pipeline([("impute", imp()),
                                       ("model", GradientBoostingClassifier(n_estimators=250, learning_rate=0.03,
                                                                            max_depth=3, subsample=0.8, random_state=seed))]),
    }


def baseline_model() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])


def calibrated(model: Pipeline, method: str | None, seed: int = 42):
    if method is None or method == "none":
        return clone(model)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(clone(model), method=method, cv=cv, ensemble=False)


def base_pipeline(fitted):
    """Return the fitted (uncalibrated) pipeline inside a calibrated wrapper, for SHAP."""
    if isinstance(fitted, CalibratedClassifierCV):
        return fitted.calibrated_classifiers_[0].estimator
    return fitted


def select_model(X: pd.DataFrame, y: pd.Series, masks: dict, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """Fit every model x calibration on train, score on validation; pick the lowest validation Brier."""
    rows = []
    tr, va = masks["train"], masks["validate"]
    for name, mdl in candidate_models(seed).items():
        for method in ["none", "sigmoid", "isotonic"]:
            m = calibrated(mdl, method, seed).fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
            rows.append(dict(model=name, calibration=method, val_auc=roc_auc_score(y[va], p),
                             val_brier=brier_score_loss(y[va], p),
                             val_cal_gap=float(p.mean() - y[va].mean())))
    table = pd.DataFrame(rows).sort_values(["val_brier", "val_auc"], ascending=[True, False]).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    return best, table


def fit_final(X: pd.DataFrame, y: pd.Series, mask: pd.Series, choice: dict, seed: int = 42):
    mdl = candidate_models(seed)[choice["model"]]
    return calibrated(mdl, choice["calibration"], seed).fit(X[mask], y[mask])


def fit_baseline(X: pd.DataFrame, y: pd.Series, mask: pd.Series):
    return baseline_model().fit(X.loc[mask, BASELINE_FEATURES], y[mask])


def bootstrap_ensemble(X: pd.DataFrame, y: pd.Series, mask: pd.Series, choice: dict, n: int, seed: int = 42) -> list:
    """Refit the chosen model on bootstrap resamples of the training classes (player uncertainty ranges)."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask.values)
    out = []
    for b in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        Xi, yi = X.iloc[s], y.iloc[s]
        mdl = candidate_models(seed + b)[choice["model"]]
        m = calibrated(mdl, choice["calibration"], seed + b).fit(Xi, yi)
        out.append(m)
    return out


def ensemble_interval(models: list, X: pd.DataFrame, p: np.ndarray, lo: float = 0.10, hi: float = 0.90) -> pd.DataFrame:
    """80% uncertainty range around the champion's score p.

    The spread of the bootstrap refits (each quantile minus their median) is added to p, so the range
    always contains the score that is reported and is as wide as the model's sensitivity to the data.
    """
    P = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    med = np.median(P, axis=1)
    return pd.DataFrame({"p_low": np.clip(p + np.quantile(P, lo, axis=1) - med, 0, 1),
                         "p_high": np.clip(p + np.quantile(P, hi, axis=1) - med, 0, 1)}, index=X.index)
