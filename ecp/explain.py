"""Step 5 - Explanations: why each player scored as he did, and which past players looked similar.

* Champion (logistic regression): exact additive contributions on the log-odds scale,
  coefficient x (standardised value - training average). For a linear model these are its SHAP values.
* Challenger (gradient boosting): TreeSHAP (Lundberg et al. 2020) for a model-agnostic cross-check of
  which signals matter.
* Comparables: the five most similar historical players at the same career stage, with their outcomes,
  so scouts can test the score against cases they know.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .features import PLAIN_NAMES
from .models import base_pipeline

COMPARABLE_FEATURES = ["best_rate_pct", "last_rate_pct", "best_gmsc_pct", "mpg", "games", "gmsc_100",
                       "ts_vs_league", "age_debut", "height", "draft_pick"]


def linear_contributions(fitted, X: pd.DataFrame, X_ref: pd.DataFrame) -> pd.DataFrame:
    """Per-player log-odds contributions for a (calibrated) logistic pipeline, grouped by input feature."""
    pipe = base_pipeline(fitted)
    pre = pipe[:-1]
    coef = pipe[-1].coef_.ravel()
    names = pre.get_feature_names_out() if hasattr(pre, "get_feature_names_out") else pipe[0].get_feature_names_out()
    Z = pre.transform(X)
    Zref = pre.transform(X_ref).mean(axis=0)
    C = pd.DataFrame((Z - Zref) * coef, index=X.index, columns=names)
    base = [n.replace("missingindicator_", "") for n in names]
    return C.T.groupby(base).sum().T[X.columns]


def tree_shap(fitted, X: pd.DataFrame) -> pd.DataFrame:
    import shap

    pipe = base_pipeline(fitted)
    Z = pipe[:-1].transform(X)
    names = pipe[:-1].get_feature_names_out()
    sv = shap.TreeExplainer(pipe[-1]).shap_values(Z)
    sv = sv[1] if isinstance(sv, list) else sv
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    S = pd.DataFrame(sv, index=X.index, columns=names)
    base = [n.replace("missingindicator_", "") for n in names]
    return S.T.groupby(base).sum().T[X.columns]


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def format_value(f: str, v) -> str:
    if pd.isna(v):
        return "not enough minutes"
    if f.endswith("_pct"):
        return f"{ordinal(int(round(v * 100)))} percentile"
    if f in ("gmsc_pct_trend",):
        return f"{v * 100:+.0f} percentile points"
    if f in ("ts_vs_league", "three_rate_vs_league", "ft_rate_vs_league"):
        return f"{v * 100:+.1f} pts vs league"
    if f.endswith("_100"):
        return f"{v:+.1f} vs league"
    if f == "mpg_trend":
        return f"{v:+.1f} min"
    if f in ("draft_pick",):
        return "undrafted" if v >= 61 else f"pick {int(v)}"
    if f in ("undrafted", "is_guard", "is_forward", "is_center"):
        return "yes" if v >= 0.5 else "no"
    if f == "height":
        return f"{int(v // 12)}'{int(v % 12)}\""
    if f in ("games", "minutes", "weight", "seasons_played"):
        return f"{v:,.0f}"
    if f == "age_debut":
        return f"{v:.1f} years"
    return f"{v:.1f}"


BEST_OF_LATEST = {"last_gmsc_pct": "best_gmsc_pct", "last_rate_pct": "best_rate_pct"}


def merge_equal_ranks(c: pd.Series, x: pd.Series) -> pd.Series:
    """After one season (or a flat career) best and latest season rank are the same number: show it once."""
    for last, best in BEST_OF_LATEST.items():
        if last in c and best in c and x[last] == x[best]:
            c = c.drop(labels=[last]).add(pd.Series({best: c[last]}), fill_value=0)
    return c


def top_drivers(contrib: pd.DataFrame, X: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Plain-language top-k features pushing each player's score up and down."""
    rows = []
    for pid, c in contrib.iterrows():
        c = c.drop(labels=["undrafted"], errors="ignore")  # draft information is shown once, via draft_pick
        c = merge_equal_ranks(c, X.loc[pid])
        up = c[c > 0].sort_values(ascending=False, kind="stable").head(k)
        dn = c[c < 0].sort_values(kind="stable").head(k)
        fmt = lambda s: [dict(feature=f, label=PLAIN_NAMES.get(f, f), value=format_value(f, X.at[pid, f]),
                              effect=round(float(v), 3)) for f, v in s.items()]
        rows.append(dict(personId=pid, drivers_up=fmt(up), drivers_down=fmt(dn)))
    return pd.DataFrame(rows).set_index("personId")


def global_importance(contrib: pd.DataFrame) -> pd.Series:
    return contrib.abs().mean().sort_values(ascending=False)


def comparables(X_query: pd.DataFrame, X_ref: pd.DataFrame, ref_info: pd.DataFrame, k: int = 5) -> dict:
    """k nearest historical players (standardised early-career profile) with their outcomes."""
    cols = COMPARABLE_FEATURES
    imp = SimpleImputer(strategy="median").fit(X_ref[cols])
    sc = StandardScaler().fit(imp.transform(X_ref[cols]))
    R = sc.transform(imp.transform(X_ref[cols]))
    Q = sc.transform(imp.transform(X_query[cols]))
    nn = NearestNeighbors(n_neighbors=k).fit(R)
    _, idx = nn.kneighbors(Q)
    out = {}
    for pid, row in zip(X_query.index, idx):
        refs = ref_info.iloc[row]
        out[pid] = [dict(name=r["name"], debut=int(r["debut"]), contributor=int(r["y"]),
                         high_impact_seasons=int(r["hi_seasons"])) for _, r in refs.iterrows()]
    return out
