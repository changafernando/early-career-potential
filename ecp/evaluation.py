"""Step 4 - Evaluation: technical metrics with uncertainty, and the business translation.

Technical: ROC-AUC, PR-AUC, Brier score and calibration, each with a 95% bootstrap interval.
Business:  shortlist hit rate, missed contributors, cost-based thresholds, decision-curve net benefit.
Fairness:  error and calibration by position, draft tier, origin and height band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(p) * abs(p[c].mean() - y[c].mean()) for c in chunks if len(c)))


def point_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                ece=ece(y, p), mean_pred=float(p.mean()), observed=float(y.mean()))


def bootstrap_metrics(y, p, p_ref=None, n: int = 2000, seed: int = 42) -> dict:
    """Point estimate and 95% interval for each metric; optional paired AUC difference vs a reference."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    est = point_metrics(y, p)
    draws = {k: [] for k in ["auc", "pr_auc", "brier", "ece"]}
    diff = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        draws["auc"].append(roc_auc_score(y[i], p[i]))
        draws["pr_auc"].append(average_precision_score(y[i], p[i]))
        draws["brier"].append(brier_score_loss(y[i], p[i]))
        draws["ece"].append(ece(y[i], p[i]))
        if p_ref is not None:
            diff.append(draws["auc"][-1] - roc_auc_score(y[i], np.asarray(p_ref)[i]))
    out = {k: dict(est=round(est[k], 6), lo=round(float(np.percentile(v, 2.5)), 6), hi=round(float(np.percentile(v, 97.5)), 6))
           for k, v in draws.items()}
    out["mean_pred"], out["observed"], out["n"], out["positives"] = round(est["mean_pred"], 6), round(est["observed"], 6), int(len(y)), int(y.sum())
    if p_ref is not None:
        out["auc_gain_vs_reference"] = dict(est=round(est["auc"] - roc_auc_score(y, p_ref), 6),
                                            lo=round(float(np.percentile(diff, 2.5)), 6), hi=round(float(np.percentile(diff, 97.5)), 6),
                                            share_of_resamples_positive=round(float(np.mean(np.array(diff) > 0)), 6))
    return out


def calibration_table(y, p, edges=(0, .05, .1, .2, .3, .5, .7, 1.0)) -> pd.DataFrame:
    y, p = pd.Series(np.asarray(y)), pd.Series(np.asarray(p))
    b = pd.cut(p, list(edges), include_lowest=True)
    t = pd.DataFrame({"y": y, "p": p, "bin": b}).groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    t.index = t.index.astype(str)
    return t.reset_index()


def net_benefit(y, p, thresholds) -> np.ndarray:
    """Decision-curve net benefit: TP/n - FP/n * t/(1-t) (Vickers and Elkin 2006)."""
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    out = []
    for t in thresholds:
        flag = p >= t
        tp, fp = np.sum(flag & (y == 1)), np.sum(flag & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def decision_curve(y, preds: dict, thresholds=None) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.02, 0.61, 0.01), 2) if thresholds is None else thresholds
    y = np.asarray(y)
    d = {"threshold": thresholds, "Flag everyone": y.mean() - (1 - y.mean()) * thresholds / (1 - thresholds),
         "Flag no one": np.zeros(len(thresholds))}
    for k, p in preds.items():
        d[k] = net_benefit(y, p, thresholds)
    return pd.DataFrame(d)


def threshold_outcomes(y, p, t) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    flag = p >= t
    tp, fp, fn = int(np.sum(flag & (y == 1))), int(np.sum(flag & (y == 0))), int(np.sum(~flag & (y == 1)))
    return dict(threshold=round(float(t), 3), flagged=int(flag.sum()), caught=tp, false_flags=fp, missed=fn,
                recall=round(tp / max(tp + fn, 1), 3), precision=round(tp / max(tp + fp, 1), 3))


def cost_threshold_table(y, p, ratios=(1, 2, 3, 5)) -> pd.DataFrame:
    """Threshold implied by each cost ratio. With calibrated probabilities, t* = 1 / (1 + ratio)."""
    rows = []
    for r in ratios:
        t = 1 / (1 + r)
        o = threshold_outcomes(y, p, t)
        o["cost_ratio"] = f"{r}:1"
        o["weighted_cost"] = o["false_flags"] + r * o["missed"]
        rows.append(o)
    return pd.DataFrame(rows)[["cost_ratio", "threshold", "flagged", "caught", "missed", "false_flags", "recall", "precision", "weighted_cost"]]


def shortlist(y: pd.Series, score: pd.Series, debut: pd.Series, share: float) -> dict:
    sel = []
    for _, g in debut.groupby(debut):
        k = int(round(share * len(g)))
        sel += list(score[g.index].sort_values(ascending=False).index[:k])
    s = pd.Index(sel)
    return dict(share=share, size=len(s), contributors=int(y[s].sum()), hit_rate=round(float(y[s].mean()), 3),
                missed=int(y.sum() - y[s].sum()), total_contributors=int(y.sum()))


def subgroup_audit(df: pd.DataFrame, groups: list[str], threshold: float, min_n: int) -> pd.DataFrame:
    """df needs columns y, p plus the group columns."""
    rows = []
    for g in groups:
        for level, d in df.groupby(g):
            y, p = d.y.values, d.p.values
            pos, neg = y == 1, y == 0
            flag = p >= threshold
            rows.append(dict(
                dimension=g, group=str(level), n=len(d), contributors=int(pos.sum()),
                auc=round(roc_auc_score(y, p), 3) if 0 < pos.sum() < len(y) else np.nan,
                mean_pred=round(float(p.mean()), 3), observed=round(float(y.mean()), 3),
                calibration_gap=round(float(p.mean() - y.mean()), 3),
                miss_rate=round(float((~flag & pos).sum() / pos.sum()), 3) if pos.sum() else np.nan,
                false_flag_rate=round(float((flag & neg).sum() / neg.sum()), 3) if neg.sum() else np.nan,
                judged=bool(len(d) >= min_n and pos.sum() >= 5)))
    return pd.DataFrame(rows)


def by_cohort(y: pd.Series, p: pd.Series, debut: pd.Series) -> pd.DataFrame:
    rows = []
    for d, idx in debut.groupby(debut).groups.items():
        yy, pp = y[idx], p[idx]
        rows.append(dict(debut=int(d), n=len(idx), contributors=int(yy.sum()), auc=round(roc_auc_score(yy, pp), 3),
                         mean_pred=round(float(pp.mean()), 3), observed=round(float(yy.mean()), 3)))
    return pd.DataFrame(rows)
