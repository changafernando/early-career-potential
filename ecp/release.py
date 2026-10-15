"""Model comparison (accuracy, calibration, complexity) and the release gates a new version must pass.

Release gates turn the governance promises into numbers. Each gate has a threshold in config.yaml
(release_gates) and a corrective action. The Model Review Board sees this table before approving any version.
"""
from __future__ import annotations

import io
import time

import joblib
import numpy as np
import pandas as pd

from . import evaluation as ev
from .models import BASELINE_FEATURES, base_pipeline, candidate_models, fit_baseline, fit_final

EXPLANATION = {"Logistic regression": "Exact additive contributions (readable coefficients)",
               "Random forest": "TreeSHAP over 400 trees (exact, but spread over thousands of splits)",
               "Gradient boosting": "TreeSHAP over 250 sequential trees"}


def _complexity(fitted) -> dict:
    pipe = base_pipeline(fitted)
    est = pipe[-1]
    if hasattr(est, "coef_"):
        params, unit = int(est.coef_.size + est.intercept_.size), "coefficients"
    elif hasattr(est, "estimators_"):
        trees = np.ravel(est.estimators_)
        params, unit = int(sum(t.tree_.node_count for t in trees)), "tree nodes"
    else:
        params, unit = None, ""
    buf = io.BytesIO()
    joblib.dump(fitted, buf, compress=3)
    return {"parameters": params, "parameter_unit": unit, "file_kb": round(len(buf.getvalue()) / 1024, 1)}


def model_comparison(cfg: dict, D: dict, sel: dict) -> list[dict]:
    """Best calibration per family (chosen on validation), then scored once on the 2015-18 classes (h = 3)."""
    coh, masks, seed = D["coh"], D["masks"], cfg["random_seed"]
    lab = coh.labelled
    X, y = D["X"][3][lab], coh.y[lab].astype(int)
    tr, te = masks["train_full"][lab], masks["test"][lab]
    summ = pd.DataFrame(sel["summary"])
    rows = []
    for fam in candidate_models(seed):
        best = summ[summ.model == fam].sort_values("val_brier").iloc[0]
        choice = {"model": fam, "calibration": best.calibration}
        t0 = time.time()
        m = fit_final(X, y, tr, choice, seed)
        secs = time.time() - t0
        p = m.predict_proba(X[te])[:, 1]
        pm = ev.point_metrics(y[te], p)
        rows.append(dict(model=fam, calibration=best.calibration, role=("Champion" if choice == sel["champion"] else
                         ("Challenger" if choice == sel["challenger"] else "Not selected")),
                         val_brier=round(float(best.val_brier), 4), val_auc=round(float(best.val_auc), 3),
                         test_auc=round(pm["auc"], 3), test_brier=round(pm["brier"], 4), test_ece=round(pm["ece"], 3),
                         test_calibration_gap=round(pm["mean_pred"] - pm["observed"], 3), fit_seconds=round(secs, 1),
                         explanation=EXPLANATION[fam], **_complexity(m)))
    b = fit_baseline(X, y, tr)
    pb = b.predict_proba(X.loc[te, BASELINE_FEATURES])[:, 1]
    pm = ev.point_metrics(y[te], pb)
    rows.append(dict(model="Draft order only", calibration="none", role="Baseline", val_brier=None, val_auc=None,
                     test_auc=round(pm["auc"], 3), test_brier=round(pm["brier"], 4), test_ece=round(pm["ece"], 3),
                     test_calibration_gap=round(pm["mean_pred"] - pm["observed"], 3), fit_seconds=None,
                     explanation="Draft slot only", parameters=3, parameter_unit="coefficients", file_kb=None))
    pd.DataFrame(rows).to_csv(cfg["paths"]["outputs_dir"] / "model_comparison_h3.csv", index=False)
    return rows


def release_gates(cfg: dict, D: dict, bt: dict, P3: pd.DataFrame, live: list[dict]) -> list[dict]:
    g = cfg["release_gates"]
    hz = bt["horizons"]
    h3 = hz[3] if 3 in hz else hz["3"]
    thr = bt["decision_threshold"]
    gates = []
    gain = h3["champion"]["auc_gain_vs_reference"]
    ok = gain["est"] >= g["min_auc_gain_vs_draft"] and gain["lo"] > 0
    gates.append(dict(gate="Beats draft order", rule=f"AUC gain >= {g['min_auc_gain_vs_draft']} and 95% CI above 0",
                      value=f"+{gain['est']:.3f} (CI {gain['lo']:.3f} to {gain['hi']:.3f})", status="PASS" if ok else "FAIL",
                      if_failed="Block release; return to design (features or outcome definition)"))
    gap = h3["champion"]["mean_pred"] - h3["champion"]["observed"]
    gates.append(dict(gate="Calibrated overall", rule=f"|predicted - observed| <= {g['max_calibration_gap']}",
                      value=f"{gap:+.3f}", status="PASS" if abs(gap) <= g["max_calibration_gap"] else "FAIL",
                      if_failed="Recalibrate on the newest labelled classes; re-test before release"))
    # subgroup calibration and miss rates, unrounded, judged groups only
    df = P3.join(D["coh"][["position", "draft_tier", "origin", "height_band"]])
    min_n = cfg["evaluation"]["min_subgroup_size"]
    overall_miss = float(((df.p < thr) & (df.y == 1)).sum() / max(df.y.sum(), 1))
    worst_gap, worst_miss = (None, 0.0), (None, 0.0)
    for dim in ["position", "draft_tier", "origin", "height_band"]:
        for lvl, d in df.groupby(dim):
            if lvl == "Unknown" or len(d) < min_n or d.y.sum() < 5:
                continue
            gp = float(d.p.mean() - d.y.mean())
            ms = float(((d.p < thr) & (d.y == 1)).sum() / d.y.sum()) - overall_miss
            if abs(gp) > abs(worst_gap[1]):
                worst_gap = (lvl, gp)
            if ms > worst_miss[1]:
                worst_miss = (lvl, ms)
    st = "PASS" if abs(worst_gap[1]) <= g["subgroup_gap_watch"] else ("WATCH" if abs(worst_gap[1]) <= g["subgroup_gap_fail"] else "FAIL")
    gates.append(dict(gate="Calibrated for every judged subgroup",
                      rule=f"|gap| <= {g['subgroup_gap_watch']} pass; <= {g['subgroup_gap_fail']} watch; above = fail",
                      value=f"worst: {worst_gap[0]} {worst_gap[1]:+.4f}", status=st,
                      if_failed="WATCH: report to the Model Review Board and add a scout review band. "
                                "FAIL: recalibrate by group or block release"))
    gates.append(dict(gate="No group missed far more often",
                      rule=f"subgroup miss rate - overall ({overall_miss:.0%}) <= {g['max_miss_rate_excess']:.2f}",
                      value=f"worst: {worst_miss[0]} +{worst_miss[1]:.3f}",
                      status="PASS" if worst_miss[1] <= g["max_miss_rate_excess"] else "FAIL",
                      if_failed="Mandatory scout review for that group between 0.10 and the threshold; "
                                "test a position-relative outcome definition"))
    dis = float(np.mean([r["models_disagree"] for r in live])) if live else 0.0
    gates.append(dict(gate="Champion and challenger agree", rule=f"share of live players >= 0.15 apart <= {g['max_challenger_disagreement']}",
                      value=f"{dis:.1%}", status="PASS" if dis <= g["max_challenger_disagreement"] else "FAIL",
                      if_failed="Analyst reviews every disagreeing player; investigate before release"))
    return gates
