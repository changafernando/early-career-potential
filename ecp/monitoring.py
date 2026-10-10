"""Step 6 - Drift monitoring: detect, understand, adapt (Lu et al. 2019).

Checks run after every data refresh and before every off-season retrain:
0. Field integrity - share of appearances flagged as starts, by season. A 5-on-5 game has a start
                  share near 5 / players used (about 0.45); values far from that mean the field changed.
1. Data drift   - population stability index (PSI) of first-season features for each new debut class
                  against the training classes, judged against empirical alarm levels (how much historical
                  classes differed from each other), because a single class is only ~100 players.
2. Concept drift - contributor base rate by debut class, and calibration on the newest labelled classes.
3. League drift  - three-point share and pace by season (how the game itself is changing).
4. Model health  - champion vs challenger disagreement on the live watchlist.
Each check returns GREEN / AMBER / RED with the action the runbook prescribes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_FEATURES = ["mpg", "games", "gmsc_100", "pts_100", "ast_100", "ts_vs_league", "three_rate_vs_league",
                "age_debut", "height", "best_rate_pct"]


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    if len(cur) < 20 or len(ref) < 20:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def status(v: float, amber: float, red: float) -> str:
    if pd.isna(v):
        return "n/a"
    return "RED" if v >= red else ("AMBER" if v >= amber else "GREEN")


def drift_thresholds(X1: pd.DataFrame, coh: pd.DataFrame) -> dict:
    """Empirical alarm levels. Each debut class is only ~100 players, so PSI is noisy. For every feature,
    each historical class is compared with all other historical classes (leave-one-class-out); the 75th
    and 95th percentiles of those PSIs become the AMBER and RED levels. A new class is flagged only when it
    differs from history more than historical classes differed from each other."""
    lab = coh[coh.labelled]
    out = {}
    for f in PSI_FEATURES:
        vals = []
        for d in sorted(lab.debut.unique()):
            cur = X1.loc[lab.index[lab.debut == d], f]
            rest = X1.loc[lab.index[lab.debut != d], f]
            vals.append(psi(rest, cur))
        out[f] = {"amber": round(float(np.nanpercentile(vals, 75)), 3), "red": round(float(np.nanpercentile(vals, 95)), 3)}
    return out


def run_monitoring(cfg: dict, D: dict, backtest_res: dict, live: list[dict], drivers: dict | None = None) -> dict:
    from .features import PLAIN_NAMES

    coh, X1 = D["coh"], D["X"][1]
    ref_idx = coh.index[coh.labelled]
    ref = X1.loc[ref_idx]
    lim = drift_thresholds(X1, coh)
    rows = []
    for d in sorted(coh.debut.unique()):
        if d <= cfg["cohort"]["last_labelled_debut"]:
            continue
        cur = X1.loc[coh.index[coh.debut == d]]
        vals = {f: psi(ref[f], cur[f]) for f in PSI_FEATURES}
        lv = {f: status(v, lim[f]["amber"], lim[f]["red"]) for f, v in vals.items()}
        ratio = {f: v / lim[f]["red"] for f, v in vals.items() if pd.notna(v)}
        worst = max(ratio, key=ratio.get)
        rows.append(dict(debut=int(d), n=len(cur), worst_feature=worst, worst_psi=round(vals[worst], 3),
                         worst_red_level=lim[worst]["red"], features_red=sum(v == "RED" for v in lv.values()),
                         features_amber=sum(v == "AMBER" for v in lv.values()),
                         **{f"psi_{k}": round(v, 3) for k, v in vals.items()}))
    data_drift = pd.DataFrame(rows)

    lab = coh[coh.labelled]
    base = lab.groupby("debut").y.mean().round(3)
    hz = backtest_res["horizons"]
    h3 = (hz[3] if 3 in hz else hz["3"])["champion"]
    gap = h3["mean_pred"] - h3["observed"]
    league = D["league"]
    disagree = float(np.mean([r["models_disagree"] for r in live])) if live else np.nan
    mon = cfg["monitoring"]
    top = [f for f in (drivers or {}).get("champion_mean_abs_logodds", {}) if f in PSI_FEATURES][:4]
    latest = data_drift.iloc[-1]
    lv_top = {f: status(latest[f"psi_{f}"], lim[f]["amber"], lim[f]["red"]) for f in top}
    lv_all = {f: status(latest[f"psi_{f}"], lim[f]["amber"], lim[f]["red"]) for f in PSI_FEATURES}
    worst_of = lambda lv: "RED" if "RED" in lv.values() else ("AMBER" if "AMBER" in lv.values() else "GREEN")
    names = lambda fs: ", ".join(PLAIN_NAMES.get(f, f) for f in fs)
    newest = int(latest.debut)
    checks = [
        dict(check=f"Data drift on the four most influential monitored features ({names(top)}), {newest} class vs history",
             value=f"{sum(v != 'GREEN' for v in lv_top.values())} of {len(top)} outside historical range", status=worst_of(lv_top),
             action="Check calibration on the newest labelled classes; retrain off-season if RED"),
        dict(check=f"Data drift on any monitored feature, {newest} class vs history",
             value=f"{sum(v == 'RED' for v in lv_all.values())} red, {sum(v == 'AMBER' for v in lv_all.values())} amber of {len(PSI_FEATURES)}",
             status=worst_of(lv_all), action="Investigate the feature at the next monthly review; note in the model card"),
        dict(check="Calibration gap on newest labelled classes (2015-18)", value=round(gap, 3),
             status=status(abs(gap), mon["calibration_gap_act"] / 2, mon["calibration_gap_act"]),
             action="Recalibrate if RED; report at Model Review Board if AMBER"),
        dict(check="Contributor base rate, last 4 labelled classes vs training", value=round(float(base.tail(4).mean() - base.mean()), 3),
             status=status(abs(float(base.tail(4).mean() - base.mean())), 0.02, 0.05),
             action="Review construct thresholds with steering if RED"),
        dict(check="Champion-challenger disagreement on live watchlist (share >= 0.15 apart)", value=round(disagree, 3),
             status=status(disagree, 0.10, 0.20), action="Analyst reviews disagreeing players before briefs go out"),
    ]
    ps = D["ps"]
    starts = (ps.groupby("season").started.sum() / ps.groupby("season").gp.sum()).round(3)
    bad = starts[(starts < 0.40) | (starts > 0.55)]
    checks.insert(0, dict(check="Field integrity: games-started flag plausible in every season",
                          value=f"{len(bad)} of {len(starts)} seasons implausible",
                          status="RED" if len(bad) else "GREEN",
                          action="Field excluded from the model; raise a data-quality ticket with the data provider"))
    out = {"psi_levels": lim, "start_flag_share_by_season": {int(k): float(v) for k, v in starts.items()},
           "start_flag_unreliable_seasons": [int(k) for k in bad.index], "checks": checks, "data_drift_by_class": data_drift.to_dict("records"),
           "base_rate_by_class": {int(k): float(v) for k, v in base.items()},
           "league_by_season": league.round(4).to_dict("records")}
    data_drift.to_csv(cfg["paths"]["outputs_dir"] / "monitoring_psi_by_class.csv", index=False)
    return out
