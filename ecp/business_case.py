"""Business case: what the model costs, what its errors cost, and when it pays back.

All money figures come from config.yaml -> business_case and are ILLUSTRATIVE ASSUMPTIONS (USD) until
finance replaces them. The error counts come from the backtest, so only the prices are assumed.

Decision population: the rookie-scale extension decision. These are first-round picks (only first-round
picks sign rookie-scale contracts) from the untouched 2015-18 classes who were still in the league in
career year 3, scored after three seasons.

Error costs:
    missed contributor (false negative) = cost_missed_contributor
    false flag (false positive)         = cost_missed_contributor / decision.cost_ratio
so the money view and the threshold (1 / (1 + cost ratio)) always agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def costs(cfg: dict) -> dict:
    b = cfg["business_case"]
    delivery = {
        "External consultants (build, 12 weeks)": b["consultant_person_weeks"] * 5 * b["consultant_day_rate"],
        "Post-handover support": b["post_handover_support_days"] * b["consultant_day_rate"],
        "Franchise staff time during the build": b["internal_person_weeks"] * b["internal_cost_per_week"],
    }
    annual = {
        "Franchise analyst (business as usual)": b["analyst_bau_fte"] * b["analyst_salary_loaded"],
        "Governance and scout review time": b["governance_per_year"],
        "Platform and contingency": b["platform_per_year"],
    }
    one_off, per_year = sum(delivery.values()), sum(annual.values())
    return {"delivery": delivery, "annual": annual, "one_off": one_off, "annual_total": per_year,
            "horizon_years": b["horizon_years"], "total_over_horizon": one_off + per_year * b["horizon_years"]}


def _confusion(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    f = p >= t
    return dict(caught=int((f & (y == 1)).sum()), missed=int((~f & (y == 1)).sum()),
                false_flags=int((f & (y == 0)).sum()), correctly_passed=int((~f & (y == 0)).sum()))


def decision_impact(cfg: dict, D: dict, P3: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Errors and their cost in the rookie-scale extension decision, model vs draft order."""
    b, ratio = cfg["business_case"], cfg["decision"]["cost_ratio"]
    c_miss = b["cost_missed_contributor"]
    c_flag = c_miss / ratio
    thr = 1 / (1 + ratio)
    coh, ps = D["coh"], D["ps"]
    alive = set(ps.loc[ps.cy == 3, "personId"])
    g = P3.join(coh[["name", "draft_pick", "debut", "y_volume"]])
    g = g[(g.draft_pick <= 30) & g.index.isin(alive)]
    y = g.y.values.astype(int)
    out = {"population": "First-round picks from the 2015-18 classes still in the league after three seasons",
           "n": int(len(g)), "contributors": int(y.sum()), "threshold": round(thr, 3),
           "cost_missed_contributor": c_miss, "cost_false_flag": round(c_flag)}
    for k, col in [("model", "p"), ("draft_order", "p_draft")]:
        cm = _confusion(y, g[col].values, thr)
        cm["error_cost"] = cm["missed"] * c_miss + cm["false_flags"] * c_flag
        cm["error_cost_per_decision"] = cm["error_cost"] / len(g)
        out[k] = cm
    out["saving_per_decision"] = out["draft_order"]["error_cost_per_decision"] - out["model"]["error_cost_per_decision"]

    # uncertainty in the saving: bootstrap the 2015-18 decisions
    rng = np.random.default_rng(cfg["random_seed"])
    pm, pd_ = g.p.values >= thr, g.p_draft.values >= thr
    unit = lambda f, yy: ((~f & (yy == 1)) * c_miss + (f & (yy == 0)) * c_flag)
    per_player = unit(pd_, y) - unit(pm, y)
    draws = [per_player[rng.integers(0, len(y), len(y))].mean() for _ in range(n_boot)]
    out["saving_per_decision_90ci"] = [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))]

    fp = g[(g.p >= thr) & (g.y == 0)].sort_values("p", ascending=False)
    fn = g[(g.p < thr) & (g.y == 1)].sort_values("p")
    out["false_flag_examples"] = [dict(name=r["name"], pick=int(r.draft_pick), p=round(r.p, 2),
                                       volume_contributor=bool(r.y_volume == 1)) for _, r in fp.head(8).iterrows()]
    out["false_flags_that_meet_volume_definition"] = fp[fp.y_volume == 1].name.tolist()
    out["missed_examples"] = [dict(name=r["name"], pick=int(r.draft_pick), p=round(r.p, 2)) for _, r in fn.iterrows()]
    return out


def payback(cfg: dict, cost: dict, impact: dict) -> dict:
    """Cumulative cost vs cumulative avoided error cost, plus sensitivity to the price of a miss."""
    b = cfg["business_case"]
    per_year_benefit = impact["saving_per_decision"] * b["extension_decisions_per_year"]
    years = b["horizon_years"]
    cum_cost = [cost["one_off"] + cost["annual_total"] * (t + 1) for t in range(years)]
    cum_benefit = [per_year_benefit * (t + 1) for t in range(years)]
    net = per_year_benefit - cost["annual_total"]
    breakeven_years = cost["one_off"] / net if net > 0 else None
    sens = []
    for m in [1_000_000, 3_000_000, 5_000_000]:
        s = impact["saving_per_decision"] * m / b["cost_missed_contributor"]
        n = s * b["extension_decisions_per_year"] - cost["annual_total"]
        sens.append(dict(cost_missed_contributor=m, saving_per_decision=s,
                         breakeven_years=(cost["one_off"] / n) if n > 0 else None,
                         net_over_horizon=s * b["extension_decisions_per_year"] * years - cost["total_over_horizon"]))
    lo, hi = impact["saving_per_decision_90ci"]
    return {"benefit_per_year": per_year_benefit, "cumulative_cost": cum_cost, "cumulative_benefit": cum_benefit,
            "net_over_horizon": cum_benefit[-1] - cum_cost[-1], "breakeven_years": breakeven_years,
            "breakeven_decisions": cost["total_over_horizon"] / impact["saving_per_decision"],
            "net_over_horizon_90ci": [lo * years - cost["total_over_horizon"], hi * years - cost["total_over_horizon"]],
            "cost_of_one_season_delay": per_year_benefit, "sensitivity": sens,
            "baseline_caveat": "Compared with draft order. Scouts add information that draft order lacks, so this is "
                               "the most the model could add until the scout blind review (E3) measures the real baseline."}


CAPACITY_TASKS = [  # (task, hours per year, who) - business-as-usual operating load
    ("Check the nightly/weekly scoring run", 10, "Analyst"),
    ("Monthly monitoring review and minutes", 20, "Analyst"),
    ("Briefs before the trade deadline and the draft", 20, "Analyst"),
    ("Extension-window briefs (about 15 flagged players)", 15, "Analyst"),
    ("Off-season retrain, tests and release pack", 24, "Analyst"),
    ("Model Review Board papers (quarterly)", 16, "Analyst"),
    ("Season-end review of overrides against outcomes", 8, "Analyst"),
    ("Questions from scouts and coaches", 40, "Analyst"),
    ("Monthly watchlist review (3 scouts x 1 hour)", 30, "Scouting panel"),
]


def capacity(cfg: dict) -> dict:
    b = cfg["business_case"]
    analyst_hours = sum(h for _, h, who in CAPACITY_TASKS if who == "Analyst")
    allocated = b["analyst_bau_fte"] * 1650
    return {"tasks": [dict(task=t, hours=h, who=w) for t, h, w in CAPACITY_TASKS],
            "analyst_hours": analyst_hours, "analyst_fte_needed": round(analyst_hours / 1650, 2),
            "analyst_hours_allocated": allocated, "headroom_hours": allocated - analyst_hours}


def build_business_case(cfg: dict, D: dict, P3: pd.DataFrame) -> dict:
    c = costs(cfg)
    imp = decision_impact(cfg, D, P3)
    return {"assumptions_note": "Illustrative USD assumptions in config.yaml; error counts from the 2015-18 backtest.",
            "costs": c, "decision_impact": imp, "payback": payback(cfg, c, imp), "capacity": capacity(cfg)}
