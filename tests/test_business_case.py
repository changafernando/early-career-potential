"""The business case must stay consistent with the decision rule and add up."""
import json

import pytest

from ecp.business_case import CAPACITY_TASKS, costs
from ecp.config import load_config


def test_costs_add_up(cfg):
    c = costs(cfg)
    assert c["one_off"] == pytest.approx(sum(c["delivery"].values()))
    assert c["annual_total"] == pytest.approx(sum(c["annual"].values()))
    assert c["total_over_horizon"] == pytest.approx(c["one_off"] + c["annual_total"] * c["horizon_years"])


def test_capacity_fits_the_allocated_analyst_time(cfg):
    hours = sum(h for _, h, who in CAPACITY_TASKS if who == "Analyst")
    assert hours <= cfg["business_case"]["analyst_bau_fte"] * 1650


def test_saved_business_case_matches_decision_rule():
    cfg = load_config()
    p = cfg["paths"]["outputs_dir"] / "metrics.json"
    if not p.exists():
        pytest.skip("run the pipeline first")
    m = json.loads(p.read_text())
    imp = m["business_case"]["decision_impact"]
    # a false flag is priced at miss cost / cost ratio, so money and threshold agree
    assert imp["cost_false_flag"] == pytest.approx(imp["cost_missed_contributor"] / cfg["decision"]["cost_ratio"], rel=1e-6)
    assert imp["threshold"] == pytest.approx(1 / (1 + cfg["decision"]["cost_ratio"]), abs=1e-3)
    for k in ("model", "draft_order"):
        r = imp[k]
        assert r["caught"] + r["missed"] == imp["contributors"]
        assert r["caught"] + r["missed"] + r["false_flags"] + r["correctly_passed"] == imp["n"]
    assert {g["status"] for g in m["release_gates"]} <= {"PASS", "WATCH", "FAIL"}
