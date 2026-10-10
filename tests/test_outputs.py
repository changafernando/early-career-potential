"""Integration checks on the shipped pipeline outputs (run `python run_pipeline.py all` first)."""
import json

import pytest

from ecp.config import load_config


@pytest.fixture(scope="module")
def out():
    cfg = load_config()
    p = cfg["paths"]["outputs_dir"]
    if not (p / "metrics.json").exists():
        pytest.skip("run the pipeline first")
    return cfg, json.loads((p / "metrics.json").read_text()), json.loads((p / "watchlist_current.json").read_text())


def test_model_beats_draft_order_on_untouched_classes(out):
    _, m, _ = out
    for h, r in m["backtest"]["horizons"].items():
        assert r["champion"]["auc_gain_vs_reference"]["lo"] > 0, f"no reliable gain at horizon {h}"


def test_probabilities_are_calibrated_within_tolerance(out):
    cfg, m, _ = out
    c = m["backtest"]["horizons"]["3"]["champion"]
    assert abs(c["mean_pred"] - c["observed"]) <= cfg["monitoring"]["calibration_gap_act"]


def test_registry_lists_every_horizon(out):
    cfg, _, _ = out
    reg = json.loads((cfg["paths"]["models_dir"] / "registry.json").read_text())
    assert set(reg["models"]) == {f"h{h}" for h in cfg["horizons"]}
    from ecp.pipeline import file_hash
    for h, v in reg["models"].items():
        assert (cfg["paths"]["models_dir"] / v["champion"]).exists()
        for n, digest in v["sha256"].items():   # model files are unaltered since registration
            assert file_hash(cfg["paths"]["models_dir"] / f"{n}_{h}.joblib") == digest


def test_watchlist_rules(out):
    cfg, m, live = out
    thr = m["backtest"]["decision_threshold"]
    for r in live:
        assert 0 <= r["p_low"] <= r["p"] <= r["p_high"] <= 1
        if r["minutes"] < cfg["decision"]["evidence_min_minutes"]:
            assert r["status"] == "Insufficient evidence"
        elif r["p"] >= thr:
            assert r["status"] == "Flag for review"
        else:
            assert r["status"] == "Below threshold"


def test_dashboard_is_self_contained(out):
    cfg, _, _ = out
    html = cfg["paths"]["dashboard"].read_text()
    assert "<script src=" not in html and "http://" not in html.replace("http://www.w3.org", "")
