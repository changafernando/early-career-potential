import numpy as np

from ecp import evaluation as ev


def test_threshold_follows_cost_ratio():
    y = np.array([1, 0, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.3, 0.26, 0.1])
    t = ev.cost_threshold_table(y, p, ratios=(3,))
    assert abs(t.threshold.iloc[0] - 0.25) < 1e-9
    assert t.caught.iloc[0] == 2 and t.false_flags.iloc[0] == 1


def test_net_benefit_matches_hand_calculation():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.8, 0.1, 0.6, 0.1])
    nb = ev.net_benefit(y, p, [0.5])[0]
    assert abs(nb - (1 / 4 - 1 / 4 * 0.5 / 0.5)) < 1e-12


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 300), 0, 1)
    m = ev.bootstrap_metrics(y, p, n=200)
    assert m["auc"]["lo"] <= m["auc"]["est"] <= m["auc"]["hi"]
