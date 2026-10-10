import numpy as np

from ecp.constructs import build_cohort
from ecp.features import window_features
from ecp.seasons import build_player_seasons


def _features(cfg, pts, teams, players, h):
    ps = build_player_seasons(pts, teams, cfg)
    coh = build_cohort(ps, players, cfg)
    return window_features(ps, coh, players, h, cfg)


def test_no_leakage_from_later_seasons(cfg, toy_league):
    """Nothing played after the decision season (end of career year h) may change horizon-h features.

    League averages for a season are known once that season ends, so only seasons strictly after the
    decision season are altered here, for every player in the league.
    """
    pts, teams, players = toy_league
    h = 2
    before = _features(cfg, pts, teams, players, h=h)
    cols = ["games", "minutes", "mpg", "pts_100", "ast_100", "best_gmsc_pct", "last_rate_pct", "age_debut", "draft_pick"]
    for debut in (1996, 1997):
        decision_season = debut + h - 1
        later = pts.copy()
        future = later.season > decision_season
        later.loc[future, ["points", "assists", "numMinutes", "steals"]] *= 3
        after = _features(cfg, later, teams, players, h=h)
        ids = [1, 2, 5] if debut == 1996 else [3, 4, 6]
        assert np.allclose(before.loc[ids, cols].values, after.loc[ids, cols].values, equal_nan=True), debut


def test_country_is_never_a_feature(cfg, toy_league):
    X = _features(cfg, *toy_league, h=3)
    assert not any("country" in c or "origin" in c for c in X.columns)


def test_started_flag_is_excluded(cfg, toy_league):
    X = _features(cfg, *toy_league, h=3)
    assert "start_share" not in X.columns


def test_small_samples_have_no_rates(cfg, toy_league):
    pts, teams, players = toy_league
    pts = pts.copy()
    pts.loc[(pts.personId == 2), "numMinutes"] = 20   # 3 seasons x 20 minutes = 60 < 100
    X = _features(cfg, pts, teams, players, h=3)
    assert np.isnan(X.loc[2, "pts_100"]) and np.isnan(X.loc[2, "ts_vs_league"])
