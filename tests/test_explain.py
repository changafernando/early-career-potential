import pandas as pd

from ecp.explain import top_drivers


def test_equal_best_and_latest_rank_is_one_driver_with_the_summed_effect():
    contrib = pd.DataFrame({"best_gmsc_pct": [0.3, 0.3], "last_gmsc_pct": [0.2, 0.2], "age_debut": [-0.1, -0.1]},
                           index=[1, 2])
    X = pd.DataFrame({"best_gmsc_pct": [0.5, 0.9], "last_gmsc_pct": [0.5, 0.4], "age_debut": [20.0, 21.0]},
                     index=[1, 2])
    d = top_drivers(contrib, X)

    one_season = d.loc[1, "drivers_up"]
    assert [r["feature"] for r in one_season] == ["best_gmsc_pct"]
    assert one_season[0]["effect"] == 0.5

    two_seasons = d.loc[2, "drivers_up"]
    assert [r["feature"] for r in two_seasons] == ["best_gmsc_pct", "last_gmsc_pct"]
