import pandas as pd

from ecp.ingest import parse_minutes, season_from_game_id


def test_parse_minutes_handles_decimals_and_clock_strings():
    s = pd.Series(["31.5", "23:21", None, "0"])
    out = parse_minutes(s)
    assert out.iloc[0] == 31.5
    assert abs(out.iloc[1] - (23 + 21 / 60)) < 1e-9
    assert pd.isna(out.iloc[2])
    assert out.iloc[3] == 0


def test_season_comes_from_game_id_not_calendar_month():
    # 0021901200 is a 2019-20 regular-season game played in the August 2020 'bubble'
    prefix, season = season_from_game_id(pd.Series(["21901200", "42500405", "22101219", "29700001"]))
    assert list(prefix) == ["002", "004", "002", "002"]
    assert list(season) == [2019, 2025, 2021, 1997]
