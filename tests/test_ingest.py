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


def test_ingest_end_to_end_repairs_the_known_data_defects(tmp_path):
    from ecp.ingest import PLAYER_BIO_COLS, PLAYER_COLS, TEAM_COLS, run_ingest

    celtics = dict(playerteamCity="Boston", playerteamName="Celtics")
    games = [  # gameId, gameType, teamId, minutes, points
        ("21900001", "Regular Season", 1610612738, "30:00", 20),  # 2019-20, clock-string minutes
        ("21901200", "Regular Season", 1610612738, "20.5", 10),   # August 2020 bubble game, still 2019-20
        ("41900001", "Playoffs", 1610612738, "40", 30),           # playoffs: excluded
        ("22100001", None, None, "25", 12),                       # 2021-22: blank game type and team ID
        ("22100002", None, None, None, 0),                        # 2021-22 did-not-play row
        ("62300001", "NBA Emirates Cup", 1610612738, "35", 25),   # NBA Cup final: excluded
    ]
    raw = tmp_path / "raw"
    raw.mkdir()
    players = pd.DataFrame([{**dict.fromkeys(PLAYER_COLS, 1), **celtics, "personId": 7, "gameId": g, "gameType": gt,
                             "playerteamId": t, "numMinutes": m, "points": p, "startingPosition": None}
                            for g, gt, t, m, p in games])
    players.to_csv(raw / "PlayerStatistics.csv", index=False)
    team = dict(numMinutes=240, fieldGoalsAttempted=80, freeThrowsAttempted=20, reboundsOffensive=10, turnovers=15,
                teamScore=110, opponentScore=100, threePointersAttempted=30)
    pd.DataFrame([{**team, "gameId": g, "teamId": t, "gameType": gt} for g, gt, t in [
        ("21900001", "Regular Season", 1610612738), ("21901200", "Regular Season", 1610612738),
        ("22100001", None, 1610612738), ("41900001", "Playoffs", 1610612738), ("22100099", None, 0),
    ]], columns=TEAM_COLS).to_csv(raw / "TeamStatistics.csv", index=False)
    pd.DataFrame([dict.fromkeys(PLAYER_BIO_COLS, 1) | {"personId": 7}]).to_csv(raw / "Players.csv", index=False)

    report = run_ingest(raw, tmp_path / "processed", first_season=2015)

    pts = pd.read_csv(tmp_path / "processed" / "player_team_seasons.csv.gz").set_index(["season", "teamId"])
    assert list(pts.index) == [(2019, 1610612738), (2021, 1610612738)]   # team ID restored, no -1 rows
    assert pts.loc[(2019, 1610612738), ["gp", "numMinutes", "points"]].tolist() == [2, 50.5, 30]
    assert pts.loc[(2021, 1610612738), ["gp", "rows"]].tolist() == [1, 2]   # DNP row kept, not counted as played
    teams = pd.read_csv(tmp_path / "processed" / "team_seasons.csv").set_index("season")
    assert teams.games.to_dict() == {2019: 2, 2021: 1}                   # playoffs and placeholder team dropped
    assert abs(teams.loc[2019, "poss"] - 2 * 93.8) < 1e-9
    dq = report["data_quality"]
    assert (dq["rows_read"], dq["regular_season_rows"], dq["cup_final_rows_excluded"]) == (6, 4, 1)
    assert (dq["rows_team_id_restored"], dq["rows_team_id_unresolved"]) == (2, 0)
    assert report["player_rows_without_team_match"] == 0
