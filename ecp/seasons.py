"""Step 2a - Player-season table: possessions, Game Score and era-relative percentiles.

Why these transformations (the case study's 'interpreting the numbers' challenge):
* Raw totals mislead, so production is expressed per 100 possessions. A player's possessions are
  estimated from his minutes and his team's pace that season, which also removes pace differences
  between teams and eras.
* Eras differ (three-point attempts doubled between 1996 and 2025), so every season-level measure is
  also expressed as a percentile within its own season. A 70th-percentile season means the same thing
  in 1998 and in 2024.
* Shortened seasons (1998-99, 2011-12, 2019-20, 2020-21) scale the rotation-minutes bar by games played.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BOX = ["numMinutes", "points", "assists", "blocks", "steals", "fieldGoalsAttempted", "fieldGoalsMade",
       "threePointersAttempted", "threePointersMade", "freeThrowsAttempted", "freeThrowsMade",
       "reboundsDefensive", "reboundsOffensive", "reboundsTotal", "foulsPersonal", "turnovers",
       "plusMinusPoints", "started", "gp", "win"]


def game_score(d: pd.DataFrame) -> pd.Series:
    """Hollinger Game Score, applied to season totals."""
    return (d.points + 0.4 * d.fieldGoalsMade - 0.7 * d.fieldGoalsAttempted
            - 0.4 * (d.freeThrowsAttempted - d.freeThrowsMade) + 0.7 * d.reboundsOffensive
            + 0.3 * d.reboundsDefensive + d.steals + 0.7 * d.assists + 0.7 * d.blocks
            - 0.4 * d.foulsPersonal - d.turnovers)


def build_player_seasons(pts: pd.DataFrame, teams: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Collapse player x team x season rows to player x season, with possessions and percentiles."""
    teams = teams[(teams.teamId > 0) & (teams.poss > 0)].copy()
    teams["poss_per_player_min"] = teams.poss / (teams.team_minutes / 5)
    season_games = teams.groupby("season").games.max().rename("season_games")

    d = pts.merge(teams[["teamId", "season", "poss_per_player_min"]], on=["teamId", "season"], how="left")
    league_rate = teams.groupby("season").apply(lambda t: t.poss.sum() / (t.team_minutes.sum() / 5), include_groups=False)
    d["poss_per_player_min"] = d.poss_per_player_min.fillna(d.season.map(league_rate))
    d["poss"] = d.numMinutes.fillna(0) * d.poss_per_player_min

    ps = d.groupby(["personId", "season"])[BOX + ["poss"]].sum(min_count=1)
    ps["teams"] = d.groupby(["personId", "season"]).teamId.nunique()
    ps = ps.reset_index()
    ps = ps[ps.gp.fillna(0) > 0].copy()
    ps = ps.join(season_games, on="season")

    ps["gmsc"] = game_score(ps)
    ps["gmsc_per100"] = np.where(ps.poss > 0, 100 * ps.gmsc / ps.poss.where(ps.poss > 0), np.nan)
    ps["mpg"] = ps.numMinutes / ps.gp

    # volume view: rank on season total Game Score among everyone who played
    ps["gmsc_pct"] = ps.groupby("season").gmsc.rank(pct=True)
    # rate view (features): rank on Game Score per 100 among players with a meaningful sample
    sample = ps.numMinutes >= 250
    ps["rate_pct"] = np.nan
    ps.loc[sample, "rate_pct"] = ps[sample].groupby("season").gmsc_per100.rank(pct=True)
    # rate view (outcome): rank among rotation players only, minutes bar scaled for short seasons
    rc = cfg["construct"]["rate"]
    ps["rotation_minutes_bar"] = rc["min_minutes_82_games"] * ps.season_games / 82
    ps["rotation"] = ps.numMinutes >= ps.rotation_minutes_bar
    ps["rotation_rate_pct"] = np.nan
    ps.loc[ps.rotation, "rotation_rate_pct"] = ps[ps.rotation].groupby("season").gmsc_per100.rank(pct=True)

    # era context: league three-point attempt share by season (used for era-relative shot profile)
    lg = ps.groupby("season")[["threePointersAttempted", "fieldGoalsAttempted"]].sum()
    ps["league_3pa_share"] = ps.season.map(lg.threePointersAttempted / lg.fieldGoalsAttempted)

    debut = ps.groupby("personId").season.min().rename("debut")
    ps = ps.join(debut, on="personId")
    ps["cy"] = ps.season - ps.debut + 1   # calendar career year: a missed season still counts as a year
    return ps.sort_values(["personId", "season"]).reset_index(drop=True)


def league_context(ps: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Season-level league trends for drift monitoring and the report."""
    teams = teams[(teams.teamId > 0) & (teams.poss > 0)]
    lg = ps.groupby("season").agg(players=("personId", "nunique"),
                                  three_pa=("threePointersAttempted", "sum"), fga=("fieldGoalsAttempted", "sum"))
    lg["three_pa_share"] = lg.three_pa / lg.fga
    lg["pace"] = teams.groupby("season").apply(lambda t: 48 * t.poss.sum() / (t.team_minutes.sum() / 5), include_groups=False)
    return lg.drop(columns=["three_pa", "fga"]).reset_index()
