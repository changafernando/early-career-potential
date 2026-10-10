"""Step 1 - Ingest: raw NBA box scores -> tidy player-team-season and team-season tables.

Uses pandas and numpy only, so it can run on a lightweight machine next to the raw files.

Inputs (from the franchise's NBA dataset folder):
    PlayerStatistics.csv  game-by-game player box scores (about 1.7 million rows)
    TeamStatistics.csv    game-by-game team box scores (used for possessions / pace)
    Players.csv           biographical data, joined on personId

Outputs (written to ``processed_dir``):
    player_team_seasons.csv.gz  one row per player x season x team (regular season only)
    team_seasons.csv            one row per team x season, with estimated possessions
    players.csv                 the biographical columns the model uses
    ingest_report.json          row counts and data-quality checks for the audit trail

Data-quality rules applied here (see docs/data_dictionary.md):
    * Regular-season games only, identified by the game ID prefix '002'. The text column gameType
      is not reliable: it is blank for every 2021-22 team row and labels some NBA Cup group games
      (which do count towards season statistics) as 'NBA Emirates Cup'. Prefix '006' (the NBA Cup
      final, which does not count) is excluded.
    * The season is read from the game ID, not the calendar month. The month rule wrongly moves
      the August 2020 'bubble' games into the 2020-21 season.
    * playerteamId is blank for almost all 2021-22 rows and half of 2000-01 rows. It is restored
      from the team city and name, using the mapping observed in every other season.
    * Minutes may be stored as decimals or as 'MM:SS' strings; both are parsed.
    * A game counts as 'played' only when minutes > 0 (did-not-play rows are kept as rows only).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PLAYER_COLS = [
    "personId", "gameId", "gameType", "win", "numMinutes", "points", "assists", "blocks", "steals",
    "fieldGoalsAttempted", "fieldGoalsMade", "threePointersAttempted", "threePointersMade",
    "freeThrowsAttempted", "freeThrowsMade", "reboundsDefensive", "reboundsOffensive",
    "reboundsTotal", "foulsPersonal", "turnovers", "plusMinusPoints", "startingPosition",
    "playerteamId", "playerteamCity", "playerteamName",
]
SUM_COLS = [
    "numMinutes", "points", "assists", "blocks", "steals", "fieldGoalsAttempted", "fieldGoalsMade",
    "threePointersAttempted", "threePointersMade", "freeThrowsAttempted", "freeThrowsMade",
    "reboundsDefensive", "reboundsOffensive", "reboundsTotal", "foulsPersonal", "turnovers",
    "plusMinusPoints", "started", "gp", "win",
]
TEAM_COLS = [
    "gameId", "teamId", "gameType", "numMinutes", "fieldGoalsAttempted", "freeThrowsAttempted",
    "reboundsOffensive", "turnovers", "teamScore", "opponentScore", "threePointersAttempted",
]
PLAYER_BIO_COLS = [
    "personId", "firstName", "lastName", "birthDate", "country", "heightInches", "bodyWeightLbs",
    "guard", "forward", "center", "draftYear", "draftRound", "draftNumber", "fromYear", "toYear",
]


def parse_minutes(s: pd.Series) -> pd.Series:
    """Parse minutes stored either as decimals ('31.5') or as clock strings ('31:30')."""
    s = s.astype("string")
    has_colon = s.str.contains(":", na=False)
    out = pd.to_numeric(s.where(~has_colon), errors="coerce")
    if has_colon.any():
        mm = s[has_colon].str.split(":", expand=True)
        out.loc[has_colon] = pd.to_numeric(mm[0], errors="coerce") + pd.to_numeric(mm[1], errors="coerce") / 60
    return out.astype(float)


def season_from_game_id(game_id: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (game type prefix, season start year) decoded from NBA game IDs.

    NBA game IDs are 10 digits, e.g. 0022500001: '002' = regular season, '25' = 2025-26.
    The CSV stores them as integers, so leading zeros are restored first.
    """
    gid = game_id.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(10)
    prefix = gid.str[:3]
    yy = pd.to_numeric(gid.str[3:5], errors="coerce")
    season = np.where(yy >= 46, 1900 + yy, 2000 + yy)
    return prefix, pd.Series(season, index=game_id.index)


def build_player_team_seasons(raw_dir: Path, first_season: int = 1990, chunksize: int = 400_000):
    """Aggregate regular-season player box scores to player x season x team totals."""
    parts, game_types, team_map = [], {}, {}
    dq = {"rows_read": 0, "regular_season_rows": 0, "cup_final_rows_excluded": 0,
          "rows_before_first_season": 0, "rows_missing_minutes": 0, "rows_missing_minutes_with_points": 0,
          "rows_team_id_restored": 0, "rows_team_id_unresolved": 0}
    chunks = []
    reader = pd.read_csv(raw_dir / "PlayerStatistics.csv", usecols=PLAYER_COLS, chunksize=chunksize,
                         low_memory=False, dtype={"numMinutes": "string", "gameId": "string"})
    for ch in reader:
        dq["rows_read"] += len(ch)
        prefix, season = season_from_game_id(ch.gameId)
        ch = ch.assign(prefix=prefix, season=season)
        for (pf, gt), v in ch.groupby([ch.prefix, ch.gameType.fillna("(blank)")]).size().items():
            key = f"{pf}|{gt}"
            game_types[key] = game_types.get(key, 0) + int(v)
        dq["cup_final_rows_excluded"] += int((ch.prefix == "006").sum())
        ch = ch[ch.prefix == "002"].copy()
        dq["regular_season_rows"] += len(ch)
        dq["rows_before_first_season"] += int((ch.season < first_season).sum())
        ch = ch[ch.season >= first_season].copy()
        ch["playerteamId"] = pd.to_numeric(ch.playerteamId, errors="coerce")
        known = ch.dropna(subset=["playerteamId", "playerteamName"])
        for k, v in known.groupby(["playerteamCity", "playerteamName"]).playerteamId.agg(lambda x: x.mode().iat[0]).items():
            team_map.setdefault(k, v)
        chunks.append(ch)
    # second pass (in memory, regular season since first_season only): restore missing team IDs
    for ch in chunks:
        miss = ch.playerteamId.isna()
        if miss.any():
            keys = list(zip(ch.loc[miss, "playerteamCity"], ch.loc[miss, "playerteamName"]))
            ch.loc[miss, "playerteamId"] = [team_map.get(k, np.nan) for k in keys]
            dq["rows_team_id_restored"] += int(ch.loc[miss, "playerteamId"].notna().sum())
            dq["rows_team_id_unresolved"] += int(ch.loc[miss, "playerteamId"].isna().sum())
        ch["playerteamId"] = ch.playerteamId.fillna(-1)
        ch["numMinutes"] = parse_minutes(ch.numMinutes)
        pts_num = pd.to_numeric(ch.points, errors="coerce").fillna(0)
        dq["rows_missing_minutes"] += int(ch.numMinutes.isna().sum())
        dq["rows_missing_minutes_with_points"] += int((ch.numMinutes.isna() & (pts_num > 0)).sum())
        ch["started"] = ch.startingPosition.notna() & (ch.startingPosition.astype(str).str.strip() != "")
        ch["gp"] = (ch.numMinutes.fillna(0) > 0).astype(int)
        for c in SUM_COLS:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        keys = ["personId", "season", "playerteamId"]
        g = ch.groupby(keys)[SUM_COLS].sum(min_count=1)
        g["rows"] = ch.groupby(keys).size()
        parts.append(g)
    df = pd.concat(parts).groupby(level=[0, 1, 2]).sum(min_count=1).reset_index()
    df = df.rename(columns={"playerteamId": "teamId"})
    df["personId"] = df.personId.astype("int64")
    df["teamId"] = df.teamId.astype("int64")
    return df, game_types, dq


def build_team_seasons(raw_dir: Path, first_season: int = 1990) -> pd.DataFrame:
    """Team x season totals with estimated possessions (FGA + 0.44 FTA - OREB + TOV)."""
    t = pd.read_csv(raw_dir / "TeamStatistics.csv", usecols=TEAM_COLS, low_memory=False, dtype={"gameId": "string"})
    prefix, season = season_from_game_id(t.gameId)
    t = t.assign(prefix=prefix, season=season)
    t = t[(t.prefix == "002") & (t.season >= first_season)].copy()  # gameType is blank for 2021-22
    t["teamId"] = pd.to_numeric(t.teamId, errors="coerce")
    t = t[(t.teamId > 0)]  # drops placeholder rows (teamId 0) for games not yet played
    for c in TEAM_COLS[3:]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["numMinutes"] = t.numMinutes.fillna(240)
    t["poss"] = t.fieldGoalsAttempted + 0.44 * t.freeThrowsAttempted - t.reboundsOffensive + t.turnovers
    g = t.groupby(["teamId", "season"]).agg(
        games=("gameId", "nunique"), team_minutes=("numMinutes", "sum"), poss=("poss", "sum"),
        fga=("fieldGoalsAttempted", "sum"), fg3a=("threePointersAttempted", "sum"),
        points_for=("teamScore", "sum"), points_against=("opponentScore", "sum")).reset_index()
    # pace = possessions per 48 minutes (team minutes are 5 players x 48 = 240 per regulation game)
    g["pace"] = 48 * g.poss / (g.team_minutes / 5)
    g["teamId"] = g.teamId.astype("int64")
    return g


def run_ingest(raw_dir: str | Path, processed_dir: str | Path, first_season: int = 1990) -> dict:
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    pts, game_types, dq = build_player_team_seasons(raw_dir, first_season)
    teams = build_team_seasons(raw_dir, first_season)
    players = pd.read_csv(raw_dir / "Players.csv", usecols=PLAYER_BIO_COLS, low_memory=False)

    pts.to_csv(processed_dir / "player_team_seasons.csv.gz", index=False, compression="gzip")
    teams.to_csv(processed_dir / "team_seasons.csv", index=False)
    players.to_csv(processed_dir / "players.csv", index=False)

    matched = pts.merge(teams[["teamId", "season"]], on=["teamId", "season"], how="left", indicator=True)
    report = {
        "game_types_seen (gameId prefix|gameType)": game_types,
        "data_quality": dq,
        "player_team_seasons": int(len(pts)),
        "players_with_minutes": int(pts.loc[pts.gp > 0, "personId"].nunique()),
        "seasons": [int(pts.season.min()), int(pts.season.max())],
        "team_seasons": int(len(teams)),
        "player_rows_without_team_match": int((matched._merge == "left_only").sum()),
        "players_in_bio_file": int(len(players)),
        "bio_missing_height_share": round(float(players.heightInches.isna().mean()), 3),
    }
    (processed_dir / "ingest_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate raw NBA box scores for the early-career potential model")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--first-season", type=int, default=1990)
    a = ap.parse_args()
    print(json.dumps(run_ingest(a.raw_dir, a.processed_dir, a.first_season), indent=2))
