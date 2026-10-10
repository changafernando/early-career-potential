"""Step 2c - Early-career features: what the front office knows after 1, 2 or 3 seasons.

Leakage rule: a feature for horizon h may only use career years 1..h. tests/test_features.py checks it.
Fairness rule: country of origin is never a feature. It is used only to audit errors by subgroup.
Era rule: production and efficiency are measured against the league average in the same seasons
          ('per 100 possessions vs league'), so a 1998 rookie and a 2024 rookie are compared fairly.
Integrity rule: the games-started field changes meaning across seasons (recorded for ~90% of
          appearances in 1996-2016, ~47% from 2017, and blank in 1990-95 and 2021-22), so it is
          excluded from the model. See docs/data_dictionary.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RATE_STATS = {  # per-100-possession rates: column -> feature name
    "points": "pts_100", "assists": "ast_100", "reboundsOffensive": "oreb_100", "reboundsDefensive": "dreb_100",
    "steals": "stl_100", "blocks": "blk_100", "turnovers": "tov_100", "fieldGoalsAttempted": "fga_100",
    "foulsPersonal": "pf_100", "plusMinusPoints": "pm_100", "gmsc": "gmsc_100",
}

PLAIN_NAMES = {
    "games": "games played", "mpg": "minutes per game",
    "minutes": "total minutes", "seasons_played": "seasons played", "pts_100": "points per 100 possessions",
    "ast_100": "assists per 100", "oreb_100": "offensive rebounds per 100", "dreb_100": "defensive rebounds per 100",
    "stl_100": "steals per 100", "blk_100": "blocks per 100", "tov_100": "turnovers per 100",
    "fga_100": "shot attempts per 100", "pf_100": "fouls per 100", "pm_100": "plus-minus per 100",
    "gmsc_100": "Game Score per 100", "ts_vs_league": "true shooting vs league", "three_rate_vs_league": "3-point share vs league",
    "ft_rate_vs_league": "free-throw rate vs league", "best_gmsc_pct": "best season rank (total impact)",
    "last_gmsc_pct": "latest season rank (total impact)", "gmsc_pct_trend": "trend in season rank",
    "best_rate_pct": "best season rank (per possession)", "last_rate_pct": "latest season rank (per possession)",
    "mpg_trend": "trend in minutes per game", "age_debut": "age at debut", "height": "height",
    "weight": "weight", "draft_pick": "draft pick", "undrafted": "undrafted",
    "is_guard": "listed guard", "is_forward": "listed forward", "is_center": "listed center",
}


def window_features(ps: pd.DataFrame, coh: pd.DataFrame, players: pd.DataFrame, h: int, cfg: dict) -> pd.DataFrame:
    """Feature matrix (one row per cohort player) using career years 1..h only."""
    w = ps[(ps.cy <= h) & ps.personId.isin(coh.index)]
    stats = list(RATE_STATS) + ["threePointersAttempted", "freeThrowsAttempted"]
    tot = w.groupby("personId")[stats + ["numMinutes", "poss", "gp"]].sum().reindex(coh.index)
    # league rates per possession / per shot in each season, used to build the 'expected' line
    lg = ps.groupby("season")[stats + ["poss"]].sum()
    lg_per_poss = lg[list(RATE_STATS)].div(lg.poss, axis=0)
    lg_ts = lg.points / (2 * (lg.fieldGoalsAttempted + 0.44 * lg.freeThrowsAttempted))
    lg_3r = lg.threePointersAttempted / lg.fieldGoalsAttempted
    lg_ftr = lg.freeThrowsAttempted / lg.fieldGoalsAttempted
    ww = w[["personId", "season", "poss", "fieldGoalsAttempted", "freeThrowsAttempted"]].copy()
    exp = pd.DataFrame({c: ww.poss * ww.season.map(lg_per_poss[c]) for c in RATE_STATS}, index=ww.index)
    exp["tsa"] = 2 * (ww.fieldGoalsAttempted + 0.44 * ww.freeThrowsAttempted)
    exp["ts_points"] = exp.tsa * ww.season.map(lg_ts)
    exp["fg3a"] = ww.fieldGoalsAttempted * ww.season.map(lg_3r)
    exp["fta"] = ww.fieldGoalsAttempted * ww.season.map(lg_ftr)
    exp = exp.groupby(ww.personId).sum().reindex(coh.index)

    f = pd.DataFrame(index=coh.index)
    f["games"] = tot.gp.fillna(0)
    f["minutes"] = tot.numMinutes.fillna(0)
    f["mpg"] = tot.numMinutes / tot.gp.where(tot.gp > 0)
    poss = tot.poss.where(tot.poss > 0)
    for col, name in RATE_STATS.items():
        f[name] = 100 * (tot[col] - exp[col]) / poss          # per 100 possessions, above/below league
    fga = tot.fieldGoalsAttempted.where(tot.fieldGoalsAttempted > 0)
    tsa = exp.tsa.where(exp.tsa > 0)
    f["ts_vs_league"] = (tot.points - exp.ts_points) / tsa
    f["three_rate_vs_league"] = (tot.threePointersAttempted - exp.fg3a) / fga
    f["ft_rate_vs_league"] = (tot.freeThrowsAttempted - exp.fta) / fga

    by_cy = lambda col: w.pivot_table(index="personId", columns="cy", values=col).reindex(coh.index)
    gp_ = by_cy("gmsc_pct"); rp = by_cy("rate_pct"); mp = by_cy("mpg")
    f["seasons_played"] = gp_.notna().sum(axis=1)
    f["best_gmsc_pct"] = gp_.max(axis=1)
    f["last_gmsc_pct"] = gp_[h] if h in gp_ else np.nan
    f["best_rate_pct"] = rp.max(axis=1)
    f["last_rate_pct"] = rp[h] if h in rp else np.nan
    if h > 1:
        first = gp_[1] if 1 in gp_ else np.nan
        f["gmsc_pct_trend"] = f.last_gmsc_pct - first
        f["mpg_trend"] = (mp[h] if h in mp else np.nan) - (mp[1] if 1 in mp else np.nan)

    # small samples: per-possession rates on a handful of minutes are noise, so treat them as unknown
    small = f.minutes < cfg["data"]["min_minutes_for_rates"]
    rate_cols = list(RATE_STATS.values()) + ["ts_vs_league", "three_rate_vs_league", "ft_rate_vs_league"]
    f.loc[small, rate_cols] = np.nan

    bio = players.drop_duplicates("personId").set_index("personId").reindex(coh.index)
    birth = pd.to_datetime(bio.birthDate, errors="coerce")
    f["age_debut"] = (pd.to_datetime(coh.debut.astype(int).astype(str) + "-10-15") - birth).dt.days / 365.25
    f["height"] = pd.to_numeric(bio.heightInches, errors="coerce")
    f["weight"] = pd.to_numeric(bio.bodyWeightLbs, errors="coerce")
    f["draft_pick"] = coh.draft_pick
    f["undrafted"] = coh.undrafted
    f["is_guard"] = pd.to_numeric(bio.guard, errors="coerce").fillna(0)
    f["is_forward"] = pd.to_numeric(bio.forward, errors="coerce").fillna(0)
    f["is_center"] = pd.to_numeric(bio.center, errors="coerce").fillna(0)
    return f


def evidence_flag(X: pd.DataFrame, cfg: dict) -> pd.Series:
    """True where the player has too few NBA minutes for a reliable score."""
    return X.minutes < cfg["decision"]["evidence_min_minutes"]
