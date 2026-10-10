"""Step 2b - Cohort and outcome construct ('high-impact contributor').

The construct is the single most consequential design decision in the project: it defines what the
model learns to predict. Two definitions are implemented so the steering committee can compare them.

volume (Assessment 2 prototype)
    A high-impact season is one whose total Game Score ranks in the league's top 20%. It rewards
    playing time, and playing time is partly a coaching decision, so early minutes 'predict' later
    minutes. That inflates accuracy and risks a self-fulfilling loop if the model influences minutes.

rate (recommended)
    A high-impact season is a rotation season (>= 1,000 minutes, scaled for short seasons) whose Game
    Score per 100 possessions ranks in the top third of rotation players. It still needs a real role,
    but it rewards what a player does with his possessions rather than how many he is given.

For both: a contributor has at least two high-impact seasons in career years 4-8. Players who leave
the league count as non-contributors, so the model does not only learn from survivors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def high_impact_flag(ps: pd.DataFrame, construct: str, cfg: dict) -> pd.Series:
    c = cfg["construct"]
    if construct == "volume":
        return ps.gmsc_pct >= 1 - c["volume"]["top_share"]
    if construct == "rate":
        return ps.rotation & (ps.rotation_rate_pct >= 1 - c["rate"]["top_share_of_qualified"])
    raise ValueError(f"unknown construct: {construct}")


def build_cohort(ps: pd.DataFrame, players: pd.DataFrame, cfg: dict, last_debut: int | None = None) -> pd.DataFrame:
    """One row per player who debuted in the modelling window, with position, draft tier and labels."""
    first = cfg["cohort"]["first_debut"]
    last = last_debut if last_debut is not None else ps.season.max()
    debut = ps.groupby("personId").debut.first()
    coh = debut[(debut >= first) & (debut <= last)].to_frame()

    bio = players.drop_duplicates("personId").set_index("personId").reindex(coh.index)
    from_year = pd.to_numeric(bio.fromYear, errors="coerce")
    coh = coh[~(from_year < cfg["data"]["exclude_if_bio_from_year_before"]).values]
    bio = bio.reindex(coh.index)

    coh["name"] = (bio.firstName.fillna("") + " " + bio.lastName.fillna("")).str.strip()
    g, f, cn = (pd.to_numeric(bio[k], errors="coerce").fillna(0) for k in ["guard", "forward", "center"])
    coh["position"] = np.select([cn == 1, (f == 1) & (g == 1), f == 1, g == 1],
                                ["Center", "Wing (G/F)", "Forward", "Guard"], "Unknown")
    pick = pd.to_numeric(bio.draftNumber, errors="coerce")
    dyear = pd.to_numeric(bio.draftYear, errors="coerce")
    undrafted = pick.isna() | (pick <= 0) | (dyear <= 0)
    coh["draft_pick"] = np.where(undrafted, 61, pick.clip(upper=60))
    coh["undrafted"] = undrafted.astype(int)
    coh["draft_tier"] = pd.cut(coh.draft_pick, [0, 14, 30, 61],
                               labels=["Lottery (1-14)", "Late 1st (15-30)", "2nd round / undrafted"]).astype(str)
    country = bio.country.fillna("Unknown").astype(str).str.strip()
    coh["origin"] = np.where(country.isin(["USA", "Unknown", ""]), np.where(country == "USA", "USA", "Unknown"),
                             "International")
    h = pd.to_numeric(bio.heightInches, errors="coerce")
    coh["height_band"] = pd.cut(h, [0, 76, 80, 84, 100], labels=["<=6'4\"", "6'5\"-6'8\"", "6'9\"-7'0\"", "7'1\"+"]).astype(str)
    coh.loc[h.isna().values, "height_band"] = "Unknown"
    coh["labelled"] = coh.debut <= cfg["cohort"]["last_labelled_debut"]

    lo, hi = cfg["construct"]["outcome_years"]
    need = cfg["construct"]["min_high_impact_seasons"]
    win = ps[(ps.cy >= lo) & (ps.cy <= hi) & ps.personId.isin(coh.index)]
    for name in ["volume", "rate"]:
        hits = win[high_impact_flag(win, name, cfg)].groupby("personId").size()
        y = (hits.reindex(coh.index).fillna(0) >= need).astype(int)
        coh[f"y_{name}"] = y.where(coh.labelled)
        coh[f"hi_seasons_{name}"] = hits.reindex(coh.index).fillna(0).astype(int)
    coh["y"] = coh[f"y_{cfg['construct']['active']}"]
    return coh


def split_masks(coh: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    s = cfg["split"]
    d = coh.debut
    return {
        "train": d.between(*s["train"]) & coh.labelled,
        "validate": d.between(*s["validate"]) & coh.labelled,
        "train_full": d.between(s["train"][0], s["validate"][1]) & coh.labelled,
        "test": d.between(*s["test"]) & coh.labelled,
        "all_labelled": coh.labelled,
    }
