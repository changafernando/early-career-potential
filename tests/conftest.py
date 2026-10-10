"""Shared fixtures: a tiny synthetic league so unit tests run in seconds without the real data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecp.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture()
def toy_league(cfg):
    """Six players over 1996-2008 on two teams, with known outcomes."""
    rng = np.random.default_rng(0)
    rows = []
    careers = {  # personId: (debut, last season, minutes per season, scoring per minute)
        1: (1996, 2008, 2400, 0.70),   # long, efficient career -> contributor
        2: (1996, 1998, 300, 0.30),    # washes out after 3 seasons -> non-contributor
        3: (1997, 2008, 1800, 0.40),   # long but inefficient -> not a rate contributor
        4: (1997, 2008, 2600, 0.65),
        5: (1996, 2008, 1500, 0.35),
        6: (1997, 2008, 2000, 0.45),
    }
    for pid, (d, last, mins, ppm) in careers.items():
        for s in range(d, last + 1):
            pts = mins * ppm
            rows.append(dict(personId=pid, season=s, teamId=100 + pid % 2, numMinutes=mins, points=pts,
                             assists=mins * 0.05, blocks=mins * 0.01, steals=mins * 0.02,
                             fieldGoalsAttempted=pts / 1.1, fieldGoalsMade=pts / 2.3,
                             threePointersAttempted=pts * 0.1, threePointersMade=pts * 0.035,
                             freeThrowsAttempted=pts * 0.2, freeThrowsMade=pts * 0.15,
                             reboundsDefensive=mins * 0.12, reboundsOffensive=mins * 0.04, reboundsTotal=mins * 0.16,
                             foulsPersonal=mins * 0.05, turnovers=mins * 0.04, plusMinusPoints=rng.normal(0, 50),
                             started=0, gp=max(1, int(mins / 30)), win=20, rows=82))
    pts = pd.DataFrame(rows)
    teams = pd.DataFrame([dict(teamId=t, season=s, games=82, team_minutes=82 * 240, poss=82 * 95.0, fga=0, fg3a=0,
                               points_for=0, points_against=0, pace=95.0)
                          for t in (100, 101) for s in range(1996, 2009)])
    players = pd.DataFrame([dict(personId=p, firstName=f"P{p}", lastName="Test", birthDate="1975-01-01", country="USA" if p % 2 else "France",
                                 heightInches=78, bodyWeightLbs=220, guard=1, forward=0, center=0, draftYear=1996,
                                 draftRound=1, draftNumber=p * 5, fromYear=1996, toYear=2008) for p in careers])
    return pts, teams, players
