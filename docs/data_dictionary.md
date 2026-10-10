# Data dictionary and data-quality log

## Sources (franchise NBA dataset)

| File | Grain | Used for |
|---|---|---|
| `PlayerStatistics.csv` | player × game (1.67 million rows, 1946–2026) | Box-score totals, minutes, games, plus-minus |
| `TeamStatistics.csv` | team × game | Team possessions and pace (per-100-possession rates) |
| `Players.csv` | player (6,692 rows) | Birth date (age at debut), height, weight, listed positions, draft year and pick, country (audit only) |

## Processed tables (`data/processed/`)

| File | Grain | Key columns |
|---|---|---|
| `player_team_seasons.csv.gz` | player × season × team, regular season, 1990–2025 | `personId`, `season` (start year), `teamId`, summed box score, `gp` (games with minutes > 0), `rows` |
| `team_seasons.csv` | team × season | `games`, `team_minutes`, `poss` (FGA + 0.44·FTA − OREB + TOV), `pace` (possessions per 48 minutes) |
| `players.csv` | player | biographical columns listed above |
| `ingest_report.json` | run | row counts, game-type crosstab, data-quality counters |

## Derived concepts

| Term | Definition |
|---|---|
| Season | Start year of the NBA season (2025 = 2025-26), read from digits 4–5 of the game ID |
| Debut / debut class | First season with at least one game played (minutes > 0). Classes are by debut, not draft year: a player drafted in 2022 who first played in 2023-24 belongs to the 2023 class |
| Career year (cy) | season − debut + 1. A missed season still counts as a year |
| Possessions (player) | minutes × team possessions per player-minute that season (team-specific pace) |
| Game Score | Hollinger: PTS + 0.4·FGM − 0.7·FGA − 0.4·(FTA − FTM) + 0.7·OREB + 0.3·DREB + STL + 0.7·AST + 0.7·BLK − 0.4·PF − TOV |
| Rotation season | minutes ≥ 1,000 × (team games that season ÷ 82) |
| High-impact season (rate) | rotation season with Game Score per 100 possessions in the top third of rotation players that season |
| High-impact season (volume) | season Game Score total in the top 20% of all players who played |
| Contributor | at least 2 high-impact seasons in career years 4–8; players who leave count as 0 |
| Labelled | debut ≤ 2018 (career year 8 = 2025-26 observed) |

## Features (horizon h uses career years 1..h only)

| Feature | Meaning |
|---|---|
| `games`, `minutes`, `mpg`, `seasons_played` | Opportunity in the window |
| `pts_100` … `gmsc_100` | Per 100 possessions, **minus the league rate in the same seasons** (era-relative) |
| `ts_vs_league` | True shooting minus league true shooting in the same seasons |
| `three_rate_vs_league`, `ft_rate_vs_league` | Three-point and free-throw attempts per shot, minus the league's |
| `best_gmsc_pct`, `last_gmsc_pct` | Best and latest season percentile of total Game Score among all players |
| `best_rate_pct`, `last_rate_pct` | Best and latest season percentile of Game Score per 100 among players with 250+ minutes |
| `gmsc_pct_trend`, `mpg_trend` | Latest minus first season (h ≥ 2) |
| `age_debut`, `height`, `weight` | Biography (age at 15 October of the debut season) |
| `draft_pick`, `undrafted` | Pick 1–60; 61 = undrafted |
| `is_guard`, `is_forward`, `is_center` | Listed positions (a player can have more than one) |

Rates and efficiency are set to missing below 100 minutes. Missing values are imputed with the training
median, with a missing flag the model can use.

## Data-quality log

| # | Issue found | Evidence | Treatment |
|---|---|---|---|
| DQ1 | `gameType` blank for 2021-22 in `TeamStatistics.csv` | only 2 of 2,460 regular-season team rows labelled, so a 'Regular Season' filter drops the whole season | Regular season identified by game-ID prefix `002` in both files |
| DQ2 | `playerteamId` blank for 2021-22 (99.9%) and 2000-01 (50%) | 45,935 rows | Restored from (`playerteamCity`, `playerteamName`) using the mapping seen in other seasons; 0 unresolved |
| DQ3 | Month-based season rule misplaces the August 2020 bubble games | 2,050 player rows | Season read from the game ID |
| DQ4 | Games-started flag changes meaning over time | about 90% of appearances flagged 1996–2016, about 47% from 2017, 0% in 1990–95 and 2021-22 | Excluded from features. The A2 prototype used it, and current players would have looked like bench players to a model trained on 1996–2016. Monitored as a field-integrity check |
| DQ5 | Minutes missing on 146,318 regular-season rows since 1990 | all have 0 points (did-not-play rows) | Kept as rows, not counted as games played |
| DQ6 | NBA Cup final (prefix `006`) labelled 'Regular Season' | 27 rows | Excluded (it does not count towards season statistics). Cup group games (prefix `002`) are kept |
| DQ7 | Placeholder rows (teamId 0) for unplayed games | 2 team rows in 2025-26 | Dropped |
| DQ8 | Height and weight missing for 24% of all players, but under 1% of the modelling cohort | `Players.csv` | Median imputation with missing flag |
| DQ9 | Zero-filled legacy statistics (steals before 1973, plus-minus before 1996) | A1 data profile | Modelling cohort starts with the 1996 debut class |
