# Early-Career Potential Model (v1.0.0)

Decision support for an NBA front office. After a player's first one, two or three seasons, the model
estimates the probability that he becomes a **high-impact contributor** in career years 4–8. It gives an
uncertainty range, the statistics behind the score and the most similar historical players. The output
feeds rookie-scale extension, trade-deadline and development decisions as a **screening input**. People
make the decisions.

> Prepared by Changa Fernando (S3842381) for ISYS3419 Analytics and AI in Practice, Assessment 3.
> Client: NBA franchise front office (Deloitte case study). Everything here runs on public box-score data.

---

## What is in the artefact

| Component | Where | What it is for |
|---|---|---|
| Reproducible pipeline | `run_pipeline.py`, `ecp/` | Raw box scores to features, model selection, backtest, training, scoring, monitoring |
| Registered models | `models/` | Champion (logistic regression, Platt-calibrated), challenger (random forest), uncertainty ensemble, `registry.json` |
| Interactive dashboard | `dashboard/ecp_dashboard.html` | Watchlist, player briefs, replay of 2015-18, threshold explorer, business case, model evidence, release gates and monitoring, governance |
| Decision briefs | `outputs/briefs/` | One-page printable briefs (live extension-window class and historical replays) |
| Results | `outputs/` | `metrics.json` (incl. business case and release gates), model-selection and model-comparison tables, backtest predictions, fairness audit, drift report, figures |
| Walkthrough notebook | `notebooks/walkthrough.ipynb` (+ `.html`) | The build and evaluation story, step by step, with outputs |
| Documentation | `docs/` | Model card, runbook, data dictionary, user guide, evaluation protocol, changelog |
| Tests | `tests/` | 23 unit and integration tests (leakage, construct logic, thresholds, output rules) |

## Quick start

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt
python run_pipeline.py all        # about 1 minute on a laptop: select, backtest, train, score, monitor, build dashboard
open dashboard/ecp_dashboard.html # or double-click it; it works offline in any modern browser
python -m pytest -q               # 23 tests
```

The processed data ships in `data/processed/`, so the raw dataset is **not** needed to run the model.
To rebuild the processed data from the raw files (for example after a data refresh):

```bash
python run_pipeline.py ingest     # reads PlayerStatistics.csv, TeamStatistics.csv, Players.csv from paths.raw_dir
```

### Commands

| Command | When | What it does | Time |
|---|---|---|---|
| `python run_pipeline.py ingest` | After each data refresh | Raw CSVs to `data/processed/` and an ingest data-quality report | ~10 s |
| `python run_pipeline.py score` | Nightly or monthly | Re-scores active young players with the **registered** models, re-runs monitoring and rebuilds briefs and dashboard. No retraining | ~3 s |
| `python run_pipeline.py all` | Off-season retrain, or after a config change | Full rebuild: selection, backtest, training, registry, scoring, monitoring, figures, dashboard | ~1 min |
| `python run_pipeline.py report` | Presentation tweaks | Rebuilds figures, briefs and dashboard from saved outputs | ~2 s |
| `python run_pipeline.py all --fast` | Quick check | As `all` with fewer bootstrap resamples | ~40 s |

## Results snapshot (untouched 2015-18 debut classes, 385 players, 46 contributors)

The model was trained on the 1996-2014 classes and scored the 2015-18 classes once.

| After 3 seasons | Model | Draft order only |
|---|---|---|
| ROC-AUC (95% CI) | **0.96** (0.94–0.98) | 0.83 (0.77–0.89) |
| PR-AUC | **0.76** | 0.44 |
| Brier score (lower is better) | **0.050** | 0.084 |
| Contributors in a top-10% shortlist (38 players) | **27** | 19 |
| Contributors missed by a top-20% shortlist | **4 of 46** | 18 of 46 |
| At the 3:1 cost threshold (0.25): caught / missed / false flags | **41 / 5 / 29** | 27 / 19 / 46 |
| Net benefit per 100 players at 0.25 | **8.1** | 3.0 |

After one season the AUC is 0.90 and after two it is 0.95; the gain over draft order is positive in at
least 99.7% of bootstrap resamples at every horizon. The full results are in `outputs/metrics.json` and the dashboard.

## How it works

```
1 Ingest -> 2 Prepare -> 3 Select & test -> 4 Train & register -> 5 Score & explain -> 6 Monitor
   ^                                                                                   |
   +--------------- off-season retrain / recalibrate / roll back (Model Review Board) --+
```

1. **Ingest** (`ecp/ingest.py`, pandas only). Regular-season games are identified by game-ID prefix
   `002`, and the season is read from the game ID. Missing 2021-22 team IDs are restored. Team possessions
   are estimated for pace adjustment.
2. **Prepare** (`ecp/seasons.py`, `ecp/constructs.py`, `ecp/features.py`). Production is measured per
   100 possessions **against the league average in the same seasons**, plus season percentiles, so eras are
   comparable. The outcome construct and the 32 features use career years 1..h only.
3. **Select & test** (`ecp/models.py`, `ecp/pipeline.py`). Logistic regression, random forest and
   gradient boosting are each tried with no, Platt or isotonic calibration. Selection uses average
   validation Brier score (2011-14 classes) across horizons. The test classes (2015-18) are never used for
   fitting or selection.
4. **Train & register**. The champion and challenger are refit on all labelled classes (1996-2018),
   along with a 40-model bootstrap ensemble for player-level uncertainty ranges. Data hashes and
   settings are recorded in `models/registry.json`.
5. **Score & explain** (`ecp/explain.py`). Each player gets a probability, an 80% range, the challenger's
   view, his top drivers up and down, five comparables and a status. Under 500 minutes the status is
   *Insufficient evidence*.
6. **Release gates and business case** (`ecp/release.py`, `ecp/business_case.py`). Five quantified gates
   (beats draft order, calibrated overall and by subgroup, no group missed far more often, champion and challenger
   agree), each with a corrective action, plus a model comparison on accuracy, calibration and complexity, and the
   cost and payback of the decisions the model supports.
7. **Monitor** (`ecp/monitoring.py`). Checks field integrity, feature drift against empirical alarm
   levels, calibration, base-rate drift and champion–challenger disagreement, each with a runbook action.

## Business choices live in `config.yaml`

| Setting | Default | Who owns it |
|---|---|---|
| `construct.active` | `rate` (or `volume`, the Assessment 2 prototype definition) | Steering committee |
| `decision.cost_ratio` | 3 (missing a contributor costs 3x a wasted offer, so the threshold is 0.25) | Steering committee, with finance |
| `decision.evidence_min_minutes` | 500 | Model Review Board |
| `split`, `horizons`, `cohort` | Forward-chaining by debut class | Head of Analytics |
| `release_gates` | AUC gain ≥ 0.05; calibration gap ≤ 0.05; subgroup gap watch 0.05 / fail 0.08; miss-rate excess ≤ 0.15 | Model Review Board |
| `business_case` | Illustrative USD assumptions (day rate, staff costs, cost of a missed contributor) | Finance |

Change a value and run `python run_pipeline.py all`. Every model, metric, brief and dashboard view updates.

## Using the dashboard

* **Overview**: headline evidence, the current extension-window class and model health.
* **Watchlist**: active players in their first three seasons. Filter by class or status, search by name, export to CSV.
* **Player brief**: probability and range, drivers, comparables, and a scout/coach note and decision log
  (an override requires a reason). The log stays in your browser and can be exported. Print to PDF for meetings.
* **Replay 2015-18**: the same view for classes whose outcomes are now known. Use it to build trust and to train new users.
* **Threshold explorer**: move the cost ratio to see contributors caught, missed and false flags.
* **Business case**: build and running costs, error cost per extension decision (model vs draft order), payback,
  and analyst capacity. All prices are illustrative assumptions in `config.yaml` → `business_case`; move the
  slider to price a missed contributor differently.
* **Model evidence**, **Monitoring**, **About & governance**: the technical case, health checks and decision rights.

## Data and data-quality notes

The data comes from the franchise's NBA dataset: `PlayerStatistics.csv`, `TeamStatistics.csv` and `Players.csv`, joined on
`personId` and `teamId`. The ingest step found and handled four defects (details in `docs/data_dictionary.md`):

1. `gameType` is blank for 2021-22 team rows, so regular season is identified from the game ID.
2. `playerteamId` is blank for 2021-22 and half of 2000-01, so it is restored from the team city and name.
3. Calendar-month season rules misplace the August 2020 bubble games, so the season comes from the game ID.
4. The **games-started flag changes meaning**: it is set for about 90% of appearances in 1996-2016 and about 47% from
   2017, and it is blank in 2021-22. The Assessment 2 prototype used it as a feature. It is now excluded, and a
   monitoring check watches it.

## Responsible use

* Use the model for screening and prioritisation only. Every decision has a named human owner and a logged rationale.
* Scores are confidential to the front office, expire each season and are never shared with players, agents or other teams.
* Country of origin is audited but never used as an input. Fairness results are in the Model evidence tab and
  `outputs/subgroup_audit_h3.csv`.
* Known limits: no injury, salary, tracking or scouting data; box-score Game Score under-credits defensive
  specialists; small guards and late bloomers are missed more often. See `docs/model_card.md`.

## Repository layout

```
early-career-potential/
├── README.md               this file
├── config.yaml             business and technical settings
├── requirements.txt
├── run_pipeline.py         command-line entry point
├── ecp/                    the package: ingest, seasons, constructs, features, models,
│                           evaluation, explain, pipeline, monitoring, release, business_case,
│                           reporting, dashboard
├── data/processed/         processed inputs (ships with the repo) + ingest_report.json
├── models/                 registered models + registry.json
├── outputs/                metrics, tables, figures/, briefs/
├── dashboard/              ecp_dashboard.html (self-contained)
├── notebooks/              walkthrough.ipynb and walkthrough.html
├── docs/                   model_card, runbook, data_dictionary, user_guide, evaluation_protocol, CHANGELOG
└── tests/                  pytest suite
```

Tested with Python 3.11, pandas 3.0, scikit-learn 1.9, numpy 2.4, shap 0.51 and matplotlib 3.11. Saved `.joblib`
models are tied to the scikit-learn version, which `models/registry.json` records; `score` warns when yours differs.
If it does, run `python run_pipeline.py all` to retrain locally.
