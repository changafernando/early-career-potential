# Changelog

## Unreleased

* Player briefs show best and latest season rank once when they are the same number (every one-season player),
  with their effects combined, instead of listing one fact as two drivers. Driver order is now stable, so
  `score` no longer rewrites unchanged outputs.
* `models/registry.json` records the scikit-learn, numpy, pandas and joblib versions used for training, and `score`
  warns before loading the models when the current environment differs. Models retrained with Python 3.11 and
  scikit-learn 1.9.1, so the registry now records them.
* Tests: an end-to-end `ingest` test on a tiny raw sample (bubble game, 2021-22 blank game type and team ID,
  playoff and NBA Cup final rows, did-not-play row), and a smoke test that runs `score` on a temporary copy.

## v1.0.0 (Assessment 3): production-ready artefact

**Design changes since the Assessment 2 prototype (v0.1)**

| Area | v0.1 prototype (A2) | v1.0.0 (A3) | Why |
|---|---|---|---|
| Outcome construct | Volume only (top 20% total Game Score) | Rate (recommended) and volume, set in config | A2 promised to test a stricter construct; rate rewards production per possession and weakens the minutes-predict-minutes loop |
| Data scope | Calendar-month season rule; player file only | Season and regular season read from the game ID; team pace added, with 2021-22 game types and team IDs repaired | DQ3: the month rule moved the August 2020 bubble games into 2020-21. DQ1–DQ2 had to be fixed before team data could be used |
| Features | Per-36 raw rates, started share | Per-100-possession rates relative to the league; era-relative efficiency; started flag removed | Case-study challenge: comparing eras and pace. DQ4: the started flag changes meaning in 2017 |
| Model | Random forest (selected on validation Brier at h = 3) | Logistic regression + Platt calibration as champion; random forest as challenger | Best average validation Brier across horizons; simpler, more transparent and easier to maintain |
| Threshold | Searched on the backtest | 1 / (1 + cost ratio) from calibrated probabilities | Business sets the ratio; no tuning on test data |
| Uncertainty | Bootstrap CI on AUC only | AUC, PR-AUC and Brier CIs, paired gain over draft order, and per-player 80% ranges | A2 feedback: report uncertainty |
| Low-minute players | Scored like everyone else | "Insufficient evidence" below 500 minutes | Late bloomers (e.g. Christian Wood) |
| Explanations | Global SHAP | Per-player drivers and five historical comparables in every brief | Scouts need to interrogate a score |
| Fairness | Position AUC | Calibration gap, miss rate and false-flag rate by position, draft tier, origin and height | Responsible deployment |
| Monitoring | Planned | Implemented: field integrity, PSI with empirical alarm levels, calibration, base rate, champion–challenger | Sustainability after hand-over |
| Release gates | Proposed stop/go criteria | Five quantified gates with corrective actions, checked on every run | A2 feedback (C4): quantitative thresholds and contingencies |
| Model comparison | AUC and Brier by model | Accuracy, calibration error and complexity (parameters, file size, fit time, explanation) | A2 feedback (C3): compare calibration and complexity where accuracy is similar |
| Business case | Benefits register, no costs | Build and running costs, error cost per extension decision vs draft order, payback with 90% range, sensitivity, analyst capacity | A2 feedback (C2, C4, C5): investment summary and financial consequences |
| Delivery | Scripts | Package, CLI (`ingest`, `score`, `all`, `report`), registry, dashboard, briefs, tests, docs | Usable and maintainable by the franchise |

**Headline result (2015-18 test classes, three seasons of evidence):** AUC 0.959 (95% CI 0.937–0.978)
against 0.834 for draft order; Brier 0.050; 41 of 46 contributors caught at the 0.25 threshold.
(v0.1 reported AUC 0.952 on the volume construct.)

## v0.1 (Assessment 2): prototype

Volume construct, random forest, 2015-18 backtest AUC 0.95. Evidence for project approval.
