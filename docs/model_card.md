# Model card: Early-Career Potential Model v1.0.0

| | |
|---|---|
| **Owner** | Head of Analytics (business-as-usual owner: franchise analyst) |
| **Approver** | Model Review Board: GM (chair), Head of Analytics, Director of Scouting, Player Relations/Legal |
| **Status** | Candidate, awaiting Model Review Board approval for the shadow pilot |
| **Version** | 1.0.0, trained on debut classes 1996-2018; data through the 2025-26 season |
| **Registry** | `models/registry.json` (settings, data hashes, model-file hashes checked before every load) |

## 1. What the model does

It estimates the probability that a player becomes a **high-impact contributor**, defined as at least two
**high-impact seasons** in career years 4–8. It uses only what is known after his first 1, 2 or 3 NBA
seasons. There is one model per horizon (h = 1, 2, 3).

**High-impact season (rate construct, active):** a rotation season (at least 1,000 minutes, scaled for
shortened seasons) in which Game Score per 100 possessions ranks in the top third of rotation players that
season. Players who leave the league count as non-contributors.

The alternative **volume construct** (season Game Score total in the league's top 20%, the Assessment 2
prototype definition) can be selected in `config.yaml`. Both are reported in `outputs/metrics.json`.

## 2. Intended use

* Screening and prioritisation before rookie-scale extension talks, the trade deadline and development planning.
* One input to a decision that a named person owns, alongside scout and coach judgement, with the rationale logged.
* Building consistent briefs: score, uncertainty, drivers and historical comparables in one page.

**Out of scope:** automated decisions; contract valuation (no salary data); injury or medical judgement;
public statements; sharing with players, agents or other teams; players with under 500 NBA minutes (shown as
*Insufficient evidence*).

## 3. Model

| | Champion | Challenger |
|---|---|---|
| Algorithm | Logistic regression (L2, C = 0.3), median imputation with missing-value flags, standardised inputs | Random forest (400 trees, min leaf 5) |
| Calibration | Platt (sigmoid), fitted on 5-fold out-of-fold training predictions | None (best uncalibrated tree model) |
| Why | Lowest average validation Brier score across the three horizons (0.0750) and the highest validation AUC (0.920). It is also the most transparent and the easiest for the franchise to maintain | Different model family, used to cross-check scores. Large disagreement triggers analyst review |
| Explanations | Exact additive log-odds contributions per feature (SHAP for a linear model) | TreeSHAP (global importance cross-check) |
| Uncertainty | 40 bootstrap refits; the 10th–90th percentile spread is placed around the champion's score | — |

**Inputs (32 at h = 3):** opportunity (games, minutes, minutes per game, seasons played); production per
100 possessions **relative to the league in the same seasons** (points, assists, offensive and defensive
rebounds, steals, blocks, turnovers, shot attempts, fouls, plus-minus, Game Score); efficiency relative to the
league (true shooting, three-point share, free-throw rate); season percentiles (best and latest, total impact
and per possession) and trends; and age at debut, height, weight, draft pick, undrafted flag and listed
positions. **Excluded:** country of origin (audit only) and the games-started flag (unreliable across seasons).

## 4. Data

* Training: 1,718 players who debuted 1996–2018 (base rate 14.5%). The 2018 class is the last with an observed career year 8.
* Validation for model selection: 2011–14 classes (319 players). Test: 2015–18 classes (385 players, 46 contributors), never used for fitting or model selection.
* Source: franchise NBA dataset (`PlayerStatistics.csv`, `TeamStatistics.csv`, `Players.csv`), regular season only. See `docs/data_dictionary.md`.

## 5. Performance (test classes 2015–18; trained on 1996–2014)

| Horizon | ROC-AUC (95% CI) | PR-AUC | Brier | Draft-order AUC | AUC gain (95% CI) |
|---|---|---|---|---|---|
| 1 season | 0.895 (0.838–0.944) | 0.64 | 0.066 | 0.834 | +0.061 (0.021–0.105) |
| 2 seasons | 0.947 (0.919–0.971) | 0.74 | 0.055 | 0.834 | +0.113 (0.067–0.162) |
| 3 seasons | 0.959 (0.937–0.978) | 0.76 | 0.050 | 0.834 | +0.125 (0.075–0.179) |

* **Calibration (h = 3):** mean prediction 13.2% against 11.9% observed. Scores of 0.5 and above match outcomes closely (0.60 → 0.57; 0.88 → 0.88).
* **Extension-decision group** (still in the league in year 3, n = 232): AUC 0.94 against 0.78 for draft order; PR-AUC 0.77 against 0.46.
* **Harder comparison** (players with six or more seasons, the population used by Milošević et al. 2026, n = 154): AUC 0.89 against 0.71 for draft order. The headline AUC is higher partly because players who wash out early are easy to rank.
* **At the 3:1 cost ratio (threshold 0.25):** 41 of 46 contributors caught, 29 false flags. Draft order catches 27 with 46 false flags.
* **Construct comparison (h = 3):** rate AUC 0.959, Brier 0.050, calibration error 0.014; volume AUC 0.954, Brier 0.057, calibration error 0.023.

## 5b. Model comparison and release gates

The three learners score within 0.006 AUC and 0.003 Brier of each other on the test classes. The champion
was chosen on validation Brier and on simplicity (`outputs/model_comparison_h3.csv`):

| Model | Calibration | Validation Brier | Test AUC | Test Brier | Calibration error | Size |
|---|---|---|---|---|---|---|
| Logistic regression (champion) | Platt | 0.0750 | 0.959 | 0.050 | 0.014 | 54 coefficients, 3 KB |
| Random forest (challenger) | none | 0.0770 | 0.954 | 0.050 | 0.023 | 34,200 tree nodes, 697 KB |
| Gradient boosting | Platt | 0.0775 | 0.953 | 0.048 | 0.016 | 3,516 tree nodes, 97 KB |
| Draft order only | none | — | 0.834 | 0.084 | 0.023 | 3 coefficients |

Release gates for v1.0.0 (`config.yaml` → `release_gates`, results in `metrics.json`): beats draft order PASS
(+0.125, CI 0.075–0.179); calibrated overall PASS (+0.012); calibrated by subgroup **WATCH** (late first-round
picks +0.0504, so they get a scout review band and are reported to the Model Review Board); no group missed far
more often PASS (worst: second round / undrafted, +0.141 against 0.15); champion–challenger agreement PASS (5.6%).

## 6. Fairness audit (h = 3, threshold 0.25; full table in `outputs/subgroup_audit_h3.csv`)

| Group | n | Contributors | AUC | Predicted vs observed | Missed |
|---|---|---|---|---|---|
| Centers | 70 | 20 | 0.92 | 33% vs 29% | 5% |
| Forwards | 106 | 7 | 0.95 | 7% vs 7% | 14% |
| Guards | 157 | 16 | 0.97 | 10% vs 10% | 19% |
| International | 90 | 15 | 0.96 | 18% vs 17% | 7% |
| USA | 295 | 31 | 0.96 | 12% vs 11% | 13% |
| 6'4" and under | 110 | 13 | 0.96 | 11% vs 12% | 23% |
| 2nd round / undrafted | 264 | 12 | 0.96 | 5% vs 5% | 25% |

Ranking quality is similar across groups and every judged group is at or within the ±0.05 calibration tolerance (late first-round picks sit at the edge, +0.05).
**Watch-points:** small guards and late picks are missed more often (for example Fred VanVleet at 0.19 and
Norman Powell at 0.05), and centres are flagged more generously. Part of this comes from the construct
itself, because Game Score credits rebounds and blocks heavily. Mitigation: any guard or late pick scoring
0.10–0.25 gets a scout review, and the steering committee will test a position-relative construct in v1.1.

## 7. Known limitations

* Box-score data only: no injuries, medical, tracking, salary or scouting inputs. Robert Williams III (0.92 in the 2018 replay, injured) shows why.
* Game Score under-credits defence. Mikal Bridges and OG Anunoby are contributors under the volume construct but not under the rate construct.
* Late bloomers with few early minutes are invisible to any early-career model (Christian Wood, 238 minutes). They are shown as *Insufficient evidence*, not as low scores.
* The contributor base rate is drifting down (15.6% for the 1996–2010 classes, 14.1% for 2011–14, 11.9% for 2015–18), and the game keeps changing (three-point share of shots 21% in 1996-97, 42% in 2025-26). Off-season retraining is mandatory.
* Test positives are few (46), so treat the intervals, not the point estimates, as the evidence.

## 8. Ethical considerations

Early labels can follow a player and become self-fulfilling. Morgulev and Azar (2026) call this digital
determinism. The controls are: screening-only use; scores expire each season; no external sharing;
override with a recorded reason; Player Relations sits on the Model Review Board; a subgroup audit runs at every release; and
the rate construct, which reduces the loop in which early minutes predict later minutes.

## 9. Maintenance

See `docs/runbook.md`. In short: score nightly or monthly with registered models; review monitoring monthly;
retrain and recalibrate each off-season; every release goes through the Model Review Board with this card updated.
