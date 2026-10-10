# Runbook: operating the Early-Career Potential Model

Audience: the franchise analyst (business-as-usual owner) and the Head of Analytics. Each routine step is a
single command run from the repository root.

## Calendar

| When | Task | Command | Owner | Output checked by |
|---|---|---|---|---|
| Nightly in season (or weekly) | Refresh data and re-score | `python run_pipeline.py ingest` then `python run_pipeline.py score` | Analyst (scheduled job) | Analyst |
| Monthly in season | Monitoring review: read the Monitoring tab, act on any RED, and note AMBER items in the minutes | (dashboard) | Analyst | Steering committee |
| Before the trade deadline (early Feb) | Watchlist briefs for targets | `score`, then print briefs from the dashboard | Analyst | Assistant GM |
| After the regular season closes (Apr–May) | Off-season retrain and recalibrate with the newly labelled class | `ingest`, then edit `cohort.last_labelled_debut` (+1) and `split` (+1), then `all` | Analyst | Model Review Board |
| Before the extension window (July) | Release decision for the new version; refresh model card | (meeting) | Head of Analytics | Model Review Board |
| Extension window (July–Oct) | Brief for every eligible player; decisions logged | dashboard → Player brief → Log decision → Export | Assistant GM | GM |
| Season end | Review overrides against outcomes; benefits register | export decision logs | Analyst | Steering committee |

## Monitoring checks and what to do

| Check | GREEN | AMBER | RED | Action on RED |
|---|---|---|---|---|
| Field integrity (games-started flag) | All seasons 40–55% | — | Any season outside | Field stays excluded. Raise a ticket with the data provider. Do **not** add the field back as a feature |
| Data drift on top-4 features (PSI vs empirical levels) | All inside the 75th percentile of history | Above the 75th | Above the 95th | Check calibration on the newest labelled classes; if they are also off, retrain at the next off-season; note in the model card |
| Data drift on any monitored feature | — | — | — | Investigate at the monthly review. Low-influence features need no action beyond a note |
| Calibration gap, newest labelled classes | under 0.025 | 0.025–0.05 | over 0.05 | Recalibrate (rerun `all`); if still over 0.05, escalate to the Model Review Board |
| Base-rate drift (last 4 classes vs all) | under 0.02 | 0.02–0.05 | over 0.05 | Steering committee reviews whether the construct thresholds still mean "high impact" |
| Champion–challenger disagreement | under 10% of players | 10–20% | over 20% | Analyst reviews disagreeing players before any brief goes out; investigate the cause before the next release |

Monitoring follows the *detect → understand → adapt* cycle for concept drift (Lu et al. 2019): the checks
detect, the analyst investigates and documents the cause, and the off-season retrain adapts.

## Off-season retrain step by step

1. `python run_pipeline.py ingest`. Check `data/processed/ingest_report.json`: row counts are similar to last
   year's and `rows_team_id_unresolved` is 0.
2. In `config.yaml`, add one year to `cohort.last_labelled_debut` and to every `split` boundary, and set a new `version`.
3. `python run_pipeline.py all`, then `python -m pytest -q` (all tests must pass).
4. Compare the new `outputs/metrics.json` with the previous release. The model must still beat draft order
   (the AUC-gain interval stays above 0) and calibration must stay within 0.05.
5. Update `docs/model_card.md` and `docs/CHANGELOG.md`, and take both to the Model Review Board.
6. **Rollback:** keep the previous release folder (`models/`, `outputs/`, `config.yaml`) with its version
   number. If a release is rejected or misbehaves, restore that folder and run `python run_pipeline.py score`.

## Changing a business setting

The steering committee owns the outcome construct (`construct.active`) and the cost ratio
(`decision.cost_ratio`). A change goes through minutes, then `config.yaml`, then `all`, then a Model Review
Board sign-off, then the new version is used. The threshold is never edited by hand; it always follows from
the cost ratio.

## Incidents

| Incident | First response |
|---|---|
| A score is challenged by a scout or coach | Open the brief. Check the drivers and comparables. Log the override with its reason. The GM decides |
| A player, agent or the media asks about a score | Scores are confidential and not shared. Refer to Player Relations/Legal |
| Pipeline fails | Run `python -m pytest -q` to find the failing step. Restore the previous release if a decision is due |
| `score` refuses to load a model (hash mismatch) | Treat as a security incident: do not bypass. Restore the model files from the release folder and tell the Head of Analytics |
| Suspected bias against a group | Model Review Board can suspend use immediately. Re-run the subgroup audit and document the outcome |

## Hand-over checklist (consultant → franchise)

- [ ] Analyst has run `ingest`, `score` and `all` unassisted and explained each monitoring check
- [ ] Analyst can change the construct and cost ratio and explain the effect using the Threshold explorer
- [ ] Model card, runbook and data dictionary reviewed together
- [ ] Previous-release folder convention agreed; first rollback rehearsed
- [ ] Three months of support scheduled
