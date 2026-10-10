# Evaluation protocol

Evaluation follows the FEDS framework (Venable et al. 2016): artificial episodes de-risk the technical
design first, then naturalistic episodes test the artefact with real users in their real setting. The main
remaining risk is adoption, not accuracy (Dietvorst et al. 2015).

| Episode | Type | Status | Measures | Pass rule |
|---|---|---|---|---|
| E1 Prototype backtest (A2) | Artificial, formative | Done | Discrimination, calibration | Beats draft order (met) |
| E2 Construct comparison, recalibration, subgroup audit (v1.0) | Artificial, formative | **Done** (this release) | AUC with bootstrap CI, Brier, calibration error, subgroup gaps, both constructs | AUC gain over draft ≥ 0.05 with CI above 0 (met: +0.125, CI 0.075–0.179); judged subgroups within ±0.05 calibration (met; late first-round picks at the edge, +0.05) |
| E3 Scout blind review | Naturalistic, formative | Ready to run (kit below) | Agreement between evaluators with and without the brief | Agreement improves |
| E4 Shadow pilot | Naturalistic, formative | Ready to run | Usefulness rating, time to brief, override reasons | ≥ 70% rate the brief useful |
| E5 First live extension window | Naturalistic, summative | Planned | Decisions documented; outcomes over three seasons | Benefits register targets |

## E3 Scout blind review kit

**Purpose.** Test whether the brief makes evaluations more consistent. This is the "reduced evaluator
disagreement" benefit.

**Material.** 20 players from the 2015-18 replay, sampled by the analyst: 8 contributors and 12
non-contributors, at least 5 second-round or undrafted, at least 5 lottery picks. Identities are hidden
(name, team and draft slot removed; statistics shown as a plain table). Outcomes are not shown.

**Procedure.**
1. Round A (without the model): each of the 3–5 panel members rates each player 1–5 ("likelihood of becoming a
   high-impact contributor in years 4–8") from the statistics table and their own knowledge.
2. Two weeks later, Round B (with the brief): the same players are re-ordered and shown with the model brief
   (score, range, drivers, comparables). Panel members rate again and may write an override reason.
3. The analyst records time spent per player in each round.

**Analysis.**
* Consistency: compare inter-rater agreement (intraclass correlation) between Round A and Round B.
* Accuracy: AUC of each round's mean rating against the known outcomes.
* Value of judgement: for players where the panel overrode the model, who was right?
* Time: minutes per player, Round A against Round B.

**Pass rule (proposed, agreed at steering):** agreement improves and accuracy does not fall.

## E4 Shadow pilot

Run the nightly `score` step for four weeks before the trade deadline. The analyst prepares briefs for every player
the front office discusses. No decision depends on them yet. Collect a three-question usefulness rating after each
meeting ("useful / partly / not useful", "what was missing", "did it change the conversation"), the time to
prepare each brief, and every disagreement with its reason.
