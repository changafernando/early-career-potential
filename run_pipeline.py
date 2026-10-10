#!/usr/bin/env python3
"""Early-Career Potential Model - command-line entry point.

Usage (from the repository root):
    python run_pipeline.py ingest      # raw NBA CSVs -> data/processed (needs the raw dataset)
    python run_pipeline.py all         # model selection, backtest, training, scoring, monitoring,
                                       # figures, decision briefs and the dashboard
    python run_pipeline.py score       # nightly/monthly run: re-score with the REGISTERED models, re-run
                                       # monitoring and rebuild the dashboard (no retraining)
    python run_pipeline.py report      # rebuild figures, briefs and dashboard from saved outputs (seconds)
    python run_pipeline.py all --config config.yaml --fast   # fewer bootstrap resamples (quick check)
"""
from __future__ import annotations

import argparse
import json
import time

from ecp.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ingest", "all", "score", "report"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--fast", action="store_true", help="200 bootstrap resamples and 10 uncertainty models")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.fast:
        cfg["evaluation"]["bootstrap_resamples"] = 200
        cfg["evaluation"]["uncertainty_models"] = 10

    if a.step == "ingest":
        from ecp.ingest import run_ingest
        raw = cfg["paths"]["raw_dir"]
        if not (raw / "PlayerStatistics.csv").exists():
            raise SystemExit(f"Raw data not found in {raw}.\nSet paths.raw_dir in config.yaml to the folder holding "
                             "PlayerStatistics.csv, TeamStatistics.csv and Players.csv. (The processed data already "
                             "ships in data/processed, so every other step runs without it.)")
        print(json.dumps(run_ingest(cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"], cfg["data"]["first_season"]), indent=2))
        return

    from ecp import pipeline as pl
    from ecp.monitoring import run_monitoring
    from ecp.reporting import build_figures, write_briefs
    from ecp.dashboard import build_dashboard

    t0 = time.time()
    out = cfg["paths"]["outputs_dir"]
    out.mkdir(parents=True, exist_ok=True)

    if a.step == "score":
        metrics = json.loads((out / "metrics.json").read_text())
        D = pl.prepare(cfg)
        M, reg = pl.load_registered(cfg)
        if reg["data_hash"] != D["data_hash"]:
            print("note: processed data changed since the models were trained (expected after a data refresh)")
        live = pl.score_live(cfg, D, M)
        sc = {"live": live, "replay": json.loads((out / "replay_2015_2018.json").read_text()), "latest_season": D["latest_season"]}
        metrics["monitoring"] = run_monitoring(cfg, D, metrics["backtest"], live, metrics["drivers"])
        metrics["latest_season"] = D["latest_season"]
        pl._json(metrics, out / "metrics.json")
        write_briefs(cfg, sc, metrics)
        build_dashboard(cfg, metrics, sc)
        flagged = sum(r["status"] == "Flag for review" for r in live)
        print(f"scored {len(live)} active young players with model v{reg['version']} ({flagged} flagged) in {time.time() - t0:.1f}s")
        for c in metrics["monitoring"]["checks"]:
            print(f"  [{c['status']:5}] {c['check']}: {c['value']}")
        return

    if a.step == "report":
        metrics = json.loads((out / "metrics.json").read_text())
        sc = {"live": json.loads((out / "watchlist_current.json").read_text()),
              "replay": json.loads((out / "replay_2015_2018.json").read_text()), "latest_season": metrics["latest_season"]}
        build_figures(cfg, None, metrics)
        write_briefs(cfg, sc, metrics)
        build_dashboard(cfg, metrics, sc)
        print(f"rebuilt figures, briefs and dashboard in {time.time() - t0:.1f}s")
        return

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("1/8 preparing player-seasons, cohort, outcome labels and features")
    D = pl.prepare(cfg)
    step(f"    cohort: {int(D['coh'].labelled.sum())} labelled players, construct = {cfg['construct']['active']}")
    step("2/8 selecting the model on the 2011-14 validation classes")
    sel = pl.select(cfg, D)
    step(f"    champion = {sel['champion']}, challenger = {sel['challenger']}")
    step("3/8 backtesting on the untouched 2015-18 classes")
    bt = pl.backtest(cfg, D, sel)
    step("4/8 training production models on all labelled classes (1996-2018)")
    M = pl.train_production(cfg, D, sel)
    step("5/8 scoring the live watchlist and replaying 2015-18")
    sc = pl.score(cfg, D, M, sel)
    drivers = pl.drivers_summary(cfg, D, M)
    step("6/8 running drift monitoring")
    mon = run_monitoring(cfg, D, bt, sc["live"], drivers)
    metrics = {"version": cfg["version"], "construct": cfg["construct"]["active"], "selection": sel,
               "backtest": bt, "drivers": drivers, "monitoring": mon, "latest_season": sc["latest_season"]}
    pl._json(metrics, out / "metrics.json")
    metrics = json.loads((out / "metrics.json").read_text())   # same JSON types as the saved file
    step("7/8 drawing figures and writing decision briefs")
    build_figures(cfg, D, metrics)
    write_briefs(cfg, sc, metrics)
    step("8/8 building the dashboard")
    build_dashboard(cfg, metrics, sc)
    h3 = bt["horizons"][3]
    step(f"done. Test AUC (3 seasons) {h3['champion']['auc']['est']:.3f} vs draft order {h3['draft_baseline']['auc']['est']:.3f}; "
         f"dashboard: {cfg['paths']['dashboard']}")


if __name__ == "__main__":
    main()
