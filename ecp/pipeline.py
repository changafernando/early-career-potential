"""Orchestrates the pipeline: prepare -> select -> backtest -> train -> score -> monitor -> report assets.

Run from the repository root:  python run_pipeline.py all
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import evaluation as ev
from .constructs import build_cohort, split_masks
from .explain import comparables, global_importance, linear_contributions, top_drivers, tree_shap
from .features import evidence_flag, window_features
from .models import (BASELINE_FEATURES, bootstrap_ensemble, ensemble_interval, fit_baseline, fit_final,
                     select_model)
from .seasons import build_player_seasons, league_context

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TREE_MODELS = ("Random forest", "Gradient boosting")


def library_versions() -> dict:
    """Versions the saved .joblib models depend on; a mismatch can change or break them on load."""
    import sklearn
    return {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "joblib": joblib.__version__}


def _json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# --------------------------------------------------------------------------------------------- prepare
def prepare(cfg: dict) -> dict:
    pdir = cfg["paths"]["processed_dir"]
    pts = pd.read_csv(pdir / "player_team_seasons.csv.gz")
    teams = pd.read_csv(pdir / "team_seasons.csv")
    players = pd.read_csv(pdir / "players.csv", low_memory=False)
    ps = build_player_seasons(pts, teams, cfg)
    coh = build_cohort(ps, players, cfg)
    X = {h: window_features(ps, coh, players, h, cfg) for h in cfg["horizons"]}
    return dict(ps=ps, coh=coh, players=players, teams=teams, X=X, masks=split_masks(coh, cfg),
                league=league_context(ps, teams), latest_season=int(ps.season.max()),
                data_hash={p.name: file_hash(p) for p in sorted(pdir.glob("*")) if p.suffix in (".gz", ".csv")})


# ---------------------------------------------------------------------------------------------- select
def select(cfg: dict, D: dict) -> dict:
    """Champion = best average validation Brier across horizons; challenger = best tree model."""
    coh, masks = D["coh"], D["masks"]
    lab = coh.labelled
    y = coh.y[lab].astype(int)
    m = {k: v[lab] for k, v in masks.items()}
    tables = []
    for h in cfg["horizons"]:
        _, t = select_model(D["X"][h][lab], y, m, cfg["random_seed"])
        t["horizon"] = h
        tables.append(t)
    T = pd.concat(tables, ignore_index=True)
    agg = T.groupby(["model", "calibration"])[["val_brier", "val_auc", "val_cal_gap"]].mean().reset_index()
    agg = agg.sort_values(["val_brier", "val_auc"], ascending=[True, False]).reset_index(drop=True)
    champion = agg.iloc[0][["model", "calibration"]].to_dict()
    challenger = agg[agg.model.isin(TREE_MODELS) & (agg.model != champion["model"])].iloc[0][["model", "calibration"]].to_dict()
    out_dir = cfg["paths"]["outputs_dir"]
    T.round(4).to_csv(out_dir / "model_selection_by_horizon.csv", index=False)
    agg.round(4).to_csv(out_dir / "model_selection_summary.csv", index=False)
    return dict(champion=champion, challenger=challenger, summary=agg.round(4).to_dict("records"))


# -------------------------------------------------------------------------------------------- backtest
def backtest(cfg: dict, D: dict, sel: dict) -> dict:
    """Train on 1996-2014 debut classes, score the untouched 2015-18 classes once."""
    coh, masks, seed = D["coh"], D["masks"], cfg["random_seed"]
    tr, te = masks["train_full"], masks["test"]
    nb = cfg["evaluation"]["bootstrap_resamples"]
    r = cfg["decision"]["cost_ratio"]
    thr = 1 / (1 + r)
    res = {"construct": cfg["construct"]["active"], "train_classes": cfg["split"]["train"][0], "test_classes": cfg["split"]["test"],
           "n_train": int(tr.sum()), "n_test": int(te.sum()), "base_rate_train": round(float(coh.y[tr].mean()), 6),
           "base_rate_test": round(float(coh.y[te].mean()), 6), "positives_test": int(coh.y[te].sum()),
           "decision_threshold": round(thr, 3), "horizons": {}}
    preds = []
    yte = coh.y[te].astype(int)
    alive_y3 = coh.index.isin(D["ps"].loc[D["ps"].cy == 3, "personId"])
    seasons_total = D["ps"].groupby("personId").size().reindex(coh.index).fillna(0)
    for h in cfg["horizons"]:
        X = D["X"][h]
        y = coh.y.astype(float)
        champ = fit_final(X[coh.labelled], y[coh.labelled].astype(int), tr[coh.labelled], sel["champion"], seed)
        chall = fit_final(X[coh.labelled], y[coh.labelled].astype(int), tr[coh.labelled], sel["challenger"], seed)
        base = fit_baseline(X[coh.labelled], y[coh.labelled].astype(int), tr[coh.labelled])
        p = pd.Series(champ.predict_proba(X[te])[:, 1], index=X.index[te])
        pc = pd.Series(chall.predict_proba(X[te])[:, 1], index=X.index[te])
        pb = pd.Series(base.predict_proba(X.loc[te, BASELINE_FEATURES])[:, 1], index=X.index[te])
        hres = {"champion": ev.bootstrap_metrics(yte, p, pb, nb, seed),
                "challenger": ev.bootstrap_metrics(yte, pc, pb, nb, seed),
                "draft_baseline": ev.bootstrap_metrics(yte, pb, None, nb, seed)}
        dec = pd.Series(alive_y3, index=coh.index)[te]
        surv = (seasons_total[te] >= 6)
        hres["extension_population"] = {"n": int(dec.sum()), "positives": int(yte[dec].sum()),
                                        "champion": ev.point_metrics(yte[dec], p[dec]), "draft": ev.point_metrics(yte[dec], pb[dec])}
        hres["six_season_survivors"] = {"n": int(surv.sum()), "positives": int(yte[surv].sum()),
                                        "champion_auc": round(float(ev.roc_auc_score(yte[surv], p[surv])), 6),
                                        "draft_auc": round(float(ev.roc_auc_score(yte[surv], pb[surv])), 6)}
        hres["cost_thresholds"] = ev.cost_threshold_table(yte, p).to_dict("records")
        hres["cost_thresholds_draft"] = ev.cost_threshold_table(yte, pb).to_dict("records")
        hres["shortlists"] = {f"top_{int(s * 100)}pct": {"model": ev.shortlist(yte, p, coh.debut[te], s),
                                                         "draft_order": ev.shortlist(yte, -coh.draft_pick[te].astype(float), coh.debut[te], s)}
                              for s in cfg["decision"]["shortlist_shares"]}
        hres["calibration"] = ev.calibration_table(yte, p).round(4).to_dict("records")
        hres["by_cohort"] = ev.by_cohort(yte, p, coh.debut[te]).to_dict("records")
        hres["champion_challenger_correlation"] = round(float(np.corrcoef(p, pc)[0, 1]), 3)
        ef = evidence_flag(X[te], cfg)
        hres["insufficient_evidence"] = {"players": int(ef.sum()), "contributors": int(yte[ef].sum()),
                                         "mean_pred": round(float(p[ef].mean()), 3) if ef.any() else None}
        res["horizons"][h] = hres
        preds.append(pd.DataFrame({"personId": X.index[te], "horizon": h, "p": p.values, "p_challenger": pc.values,
                                   "p_draft": pb.values, "y": yte.values}))
    P = pd.concat(preds, ignore_index=True)
    out = cfg["paths"]["outputs_dir"]
    P.round(5).to_csv(out / "backtest_predictions.csv", index=False)

    # horizon-3 deep dives: decision curve, fairness audit, construct comparison, named examples
    h3 = P[P.horizon == 3].set_index("personId")
    dc = ev.decision_curve(h3.y, {"Model": h3.p, "Draft order": h3.p_draft})
    dc.round(5).to_csv(out / "decision_curve_h3.csv", index=False)
    res["net_benefit_at_threshold"] = {k: round(float(dc.loc[(dc.threshold - round(thr, 2)).abs().idxmin(), k]) * 100, 2)
                                       for k in ["Model", "Draft order", "Flag everyone"]}
    audit_df = h3.join(coh[["position", "draft_tier", "origin", "height_band"]])
    audit = ev.subgroup_audit(audit_df, ["position", "draft_tier", "origin", "height_band"], thr, cfg["evaluation"]["min_subgroup_size"])
    audit.to_csv(out / "subgroup_audit_h3.csv", index=False)
    res["subgroup_audit_h3"] = audit.to_dict("records")

    late = coh.loc[h3.index, "draft_tier"] == "2nd round / undrafted"
    gems = h3[late & (h3.y == 1)].join(coh[["name", "draft_pick", "debut"]]).sort_values("p", ascending=False)
    res["late_pick_contributors"] = {"count": int(len(gems)), "flagged_at_threshold": int((gems.p >= thr).sum()),
                                     "players": [dict(name=r["name"], pick=int(r.draft_pick), debut=int(r.debut), p=round(r.p, 3))
                                                 for _, r in gems.iterrows()],
                                     "late_pick_non_contributors_mean_p": round(float(h3[late & (h3.y == 0)].p.mean()), 3)}
    lottery = coh.loc[h3.index, "draft_tier"] == "Lottery (1-14)"
    low = h3[lottery & (h3.y == 0)].join(coh[["name", "draft_pick", "debut"]]).sort_values("p").head(6)
    res["lottery_low_scores"] = [dict(name=r["name"], pick=int(r.draft_pick), debut=int(r.debut), p=round(r.p, 3)) for _, r in low.iterrows()]
    missed = h3[(h3.y == 1) & (h3.p < thr)].join(coh[["name", "draft_pick", "debut"]]).sort_values("p")
    res["missed_contributors"] = [dict(name=r["name"], pick=int(r.draft_pick), debut=int(r.debut), p=round(r.p, 3),
                                       minutes=int(D["X"][3].at[i, "minutes"])) for i, r in missed.iterrows()]
    res["construct_comparison"] = construct_comparison(cfg, D, sel)
    return res


def construct_comparison(cfg: dict, D: dict, sel: dict) -> dict:
    """Same champion model, both outcome definitions, horizon 3 test classes."""
    coh, masks, seed = D["coh"], D["masks"], cfg["random_seed"]
    out = {}
    lab = coh.labelled
    tr, te = masks["train_full"], masks["test"]
    X = D["X"][3]
    for c in ["volume", "rate"]:
        y = coh[f"y_{c}"]
        m = fit_final(X[lab], y[lab].astype(int), tr[lab], sel["champion"], seed)
        b = fit_baseline(X[lab], y[lab].astype(int), tr[lab])
        p = m.predict_proba(X[te])[:, 1]
        pb = b.predict_proba(X.loc[te, BASELINE_FEATURES])[:, 1]
        out[c] = {"base_rate_all": round(float(y[lab].mean()), 6), "base_rate_test": round(float(y[te].mean()), 6),
                  "model": {k: round(v, 6) for k, v in ev.point_metrics(y[te].astype(int), p).items()},
                  "draft_auc": round(float(ev.roc_auc_score(y[te].astype(int), pb)), 6)}
    both = coh[lab]
    out["agreement"] = {"both": int(((both.y_volume == 1) & (both.y_rate == 1)).sum()),
                        "volume_only": int(((both.y_volume == 1) & (both.y_rate == 0)).sum()),
                        "rate_only": int(((both.y_volume == 0) & (both.y_rate == 1)).sum()),
                        "neither": int(((both.y_volume == 0) & (both.y_rate == 0)).sum())}
    vo = both[(both.y_volume == 1) & (both.y_rate == 0)]
    ro = both[(both.y_volume == 0) & (both.y_rate == 1)]
    out["examples_volume_only"] = vo.sort_values("debut", ascending=False).name.head(5).tolist()
    out["examples_rate_only"] = ro.sort_values("debut", ascending=False).name.head(5).tolist()
    return out


# --------------------------------------------------------------------------------------- train + score
def train_production(cfg: dict, D: dict, sel: dict) -> dict:
    """Refit champion and challenger on every labelled class (1996-2018), plus an uncertainty ensemble."""
    coh, seed = D["coh"], cfg["random_seed"]
    lab = coh.labelled
    mdir = cfg["paths"]["models_dir"]
    mdir.mkdir(parents=True, exist_ok=True)
    reg = {"version": cfg["version"], "created": dt.date.today().isoformat(), "construct": cfg["construct"]["active"],
           "construct_description": cfg["construct"][cfg["construct"]["active"]]["description"],
           "training_classes": [cfg["cohort"]["first_debut"], cfg["cohort"]["last_labelled_debut"]],
           "n_training_players": int(lab.sum()), "base_rate": round(float(coh.y[lab].mean()), 4),
           "champion": sel["champion"], "challenger": sel["challenger"], "data_hash": D["data_hash"],
           "library_versions": library_versions(),
           "decision_threshold": round(1 / (1 + cfg["decision"]["cost_ratio"]), 3), "status": "Candidate - awaiting Model Review Board",
           "models": {}}
    M = {}
    for h in cfg["horizons"]:
        X, y = D["X"][h][lab], coh.y[lab].astype(int)
        all_ = pd.Series(True, index=X.index)
        champ = fit_final(X, y, all_, sel["champion"], seed)
        chall = fit_final(X, y, all_, sel["challenger"], seed)
        ens = bootstrap_ensemble(X, y, all_, sel["champion"], cfg["evaluation"]["uncertainty_models"], seed)
        for name, obj in [("champion", champ), ("challenger", chall), ("uncertainty", ens)]:
            joblib.dump(obj, mdir / f"{name}_h{h}.joblib", compress=3)
        reg["models"][f"h{h}"] = {"champion": f"champion_h{h}.joblib", "challenger": f"challenger_h{h}.joblib",
                                  "uncertainty_ensemble": f"uncertainty_h{h}.joblib", "features": list(X.columns),
                                  "sha256": {n: file_hash(mdir / f"{n}_h{h}.joblib") for n in ["champion", "challenger", "uncertainty"]}}
        M[h] = dict(champion=champ, challenger=chall, ensemble=ens)
    _json(reg, mdir / "registry.json")
    return M


def _brief_rows(cfg, D, h, X, champ, chall, ens, X_ref, ref_info, info_cols) -> list[dict]:
    coh = D["coh"]
    thr = 1 / (1 + cfg["decision"]["cost_ratio"])
    p = champ.predict_proba(X)[:, 1]
    pc = chall.predict_proba(X)[:, 1]
    iv = ensemble_interval(ens, X, p)
    contrib = linear_contributions(champ, X, X_ref)
    drv = top_drivers(contrib, X)
    comps = comparables(X, X_ref, ref_info)
    ef = evidence_flag(X, cfg)
    rows = []
    for i, pid in enumerate(X.index):
        c = coh.loc[pid]
        status = "Insufficient evidence" if ef[pid] else ("Flag for review" if p[i] >= thr else "Below threshold")
        row = dict(personId=int(pid), name=c["name"], debut=int(c.debut), horizon=h, position=c.position,
                   draft=("Undrafted" if c.draft_pick >= 61 else f"Pick {int(c.draft_pick)}"), draft_tier=c.draft_tier,
                   p=round(float(p[i]), 3), p_low=round(float(iv.p_low.iat[i]), 3), p_high=round(float(iv.p_high.iat[i]), 3),
                   p_challenger=round(float(pc[i]), 3), models_disagree=bool(abs(p[i] - pc[i]) >= 0.15),
                   status=status, games=int(X.at[pid, "games"]), minutes=int(X.at[pid, "minutes"]),
                   mpg=round(float(X.at[pid, "mpg"]), 1) if pd.notna(X.at[pid, "mpg"]) else None,
                   gmsc_100=round(float(X.at[pid, "gmsc_100"]), 1) if pd.notna(X.at[pid, "gmsc_100"]) else None,
                   ts_vs_league=round(float(X.at[pid, "ts_vs_league"]), 3) if pd.notna(X.at[pid, "ts_vs_league"]) else None,
                   best_rate_pct=round(float(X.at[pid, "best_rate_pct"]), 3) if pd.notna(X.at[pid, "best_rate_pct"]) else None,
                   age_debut=round(float(X.at[pid, "age_debut"]), 1) if pd.notna(X.at[pid, "age_debut"]) else None,
                   drivers_up=drv.at[pid, "drivers_up"], drivers_down=drv.at[pid, "drivers_down"], comparables=comps[pid])
        for col in info_cols:
            row[col] = None if pd.isna(c[col]) else int(c[col])
        rows.append(row)
    # rank within debut class (1 = highest probability)
    df = pd.DataFrame(rows)
    df["class_rank"] = df.groupby("debut").p.rank(ascending=False, method="min").astype(int)
    df["class_size"] = df.groupby("debut").p.transform("size")
    return df.to_dict("records")


LIST_COLS = ["personId", "name", "debut", "horizon", "position", "draft", "p", "p_low", "p_high", "p_challenger",
             "models_disagree", "status", "class_rank", "class_size", "games", "minutes", "mpg"]


def score_live(cfg: dict, D: dict, M: dict) -> list[dict]:
    """Score every active player in his first three seasons with the registered production models."""
    coh, ps = D["coh"], D["ps"]
    S = D["latest_season"]
    lab = coh.labelled
    active = set(ps.loc[ps.season == S, "personId"])
    info = coh.assign(y=coh.y.fillna(0), hi_seasons=coh[f"hi_seasons_{cfg['construct']['active']}"])
    live = []
    for h in cfg["horizons"]:
        d = S - h + 1
        idx = coh.index[(coh.debut == d) & coh.index.isin(active)]
        X = D["X"][h].loc[idx]
        X_ref = D["X"][h][lab]
        live += _brief_rows(cfg, D, h, X, M[h]["champion"], M[h]["challenger"], M[h]["ensemble"], X_ref, info[lab], [])
    out = cfg["paths"]["outputs_dir"]
    _json(live, out / "watchlist_current.json")
    pd.DataFrame(live)[LIST_COLS].sort_values(["debut", "class_rank"]).to_csv(out / "watchlist_current.csv", index=False)
    return live


def score_replay(cfg: dict, D: dict, sel: dict) -> list[dict]:
    """Replay the 2015-18 classes at horizon 3 with a model that only saw the 1996-2014 classes."""
    coh = D["coh"]
    lab = coh.labelled
    info = coh.assign(y=coh.y.fillna(0), hi_seasons=coh[f"hi_seasons_{cfg['construct']['active']}"])
    tr, te = D["masks"]["train_full"], D["masks"]["test"]
    X3 = D["X"][3]
    seed = cfg["random_seed"]
    y = coh.y[lab].astype(int)
    champ = fit_final(X3[lab], y, tr[lab], sel["champion"], seed)
    chall = fit_final(X3[lab], y, tr[lab], sel["challenger"], seed)
    ens = bootstrap_ensemble(X3[lab], y, tr[lab], sel["champion"], cfg["evaluation"]["uncertainty_models"], seed)
    replay = _brief_rows(cfg, D, 3, X3[te], champ, chall, ens, X3[tr], info[tr], [])
    for r in replay:
        r["outcome"] = int(coh.y[r["personId"]])
        r["high_impact_seasons"] = int(info.at[r["personId"], "hi_seasons"])
    out = cfg["paths"]["outputs_dir"]
    _json(replay, out / "replay_2015_2018.json")
    pd.DataFrame(replay)[LIST_COLS + ["outcome"]].sort_values(["debut", "class_rank"]).to_csv(out / "replay_2015_2018.csv", index=False)
    return replay


def score(cfg: dict, D: dict, M: dict, sel: dict) -> dict:
    return {"live": score_live(cfg, D, M), "replay": score_replay(cfg, D, sel), "latest_season": D["latest_season"]}


def load_registered(cfg: dict) -> tuple[dict, dict]:
    """Load the registered production models (no retraining).

    Security: .joblib files can execute code when loaded, so each file's hash is checked against the
    registry first and loading stops if any file has been altered or replaced.
    """
    mdir = cfg["paths"]["models_dir"]
    reg = json.loads((mdir / "registry.json").read_text())
    trained_with, here = reg.get("library_versions"), library_versions()
    if trained_with is None:
        print("note: the registry does not record library versions; run `python run_pipeline.py all` to record them")
    elif trained_with != here:
        print(f"warning: models were trained with {trained_with} but this environment has {here}. "
              "Saved models are tied to these versions: run `python run_pipeline.py all` to retrain locally.", flush=True)
    M = {}
    for h in cfg["horizons"]:
        e = reg["models"][f"h{h}"]
        for n, expected in e.get("sha256", {}).items():
            if file_hash(mdir / f"{n}_h{h}.joblib") != expected:
                raise RuntimeError(f"{n}_h{h}.joblib does not match the registry hash - refusing to load it")
        M[h] = dict(champion=joblib.load(mdir / e["champion"]), challenger=joblib.load(mdir / e["challenger"]),
                    ensemble=joblib.load(mdir / e["uncertainty_ensemble"]))
    return M, reg


# ----------------------------------------------------------------------------------- global drivers
def drivers_summary(cfg: dict, D: dict, M: dict) -> dict:
    coh = D["coh"]
    lab = coh.labelled
    X = D["X"][3][lab]
    lin = global_importance(linear_contributions(M[3]["champion"], X, X))
    tree = global_importance(tree_shap(M[3]["challenger"], X))
    out = {"champion_mean_abs_logodds": lin.round(4).to_dict(), "challenger_mean_abs_shap": tree.round(4).to_dict()}
    pd.DataFrame({"champion": lin, "challenger": tree}).round(4).to_csv(cfg["paths"]["outputs_dir"] / "global_drivers_h3.csv")
    return out
