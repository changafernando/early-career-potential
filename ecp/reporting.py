"""Step 7 - Report assets: figures (PNG) and one-page decision briefs (HTML).

Colour roles are fixed across every figure and the dashboard:
model = blue, draft order = orange, challenger = aqua, reference lines = grey.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
RED, GREEN, AMBER = "#d03b3b", "#0ca30c", "#fab219"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_auc_by_horizon(bt: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    H = [1, 2, 3]
    series = [("Model (champion)", "champion", BLUE, -0.12), ("Challenger (random forest)", "challenger", AQUA, 0.0),
              ("Draft order only", "draft_baseline", ORANGE, 0.12)]
    for label, key, col, dx in series:
        est = [bt["horizons"][str(h)][key]["auc"]["est"] for h in H]
        lo = [bt["horizons"][str(h)][key]["auc"]["lo"] for h in H]
        hi = [bt["horizons"][str(h)][key]["auc"]["hi"] for h in H]
        x = np.array(H) + dx
        ax.errorbar(x, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="o", color=col, ms=6,
                    elinewidth=1.6, capsize=0, label=label)
        if key != "challenger":
            for xi, e in zip(x, est):
                ax.annotate(f"{e:.2f}", (xi, e), xytext=(7 if dx > 0 else -7, 0), textcoords="offset points",
                            ha="left" if dx > 0 else "right", va="center", fontsize=8, color=INK2)
    ax.set_xticks(H, ["After 1 season", "After 2 seasons", "After 3 seasons"])
    ax.set_xlim(0.6, 3.4)
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("ROC-AUC on 2015-18 classes (95% CI)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def fig_shortlist(bt: dict, path: Path):
    s = bt["horizons"]["3"]["shortlists"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.1))
    a = s["top_10pct"]
    b = s["top_20pct"]
    panels = [(axes[0], f"Contributors in a top-10% shortlist (of {a['model']['size']})",
               [a["draft_order"]["contributors"], a["model"]["contributors"]]),
              (axes[1], f"Contributors missed by a top-20% shortlist (of {b['model']['total_contributors']})",
               [b["draft_order"]["missed"], b["model"]["missed"]])]
    for ax, title, vals in panels:
        ax.barh([0, 1], vals, color=[ORANGE, BLUE], height=0.55)
        ax.set_yticks([0, 1], ["Draft order", "Model"])
        for i, v in enumerate(vals):
            ax.text(v + 0.4, i, str(v), va="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.25)
    fig.tight_layout(w_pad=2.5)
    _save(fig, path)


def fig_calibration(bt: dict, path: Path):
    cal = pd.DataFrame(bt["horizons"]["3"]["calibration"])
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", label="Perfect calibration")
    ax.scatter(cal.mean_pred, cal.observed, s=20 + cal.n * 0.9, color=BLUE, edgecolor="white", lw=1.5, zorder=3,
               label="Model (point size = players)")
    for _, r in cal.iterrows():
        if r.mean_pred >= 0.2:
            ax.annotate(f"n={int(r.n)}", (r.mean_pred, r.observed), xytext=(9, -3), textcoords="offset points", fontsize=7, color=INK2)
    low = cal[cal.mean_pred < 0.2]
    ax.text(0.98, 0.03, "Lowest bands: n = " + ", ".join(str(int(n)) for n in low.n), transform=ax.transAxes,
            ha="right", fontsize=7, color=INK2)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Share who became contributors")
    ax.set_xlim(-0.02, 1)
    ax.set_ylim(-0.02, 1)
    ax.legend(loc="upper left", fontsize=7.5)
    _save(fig, path)


def fig_decision_curve(dc: pd.DataFrame, thr: float, path: Path):
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(dc.threshold, dc["Model"] * 100, color=BLUE, lw=2, label="Model")
    ax.plot(dc.threshold, dc["Draft order"] * 100, color=ORANGE, lw=2, label="Draft order")
    ax.plot(dc.threshold, dc["Flag everyone"] * 100, color=GREY, lw=1.2, ls="--", label="Flag every player")
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(thr, color=INK2, lw=0.8, ls=":")
    ax.set_ylim(-4, max(dc["Model"].max() * 100 * 1.15, 5))
    ax.text(thr + 0.008, -3.6, f"threshold {thr:.2f}\n(3:1 cost ratio)", fontsize=7.5, color=INK2, va="bottom")
    ax.set_xlabel("Decision threshold (probability)")
    ax.set_ylabel("Net benefit per 100 players")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, path)


def fig_subgroups(audit: pd.DataFrame, tol: float, path: Path):
    a = audit[audit.group != "Unknown"].copy()
    a["label"] = a.group + "  (n=" + a.n.astype(str) + ")"
    order = ["position", "draft_tier", "origin", "height_band"]
    names = {"position": "Position", "draft_tier": "Draft tier", "origin": "Origin", "height_band": "Height"}
    rows, ypos, heads, y = [], [], [], 0.0
    for d in order:
        g = a[a.dimension == d]
        heads.append((y, names[d]))
        y -= 0.9
        for _, r in g.iterrows():
            rows.append(r)
            ypos.append(y)
            y -= 1
        y -= 0.4
    a = pd.DataFrame(rows)
    yy = np.array(ypos)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.6), sharey=True)
    cols = [BLUE if j else GREY for j in a.judged]
    axes[0].axvspan(-tol, tol, color="#eef4fc", zorder=0)
    axes[0].axvline(0, color=AXIS, lw=1)
    axes[0].scatter(a.calibration_gap, yy, color=cols, s=36, zorder=3)
    axes[0].set_title("Calibration gap\n(predicted minus observed)", fontsize=8.5)
    axes[0].set_xlim(-0.12, 0.12)
    axes[1].scatter(a.miss_rate, yy, color=cols, s=36, zorder=3)
    axes[1].set_title(f"Contributors missed\nat threshold 0.25", fontsize=8.5)
    axes[1].set_xlim(-0.02, 0.4)
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_yticks(yy, a.label, fontsize=7.5)
    for y0, t in heads:
        axes[0].text(-0.36, y0 - 0.25, t, transform=axes[0].get_yaxis_transform(), fontsize=8, weight="bold", color=INK, ha="left")
    axes[0].set_ylim(yy.min() - 0.8, 0.4)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.text(0.0, -0.02, "Grey: under 25 players or 5 contributors (reported, not judged). Shaded: +/-0.05 tolerance.",
             fontsize=7, color=INK2)
    fig.tight_layout()
    _save(fig, path)


def fig_drivers(drivers: dict, plain: dict, path: Path, k: int = 10):
    s = pd.Series(drivers["champion_mean_abs_logodds"]).head(k)[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.barh([plain.get(i, i) for i in s.index], s.values, color=BLUE, height=0.6)
    ax.set_xlabel("Average influence on a player's score (|log-odds|)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def fig_drift(mon: dict, path: Path):
    lg = pd.DataFrame(mon["league_by_season"])
    lg = lg[lg.season >= 1996]
    br = pd.Series(mon["base_rate_by_class"])
    br.index = br.index.astype(int)
    br = br.sort_index()
    st = pd.Series(mon["start_flag_share_by_season"])
    st.index = st.index.astype(int)
    st = st.sort_index()
    st = st[st.index >= 1996]
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.5))
    axes[0].plot(lg.season, lg.three_pa_share * 100, color=BLUE, lw=2)
    axes[0].set_title("League 3PT share of shots (%)", fontsize=8.5)
    axes[1].plot(br.index, br.values * 100, color=BLUE, lw=1.2, marker="o", ms=3)
    axes[1].plot(br.index, br.rolling(5, center=True, min_periods=3).mean() * 100, color=INK2, lw=1.6, ls="--", label="5-class average")
    axes[1].set_title("Contributor rate by debut class (%)", fontsize=8.5)
    axes[1].legend(fontsize=7, loc="lower left")
    axes[2].bar(st.index, st.values * 100, color=[RED if (v < .40 or v > .55) else BLUE for v in st.values], width=0.8)
    axes[2].set_title("Games flagged 'started' (%)", fontsize=8.5)
    axes[2].text(1996, 102, "red = implausible season", fontsize=7, color=RED)
    axes[2].set_ylim(0, 112)
    for ax in axes:
        ax.tick_params(labelsize=7.5)
    fig.tight_layout(w_pad=2.2)
    _save(fig, path)


def fig_architecture(path: Path):
    fig, ax = plt.subplots(figsize=(11, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    boxes = [
        (0.5, "1 Ingest", "Box scores, team\npace, biography\n-> player-seasons\n+ data checks"),
        (17.2, "2 Prepare", "Era-relative\nfeatures; outcome\nconstruct (rate\nor volume)"),
        (33.9, "3 Select & test", "4 models x 3\ncalibrations;\nvalidate 2011-14,\ntest 2015-18 once"),
        (50.6, "4 Train & register", "Champion,\nchallenger,\nuncertainty\nensemble; registry"),
        (67.3, "5 Score & explain", "Watchlist, ranges,\ndrivers,\ncomparables,\nbriefs, dashboard"),
        (84.0, "6 Monitor", "Field integrity,\nPSI drift,\ncalibration,\nchallenger gap"),
    ]
    for x, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, 12), 15, 25, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#eef4fc", ec=BLUE, lw=1.2))
        ax.text(x + 7.5, 33.5, title, ha="center", va="center", fontsize=8.8, weight="bold", color=INK)
        ax.text(x + 7.5, 21.5, body, ha="center", va="center", fontsize=8, color=INK2, linespacing=1.4)
    for x, _, _ in boxes[:-1]:
        ax.annotate("", xy=(x + 16.6, 24.5), xytext=(x + 15.4, 24.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
    ax.annotate("", xy=(41.4, 11.4), xytext=(91.5, 11.4),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5, connectionstyle="arc3,rad=-0.22"))
    ax.text(50, 0.8, "Off-season retrain, recalibrate or roll back when a check turns red; every release approved by the Model Review Board",
            ha="center", fontsize=8.5, color=ORANGE)
    _save(fig, path)


def fig_business_case(bc: dict, path: Path):
    """Errors in the extension decision (model vs draft order) and payback at illustrative prices."""
    di, pb, c = bc["decision_impact"], bc["payback"], bc["costs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.38})
    cats = ["Missed contributors", "False flags"]
    m = [di["model"]["missed"], di["model"]["false_flags"]]
    d = [di["draft_order"]["missed"], di["draft_order"]["false_flags"]]
    y = np.arange(len(cats))
    a.barh(y - 0.19, m, 0.36, color=BLUE, label="Model")
    a.barh(y + 0.19, d, 0.36, color=ORANGE, label="Draft order")
    for yi, v in zip(y - 0.19, m):
        a.text(v + 0.8, yi, str(v), va="center", fontsize=8, color=INK2)
    for yi, v in zip(y + 0.19, d):
        a.text(v + 0.8, yi, str(v), va="center", fontsize=8, color=INK2)
    a.set_yticks(y, cats)
    a.invert_yaxis()
    a.set_xlim(0, max(d) * 1.25)
    a.grid(axis="y", visible=False)
    a.set_xlabel(f"Players (of {di['n']} extension decisions)")
    a.set_title("(a) Mistakes at the 0.25 threshold")
    a.legend(loc="upper right", fontsize=8)
    a.text(0.98, 0.70, f"Error cost per decision\nmodel ${di['model']['error_cost_per_decision'] / 1e3:,.0f}k\n"
           f"draft order ${di['draft_order']['error_cost_per_decision'] / 1e3:,.0f}k", transform=a.transAxes,
           ha="right", va="center", fontsize=7.5, color=INK2)
    yrs = c["horizon_years"]
    t = np.linspace(0, yrs, 61)
    cost = (c["one_off"] + c["annual_total"] * t) / 1e3
    s, (lo, hi) = di["saving_per_decision"], di["saving_per_decision_90ci"]
    k = pb["benefit_per_year"] / s if s else 1
    b.fill_between(t, lo * k * t / 1e3, hi * k * t / 1e3, color=BLUE, alpha=0.12, lw=0, label="90% interval")
    for row in pb["sensitivity"]:
        cm = row["cost_missed_contributor"]
        main = abs(cm - di["cost_missed_contributor"]) < 1
        b.plot(t, row["saving_per_decision"] * k * t / 1e3, color=BLUE, lw=2.2 if main else 1.0,
               ls="-" if main else "--", alpha=1 if main else 0.7,
               label="Avoided error cost" if main else None)
        b.annotate(f"miss = ${cm / 1e6:.0f}M", (yrs, row["saving_per_decision"] * k * yrs / 1e3), xytext=(4, 0),
                   textcoords="offset points", va="center", fontsize=7.5, color=BLUE if main else INK2)
    b.plot(t, cost, color=GREY, lw=2, label="Cumulative cost")
    if pb["breakeven_years"]:
        be = pb["breakeven_years"]
        yb = (c["one_off"] + c["annual_total"] * be) / 1e3
        b.plot([be, be], [0, yb], color=GREY, lw=0.8, ls=":")
        b.plot([be], [yb], "o", color=INK, ms=4, zorder=5)
        b.annotate(f"payback {be:.1f} yrs", (be, yb), xytext=(7, -10), textcoords="offset points",
                   fontsize=7.5, color=INK)
    b.set_xlim(0, yrs)
    b.set_xticks(range(yrs + 1), [f"Year {i}" for i in range(yrs + 1)])
    b.set_ylabel("Cumulative USD (thousands)")
    b.set_title("(b) Avoided error cost vs cost (illustrative)")
    b.legend(loc="upper left", fontsize=7.5)
    _save(fig, path)


def build_figures(cfg: dict, D: dict, metrics: dict):
    from .features import PLAIN_NAMES

    fdir = cfg["paths"]["outputs_dir"] / "figures"
    bt = metrics["backtest"]
    thr = bt["decision_threshold"]
    fig_auc_by_horizon(bt, fdir / "fig1_auc_by_horizon.png")
    fig_shortlist(bt, fdir / "fig2_shortlists.png")
    fig_calibration(bt, fdir / "fig3_calibration.png")
    fig_decision_curve(pd.read_csv(cfg["paths"]["outputs_dir"] / "decision_curve_h3.csv"), thr, fdir / "fig4_decision_curve.png")
    fig_subgroups(pd.DataFrame(bt["subgroup_audit_h3"]), 0.05, fdir / "fig5_subgroup_audit.png")
    fig_drivers(metrics["drivers"], PLAIN_NAMES, fdir / "fig6_drivers.png")
    fig_drift(metrics["monitoring"], fdir / "fig7_drift.png")
    fig_architecture(fdir / "fig8_architecture.png")
    if "business_case" in metrics:
        fig_business_case(metrics["business_case"], fdir / "fig9_business_case.png")


# ------------------------------------------------------------------------------------------- briefs
BRIEF_CSS = """
body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#0b0b0b;background:#fff;margin:0;padding:28px;max-width:780px}
h1{font-size:22px;margin:0}.sub{color:#52514e;font-size:13px;margin:4px 0 16px}
.row{display:flex;gap:16px;flex-wrap:wrap}.card{border:1px solid #e1e0d9;border-radius:10px;padding:14px 16px;flex:1;min-width:220px}
.big{font-size:40px;font-weight:700}.muted{color:#52514e;font-size:12px}.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600}
.flag{background:#e7f0fb;color:#1c5cab}.below{background:#f0efec;color:#52514e}.insuff{background:#fff4dc;color:#7a5200}
table{border-collapse:collapse;width:100%;font-size:12.5px}td,th{padding:5px 6px;border-bottom:1px solid #eeede8;text-align:left}
h2{font-size:14px;margin:18px 0 6px}.up{color:#1c5cab}.down{color:#b23a3a}.box{border:1px dashed #c3c2b7;border-radius:8px;padding:10px;min-height:46px;font-size:12px;color:#898781}
.foot{font-size:11px;color:#898781;margin-top:18px}
"""


def brief_html(r: dict, thr: float, version: str) -> str:
    e = html.escape
    tag = {"Flag for review": "flag", "Below threshold": "below", "Insufficient evidence": "insuff"}[r["status"]]
    ups = "".join(f"<tr><td class='up'>&#9650; {e(d['label'])}</td><td>{e(d['value'])}</td></tr>" for d in r["drivers_up"])
    dns = "".join(f"<tr><td class='down'>&#9660; {e(d['label'])}</td><td>{e(d['value'])}</td></tr>" for d in r["drivers_down"])
    comps = "".join(f"<tr><td>{e(c['name'])}</td><td>{c['debut']}</td><td>{'Yes' if c['contributor'] else 'No'}</td>"
                    f"<td>{c['high_impact_seasons']}</td></tr>" for c in r["comparables"])
    outcome = ""
    if "outcome" in r:
        outcome = (f"<div class='card'><div class='muted'>Actual outcome (known in replay)</div><div style='font-size:20px;font-weight:700'>"
                   f"{'Became a contributor' if r['outcome'] else 'Did not become a contributor'}</div>"
                   f"<div class='muted'>{r['high_impact_seasons']} high-impact seasons in years 4-8</div></div>")
    score = "&mdash;" if r["status"] == "Insufficient evidence" else f"{r['p'] * 100:.0f}%"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Decision brief - {e(r['name'])}</title>
<style>{BRIEF_CSS}</style></head><body>
<div class='muted'>EARLY-CAREER POTENTIAL &middot; DECISION BRIEF &middot; CONFIDENTIAL - FRONT OFFICE ONLY</div>
<h1>{e(r['name'])}</h1>
<div class='sub'>{e(r['position'])} &middot; {e(r['draft'])} &middot; debut {r['debut']}-{str(r['debut'] + 1)[-2:]} &middot; evidence: {r['horizon']} season(s), {r['games']} games, {r['minutes']:,} minutes</div>
<div class='row'>
 <div class='card'><div class='muted'>Probability of becoming a high-impact contributor (years 4-8)</div>
 <div class='big'>{score}</div><div class='muted'>80% range {r['p_low'] * 100:.0f}-{r['p_high'] * 100:.0f}% &middot; challenger model {r['p_challenger'] * 100:.0f}%
 {'&middot; <b>models disagree - analyst review</b>' if r['models_disagree'] else ''}</div></div>
 <div class='card'><div class='muted'>Status at the agreed threshold ({thr:.2f})</div>
 <div style='margin:8px 0'><span class='tag {tag}'>{e(r['status'])}</span></div>
 <div class='muted'>Ranked {r['class_rank']} of {r['class_size']} in his debut class</div></div>
 {outcome}
</div>
<h2>What moves the score</h2>
<table><tr><th>Pushes up</th><th>Value</th></tr>{ups}<tr><th>Pushes down</th><th>Value</th></tr>{dns}</table>
<h2>Most similar historical players at the same career stage</h2>
<table><tr><th>Player</th><th>Debut</th><th>Contributor?</th><th>High-impact seasons</th></tr>{comps}</table>
<h2>Scout and coach view (required before any decision)</h2>
<div class='box'>Context the model cannot see: injuries, role, fit, character, development trajectory.</div>
<h2>Decision and rationale (logged)</h2>
<div class='box'>Decision &middot; decided by &middot; date &middot; override reason if different from the model</div>
<div class='foot'>Screening input only - never a decision on its own. Model v{version}; outcome definition and threshold set by the
steering committee. Scores expire at the end of the season and are never shared outside the front office.</div>
</body></html>"""


def write_briefs(cfg: dict, sc: dict, metrics: dict, n_live: int = 8):
    bdir = cfg["paths"]["outputs_dir"] / "briefs"
    bdir.mkdir(parents=True, exist_ok=True)
    thr = metrics["backtest"]["decision_threshold"]
    S = sc["latest_season"]
    live = sorted([r for r in sc["live"] if r["horizon"] == 3 and r["debut"] == S - 2], key=lambda r: -r["p"])[:n_live]
    want = {"Jalen Brunson", "Nikola Jokic", "Fred VanVleet", "Frank Ntilikina", "Malcolm Brogdon", "Christian Wood"}
    replay = [r for r in sc["replay"] if r["name"] in want]
    links = []
    for r in live + replay:
        slug = re.sub(r"[^a-z0-9]+", "_", r["name"].lower()).strip("_")
        name = f"{'replay' if 'outcome' in r else 'live'}_{r['debut']}_{slug}.html"
        (bdir / name).write_text(brief_html(r, thr, cfg["version"]))
        links.append((("Replay (outcome known)" if "outcome" in r else f"Live - {S - 2} debut class (extension window)"), r["name"], name))
    idx = "".join(f"<tr><td>{html.escape(k)}</td><td><a href='{f}'>{html.escape(n)}</a></td></tr>" for k, n, f in links)
    (bdir / "index.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Decision briefs</title><style>{BRIEF_CSS}</style></head>"
                                     f"<body><h1>Decision briefs</h1><div class='sub'>Generated by run_pipeline.py. Open any brief and print to PDF.</div>"
                                     f"<table><tr><th>Set</th><th>Player</th></tr>{idx}</table></body></html>")
