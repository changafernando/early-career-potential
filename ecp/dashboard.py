"""Step 8 - Interactive decision dashboard: a single self-contained HTML file (no server, no internet).

Open dashboard/ecp_dashboard.html in any modern browser. Tabs:
  Overview | Watchlist | Player brief | Replay 2015-18 | Threshold explorer | Model evidence | Monitoring | About
Scout notes and decision logs are kept in the browser (localStorage) and can be exported as CSV.
"""
from __future__ import annotations

import json

import pandas as pd


def _payload(cfg: dict, metrics: dict, sc: dict) -> dict:
    bt = metrics["backtest"]
    P = pd.read_csv(cfg["paths"]["outputs_dir"] / "backtest_predictions.csv")
    h3 = P[P.horizon == 3][["p", "p_draft", "y"]].round(4)
    dc = pd.read_csv(cfg["paths"]["outputs_dir"] / "decision_curve_h3.csv").round(5)
    hz = {h: {k: bt["horizons"][h][k]["auc"] for k in ["champion", "challenger", "draft_baseline"]} for h in bt["horizons"]}
    return {
        "version": cfg["version"], "construct": cfg["construct"]["active"],
        "construct_text": cfg["construct"][cfg["construct"]["active"]]["description"],
        "cost_ratio": cfg["decision"]["cost_ratio"], "threshold": bt["decision_threshold"],
        "latest_season": sc["latest_season"], "selection": metrics["selection"],
        "auc_by_horizon": hz,
        "h3": {k: bt["horizons"]["3"][k] for k in ["champion", "draft_baseline", "shortlists", "calibration",
                                                    "extension_population", "six_season_survivors", "cost_thresholds"]},
        "net_benefit": bt["net_benefit_at_threshold"], "audit": bt["subgroup_audit_h3"],
        "late_picks": bt["late_pick_contributors"], "construct_comparison": bt["construct_comparison"],
        "bt_preds": h3.to_dict("list"), "decision_curve": dc.to_dict("list"),
        "drivers": dict(list(metrics["drivers"]["champion_mean_abs_logodds"].items())[:10]),
        "monitoring": {k: metrics["monitoring"][k] for k in ["checks", "data_drift_by_class", "base_rate_by_class", "league_by_season", "psi_levels"]},
        "live": sc["live"], "replay": sc["replay"],
    }


def build_dashboard(cfg: dict, metrics: dict, sc: dict):
    from .features import PLAIN_NAMES

    data = _payload(cfg, metrics, sc)
    data["plain"] = PLAIN_NAMES
    js = json.dumps(data, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    js = js.replace("</", "<\\/")
    out = cfg["paths"]["dashboard"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__DATA__", js))
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Early-Career Potential</title>
<style>
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,.10);--blue:#2a78d6;--blue-soft:#e7f0fb;--orange:#eb6834;--aqua:#1baf7a;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--chip:#f0efec}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--blue:#3987e5;--blue-soft:#16263a;--orange:#d95926;--aqua:#199e70;--chip:#2c2c2a}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--blue:#3987e5;--blue-soft:#16263a;--orange:#d95926;--aqua:#199e70;--chip:#2c2c2a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:18px 24px 0;max-width:1240px;margin:auto}header h1{font-size:20px;margin:0}header .sub{color:var(--ink2);font-size:13px}
nav{display:flex;gap:4px;flex-wrap:wrap;margin-top:14px;border-bottom:1px solid var(--border)}
nav button{background:none;border:0;padding:9px 12px;font:inherit;color:var(--ink2);cursor:pointer;border-bottom:2px solid transparent}
nav button.on{color:var(--ink);border-bottom-color:var(--blue);font-weight:600}
main{max-width:1240px;margin:auto;padding:18px 24px 60px}section{display:none}section.on{display:block}
.grid{display:grid;gap:14px}.g4{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}.g2{grid-template-columns:repeat(auto-fit,minmax(380px,1fr))}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
.card h3{margin:0 0 4px;font-size:14px}.muted{color:var(--muted);font-size:12px}.ink2{color:var(--ink2)}
.kpi{font-size:32px;font-weight:700;margin:6px 0 2px}.kpi small{font-size:14px;color:var(--ink2);font-weight:500}
table{border-collapse:collapse;width:100%;font-size:13px}th{color:var(--ink2);font-weight:600;text-align:left;border-bottom:1px solid var(--axis);padding:6px}
td{padding:6px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}tr.click{cursor:pointer}tr.click:hover{background:var(--blue-soft)}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;white-space:nowrap}
.t-flag{background:var(--blue-soft);color:var(--blue)}.t-below{background:var(--chip);color:var(--ink2)}.t-insuff{background:rgba(250,178,25,.18);color:#8a5d00}
:root[data-theme="dark"] .t-insuff{color:#fab219}
.bar{position:relative;height:14px;width:150px;background:var(--chip);border-radius:7px}.bar .rng{position:absolute;top:4px;height:6px;background:var(--blue);opacity:.35;border-radius:3px}
.bar .pt{position:absolute;top:1px;width:12px;height:12px;margin-left:-6px;border-radius:50%;background:var(--blue);border:2px solid var(--card)}
.bar .thr{position:absolute;top:-2px;bottom:-2px;width:1px;background:var(--ink2)}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
select,input[type=text],textarea{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--axis);border-radius:8px;padding:6px 8px}
textarea{width:100%;min-height:70px}button.btn{font:inherit;background:var(--blue);color:#fff;border:0;border-radius:8px;padding:7px 12px;cursor:pointer}
button.ghost{font:inherit;background:none;border:1px solid var(--axis);color:var(--ink);border-radius:8px;padding:6px 10px;cursor:pointer}
.status{display:inline-flex;gap:6px;align-items:center;font-weight:600;white-space:nowrap}.dot{width:10px;height:10px;border-radius:50%}
.up{color:var(--blue)}.down{color:var(--crit)}svg text{fill:var(--ink2);font-size:11px}svg .gl{stroke:var(--grid)}svg .ax{stroke:var(--axis)}
#tip{position:fixed;pointer-events:none;background:var(--card);color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:6px 9px;font-size:12px;
box-shadow:0 4px 14px rgba(0,0,0,.15);display:none;z-index:9;max-width:260px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:4px 0 6px}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.big{font-size:44px;font-weight:700;line-height:1.1}.note{background:var(--blue-soft);border-radius:10px;padding:10px 12px;font-size:13px}
input[type=range]{width:260px}.theme{float:right;margin-top:-26px}
@media (max-width:640px){.theme{float:none;margin:8px 0 0}header,main{padding-left:16px;padding-right:16px}.g2{grid-template-columns:1fr}.bar{width:90px}.hide-sm{display:none}}
@media print{nav,.controls,.theme,.noprint{display:none!important}section{display:none!important}section.print{display:block!important}body{background:#fff}}
</style></head><body>
<header><h1>Early-Career Potential &middot; Decision Support</h1>
<div class="sub" id="sub"></div><button class="ghost theme" id="theme">Theme</button>
<nav id="nav"></nav></header>
<main>
<section id="overview"></section><section id="watchlist"></section><section id="brief"></section><section id="replay"></section>
<section id="threshold"></section><section id="evidence"></section><section id="monitoring"></section><section id="about"></section>
</main><div id="tip"></div>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s), fmt=(v,d=0)=>v==null||isNaN(v)?'—':(+v).toFixed(d), pct=v=>v==null?'—':Math.round(v*100)+'%';
const ord=n=>n+((n%100>=10&&n%100<=20)?'th':({1:'st',2:'nd',3:'rd'}[n%10]||'th'));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const store={get(k,d){try{const v=localStorage.getItem(k);return v?JSON.parse(v):d}catch(e){return d}},set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}}};
const TABS=[['overview','Overview'],['watchlist','Watchlist'],['brief','Player brief'],['replay','Replay 2015-18'],['threshold','Threshold explorer'],['evidence','Model evidence'],['monitoring','Monitoring'],['about','About & governance']];
let current=null;
function show(id){document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===id));document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.id===id));store.set('ecp_tab',id);if(id==='evidence')evidence();if(id==='monitoring')monitoring();}
$('#nav').innerHTML=TABS.map(([id,l])=>`<button data-id="${id}">${l}</button>`).join('');
$('#nav').onclick=e=>{if(e.target.dataset.id)show(e.target.dataset.id)};
$('#theme').onclick=()=>{const r=document.documentElement;const dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dark?'light':'dark';store.set('ecp_theme',r.dataset.theme);drawAll();};
{const t=store.get('ecp_theme',null);if(t)document.documentElement.dataset.theme=t;}
const S=D.latest_season, EXT=S-2, season=y=>`${y}-${String(y+1).slice(-2)}`;
$('#sub').textContent=`Model v${D.version} · outcome: ${D.construct} construct · data through the ${season(S)} season · threshold ${D.threshold.toFixed(2)} (cost ratio ${D.cost_ratio}:1) · confidential, front office only`;
const tip=$('#tip');document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');if(t){tip.innerHTML=t.dataset.tip;tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-270)+'px';tip.style.top=(e.clientY+14)+'px'}else tip.style.display='none'});
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function tagFor(s){return `<span class="tag ${s==='Flag for review'?'t-flag':s==='Insufficient evidence'?'t-insuff':'t-below'}">${s}</span>`}
function bar(r){const th=D.threshold*100;if(r.status==='Insufficient evidence')return `<div class="bar" data-tip="Too few minutes (${r.minutes}) for a reliable score"><div class="thr" style="left:${th}%"></div></div>`;
return `<div class="bar" data-tip="${pct(r.p)} (80% range ${pct(r.p_low)}–${pct(r.p_high)})"><div class="rng" style="left:${r.p_low*100}%;width:${Math.max((r.p_high-r.p_low)*100,1)}%"></div><div class="thr" style="left:${th}%"></div><div class="pt" style="left:${r.p*100}%"></div></div>`}

// ------------------------------------------------------------------ overview
function overview(){const c=D.h3.champion,b=D.h3.draft_baseline,s10=D.h3.shortlists.top_10pct,s20=D.h3.shortlists.top_20pct;
const ext=D.live.filter(r=>r.debut===EXT),fl=ext.filter(r=>r.status==='Flag for review').length;
const red=D.monitoring.checks.filter(c=>c.status==='RED').length,amb=D.monitoring.checks.filter(c=>c.status==='AMBER').length;
$('#overview').innerHTML=`<div class="grid g4">
<div class="card"><h3>Ranking accuracy after 3 seasons</h3><div class="kpi">${fmt(c.auc.est,2)} <small>vs ${fmt(b.auc.est,2)} draft order</small></div><div class="muted">ROC-AUC on the untouched ${D.h3.champion.n}-player 2015-18 debut classes (95% CI ${fmt(c.auc.lo,2)}–${fmt(c.auc.hi,2)})</div></div>
<div class="card"><h3>Better shortlists</h3><div class="kpi">${s10.model.contributors} <small>of ${s10.model.size} vs ${s10.draft_order.contributors} by draft order</small></div><div class="muted">Future contributors in a top-10% shortlist of each class</div></div>
<div class="card"><h3>Fewer missed contributors</h3><div class="kpi">${s20.model.missed} <small>of ${s20.model.total_contributors} vs ${s20.draft_order.missed} by draft order</small></div><div class="muted">Contributors left outside a top-20% shortlist</div></div>
<div class="card"><h3>Net benefit at the 3:1 threshold</h3><div class="kpi">${fmt(D.net_benefit.Model,1)} <small>vs ${fmt(D.net_benefit['Draft order'],1)} per 100 players</small></div><div class="muted">Correct flags minus cost-weighted false flags (decision curve)</div></div></div>
<div class="grid g2" style="margin-top:14px"><div class="card"><h3>${season(EXT)} debut class · rookie-scale extension window</h3>
<p class="ink2">${ext.length} players still in the league. <b>${fl}</b> are flagged for review at the ${D.threshold.toFixed(2)} threshold; ${ext.filter(r=>r.status==='Insufficient evidence').length} have too few minutes to score.</p>
<table><tr><th>Player</th><th>Draft</th><th>Probability</th><th class="hide-sm">Status</th></tr>${ext.slice().sort((a,b)=>b.p-a.p).slice(0,8).map(r=>`<tr class="click" data-pid="${r.personId}"><td>${esc(r.name)}</td><td>${r.draft}</td><td>${bar(r)}</td><td class="hide-sm">${tagFor(r.status)}</td></tr>`).join('')}</table>
<p class="muted">Click a player for the full brief. Scores are screening inputs, never decisions.</p></div>
<div class="card"><h3>Model health</h3><p class="ink2">${red} red, ${amb} amber and ${D.monitoring.checks.length-red-amb} green checks. See the Monitoring tab for actions.</p>
${D.monitoring.checks.map(c=>`<div style="margin:6px 0">${statusChip(c.status)} <span class="ink2">${esc(c.check)}</span></div>`).join('')}
<div class="note" style="margin-top:12px"><b>How to read a score.</b> 40% means that, of past players with a similar first three seasons, about 4 in 10 became high-impact contributors. The shaded range shows how much the score moves if the model is retrained on slightly different history.</div></div></div>`;
document.querySelectorAll('#overview tr.click').forEach(t=>t.onclick=()=>openBrief(+t.dataset.pid,'live'));}
function statusChip(s){const col={RED:'--crit',AMBER:'--warn',GREEN:'--good'}[s]||'--muted';const icon={RED:'✕',AMBER:'!',GREEN:'✓'}[s]||'·';return `<span class="status"><span class="dot" style="background:var(${col})"></span>${icon} ${s}</span>`}

// ------------------------------------------------------------------ watchlist
function listView(sec,rows,isReplay){const classes=[...new Set(rows.map(r=>r.debut))].sort();const key='ecp_'+sec;const st=store.get(key,{cls:isReplay?'all':String(EXT),status:'all',q:''});
const el=$('#'+sec);el.innerHTML=`<div class="controls"><label>Debut class <select id="${sec}-cls"><option value="all">All</option>${classes.map(c=>`<option value="${c}">${season(c)}${c===EXT&&!isReplay?' (extension window)':''}</option>`).join('')}</select></label>
<label>Status <select id="${sec}-st"><option value="all">All</option><option>Flag for review</option><option>Below threshold</option><option>Insufficient evidence</option></select></label>
<input type="text" id="${sec}-q" placeholder="Search player"> <button class="ghost" id="${sec}-csv">Export CSV</button></div>
${isReplay?'<p class="ink2">Replay: each 2015-18 debut class scored after three seasons by a model trained only on 1996-2014 classes. Outcomes are now known.</p>':`<p class="ink2">Live scores for players active in ${season(S)}. Evidence: ${season(S-2)} debuts have 3 seasons, ${season(S-1)} debuts 2, ${season(S)} debuts 1.</p>`}
<div class="card" style="overflow-x:auto"><table id="${sec}-tbl"></table></div>`;
const f=()=>{const cls=$(`#${sec}-cls`).value,stt=$(`#${sec}-st`).value,q=$(`#${sec}-q`).value.toLowerCase();store.set(key,{cls,status:stt,q});
return rows.filter(r=>(cls==='all'||r.debut==cls)&&(stt==='all'||r.status===stt)&&r.name.toLowerCase().includes(q)).sort((a,b)=>b.p-a.p)};
const render=()=>{const rs=f();$(`#${sec}-tbl`).innerHTML=`<tr><th>#</th><th>Player</th><th>Class</th><th class="hide-sm">Pos</th><th>Draft</th><th>Probability and 80% range</th><th>Score</th><th class="hide-sm">Challenger</th><th>Status</th>${isReplay?'<th>Outcome</th>':''}</tr>`+
rs.map(r=>`<tr class="click" data-pid="${r.personId}"><td>${r.class_rank}/${r.class_size}</td><td>${esc(r.name)}${r.models_disagree?' <span class="tag t-insuff" data-tip="Champion and challenger differ by 15+ points: analyst review">review</span>':''}</td><td>${season(r.debut)}</td><td class="hide-sm">${r.position}</td><td>${r.draft}</td><td>${bar(r)}</td><td>${r.status==='Insufficient evidence'?'—':pct(r.p)}</td><td class="hide-sm">${pct(r.p_challenger)}</td><td>${tagFor(r.status)}</td>${isReplay?`<td>${r.outcome?'<b>Contributor</b>':'No'}</td>`:''}</tr>`).join('');
el.querySelectorAll('tr.click').forEach(t=>t.onclick=()=>openBrief(+t.dataset.pid,isReplay?'replay':'live'));};
$(`#${sec}-cls`).value=st.cls;$(`#${sec}-st`).value=st.status;$(`#${sec}-q`).value=st.q;['cls','st'].forEach(k=>$(`#${sec}-${k}`).onchange=render);$(`#${sec}-q`).oninput=render;
$(`#${sec}-csv`).onclick=()=>{const rs=f();const cols=['name','debut','position','draft','p','p_low','p_high','p_challenger','status','class_rank','class_size','minutes'].concat(isReplay?['outcome']:[]);
download(`${sec}.csv`,[cols.join(',')].concat(rs.map(r=>cols.map(c=>JSON.stringify(r[c]??'')).join(','))).join('\n'))};render();}
function download(name,text){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/csv'}));a.download=name;a.click();}

// ------------------------------------------------------------------ brief
function openBrief(pid,set){const r=(set==='replay'?D.replay:D.live).find(x=>x.personId===pid);if(!r)return;current={r,set};renderBrief();show('brief');window.scrollTo(0,0)}
function renderBrief(){const el=$('#brief');if(!current){el.innerHTML='<div class="card"><p>Select a player in the Watchlist or Replay tab.</p></div>';return}
const {r,set}=current;const logKey='ecp_log';const log=store.get(logKey,[]);const mine=log.filter(l=>l.personId===r.personId);
const insuff=r.status==='Insufficient evidence';
el.innerHTML=`<div class="controls noprint"><button class="ghost" onclick="show('${set==='replay'?'replay':'watchlist'}')">← Back to list</button><button class="ghost" onclick="document.querySelectorAll('section').forEach(s=>s.classList.toggle('print',s.id==='brief'));print()">Print / save PDF</button></div>
<div class="muted">DECISION BRIEF · ${set==='replay'?'REPLAY (OUTCOME KNOWN)':'LIVE'} · CONFIDENTIAL</div><h2 style="margin:4px 0 2px;font-size:24px">${esc(r.name)}</h2>
<div class="ink2">${r.position} · ${r.draft} (${r.draft_tier}) · debut ${season(r.debut)} · evidence: ${r.horizon} season(s), ${r.games} games, ${r.minutes.toLocaleString()} minutes · age at debut ${fmt(r.age_debut,1)}</div>
<div class="grid g4" style="margin-top:14px"><div class="card"><div class="muted">Probability of becoming a high-impact contributor (career years 4–8)</div><div class="big">${insuff?'—':pct(r.p)}</div>
<div class="muted">80% range ${pct(r.p_low)}–${pct(r.p_high)} · challenger ${pct(r.p_challenger)}</div>${r.models_disagree?'<div class="tag t-insuff" style="margin-top:6px">Models disagree: analyst review</div>':''}</div>
<div class="card"><div class="muted">Status at threshold ${D.threshold.toFixed(2)}</div><div style="margin:10px 0">${tagFor(r.status)}</div><div class="muted">Ranked ${r.class_rank} of ${r.class_size} in the ${season(r.debut)} debut class</div>${insuff?'<p class="muted">Below '+500+' minutes the brief shows "insufficient evidence" instead of a low score: late bloomers often look like this.</p>':''}</div>
${set==='replay'?`<div class="card"><div class="muted">Actual outcome</div><div style="font-size:20px;font-weight:700;margin-top:8px">${r.outcome?'Became a contributor':'Did not become a contributor'}</div><div class="muted">${r.high_impact_seasons} high-impact season(s) in years 4–8</div></div>`:''}
<div class="card"><div class="muted">Early-career profile</div><table><tr><td>Minutes per game</td><td>${fmt(r.mpg,1)}</td></tr><tr><td>Game Score per 100 vs league</td><td>${r.gmsc_100==null?'—':(r.gmsc_100>0?'+':'')+fmt(r.gmsc_100,1)}</td></tr><tr><td>True shooting vs league</td><td>${r.ts_vs_league==null?'—':(r.ts_vs_league>0?'+':'')+fmt(r.ts_vs_league*100,1)+' pts'}</td></tr><tr><td>Best season rank, per possession</td><td>${r.best_rate_pct==null?'—':ord(Math.round(r.best_rate_pct*100))+' percentile'}</td></tr></table></div></div>
<div class="grid g2" style="margin-top:14px"><div class="card"><h3>What moves the score</h3><table>${r.drivers_up.map(d=>`<tr><td class="up">▲ ${esc(d.label)}</td><td>${esc(d.value)}</td></tr>`).join('')}${r.drivers_down.map(d=>`<tr><td class="down">▼ ${esc(d.label)}</td><td>${esc(d.value)}</td></tr>`).join('')}</table><p class="muted">Ranked by each statistic's contribution to this player's score (champion logistic model).</p></div>
<div class="card"><h3>Most similar historical players at the same stage</h3><table><tr><th>Player</th><th>Debut</th><th>Contributor?</th><th>High-impact seasons</th></tr>${r.comparables.map(c=>`<tr><td>${esc(c.name)}</td><td>${c.debut}</td><td>${c.contributor?'<b>Yes</b>':'No'}</td><td>${c.high_impact_seasons}</td></tr>`).join('')}</table><p class="muted">${r.comparables.filter(c=>c.contributor).length} of 5 comparables became contributors.</p></div></div>
<div class="card noprint" style="margin-top:14px"><h3>Scout / coach view and decision log</h3><p class="muted">Required before any extension, trade or development decision. Saved in this browser only; export for the decision record.</p>
<div class="grid g2"><div><label class="muted">Context the model cannot see</label><textarea id="note"></textarea></div>
<div><label class="muted">Decision</label><br><select id="dec"><option>Extend / pursue</option><option>Monitor</option><option>Development plan</option><option>No action</option></select>
<label class="muted" style="margin-left:8px">Agrees with model?</label> <select id="agree"><option>Yes</option><option>No - override</option></select><br><br>
<label class="muted">Decided by</label> <input type="text" id="by" placeholder="Name / role"> <button class="btn" id="save">Log decision</button> <button class="ghost" id="exp">Export log</button></div></div>
<table style="margin-top:10px">${mine.map(l=>`<tr><td>${l.when}</td><td>${esc(l.decision)}</td><td>${esc(l.agree)}</td><td>${esc(l.by)}</td><td>${esc(l.note)}</td></tr>`).join('')}</table></div>
<p class="muted">Screening input only. Outcome definition: ${esc(D.construct_text)} in at least two of career years 4–8. Scores expire each season and are never shared outside the front office.</p>`;
$('#save').onclick=()=>{const e={personId:r.personId,name:r.name,p:r.p,status:r.status,decision:$('#dec').value,agree:$('#agree').value,by:$('#by').value,note:$('#note').value,when:new Date().toISOString().slice(0,16).replace('T',' ')};
if(e.agree.startsWith('No')&&!e.note.trim()){alert('An override needs a recorded reason.');return}const l=store.get(logKey,[]);l.push(e);store.set(logKey,l);renderBrief();};
$('#exp').onclick=()=>{const l=store.get(logKey,[]);const cols=['when','name','p','status','decision','agree','by','note'];download('decision_log.csv',[cols.join(',')].concat(l.map(x=>cols.map(c=>JSON.stringify(x[c]??'')).join(','))).join('\n'))};}

// ------------------------------------------------------------------ svg helpers
function axes(w,h,m,xs,ys,xt,yt,xl,yl){let g='';yt.forEach(v=>{const y=ys(v);g+=`<line class="gl" x1="${m.l}" x2="${w-m.r}" y1="${y}" y2="${y}"/><text x="${m.l-6}" y="${y+4}" text-anchor="end">${yl(v)}</text>`});
xt.forEach(v=>{g+=`<text x="${xs(v)}" y="${h-m.b+16}" text-anchor="middle">${xl(v)}</text>`});g+=`<line class="ax" x1="${m.l}" x2="${w-m.r}" y1="${h-m.b}" y2="${h-m.b}"/>`;return g}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
function lineChart(id,series,opt){const el=$(id);const w=el.clientWidth||520,h=opt.h||240,m={l:46,r:14,t:10,b:30};const xs=lin(opt.x0,opt.x1,m.l,w-m.r),ys=lin(opt.y0,opt.y1,h-m.b,m.t);
let s=`<svg width="${w}" height="${h}" role="img">`+axes(w,h,m,xs,ys,opt.xt,opt.yt,opt.xl||(v=>v),opt.yl||(v=>v));
(opt.vlines||[]).forEach(v=>s+=`<line x1="${xs(v)}" x2="${xs(v)}" y1="${m.t}" y2="${h-m.b}" stroke="${css('--ink2')}" stroke-dasharray="3 3"/>`);
series.forEach(se=>{const pts=se.x.map((x,i)=>[xs(x),ys(se.y[i])]).filter(p=>!isNaN(p[1]));s+=`<polyline fill="none" stroke="${se.color}" stroke-width="2" ${se.dash?'stroke-dasharray="5 4"':''} points="${pts.map(p=>p.join(',')).join(' ')}"/>`;
if(se.dots)se.x.forEach((x,i)=>s+=`<circle cx="${xs(x)}" cy="${ys(se.y[i])}" r="4" fill="${se.color}" stroke="${css('--card')}" stroke-width="2"/>`);
se.x.forEach((x,i)=>s+=`<rect x="${xs(x)-4}" y="${m.t}" width="8" height="${h-m.b-m.t}" fill="transparent" data-tip="<b>${esc(se.name)}</b><br>${opt.tipx?opt.tipx(x):x}: ${opt.tipy?opt.tipy(se.y[i]):se.y[i]}"/>`)});
el.innerHTML=s+'</svg>'}
function legend(items){return `<div class="legend">${items.map(([n,c])=>`<span><i style="background:${c}"></i>${n}</span>`).join('')}</div>`}

// ------------------------------------------------------------------ threshold explorer
function threshold(){const el=$('#threshold');el.innerHTML=`<div class="card"><h3>How the cost of mistakes sets the threshold</h3>
<p class="ink2">The steering committee agrees how much worse it is to <b>miss</b> a future contributor than to spend an offer or development slot on a player who does not develop. Because the probabilities are calibrated, that ratio fixes the threshold: <b>threshold = 1 ÷ (1 + ratio)</b>.</p>
<div class="controls"><label>Missing a contributor costs <b id="rv"></b> × a wasted offer</label><input type="range" id="ratio" min="1" max="8" step="0.5"></div>
<div class="grid g4" id="tcards"></div><p class="muted">Counts from the 2015-18 backtest (${D.bt_preds.y.length} players, ${D.bt_preds.y.reduce((a,b)=>a+b,0)} contributors), three seasons of evidence. The live column counts current ${season(EXT)}-class players above the threshold.</p></div>`;
const r=$('#ratio');r.value=store.get('ecp_ratio',D.cost_ratio);const upd=()=>{const ratio=+r.value,t=1/(1+ratio);store.set('ecp_ratio',ratio);$('#rv').textContent=ratio;
const c=(key)=>{let tp=0,fp=0,fn=0;D.bt_preds.y.forEach((y,i)=>{const f=D.bt_preds[key][i]>=t;if(f&&y)tp++;else if(f)fp++;else if(y)fn++});return {tp,fp,fn,cost:fp+ratio*fn}};
const m=c('p'),d=c('p_draft');const live=D.live.filter(x=>x.debut===EXT&&x.status!=='Insufficient evidence'&&x.p>=t).length;
$('#tcards').innerHTML=`<div class="card"><div class="muted">Threshold</div><div class="kpi">${t.toFixed(2)}</div></div>
<div class="card"><div class="muted">Model: contributors caught / missed</div><div class="kpi">${m.tp} <small>/ ${m.fn}</small></div><div class="muted">${m.fp} false flags · weighted cost ${fmt(m.cost,0)}</div></div>
<div class="card"><div class="muted">Draft order: caught / missed</div><div class="kpi">${d.tp} <small>/ ${d.fn}</small></div><div class="muted">${d.fp} false flags · weighted cost ${fmt(d.cost,0)}</div></div>
<div class="card"><div class="muted">${season(EXT)} class flagged today</div><div class="kpi">${live}</div><div class="muted">players to brief</div></div>`};r.oninput=upd;upd();}

// ------------------------------------------------------------------ evidence
function evidence(){const el=$('#evidence');const B=css('--blue'),O=css('--orange'),A=css('--aqua'),G=css('--muted');const a=D.auc_by_horizon;
el.innerHTML=`<div class="grid g2"><div class="card"><h3>Accuracy by seasons of evidence (ROC-AUC, 2015-18 classes)</h3>${legend([['Model',B],['Challenger (random forest)',A],['Draft order',O]])}<div id="c-auc"></div></div>
<div class="card"><h3>Decision curve: value of acting on the scores</h3>${legend([['Model',B],['Draft order',O],['Flag everyone',G]])}<div id="c-dc"></div><p class="muted">Net benefit = correct flags minus false flags weighted by the threshold's cost ratio (Vickers and Elkin 2006). Dashed line: threshold at the 3:1 cost ratio.</p></div>
<div class="card"><h3>Calibration: does 40% mean 40%?</h3><div id="c-cal"></div><p class="muted">Each dot is a band of predictions; the dashed diagonal is perfect calibration.</p></div>
<div class="card"><h3>What drives scores (average influence)</h3><div id="c-drv"></div></div>
<div class="card"><h3>Late picks the model found</h3><p class="ink2">${D.late_picks.flagged_at_threshold} of ${D.late_picks.count} second-round or undrafted contributors were flagged after three seasons; non-contributing late picks averaged ${pct(D.late_picks.late_pick_non_contributors_mean_p)}.</p>
<table><tr><th>Player</th><th>Pick</th><th>Score</th></tr>${D.late_picks.players.map(p=>`<tr><td>${esc(p.name)}</td><td>${p.pick>60?'Undrafted':p.pick}</td><td>${pct(p.p)}</td></tr>`).join('')}</table></div><div class="card" style="grid-column:1/-1"><h3>Fairness audit (2015-18, threshold ${D.threshold.toFixed(2)})</h3><div style="overflow-x:auto"><table><tr><th>Group</th><th>n</th><th>Contributors</th><th>AUC</th><th>Predicted</th><th>Observed</th><th>Missed</th><th>False flags</th></tr>
${D.audit.filter(r=>r.group!=='Unknown').map(r=>`<tr style="${r.judged?'':'color:var(--muted)'}"><td>${esc(r.group)}</td><td>${r.n}</td><td>${r.contributors}</td><td>${fmt(r.auc,2)}</td><td>${pct(r.mean_pred)}</td><td>${pct(r.observed)}</td><td>${pct(r.miss_rate)}</td><td>${pct(r.false_flag_rate)}</td></tr>`).join('')}</table></div>
<p class="muted">Grey rows are too small to judge. Origin is audited but never used as a model input.</p></div>
</div>`;
const H=[1,2,3];const dc=D.decision_curve;const c=D.h3.calibration;
lineChart('#c-auc',[['champion',B,'Model'],['challenger',A,'Challenger'],['draft_baseline',O,'Draft order']].map(([k,col,n])=>({name:n,color:col,dots:1,x:H,y:H.map(h=>a[h][k].est)})),{x0:.8,x1:3.2,y0:.7,y1:1,xt:H,yt:[.7,.8,.9,1],xl:v=>v+' season'+(v>1?'s':''),yl:v=>v.toFixed(1),tipx:v=>v+' season(s)',tipy:v=>v.toFixed(3)});
lineChart('#c-dc',[{name:'Model',color:B,x:dc.threshold,y:dc.Model.map(v=>v*100)},{name:'Draft order',color:O,x:dc.threshold,y:dc['Draft order'].map(v=>v*100)},{name:'Flag everyone',color:G,dash:1,x:dc.threshold,y:dc['Flag everyone'].map(v=>Math.max(v*100,-5))}],
{x0:0,x1:.6,y0:-5,y1:15,xt:[0,.1,.2,.3,.4,.5,.6],yt:[-5,0,5,10,15],xl:v=>v.toFixed(1),vlines:[D.threshold],tipx:v=>'threshold '+v.toFixed(2),tipy:v=>v.toFixed(1)+' per 100'});
{const el=$('#c-cal');const w=el.clientWidth||400,h=260,m={l:46,r:14,t:10,b:30},xs=lin(0,1,m.l,w-m.r),ys=lin(0,1,h-m.b,m.t);let s=`<svg width="${w}" height="${h}">`+axes(w,h,m,xs,ys,[0,.25,.5,.75,1],[0,.25,.5,.75,1],v=>pct(v),v=>pct(v));
s+=`<line x1="${xs(0)}" y1="${ys(0)}" x2="${xs(1)}" y2="${ys(1)}" stroke="${G}" stroke-dasharray="4 4"/>`;c.forEach(b=>s+=`<circle cx="${xs(b.mean_pred)}" cy="${ys(b.observed)}" r="${4+Math.sqrt(b.n)/2}" fill="${B}" stroke="${css('--card')}" stroke-width="2" data-tip="Predicted ${pct(b.mean_pred)}<br>Observed ${pct(b.observed)}<br>${b.n} players"/>`);el.innerHTML=s+'</svg>'}
{const el=$('#c-drv');const e=Object.entries(D.drivers);const mx=Math.max(...e.map(x=>x[1]));el.innerHTML=e.map(([k,v])=>`<div style="display:flex;align-items:center;gap:8px;margin:5px 0" data-tip="${esc(D.plain[k]||k)}: ${v.toFixed(3)}"><div style="flex:0 0 190px;font-size:12px" class="ink2">${esc(D.plain[k]||k)}</div><div style="height:12px;border-radius:0 4px 4px 0;background:${B};width:${v/mx*55}%"></div></div>`).join('')}}

// ------------------------------------------------------------------ monitoring
function monitoring(){const el=$('#monitoring');const M=D.monitoring;const lg=M.league_by_season.filter(r=>r.season>=1996);const br=Object.entries(M.base_rate_by_class);const B=css('--blue');
el.innerHTML=`<div class="card"><h3>Health checks (detect → understand → adapt)</h3><table><tr><th>Check</th><th>Value</th><th>Status</th><th>Action in the runbook</th></tr>
${M.checks.map(c=>`<tr><td>${esc(c.check)}</td><td>${typeof c.value==='number'?fmt(c.value,3):esc(c.value)}</td><td>${statusChip(c.status)}</td><td class="ink2">${esc(c.action)}</td></tr>`).join('')}</table></div>
<div class="grid g2" style="margin-top:14px"><div class="card"><h3>League three-point share of shots</h3><div id="m-3p"></div></div><div class="card"><h3>Contributor rate by debut class</h3><div id="m-br"></div></div>
<div class="card" style="grid-column:1/-1"><h3>Feature drift by debut class (population stability index, first-season features)</h3><div style="overflow-x:auto"><table><tr><th>Class</th><th>n</th>${Object.keys(M.data_drift_by_class[0]).filter(k=>k.startsWith('psi_')).map(k=>`<th>${esc((D.plain[k.slice(4)]||k.slice(4)))}</th>`).join('')}</tr>
${M.data_drift_by_class.map(r=>`<tr><td>${season(r.debut)}</td><td>${r.n}</td>${Object.keys(r).filter(k=>k.startsWith('psi_')).map(k=>{const L=M.psi_levels[k.slice(4)],v=r[k],lv=v>=L.red?'RED':v>=L.amber?'AMBER':'GREEN';return `<td data-tip="${esc(D.plain[k.slice(4)]||k)}: PSI ${fmt(v,2)}<br>amber from ${fmt(L.amber,2)}, red from ${fmt(L.red,2)}" style="${lv==='RED'?'color:var(--crit);font-weight:600':lv==='AMBER'?'color:#b07a00':''}">${fmt(v,2)}${lv==='RED'?' ✕':lv==='AMBER'?' !':''}</td>`}).join('')}</tr>`).join('')}
<tr><td colspan="2" class="muted">Red from</td>${Object.keys(M.data_drift_by_class[0]).filter(k=>k.startsWith('psi_')).map(k=>`<td class="muted">${fmt(M.psi_levels[k.slice(4)].red,2)}</td>`).join('')}</tr></table></div>
<p class="muted">Classes that debuted after 2018 have no outcomes yet, so drift is the early warning. Alarm levels come from history: a value is amber (!) above the 75th and red (✕) above the 95th percentile of how much past classes differed from each other.</p></div></div>`;
lineChart('#m-3p',[{name:'3-point share',color:B,x:lg.map(r=>r.season),y:lg.map(r=>r.three_pa_share*100)}],{x0:1996,x1:S,y0:0,y1:50,xt:[2000,2005,2010,2015,2020,2025],yt:[0,10,20,30,40,50],yl:v=>v+'%',tipx:v=>season(v),tipy:v=>v.toFixed(1)+'%'});
lineChart('#m-br',[{name:'Contributor rate',color:B,dots:1,x:br.map(x=>+x[0]),y:br.map(x=>x[1]*100)}],{x0:1995.5,x1:2018.5,y0:0,y1:30,xt:[1996,2000,2005,2010,2015,2018],yt:[0,10,20,30],yl:v=>v+'%',tipx:v=>season(v)+' class',tipy:v=>v.toFixed(1)+'%'});}

// ------------------------------------------------------------------ about
function about(){const cc=D.construct_comparison;$('#about').innerHTML=`<div class="grid g2"><div class="card"><h3>Intended use</h3><ul class="ink2"><li>Screening and prioritisation for rookie-scale extension, trade-deadline and development decisions.</li><li>One input to a documented human decision, alongside scout and coach judgement.</li></ul>
<h3>Not for</h3><ul class="ink2"><li>Automated decisions, public statements, or sharing with players, agents or other teams.</li><li>Contract valuation (no salary data), injury or medical judgements.</li><li>Players with under 500 NBA minutes (shown as insufficient evidence).</li></ul></div>
<div class="card"><h3>Decision rights</h3><table><tr><th>Decision</th><th>Decides</th></tr><tr><td>Extension or trade call</td><td>GM, with scout/coach view and logged rationale</td></tr><tr><td>Override a score</td><td>Scout/coach with recorded reason → GM</td></tr><tr><td>Outcome definition and cost ratio</td><td>Steering committee</td></tr><tr><td>Release, retrain or retire a model</td><td>Model Review Board (GM chair, Head of Analytics, Director of Scouting, Player Relations/Legal)</td></tr><tr><td>Player-equity or privacy concern</td><td>Model Review Board, which can suspend use</td></tr></table></div>
<div class="card"><h3>Outcome definitions compared (same model, 2015-18 classes)</h3><table><tr><th></th><th>Rate (active)</th><th>Volume (A2 prototype)</th></tr>
<tr><td>Contributor base rate</td><td>${pct(cc.rate.base_rate_all)}</td><td>${pct(cc.volume.base_rate_all)}</td></tr><tr><td>ROC-AUC</td><td>${fmt(cc.rate.model.auc,3)}</td><td>${fmt(cc.volume.model.auc,3)}</td></tr><tr><td>Brier score</td><td>${fmt(cc.rate.model.brier,3)}</td><td>${fmt(cc.volume.model.brier,3)}</td></tr><tr><td>Calibration error</td><td>${fmt(cc.rate.model.ece,3)}</td><td>${fmt(cc.volume.model.ece,3)}</td></tr></table>
<p class="muted">Agreement: ${cc.agreement.both} players are contributors under both; ${cc.agreement.volume_only} only under volume (e.g. ${cc.examples_volume_only.slice(0,3).map(esc).join(', ')}); ${cc.agreement.rate_only} only under rate (e.g. ${cc.examples_rate_only.slice(0,3).map(esc).join(', ')}).</p></div>
<div class="card"><h3>Model</h3><p class="ink2">Champion: ${esc(D.selection.champion.model)} with ${esc(D.selection.champion.calibration)} calibration. Challenger: ${esc(D.selection.challenger.model)}. Selected on the 2011-14 validation classes by average Brier score across 1, 2 and 3 seasons of evidence; tested once on 2015-18. Full details: docs/model_card.md.</p>
<p class="muted">Browser storage holds only your theme, filters, threshold slider and decision log on this device. Export the log for the official record.</p></div></div>`}

function drawAll(){overview();listView('watchlist',D.live,false);listView('replay',D.replay,true);renderBrief();threshold();evidence();monitoring();about();}
drawAll();show(store.get('ecp_tab','overview')==='brief'?'overview':store.get('ecp_tab','overview'));
let rz;addEventListener('resize',()=>{clearTimeout(rz);rz=setTimeout(()=>{evidence();monitoring()},200)});
</script></body></html>
"""
