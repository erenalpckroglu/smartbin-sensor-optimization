"""
Pick-size explorer: one self-contained interactive HTML page (no internet needed).

Pick a scale type and a sensitivity -> every pick-size range is shown as GREEN (reliably detected),
AMBER or RED, with its confidence interval and what that sensitivity costs in empty measurements.
A second view answers the reverse question a planner asks: "parts are taken in 5-10 g portions ->
what is the lowest sensitivity that still detects them reliably?"
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def write_pick_size_explorer(zones: pd.DataFrame, summary: pd.DataFrame, reverse: pd.DataFrame, cfg,
                             floor: dict | None = None) -> Path:
    data = {"thresholds": {"green": cfg.pick_size.green_zone, "red": cfg.pick_size.red_zone,
                           "min_picks": cfg.pick_size.min_picks_per_cell}, "scales": {}}
    synthetic = (Path(cfg.paths.measurements).parent / "ground_truth.json").exists()
    for scale, z in zones.groupby("scale_type"):
        bins = list(z.sort_values("bin_idx").pick_bin.unique())
        levels = sorted(float(x) for x in z.sensitivity.unique())
        cells = {}
        for k in levels:
            g = z[z.sensitivity == k].set_index("pick_bin")
            cells[str(int(k))] = [
                {key: _clean(g.loc[b, col]) if b in g.index else None
                 for key, col in [("picks", "picks"), ("det", "detection"), ("lo", "ci_lo"), ("hi", "ci_hi"),
                                  ("zone", "zone")]} for b in bins]
        sm = summary[summary.scale_type == scale].set_index("sensitivity")
        summ = {str(int(k)): {c: _clean(sm.loc[k, c]) for c in ["green_from_label", "red_up_to_label",
                                                                 "frac_too_many", "meas_per_day", "devices"]}
                for k in levels if k in sm.index}
        rv = reverse[reverse.scale_type == scale] if len(reverse) else pd.DataFrame()
        rev = {r.pick_bin: _clean(r.min_green_sensitivity) for r in rv.itertuples()} if len(rv) else {}
        below = sorted(set(z[z.zone == "below threshold"].pick_bin), key=bins.index)
        fl = (floor or {}).get(scale)
        data["scales"][scale] = {"bins": bins, "levels": [int(k) for k in levels], "cells": cells,
                                 "summary": summ, "reverse": rev, "below": below,
                                 "floor": None if fl is None else round(float(fl))}
    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace(
        "__BANNER__", "<p class='banner'>Synthetic demo data - the numbers are not Bossard's.</p>" if synthetic else "")
    out = cfg.paths.reports / "pick_size_explorer.html"
    out.write_text(html, encoding="utf-8")
    return out


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pick-size explorer</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#141a22;--ink2:#4a5260;--muted:#7c8594;--line:#dde2e8;
--green:#0ca30c;--amber:#fab219;--red:#d03b3b;--gray:#d7dbe0;--accent:#2a78d6;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0f1318;--card:#171c23;--ink:#e8edf3;--ink2:#b3bcc8;--muted:#8792a1;
--line:#2a313b;--gray:#333b46;--accent:#4b93ea;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:24px 18px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:26px 0 8px}
.sub{color:var(--ink2);margin:0 0 14px;max-width:75ch}.banner{background:#eaf2fc;color:#1c4f8a;padding:8px 12px;border-radius:8px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:14px 0}
label{font-size:12px;color:var(--ink2);display:flex;flex-direction:column;gap:4px}
select{font:inherit;padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .l{font-size:12px;color:var(--ink2)}.tile .v{font-size:24px;font-weight:650;margin-top:2px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
svg{display:block;width:100%;height:auto;min-width:620px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:8px 0}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left}
th{font-size:12px;color:var(--ink2)}td.c{text-align:center;font-variant-numeric:tabular-nums}
.hm td.cell{text-align:center;font-variant-numeric:tabular-nums;color:#fff;font-weight:600;cursor:pointer;border:2px solid var(--card)}
.hm td.cell.amber{color:#141a22}.hm td.cell.nd{color:var(--muted);font-weight:400}
.hatch{background:repeating-linear-gradient(45deg,var(--gray) 0 3px,transparent 3px 7px)!important}
.hm td.cell.bt{color:var(--muted);font-weight:400}
.hm tr.sel td.lvl{font-weight:700;color:var(--accent)}.hm tr.sel td.cell{outline:2px solid var(--ink);outline-offset:-2px}
.note{font-size:13px;color:var(--muted);max-width:80ch}
</style></head><body><main>
<h1>Pick-size explorer</h1>
<p class="sub">Which pick sizes does each sensitivity level detect reliably? A pick is detected when the vibration sensor
fires (AUTOMATIC + OK) and missed when only a later manual / 24 h check finds the change (TOO_FEW).</p>
__BANNER__
<div class="controls">
  <label for="scale">Scale type<select id="scale"></select></label>
  <label for="level">Sensitivity<select id="level"></select></label>
</div>
<div class="tiles" id="tiles"></div>
<div class="legend"><span><i class="sw" style="background:var(--green)"></i>green: reliably detected</span>
<span><i class="sw" style="background:var(--amber)"></i>amber</span><span><i class="sw" style="background:var(--red)"></i>red: often missed</span>
<span><i class="sw" style="background:var(--gray)"></i>not enough picks</span>
<span><i class="sw hatch"></i>below the scale's significance threshold: misses are never recorded, not measurable</span></div>
<div class="card"><svg id="bars" viewBox="0 0 900 330" role="img" aria-label="Detection rate per pick size"></svg></div>
<h2>All levels at once (click a row)</h2>
<div class="card"><table class="hm" id="heat"></table></div>
<h2>Reverse view: lowest sensitivity that is green for a pick size</h2>
<div class="card"><table id="rev"></table></div>
<p class="note">Caveat 1: one TOO_FEW can contain several missed picks added together, so missed picks look larger than they
were. Real thresholds are therefore likely a bit LOWER than shown (conservative). Bars show the Wilson 95% interval.</p>
<p class="note">Caveat 2: a change smaller than the scale's significance threshold is never labelled TOO_FEW, and an automatic
weighing of it is labelled TOO_MANY. Below that threshold detection cannot be measured, so those sizes are hatched.</p>
</main>
<script>
const D = __DATA__;
const T = D.thresholds;
const $ = (id) => document.getElementById(id);
const col = {green:"var(--green)", amber:"var(--amber)", red:"var(--red)", "no data":"var(--gray)", "below threshold":"transparent"};
const pct = (v) => v == null ? "–" : Math.round(v * 100) + "%";
function fillScales(){ const s=$("scale"); Object.keys(D.scales).forEach(k=>{const o=document.createElement("option");o.value=k;o.textContent=k;s.appendChild(o);}); }
function fillLevels(){ const S=D.scales[$("scale").value], l=$("level"); const keep=l.value; l.innerHTML="";
  S.levels.slice().reverse().forEach(k=>{const o=document.createElement("option");o.value=k;o.textContent="sensitivity "+k;l.appendChild(o);});
  if (S.levels.map(String).includes(keep)) l.value=keep; else l.value=String(S.levels.includes(8)?8:S.levels[S.levels.length-1]); }
function tiles(S, lvl){ const s=S.summary[lvl]||{};
  $("tiles").innerHTML = [["Green zone starts at", s.green_from_label==="never"?"never":(s.green_from_label||"–")+" g"],
   ["Red up to", s.red_up_to_label==="none"?"no red zone":(s.red_up_to_label||"–")+" g"],
   ["Cost: TOO_MANY fraction", s.frac_too_many==null?"–":s.frac_too_many.toFixed(2)],
   ["Cost: measurements / day", s.meas_per_day==null?"–":s.meas_per_day.toFixed(1)],
   ["Scale registers changes from", S.floor==null?"–":"~"+S.floor+" g"]]
   .map(([l,v])=>`<div class="tile"><div class="l">${l}</div><div class="v">${v}</div></div>`).join(""); }
function bars(S, lvl){ const cells=S.cells[lvl], n=S.bins.length, W=900, H=330, L=50, R=14, B=50, top=16;
  const bw=(W-L-R)/n, y=(v)=>top+(1-v)*(H-top-B); let g="";
  [0,0.25,0.5,0.75,1].forEach(t=>{g+=`<line x1="${L}" x2="${W-R}" y1="${y(t)}" y2="${y(t)}" stroke="var(--line)"/>
    <text x="${L-8}" y="${y(t)+4}" text-anchor="end" font-size="11" fill="var(--ink2)">${t*100}%</text>`;});
  g+=`<defs><pattern id="hp" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="3" height="7" fill="var(--gray)"/></pattern></defs>`;
  cells.forEach((c,i)=>{ const x=L+i*bw+bw*0.18, w=bw*0.64; const zone=c&&c.zone?c.zone:"no data";
    if(zone==="below threshold"){ g+=`<rect x="${x}" y="${y(1)}" width="${w}" height="${y(0)-y(1)}" rx="3" fill="url(#hp)"><title>${S.bins[i]} g: below the scale's significance threshold - not measurable</title></rect>`;
      g+=`<text x="${x+w/2}" y="${y(0.5)}" text-anchor="middle" font-size="10" fill="var(--ink2)" transform="rotate(-90 ${x+w/2} ${y(0.5)})">below threshold</text>`;
    } else if(c&&c.det!=null){ g+=`<rect x="${x}" y="${y(c.det)}" width="${w}" height="${y(0)-y(c.det)}" rx="3" fill="${col[zone]}"><title>${S.bins[i]} g: ${pct(c.det)} detected (${c.picks} picks, CI ${pct(c.lo)}–${pct(c.hi)})</title></rect>`;
      g+=`<line x1="${x+w/2}" x2="${x+w/2}" y1="${y(c.hi)}" y2="${y(c.lo)}" stroke="var(--ink)" stroke-width="1.2"/>`;
      g+=`<text x="${x+w/2}" y="${y(c.det)-6}" text-anchor="middle" font-size="11" fill="var(--ink)">${pct(c.det)}</text>`;
    } else { g+=`<rect x="${x}" y="${y(0)-6}" width="${w}" height="6" rx="2" fill="var(--gray)"/>`; }
    g+=`<text x="${x+w/2}" y="${H-B+18}" text-anchor="middle" font-size="11" fill="var(--ink2)">${S.bins[i]} g</text>`;
    g+=`<text x="${x+w/2}" y="${H-B+33}" text-anchor="middle" font-size="10" fill="var(--muted)">${c&&c.picks?c.picks.toLocaleString()+" picks":""}</text>`; });
  [[T.green,"green "+Math.round(T.green*100)+"%"],[T.red,"red "+Math.round(T.red*100)+"%"]].forEach(([t,lab])=>{
    g+=`<line x1="${L}" x2="${W-R}" y1="${y(t)}" y2="${y(t)}" stroke="var(--ink2)" stroke-dasharray="4 4"/><text x="${W-R}" y="${y(t)-4}" text-anchor="end" font-size="10" fill="var(--ink2)">${lab}</text>`;});
  g+=`<text x="${L}" y="${H-6}" font-size="11" fill="var(--ink2)">Pick size = weight removed in one go</text>`;
  $("bars").innerHTML=g; }
function heat(S, lvl){ let h="<tr><th></th>"+S.bins.map(b=>`<th class="c">${b} g</th>`).join("")+"<th>TOO_MANY</th></tr>";
  S.levels.slice().reverse().forEach(k=>{ const s=S.summary[k]||{};
    h+=`<tr data-l="${k}" class="${String(k)===lvl?"sel":""}"><td class="lvl">sens ${k}</td>`+S.cells[k].map(c=>{
      const z=c&&c.zone?c.zone:"no data"; const cls=z==="amber"?"amber":(z==="no data"?"nd":(z==="below threshold"?"bt hatch":""));
      const txt=z==="below threshold"?"n/m":(c&&c.det!=null&&z!=="no data"?pct(c.det):"·");
      return `<td class="cell ${cls}" style="background:${col[z]}">${txt}</td>`;}).join("")
      +`<td class="c">${s.frac_too_many==null?"–":s.frac_too_many.toFixed(2)}</td></tr>`; });
  $("heat").innerHTML=h; $("heat").querySelectorAll("tr[data-l]").forEach(tr=>tr.addEventListener("click",()=>{$("level").value=tr.dataset.l;render();})); }
function rev(S){ let h="<tr><th>Pick size</th><th>Lowest green sensitivity</th><th>Its TOO_MANY fraction</th></tr>";
  S.bins.forEach(b=>{ const k=S.reverse[b]; const s=k==null?null:S.summary[String(k)];
    const bt=(S.below||[]).includes(b);
    h+=`<tr><td>${b} g</td><td>${bt?"not measurable (below the scale's threshold)":(k==null?"none reaches green":"sensitivity "+k)}</td><td>${s&&s.frac_too_many!=null?s.frac_too_many.toFixed(2):"–"}</td></tr>`;});
  $("rev").innerHTML=h; }
function render(){ const S=D.scales[$("scale").value], lvl=$("level").value; tiles(S,lvl); bars(S,lvl); heat(S,lvl); rev(S); }
fillScales(); fillLevels(); render();
$("scale").addEventListener("change",()=>{fillLevels();render();}); $("level").addEventListener("change",render);
</script></body></html>"""
