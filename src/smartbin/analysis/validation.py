"""
Method validation on synthetic data: did the pipeline rediscover what was planted?
Runs only when ground_truth.json sits next to the measurements file (i.e. never on real data).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import comb

from . import ModuleResult


def adjusted_rand(a, b) -> float:
    ct = pd.crosstab(pd.Series(a, name="a"), pd.Series(b, name="b")).values
    n = ct.sum()
    s_ij = comb(ct, 2).sum()
    s_a = comb(ct.sum(axis=1), 2).sum()
    s_b = comb(ct.sum(axis=0), 2).sum()
    exp = s_a * s_b / comb(n, 2)
    mx = (s_a + s_b) / 2
    return float((s_ij - exp) / (mx - exp)) if mx != exp else np.nan


def run(results: dict, measurements_path: Path, cfg) -> ModuleResult | None:
    gt_path = Path(measurements_path).parent / "ground_truth.json"
    if not gt_path.exists():
        return None
    gt = json.loads(gt_path.read_text())
    r = ModuleResult("m0_validation", "Method validation on synthetic ground truth",
                     "Does the pipeline recover effects we planted on purpose?")
    rows = []

    def add(check, planted, recovered, ok):
        rows.append(dict(check=check, planted=planted, recovered=recovered, status="PASS" if ok else "FAIL"))

    imp = results.get("m3_importance")
    if imp and "shapley" in imp.tables:
        s = imp.tables["shapley"]
        for scale in [x for x in s.scope.unique() if x != "all scales"]:
            box = s[(s.scope == scale) & (s.factor == "box_name")].shapley_r2.max()
            add(f"{scale}: box type has no effect", "none", f"Shapley R2 = {box:.4f}",
                box < cfg.analysis.marginal_r2)
            sens = s[(s.scope == scale) & (s.factor == "sensitivity") & (s.kpi == "frac_too_many")].shapley_r2.max()
            add(f"{scale}: sensitivity matters", "strong", f"Shapley R2 = {sens:.3f}", sens >= 0.01)
    loc = results.get("m4_location")
    if loc and "tests" in loc.tables and len(loc.tables["tests"]):
        for _, t in loc.tables["tests"].iterrows():
            add(f"{t.scale_type}: location effect is real", "strong",
                f"perm p = {t.perm_p_frac_too_many:.3f}, split-half r = {t.split_half_r_frac_too_many:.2f}",
                str(t.verdict).startswith("REAL"))
    sen = results.get("m2_sensitivity")
    if sen and "effects" in sen.tables and len(sen.tables["effects"]):
        e = sen.tables["effects"]
        for scale, g in e.groupby("scale_type"):
            low = g[g.sensitivity <= 2].did_frac_too_many.median()
            rr = np.exp(g[g.sensitivity <= 2].beta_tf.median())
            add(f"{scale}: lowering 8 -> 1-2 cuts TOO_MANY, raises TOO_FEW", "threshold model",
                f"DiD frac TM = {low:+.3f}, TOO_FEW rate x{rr:.1f}", (low < -0.05) and (rr > 1.5))
        for scale, v in sen.facts.items():
            if isinstance(v, dict) and "anomaly_verdict" in v:
                add(f"{scale}: sensitivity-3 anomaly detected", "selection + setting quirk",
                    v["anomaly_verdict"][:90], ("SELECTION" in v["anomaly_verdict"]) or
                    ("SETTING ANOMALY" in v["anomaly_verdict"]) or ("BUG SIGNATURE" in v["anomaly_verdict"]))
    xt = results.get("m6_crosstalk")
    if xt and xt.status == "ok":
        dec = xt.tables.get("root_causes_baseline", pd.DataFrame())
        for _, d in dec.iterrows():
            add(f"{d.scale_type}: crosstalk causes a share of TOO_MANY", "rack neighbours",
                f"neighbour-induced = {d.neighbour_induced:.0%}", d.neighbour_induced > 0.10)
        racks = xt.tables.get("virtual_racks", pd.DataFrame())
        if len(racks):
            truth = gt["rack_of_device"]
            rk = racks[racks.deviceid.isin(truth)]
            ari = adjusted_rand(rk.cluster.values, [truth[d] for d in rk.deviceid])
            add("virtual racks match the hidden racks", f"{gt['n_racks']} racks",
                f"ARI = {ari:.2f} on {len(rk)} scales", ari > 0.5)
    ps = results.get("m5_pick_size")
    if ps and "detection_thresholds" in ps.tables and len(ps.tables["detection_thresholds"]):
        f = ps.tables["detection_thresholds"]
        for scale, g in f.groupby("scale_type"):
            g = g.set_index("sensitivity").D90_g
            if 8.0 in g.index and 4.0 in g.index:
                add(f"{scale}: higher sensitivity detects smaller picks", "intensity ~ log(grams)",
                    f"D90 at 8 = {g[8.0]:.0f} g vs at 4 = {g[4.0]:.0f} g", g[8.0] < g[4.0])
    v = pd.DataFrame(rows)
    r.tables["checks"] = v
    r.facts = {"passed": int((v.status == "PASS").sum()) if len(v) else 0, "total": len(v)}
    return r
