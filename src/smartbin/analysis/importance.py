"""
M3 - Which parameters matter, and how much?  (answers task 1 + task 2 in one chart)

Method: Shapley decomposition of R2 (a.k.a. LMG / dominance analysis).
The model "KPI ~ sensitivity + scale + weight bin + item weight + box + location" explains some
share of the differences between sections. Shapley splits that explained share FAIRLY among the
parameters, even when they are correlated (box is nested in scale type, boxes cluster by factory).
Reading: "location alone accounts for 18% of the variation in TOO_MANY, box for 0.3%".

Rule of thumb (Cohen): < 1% = no / marginal influence, 1-6% small, 6-14% medium, > 14% large.

Plus: the weight-in-box check the lecture asks for (same device, same sensitivity, different weight bin),
once on D ONE's subset (relevant_for_weight_analysis) and once on ALL such devices - because that
subset was selected on "something changes", which inflates the effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import viz
from ..io import WEIGHT_BIN_ORDER
from ..stats import boot_ci, shapley_r2
from . import ModuleResult

LABELS = {"sensitivity": "Sensitivity setting", "scale_type": "Scale type (6kg / 20kg)",
          "weight_bin": "Weight in box (fill level)", "item_class": "Item weight (single part)",
          "box_name": "Box type", "location_label": "Location (factory)"}


def _prepare(sec, cfg):
    s = sec[sec.main_analysis].copy()
    s["item_class"] = pd.qcut(s.item_weight.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
    s["location_label"] = s.location_label.fillna("unknown")
    big = s.groupby("location_label").deviceid.nunique()
    small = big[big < cfg.analysis.min_devices_location].index
    s.loc[s.location_label.isin(small), "location_label"] = "small locations"
    box_n = s.groupby("box_name").size()
    s.loc[s.box_name.isin(box_n[box_n < 30].index), "box_name"] = "other boxes"
    s["log_tf_week"] = np.log1p(s.too_few_per_week)
    s["w_auto"] = s.n_auto.clip(upper=200)
    s["w_days"] = s.days.clip(upper=60)
    return s


def _location_noise_floor(s, y, w, reps=20, seed=0):
    """R2 of 'location' if locations were assigned at random (devices keep their own data)."""
    rng = np.random.default_rng(seed)
    dev_loc = s.groupby("deviceid").location_label.first()
    vals = []
    for _ in range(reps):
        perm = pd.Series(rng.permutation(dev_loc.values), index=dev_loc.index)
        d = s.assign(location_label=s.deviceid.map(perm))
        vals.append(shapley_r2(d, y, ["location_label"], weights=w).attrs["r2_full"])
    return float(np.mean(vals))


def run(sections: pd.DataFrame, cfg) -> ModuleResult:
    r = ModuleResult("m3_importance", "Parameter influence - which parameters matter?",
                     "How much of the difference between scales does each parameter explain?")
    s = _prepare(sections, cfg)
    # box type is nested in scale type (each box exists for one scale only) -> only judged WITHIN a scale
    scopes = {"all scales": (s, ["sensitivity", "scale_type", "weight_bin", "item_class", "location_label"])}
    for scale in sorted(s.scale_type.dropna().unique()):
        scopes[scale] = (s[s.scale_type == scale], ["sensitivity", "weight_bin", "item_class", "box_name",
                                                    "location_label"])
    kpis = {"frac_too_many": ("frac_too_many", "w_auto"), "too_few_per_week": ("log_tf_week", "w_days")}
    rows = []
    for scope, (d, factors) in scopes.items():
        for kpi, (y, w) in kpis.items():
            if len(d) < 50:
                continue
            res = shapley_r2(d, y, factors, weights=w)
            floor = _location_noise_floor(d, y, w, seed=cfg.analysis.random_seed)
            for _, x in res.iterrows():
                rows.append(dict(scope=scope, kpi=kpi, factor=x.factor, parameter=LABELS[x.factor],
                                 shapley_r2=x.shapley_r2, share_of_explained=x.share,
                                 r2_full=res.attrs["r2_full"], n_sections=res.attrs["n"],
                                 location_noise_floor=floor if x.factor == "location_label" else np.nan))
    imp = pd.DataFrame(rows)
    if imp.empty:
        r.status = "skipped"
        r.notes.append("Not enough sections for the importance model.")
        return r
    thr = cfg.analysis.marginal_r2

    def verdict(v):
        return "large" if v >= 0.14 else "medium" if v >= 0.06 else "small" if v >= thr else "marginal / none"
    # location gets a chance correction: random labels already "explain" something with many levels
    imp["r2_corrected"] = (imp.shapley_r2 - imp.location_noise_floor.fillna(0)).clip(lower=0)
    imp["verdict"] = imp.r2_corrected.apply(verdict)
    r.tables["shapley"] = imp

    piv = imp.pivot_table(index=["scope", "parameter"], columns="kpi", values="r2_corrected").reset_index()
    piv["max_r2"] = piv[["frac_too_many", "too_few_per_week"]].max(axis=1)
    piv["verdict"] = piv.max_r2.apply(verdict)
    r.tables["verdict"] = piv.sort_values(["scope", "max_r2"], ascending=[True, False])
    for scope in scopes:
        g = imp[imp.scope == scope]
        if len(g):
            r.figures[f"importance_{scope.replace(' ', '_')}"] = _chart(g, scope, cfg, r)
            v = piv[piv.scope == scope]
            r.facts[scope] = {
                "influential": v[v.max_r2 >= thr].sort_values("max_r2", ascending=False).parameter.tolist(),
                "marginal": v[v.max_r2 < thr].parameter.tolist(),
                "shapley_r2_tm": dict(zip(g[g.kpi == "frac_too_many"].parameter,
                                          g[g.kpi == "frac_too_many"].shapley_r2.round(4))),
                "shapley_r2_tf": dict(zip(g[g.kpi == "too_few_per_week"].parameter,
                                          g[g.kpi == "too_few_per_week"].shapley_r2.round(4))),
                "r2_full_tm": float(g[g.kpi == "frac_too_many"].r2_full.max()),
            }

    # weight-in-box check (lecture method + robustness on all weight-bin movers)
    wt = _weight_check(s, cfg)
    r.tables["weight_check"] = wt
    r.facts["weight_check"] = wt.to_dict(orient="records")
    r.notes.append("Shapley R2 is descriptive: it says how much variation a parameter accounts for, "
                   "not that it causes it. Causality for sensitivity comes from M2 (same-device before/after).")
    return r


def _weight_check(s, cfg):
    order = {b: i for i, b in enumerate(WEIGHT_BIN_ORDER)}
    s = s.assign(bin_idx=s.weight_bin.map(order))
    rows = []
    for label, d in [("lecture subset (relevant_for_weight_analysis)", s[s.rel_weight.fillna(False)]),
                     ("all devices that changed weight bin", s)]:
        g = (d.groupby(["deviceid", "sensitivity", "bin_idx", "scale_type"])
             .agg(n_tm=("n_tm", "sum"), n_auto=("n_auto", "sum"), n_tf=("n_tf", "sum"), days=("days", "sum"))
             .reset_index())
        g["frac"] = g.n_tm / g.n_auto.replace(0, np.nan)
        g["tfw"] = g.n_tf / g.days * 7
        lo = g.loc[g.groupby(["deviceid", "sensitivity"]).bin_idx.idxmin()]
        hi = g.loc[g.groupby(["deviceid", "sensitivity"]).bin_idx.idxmax()]
        m = lo.merge(hi, on=["deviceid", "sensitivity", "scale_type"], suffixes=("_light", "_heavy"))
        m = m[m.bin_idx_heavy > m.bin_idx_light]
        for scale, x in m.groupby("scale_type"):
            d_tm = x.frac_heavy - x.frac_light
            d_tf = x.tfw_heavy - x.tfw_light
            lo_, hi_ = boot_ci(d_tm, reps=cfg.analysis.bootstrap_reps, seed=cfg.analysis.random_seed)
            rows.append(dict(subset=label, scale_type=scale, device_settings=len(x),
                             median_delta_frac_tm_heavier=d_tm.median(), ci_lo=lo_, ci_hi=hi_,
                             median_delta_tf_week_heavier=d_tf.median()))
    return pd.DataFrame(rows)


def _chart(g, scope, cfg, r):
    import matplotlib.pyplot as plt
    thr = cfg.analysis.marginal_r2
    order = (g[g.kpi == "frac_too_many"].sort_values("shapley_r2").parameter.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(order) + 1.8), sharey=True)
    for ax, (kpi, title) in zip(axes, [("frac_too_many", "TOO_MANY fraction"),
                                       ("too_few_per_week", "TOO_FEW per week")]):
        d = g[g.kpi == kpi].set_index("parameter").reindex(order)
        vals = d.shapley_r2.fillna(0) * 100
        corr = d.r2_corrected.fillna(0) * 100
        colors = [viz.CAT[0] if v >= thr * 100 else viz.GRAY_BAR for v in corr]
        ax.barh(order, vals, color=colors, height=0.55)
        for y, v in enumerate(vals):
            ax.text(v + vals.max() * 0.02, y, f"{v:.1f}%", va="center", fontsize=8, color=viz.INK2)
        floor = d.location_noise_floor.dropna()
        if len(floor):
            ax.plot([floor.iloc[0] * 100] * 2, [order.index(LABELS["location_label"]) - 0.35,
                                                order.index(LABELS["location_label"]) + 0.35],
                    color=viz.INK, lw=1.5)
        ax.axvline(thr * 100, color=viz.MUTED, lw=1)
        ax.set_xlim(0, max(vals.max() * 1.18, thr * 100 * 3))
        ax.set_title(f"{title}  (model R2 = {d.r2_full.iloc[0]:.0%})", fontsize=10)
        ax.set_xlabel("% of variation explained (Shapley R2)")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Which parameters move the sensor KPIs? - {scope}", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    viz.note(fig, f"Blue = influences the result (>= {thr:.0%} of variation); gray = no or marginal influence. "
                  "Black tick on location = what random location labels would explain (chance level).")
    fig.tight_layout()
    return viz.save(fig, r.fig_path(cfg.paths.figures, f"importance_{scope.replace(' ', '_')}"))
