"""
M1 - Baseline: how well do the scales work TODAY (factory default sensitivity 8)?

Unit: device-setting at the baseline sensitivity (every device counts once).
Output: Bossard's KPIs per scale type + share of scales in each situation R1/R2/R3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import viz
from ..stats import boot_ci
from . import ModuleResult


def run(ds: pd.DataFrame, cfg) -> ModuleResult:
    r = ModuleResult("m1_baseline", "Baseline - how do the scales perform today?",
                     "At the default sensitivity, how many scales meet Bossard's targets?")
    b = cfg.analysis.baseline_sensitivity
    base = ds[(ds.sensitivity == b) & ds.main_analysis].copy()
    rows = []
    for scale, g in base.groupby("scale_type"):
        lo, hi = boot_ci(g.frac_too_many, reps=cfg.analysis.bootstrap_reps, seed=cfg.analysis.random_seed)
        z = g.zone.value_counts(normalize=True)
        rows.append(dict(
            scale_type=scale, devices=len(g),
            frac_too_many_median=g.frac_too_many.median(), frac_too_many_lo=lo, frac_too_many_hi=hi,
            too_few_per_week_median=g.too_few_per_week.median(),
            miss_rate_median=g.miss_rate.median(), meas_per_day_median=g.meas_per_day.median(),
            too_many_per_week_median=g.too_many_per_week.median(),
            share_in_target=g.in_target.mean(), share_meets_tm=g.meets_tm.mean(),
            share_meets_tf=g.meets_tf.mean(), share_tm_alarm=(g.frac_too_many > cfg.kpi.tm_alarm).mean(),
            share_R2=z.get("R2 target", 0.0), share_R1=z.get("R1 too sensitive", 0.0),
            share_R3=z.get("R3 too insensitive", 0.0), share_R13=z.get("R1+R3 unstable", 0.0),
        ))
        r.figures[f"device_map_{scale}"] = _device_map(g, scale, cfg, r)
    tab = pd.DataFrame(rows)
    r.tables["kpis"] = tab
    act = _activity(base, cfg, r)
    r.tables["activity_quintiles"] = act
    r.facts = {rec["scale_type"]: rec for rec in tab.to_dict(orient="records")}
    for scale, g in act.groupby("scale_type"):
        r.facts[scale]["activity_spearman_tm_fraction"] = float(g.spearman.iloc[0])
        r.facts[scale]["tm_fraction_least_vs_most_used"] = [float(g.frac_too_many_median.iloc[0]),
                                                            float(g.frac_too_many_median.iloc[-1])]
    r.notes.append("TOO_MANY fraction = empty / (empty + real) measurements. Bins that are rarely used have few real "
                   "measurements in the denominator, so their fraction is high whatever the setting.")
    return r


def _activity(base, cfg, r):
    """How much of the TOO_MANY fraction is just 'how often is this bin used'?"""
    from scipy import stats as st
    import matplotlib.pyplot as plt
    b = base[base.frac_too_many.notna()].copy()
    b["real_picks_week"] = b.n_aok / (b.days / 7)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for scale, g in b.groupby("scale_type"):
        g = g.copy()
        g["q"] = pd.qcut(g.real_picks_week.rank(method="first"), 5,
                         labels=["Q1 least used", "Q2", "Q3", "Q4", "Q5 most used"])
        rho = st.spearmanr(g.real_picks_week, g.frac_too_many).statistic
        t = g.groupby("q", observed=True).agg(devices=("deviceid", "size"),
                                              real_picks_week=("real_picks_week", "median"),
                                              frac_too_many_median=("frac_too_many", "median"),
                                              too_many_per_week_median=("too_many_per_week", "median"),
                                              share_in_target=("in_target", "mean")).reset_index()
        t.insert(0, "scale_type", scale)
        t["spearman"] = rho
        rows.append(t)
        c = viz.SCALE_COLOR.get(scale, viz.CAT[2])
        x = range(len(t))
        axes[0].plot(x, t.frac_too_many_median, "-o", color=c, label=scale, mec=viz.SURFACE, mew=1.5)
        axes[1].plot(x, t.too_many_per_week_median, "-o", color=c, label=scale, mec=viz.SURFACE, mew=1.5)
    lab = ["Q1\nleast used", "Q2", "Q3", "Q4", "Q5\nmost used"]
    for ax in axes:
        ax.set_xticks(range(5), lab)
        ax.set_xlabel("Bins grouped by real picks per week (quintiles)")
        ax.set_ylim(bottom=0)
    viz.target_line(axes[0], cfg.kpi.tm_target, "y", "target")
    axes[0].set_title("TOO_MANY fraction (Bossard KPI)", fontsize=10)
    axes[1].set_title("TOO_MANY per week (absolute load)", fontsize=10)
    axes[0].legend()
    fig.suptitle(f"Same setting ({cfg.analysis.baseline_sensitivity}), different usage: the fraction punishes "
                 "rarely used bins", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    r.figures["activity"] = viz.save(fig, r.fig_path(cfg.paths.figures, "activity"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _device_map(g, scale, cfg, r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ymax = max(6.0, float(np.nanquantile(g.too_few_per_week, 0.98)) * 1.05)
    ax.add_patch(plt.Rectangle((0, 0), cfg.kpi.tm_target, cfg.kpi.tf_target, color=viz.GOOD, alpha=0.08, lw=0))
    y = g.too_few_per_week.clip(upper=ymax)
    viz.scatter_points(ax, g.frac_too_many, y, viz.SCALE_COLOR.get(scale, viz.CAT[0]))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    viz.target_line(ax, cfg.kpi.tm_target, "x", f"target {cfg.kpi.tm_target}")
    viz.target_line(ax, cfg.kpi.tf_target, "y", f"target {cfg.kpi.tf_target}/week")
    share = g.zone.value_counts(normalize=True)
    lab = dict(fontsize=9, color=viz.INK2)
    ax.text(0.02, cfg.kpi.tf_target * 0.5, f"R2 target\n{share.get('R2 target', 0):.0%} of scales", **lab)
    ax.text(0.98, cfg.kpi.tf_target * 0.5, f"R1 too sensitive\n{share.get('R1 too sensitive', 0):.0%}",
            ha="right", **lab)
    ax.text(0.02, ymax * 0.92, f"R3 too insensitive\n{share.get('R3 too insensitive', 0):.0%}", va="top", **lab)
    ax.text(0.98, ymax * 0.92, f"both violated\n{share.get('R1+R3 unstable', 0):.0%}", ha="right", va="top", **lab)
    ax.set_xlabel("Fraction TOO_MANY  (empty measurements / automatic measurements)")
    ax.set_ylabel("TOO_FEW per week  (missed stock changes)")
    ax.set_title(f"{scale} scales today (sensitivity {cfg.analysis.baseline_sensitivity}): one dot = one scale")
    viz.subtitle(ax, f"n = {len(g):,} scales · green box = Bossard target zone · y clipped at {ymax:.0f}")
    return viz.save(fig, r.fig_path(cfg.paths.figures, "device_map", scale))
