"""
M4 - Is the location effect real?  Three independent tests on the baseline (sensitivity 8),
so the sensitivity setting cannot be the reason locations differ.

1. ICC          - what share of the device-to-device variation sits BETWEEN locations?
2. Permutation  - shuffle location labels between devices 999x: is the real between-location
                  spread larger than chance?  (p-value without distribution assumptions)
3. Split-half   - split every location's devices randomly in two halves; if the location
                  effect is real, locations that are noisy in half A are noisy in half B too.
                  (Spearman r; Spearman-Brown = reliability of the location ranking)

Location granularity: LOCATIONID is a site (median ~50 scales, max ~1000). No rack/shelf position
exists in the data -> rack-level neighbourhoods are inferred from timing in M6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .. import viz
from ..stats import boot_ci, icc_oneway, permutation_test_groups, split_half_reliability
from . import ModuleResult


def run(ds: pd.DataFrame, cfg) -> ModuleResult:
    r = ModuleResult("m4_location", "Location - is the factory effect real?",
                     "Do scales in some factories perform systematically worse at the same setting?")
    b = cfg.analysis.baseline_sensitivity
    base = ds[(ds.sensitivity == b) & ds.main_analysis & ds.location_label.notna()].copy()
    rows, loc_tables = [], []
    for scale, g in base.groupby("scale_type"):
        cnt = g.location_label.value_counts()
        g = g[g.location_label.isin(cnt[cnt >= cfg.analysis.min_devices_location].index)]
        if g.location_label.nunique() < 3:
            continue
        lt = []
        for loc, x in g.groupby("location_label"):
            lo, hi = boot_ci(x.frac_too_many, reps=200, seed=cfg.analysis.random_seed)
            lt.append(dict(scale_type=scale, location_label=loc, devices=len(x),
                           location_size=x.location_size.iloc[0],
                           frac_too_many_median=x.frac_too_many.median(), lo=lo, hi=hi,
                           too_few_per_week_median=x.too_few_per_week.median(),
                           share_in_target=x.in_target.mean(), meas_per_day_median=x.meas_per_day.median()))
        lt = pd.DataFrame(lt).sort_values("frac_too_many_median")
        loc_tables.append(lt)
        res = dict(scale_type=scale, locations=g.location_label.nunique(), devices=len(g))
        for kpi in ["frac_too_many", "too_few_per_week"]:
            res[f"icc_{kpi}"] = icc_oneway(g[kpi], g.location_label)
            f, p = permutation_test_groups(g[kpi], g.location_label, n_perm=cfg.analysis.permutations,
                                           seed=cfg.analysis.random_seed)
            res[f"perm_F_{kpi}"], res[f"perm_p_{kpi}"] = f, p
            rr, sb, ex = split_half_reliability(g[kpi], g.location_label, reps=cfg.analysis.split_half_reps,
                                                seed=cfg.analysis.random_seed)
            res[f"split_half_r_{kpi}"], res[f"reliability_{kpi}"] = rr, sb
            if kpi == "frac_too_many" and ex is not None:
                r.figures[f"split_half_{scale}"] = _split_half_chart(ex, rr, scale, cfg, r)
        # does the factory effect survive once we account for how busy the bins are?
        act = g.n_aok / (g.days / 7)
        dec = pd.qcut(np.log1p(act).rank(method="first"), 10, labels=False)
        res["icc_frac_too_many_activity_adjusted"] = icc_oneway(
            g.frac_too_many - g.frac_too_many.groupby(dec).transform("mean"), g.location_label)
        ltm = np.log1p(g.too_many_per_week)
        res["icc_log_too_many_per_week"] = icc_oneway(ltm, g.location_label)
        res["icc_log_too_many_per_week_activity_adjusted"] = icc_oneway(ltm - ltm.groupby(dec).transform("mean"),
                                                                         g.location_label)
        rho = st.spearmanr(lt.location_size, lt.frac_too_many_median)
        res["size_vs_tm_spearman"], res["size_vs_tm_p"] = rho.statistic, rho.pvalue
        res["spread_p10_p90"] = (float(lt.frac_too_many_median.quantile(0.1)),
                                 float(lt.frac_too_many_median.quantile(0.9)))
        p_, r_ = res["perm_p_frac_too_many"], (res["split_half_r_frac_too_many"] or 0)
        res["verdict"] = ("REAL location effect (strong)" if (p_ < 0.01 and r_ > 0.4) else
                          "REAL location effect (moderate)" if (p_ < 0.05 and r_ > 0.2) else
                          "not reliable")
        rows.append(res)
        r.figures[f"caterpillar_{scale}"] = _caterpillar(lt, scale, cfg, r)
    tests = pd.DataFrame(rows)
    r.tables["tests"] = tests
    r.tables["locations"] = pd.concat(loc_tables, ignore_index=True) if loc_tables else pd.DataFrame()
    r.facts = {row["scale_type"]: row for row in tests.to_dict(orient="records")}
    return r


def _caterpillar(lt, scale, cfg, r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(lt))
    c = viz.SCALE_COLOR.get(scale, viz.CAT[0])
    ax.vlines(x, lt.lo, lt.hi, color=c, alpha=0.35, lw=1.5)
    ax.scatter(x, lt.frac_too_many_median, s=22, color=c, edgecolor=viz.SURFACE, linewidth=1, zorder=3)
    viz.target_line(ax, cfg.kpi.tm_target, "y", f"target {cfg.kpi.tm_target}")
    viz.target_line(ax, cfg.kpi.tm_alarm, "y", f"alarm {cfg.kpi.tm_alarm}")
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(lt)} locations, sorted (each with >= {cfg.analysis.min_devices_location} scales)")
    ax.set_ylabel("Fraction TOO_MANY (median of scales)")
    ax.set_title(f"{scale}: same setting (8), very different factories")
    viz.subtitle(ax, "dot = location median · line = 95% bootstrap CI")
    return viz.save(fig, r.fig_path(cfg.paths.figures, "caterpillar", scale))


def _split_half_chart(ex, rr, scale, cfg, r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    c = viz.SCALE_COLOR.get(scale, viz.CAT[0])
    ax.scatter(ex.median_a, ex.median_b, s=26, color=c, edgecolor=viz.SURFACE, linewidth=1, zorder=3)
    ax.plot([0, 1], [0, 1], color=viz.AXIS, lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Location median, random half A of its scales")
    ax.set_ylabel("Location median, other half B")
    ax.set_title(f"{scale}: split-half test (r = {rr:.2f})")
    viz.subtitle(ax, "points near the diagonal = the location effect is reproducible")
    return viz.save(fig, r.fig_path(cfg.paths.figures, "split_half", scale))
