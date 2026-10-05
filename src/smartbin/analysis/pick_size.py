"""
M5 - How small can a pick be before the sensor sleeps through it?

Every real stock change is visible in the data as a weight drop between two consecutive weighings:
  * caught by the sensor  -> AUTOMATIC + OK       (detected)
  * missed by the sensor  -> MANUAL + TOO_FEW     (found later by a person / the 24h check)
Pick size = |weight drop| in grams. Detection curve = P(detected | pick size, sensitivity, scale).
D90 = the pick size above which 90% of picks are detected (logistic fit on log grams).

Caveat (stated on every chart): a TOO_FEW row can contain SEVERAL missed picks added up,
so missed picks look bigger than they were -> the curves are conservative.

Observability floor: a change smaller than the scale's significance threshold is never labelled
TOO_FEW (and an automatic weighing of it is labelled TOO_MANY). Below that floor both detections and
misses are cut off, so a detection rate there means nothing. The floor is measured from the data
(5th percentile of TOO_FEW pick sizes per scale type); bins below it are shown as "below threshold".

Without event chains (random-row sample) the item weight from the metadata is used as a proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import viz
from ..stats import logistic_fit, wilson
from . import ModuleResult


def run(con, ds: pd.DataFrame, cfg, sample: bool) -> ModuleResult:
    r = ModuleResult("m5_pick_size", "Pick size - when does the sensor miss a pick?",
                     "Below which pick weight (grams) does each sensitivity start missing stock changes?")
    levels = [float(x) for x in cfg.analysis.main_levels]
    r.tables["item_weight_proxy"] = _item_proxy(ds, levels, cfg, r)
    if sample:
        r.notes.append("Sample mode: event chains are broken, pick sizes from weight deltas are not available. "
                       "Item-weight proxy only.")
        r.status = "partial"
        return r
    ev = con.execute(f"""
        SELECT deviceid, scale_type, sensitivity, is_auto, -dw AS pick_g, item_weight
        FROM events
        WHERE exclusion_reason IS NULL AND dw < 0 AND gap_s <= {cfg.pick_size.max_gap_days * 86400}
          AND ((is_auto AND cls = 'OK') OR (NOT is_auto AND cls = 'TOO_FEW'))
    """).df()
    if ev.empty:
        r.status = "skipped"
        return r
    ev["detected"] = ev.is_auto.astype(int)
    bins = cfg.pick_size.bins_g
    labels = [f"{bins[i]}-{bins[i + 1]}" if bins[i + 1] < 100000 else f">{bins[i]}" for i in range(len(bins) - 1)]
    ev["pick_bin"] = pd.cut(ev.pick_g, bins, labels=labels, right=False)
    upper = dict(zip(labels, bins[1:]))
    tf = ev[ev.detected == 0]
    floor = tf.groupby("scale_type").pick_g.quantile(0.05).round(0)
    floor = floor[tf.groupby("scale_type").size().reindex(floor.index) >= 100]
    curve = (ev[ev.sensitivity.isin(levels)].groupby(["scale_type", "sensitivity", "pick_bin"], observed=True)
             .agg(picks=("detected", "size"), detected=("detected", "sum")).reset_index())
    curve["miss_rate"] = 1 - curve.detected / curve.picks
    curve["measurable"] = ~(curve.pick_bin.astype(str).map(upper).astype(float)
                            <= curve.scale_type.map(floor).astype(float).fillna(0))
    zones, summary, reverse = _zones(curve, ds, cfg)
    r.tables["detection_zones"] = zones
    r.tables["zone_summary"] = summary
    r.tables["min_sensitivity_by_pick_size"] = reverse
    for scale in sorted(zones.scale_type.unique()):
        r.figures[f"zone_heatmap_{scale}"] = _heatmap(zones[zones.scale_type == scale], scale, cfg, r)
    from ..explorer import write_pick_size_explorer
    r.figures["explorer_html"] = write_pick_size_explorer(zones, summary, reverse, cfg, floor.to_dict())
    curve = curve[(curve.picks >= cfg.pick_size.min_picks_per_cell) & curve.measurable]
    r.tables["detection_curve"] = curve

    fits = []
    fit_ev = ev[ev.sensitivity.isin(levels) & (ev.pick_g >= ev.scale_type.map(floor).astype(float).fillna(0))]
    for (scale, k), g in fit_ev.groupby(["scale_type", "sensitivity"]):
        if len(g) < 200 or g.detected.nunique() < 2:
            continue
        a, b = logistic_fit(np.log(g.pick_g.clip(lower=1)), g.detected)
        lo_g, hi_g = float(g.pick_g.quantile(0.02)), float(g.pick_g.quantile(0.98))
        d50 = np.exp(-a / b) if b > 0 else np.nan
        d90 = np.exp((np.log(9) - a) / b) if b > 0 else np.nan
        d50, d90 = (np.clip(v, lo_g, hi_g) if np.isfinite(v) else v for v in (d50, d90))
        fits.append(dict(scale_type=scale, sensitivity=k, picks=len(g), slope=b, D50_g=d50, D90_g=d90,
                         pick_g_lo=lo_g, pick_g_hi=hi_g,
                         miss_rate=1 - g.detected.mean(), median_pick_g=g.pick_g.median()))
    fits = pd.DataFrame(fits)
    r.tables["detection_thresholds"] = fits

    prof = (ev.groupby("deviceid").agg(scale_type=("scale_type", "first"), picks=("pick_g", "size"),
                                       median_pick_g=("pick_g", "median")).reset_index())
    prof["pick_class"] = pd.cut(prof.median_pick_g, [0, 10, 50, np.inf], labels=["small <10 g", "medium 10-50 g",
                                                                                   "large >50 g"], right=False)
    r.tables["device_pick_profile"] = prof
    for scale in sorted(curve.scale_type.unique()):
        r.figures[f"detection_curve_{scale}"] = _curve_chart(curve[curve.scale_type == scale], fits, scale, cfg, r)
    audit = _label_audit(con)
    r.tables["label_audit"] = audit
    if len(audit):
        r.figures["label_audit"] = _audit_chart(audit, cfg, r)
    for scale, g in summary.groupby("scale_type"):
        r.facts.setdefault(scale, {})["green_from_g"] = dict(zip(g.sensitivity, g.green_from_label))
    for scale, v in floor.items():
        r.facts.setdefault(scale, {})["significance_floor_g"] = float(v)
    for scale, g in fits.groupby("scale_type"):
        r.facts.setdefault(scale, {}).update({"D90_g": dict(zip(g.sensitivity, g.D90_g.round(1))),
                          "D50_g": dict(zip(g.sensitivity, g.D50_g.round(1))),
                          "pick_class_share": prof[prof.scale_type == scale].pick_class.value_counts(
                              normalize=True).round(3).to_dict()})
    r.notes.append("A TOO_FEW weighing may contain several missed picks added up -> missed picks look larger; "
                   "detection thresholds are therefore conservative (true D90 is likely lower).")
    if len(floor):
        r.notes.append("Changes below the scale's significance threshold are never recorded as missed ("
                       + ", ".join(f"{k}: ~{v:.0f} g" for k, v in floor.items())
                       + "). Those pick sizes are marked 'below threshold' and left out of the fits.")
    return r


def _item_proxy(ds, levels, cfg, r):
    d = ds[ds.main_analysis & ds.sensitivity.isin(levels) & ds.item_weight.notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["item_class"] = pd.qcut(d.item_weight.rank(method="first"), 4,
                              labels=["Q1 lightest", "Q2", "Q3", "Q4 heaviest"])
    edges = d.groupby("item_class", observed=True).item_weight.agg(["min", "max"]).round(2)
    g = (d.groupby(["scale_type", "item_class", "sensitivity"], observed=True)
         .agg(devices=("deviceid", "nunique"), n_tf=("n_tf", "sum"), n_aok=("n_aok", "sum"),
              n_tm=("n_tm", "sum"), n_auto=("n_auto", "sum")).reset_index())
    g["miss_rate"] = g.n_tf / (g.n_tf + g.n_aok).replace(0, np.nan)
    g["frac_too_many"] = g.n_tm / g.n_auto.replace(0, np.nan)
    g = g.merge(edges.reset_index(), on="item_class", how="left")
    g = g[g.devices >= 10]
    if len(g):
        r.figures["item_weight_proxy"] = _proxy_chart(g, cfg, r)
    return g


def _label_audit(con):
    """Do the labels make physical sense? TOO_MANY should have ~0 g change, OK auto a real change."""
    return con.execute("""
        SELECT scale_type, cls,
               count(*) AS n,
               quantile_cont(abs(dw), 0.5) AS median_abs_change_g,
               quantile_cont(abs(dw), 0.9) AS p90_abs_change_g,
               avg(CASE WHEN abs(dw) <= 2 THEN 1.0 ELSE 0 END) AS share_le_2g
        FROM events
        WHERE exclusion_reason IS NULL AND is_auto AND dw IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2
    """).df()


def _curve_chart(curve, fits, scale, cfg, r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    levels = sorted(curve.sensitivity.unique())
    bins = [b for b in curve.pick_bin.cat.categories if b in set(curve.pick_bin.astype(str))]
    x = {b: i for i, b in enumerate(bins)}
    for k in levels:
        g = curve[curve.sensitivity == k]
        story = int(k) in cfg.analysis.story_levels
        ax.plot([x[b] for b in g.pick_bin.astype(str)], g.miss_rate * 100, "-o",
                color=viz.level_color(k, levels), lw=2 if story else 1.2, ms=6 if story else 3,
                mec=viz.SURFACE, label=f"sensitivity {int(k)}", alpha=1 if story else 0.7)
    ax.set_xticks(range(len(bins)), [f"{b} g" for b in bins], rotation=0, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Missed picks (%)")
    ax.set_xlabel("Size of the pick (weight removed in one go)")
    ax.set_title(f"{scale}: small picks are missed - how small depends on the sensitivity")
    f = fits[fits.scale_type == scale]
    if len(f):
        def fmt(v, sub):
            lo, hi = sub.pick_g_lo.iloc[0], sub.pick_g_hi.iloc[0]
            return f"<={lo:.0f} g" if v <= lo * 1.001 else f">={hi:.0f} g" if v >= hi * 0.999 else f"{v:.0f} g"
        txt = " · ".join(f"s{int(a)}: {fmt(b, f[f.sensitivity == a])}" for a, b in zip(f.sensitivity, f.D90_g)
                         if np.isfinite(b) and int(a) in cfg.analysis.story_levels)
        viz.subtitle(ax, f"90% detection threshold (D90) - {txt}")
    ax.legend(ncol=2, fontsize=8)
    return viz.save(fig, r.fig_path(cfg.paths.figures, "detection_curve", scale))


def _proxy_chart(g, cfg, r):
    import matplotlib.pyplot as plt
    scales = sorted(g.scale_type.unique())
    fig, axes = plt.subplots(1, len(scales), figsize=(5.4 * len(scales), 3.8), sharey=True, squeeze=False)
    levels = sorted(g.sensitivity.unique())
    for ax, scale in zip(axes[0], scales):
        d = g[g.scale_type == scale]
        cats = list(d.item_class.cat.categories)
        for k in levels:
            x = d[d.sensitivity == k]
            if int(k) not in cfg.analysis.story_levels:
                continue
            ax.plot([cats.index(c) for c in x.item_class], x.miss_rate * 100, "-o", color=viz.level_color(k, levels),
                    label=f"sensitivity {int(k)}", mec=viz.SURFACE)
        ax.set_xticks(range(len(cats)), cats, fontsize=8)
        ax.set_title(scale, fontsize=10)
        ax.set_xlabel("Weight of a single item (quartiles)")
    axes[0][0].set_ylabel("Missed changes (%)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Item weight as a proxy for pick size: light parts are missed more often", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return viz.save(fig, r.fig_path(cfg.paths.figures, "item_weight_proxy"))


def _audit_chart(audit, cfg, r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    a = audit.copy()
    a["label"] = a.scale_type.fillna("?") + " · " + a.cls
    ax.barh(a.label, a.share_le_2g * 100, color=[viz.CAT[1] if c == "TOO_MANY" else viz.CAT[0] for c in a.cls],
            height=0.5)
    for y, v in enumerate(a.share_le_2g * 100):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8, color=viz.INK2)
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of automatic measurements with weight change <= 2 g (scale resolution: 1 g)")
    ax.set_title("Label audit: TOO_MANY really means 'nothing changed'")
    ax.grid(axis="y", visible=False)
    return viz.save(fig, r.fig_path(cfg.paths.figures, "label_audit"))


ZONE_ORDER = ["green", "amber", "red", "no data", "below threshold"]
NOT_RATED = ("no data", "below threshold")


def _zones(curve, ds, cfg):
    """Green / amber / red zone per scale x sensitivity x pick-size bin (+ what each level costs)."""
    c = cfg.pick_size
    z = curve.copy()
    z["pick_bin"] = z.pick_bin.astype(str)
    order = [b for b in curve.pick_bin.cat.categories]
    z["bin_idx"] = z.pick_bin.map({b: i for i, b in enumerate(order)})
    z["detection"] = z.detected / z.picks
    z["ci_lo"], z["ci_hi"] = wilson(z.detected, z.picks)
    z["zone"] = np.select([~z.measurable, z.picks < c.min_picks_per_cell, z.detection >= c.green_zone,
                           z.detection >= c.red_zone], ["below threshold", "no data", "green", "amber"], "red")
    z = z.sort_values(["scale_type", "sensitivity", "bin_idx"])
    cost = (ds[ds.main_analysis].groupby(["scale_type", "sensitivity"])
            .agg(frac_too_many=("frac_too_many", "median"), meas_per_day=("meas_per_day", "median"),
                 devices=("deviceid", "nunique")).reset_index())
    rows = []
    for (scale, k), g in z.groupby(["scale_type", "sensitivity"]):
        g = g[~g.zone.isin(NOT_RATED)].sort_values("bin_idx")
        green_from, red_below = None, None
        for i in range(len(g)):                       # smallest bin from which everything larger is green
            if (g.zone.iloc[i:] == "green").all():
                green_from = g.pick_bin.iloc[i]
                break
        reds = g[g.zone == "red"]
        if len(reds):
            red_below = reds.pick_bin.iloc[-1]
        rows.append(dict(scale_type=scale, sensitivity=k, green_from_label=green_from or "never",
                         red_up_to_label=red_below or "none",
                         picks=int(g.picks.sum()), detection_overall=g.detected.sum() / max(g.picks.sum(), 1)))
    summary = pd.DataFrame(rows).merge(cost, on=["scale_type", "sensitivity"], how="left")
    rev = []
    for (scale, b), g in z[~z.zone.isin(NOT_RATED)].groupby(["scale_type", "pick_bin"]):
        green = g[g.zone == "green"].sensitivity
        rev.append(dict(scale_type=scale, pick_bin=b, bin_idx=g.bin_idx.iloc[0],
                        min_green_sensitivity=float(green.min()) if len(green) else np.nan,
                        best_detection=g.detection.max()))
    reverse = pd.DataFrame(rev).sort_values(["scale_type", "bin_idx"]) if rev else pd.DataFrame()
    return z, summary, reverse


def _heatmap(z, scale, cfg, r):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    colors = {"green": viz.GOOD, "amber": viz.WARNING, "red": viz.CRITICAL, "no data": viz.GRID}
    levels = sorted(z.sensitivity.unique(), reverse=True)
    bins = list(z.sort_values("bin_idx").pick_bin.unique())
    fig, ax = plt.subplots(figsize=(1.05 * len(bins) + 2.2, 0.48 * len(levels) + 1.6))
    for i, k in enumerate(levels):
        for j, b in enumerate(bins):
            row = z[(z.sensitivity == k) & (z.pick_bin == b)]
            zone = row.zone.iloc[0] if len(row) else "no data"
            if zone == "below threshold":
                ax.add_patch(Rectangle((j, i), 0.94, 0.88, facecolor=viz.SURFACE, edgecolor=viz.AXIS,
                                       hatch="///", lw=0))
                continue
            ax.add_patch(Rectangle((j, i), 0.94, 0.88, color=colors[zone], lw=0))
            if len(row) and zone != "no data":
                ax.text(j + 0.47, i + 0.44, f"{row.detection.iloc[0]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if zone in ("green", "red") else viz.INK)
    ax.set_xlim(0, len(bins))
    ax.set_ylim(0, len(levels))
    ax.set_xticks([j + 0.47 for j in range(len(bins))], [f"{b} g" for b in bins], fontsize=8)
    ax.set_yticks([i + 0.44 for i in range(len(levels))], [f"sens {int(k)}" for k in levels], fontsize=8)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Pick size (weight removed in one go)")
    ax.set_title(f"{scale}: which pick sizes each sensitivity detects reliably")
    note = (" · hatched = below the scale's significance threshold, misses never recorded"
            if (z.zone == "below threshold").any() else "")
    viz.subtitle(ax, f"green >= {cfg.pick_size.green_zone:.0%} detected · amber >= {cfg.pick_size.red_zone:.0%} · "
                     f"red below · gray = < {cfg.pick_size.min_picks_per_cell} picks{note}")
    return viz.save(fig, r.fig_path(cfg.paths.figures, "zone_heatmap", scale))
