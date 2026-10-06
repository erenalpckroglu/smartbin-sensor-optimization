"""
M2 - What does the sensitivity knob do?

Three layers of evidence, from weakest to strongest:

(a) Population curve   - devices on level k vs devices on level 8. Easy, but confounded:
                         Bossard did not assign levels at random (see the sensitivity-3 anomaly).
(b) Same device, before vs after (switchers) + difference-in-differences:
                         a device's own March KPI (level 8) vs its KPI after the switch to k,
                         minus the change that happened anyway over the same calendar weeks
                         to devices that stayed on 8 ("stayers"). Location, rack, box, item and
                         seasonality cancel out. Rate effects are estimated with a fixed-effects
                         (conditional) Poisson rate-ratio, which handles sparse counts properly.
(c) Event study        - weekly KPI of switchers aligned on their switch week: the picture a manager
                         understands in one second ("the line drops at week 0").
(d) Sensitivity-3 anomaly - were the level-3 devices special already before (selection),
                         or does the setting itself misbehave?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.optimize import brentq

from .. import viz
from ..stats import boot_ci, summarize
from . import ModuleResult

KPIS = ["frac_too_many", "too_few_per_week", "miss_rate", "meas_per_day"]
RATES = {"tm": "n_tm", "aok": "n_aok", "tf": "n_tf", "mok": "n_mok"}


def run(ds: pd.DataFrame, dw: pd.DataFrame, cfg, con=None, sample: bool = False) -> ModuleResult:
    r = ModuleResult("m2_sensitivity", "Sensitivity - what does the knob do?",
                     "How do TOO_MANY and TOO_FEW change when the sensitivity is changed?")
    levels = [float(x) for x in cfg.analysis.main_levels]
    main = ds[ds.main_analysis & ds.sensitivity.isin(levels)].copy()

    # (a) population curve -------------------------------------------------------------
    curve = summarize(main, ["scale_type", "sensitivity"], KPIS, reps=cfg.analysis.bootstrap_reps,
                      seed=cfg.analysis.random_seed)
    share = main.groupby(["scale_type", "sensitivity"]).in_target.mean().rename("share_in_target").reset_index()
    curve = curve.merge(share, on=["scale_type", "sensitivity"], how="left")
    r.tables["population_curve"] = curve
    r.figures["tradeoff_population"] = _tradeoff(curve, cfg, r)
    r.figures["kpi_vs_sensitivity"] = _kpi_small_multiples(curve, cfg, r)

    # (b) within-device + DiD ----------------------------------------------------------
    pairs, effects = _within_device(main, dw, cfg)
    r.tables["within_device_pairs"] = pairs
    r.tables["effects"] = effects
    if len(effects):
        r.figures["within_device_effects"] = _effects_chart(effects, cfg, r)

    # (c) event study -------------------------------------------------------------------
    es = _event_study(dw, ds, cfg)
    r.tables["event_study"] = es
    for scale in sorted(es.scale_type.unique()) if len(es) else []:
        r.figures[f"event_study_{scale}"] = _event_chart(es[es.scale_type == scale], scale, cfg, r)

    # (d) anomaly level(s) ----------------------------------------------------------------
    anom, verdicts = _anomaly(ds, effects, cfg)
    r.tables["anomaly"] = anom
    if con is not None and not sample and len(anom):
        sig, bug = _anomaly_signature(con, cfg)
        r.tables["anomaly_signature"] = sig
        for scale, txt in bug.items():
            if txt:
                verdicts[scale] = (verdicts.get(scale, "") + "; " + txt).strip("; ")
    if len(anom):
        r.figures["anomaly"] = _anomaly_chart(anom, cfg, r)
    sg = r.tables.get("anomaly_signature")
    if sg is not None and len(sg):
        r.figures["anomaly_bug"] = _bug_chart(sg, cfg, r)
    # (e) the same loop signature on ANY level (FLAG): scales that re-trigger themselves ------------
    if con is not None and not sample:
        lf = loop_flags(con, cfg)
        r.tables["loop_flags"] = lf
        tot = lf.n_auto.sum()
        r.facts["loops"] = {
            "scales_flagged": int(lf[lf.loop].deviceid.nunique()), "scales_with_auto": int(lf.deviceid.nunique()),
            "share_of_automatic_weighings": float(lf[lf.loop].n_auto.sum() / tot) if tot else None,
            "on_anomaly_level": float(lf[lf.loop & lf.sensitivity.isin(cfg.cleaning.anomaly_levels)].n_auto.sum() / tot)
            if tot else None}

    # facts --------------------------------------------------------------------------------
    for scale, g in curve.groupby("scale_type"):
        f = {"levels": g.sensitivity.tolist(),
             "frac_too_many_median": g.frac_too_many_median.round(3).tolist(),
             "too_few_per_week_median": g.too_few_per_week_median.round(3).tolist(),
             "share_in_target": g.share_in_target.round(3).tolist()}
        e = effects[effects.scale_type == scale] if len(effects) else pd.DataFrame()
        if len(e):
            f["did_frac_too_many"] = dict(zip(e.sensitivity, e.did_frac_too_many.round(3)))
            f["rate_ratio_too_many"] = dict(zip(e.sensitivity, np.exp(e.beta_tm).round(3)))
            f["rate_ratio_too_few"] = dict(zip(e.sensitivity, np.exp(e.beta_tf).round(3)))
            f["n_switchers"] = dict(zip(e.sensitivity, e.n_devices))
        f["anomaly_verdict"] = verdicts.get(scale, "")
        r.facts[scale] = f
    r.facts["n_switchers_total"] = int(pairs.deviceid.nunique()) if len(pairs) else 0
    r.notes.append("Population curve compares DIFFERENT devices -> confounded. Decisions use the "
                   "within-device difference-in-differences estimates.")
    return r


# ------------------------------------------------------------------------------ within-device
def _stayer_windows(dw, ds, cfg):
    """Cumulative weekly sums of stayers per scale type -> fast KPI for any calendar window."""
    stayers = ds.loc[ds.device_role == "stayer", "deviceid"].unique()
    st_ = dw[dw.deviceid.isin(stayers)].copy()
    # BALANCED cohort: only stayers observed from the first to the last weeks. The fleet grows during
    # the experiment (new scales join in May/June and behave differently) - pooling all stayers would
    # mix that composition change into the "calendar trend".
    st_["week"] = pd.to_datetime(st_.week)
    weeks = np.sort(st_.week.unique())
    if len(weeks) >= 6:
        cov = st_.groupby("deviceid").week.agg(["min", "max", "nunique"])
        keep = cov[(cov["min"] <= weeks[1]) & (cov["max"] >= weeks[-3]) & (cov["nunique"] >= 0.7 * len(weeks))].index
        if len(keep) >= 50:
            st_ = st_[st_.deviceid.isin(keep)]
    cols = ["n_auto", "n_tm", "n_tf", "n_aok", "n_mok", "n_rows", "active_days"]
    out = {}
    for scale, g in st_.groupby("scale_type"):
        w = g.groupby("week")[cols].sum().sort_index()
        # unit-safe epoch seconds (DuckDB may hand back datetime64[us] or [ns])
        t = ((pd.to_datetime(w.index) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).values.astype(float)
        c = np.vstack([np.zeros(len(cols)), np.cumsum(w.values, axis=0)])
        out[scale] = (t, c, cols)
    return out


def _window_sums(win, t0, t1):
    t, c, cols = win
    lo = np.searchsorted(t, np.asarray(t0) - 7 * 86400, side="right")
    hi = np.searchsorted(t, np.asarray(t1), side="right")
    s = c[hi] - c[lo]
    return pd.DataFrame(s, columns=cols)


def _kpis_from_sums(s):
    days = s.active_days.replace(0, np.nan)
    out = pd.DataFrame({
        "frac_too_many": s.n_tm / s.n_auto.replace(0, np.nan),
        "too_few_per_week": s.n_tf / days * 7,
        "miss_rate": s.n_tf / (s.n_aok + s.n_tf).replace(0, np.nan),
        "meas_per_day": s.n_rows / days,
    })
    for k, col in RATES.items():
        out[f"lograte_{k}"] = np.log((s[col] + 0.5) / days)
    return out


def _fe_poisson_rr(a, e0, b, e1):
    """Common rate ratio theta for paired counts (a: before, exposure e0; b: after, exposure e1),
    conditional on each device's total -> device baselines cancel (fixed effects)."""
    n = a + b
    m = n > 0
    if m.sum() < 3 or b[m].sum() == 0 or a[m].sum() == 0:
        return np.nan
    n, e0, e1, b = n[m], e0[m], e1[m], b[m]

    def score(logt):
        th = np.exp(logt)
        return b.sum() - np.sum(n * th * e1 / (e0 + th * e1))
    try:
        return brentq(score, -12, 12)
    except ValueError:
        return np.nan


def _within_device(main, dw, cfg):
    base = float(cfg.analysis.baseline_sensitivity)
    ds_all = main
    sw = ds_all[ds_all.device_role == "switcher"]
    if sw.empty:
        return pd.DataFrame(), pd.DataFrame()
    pre = sw[sw.sensitivity == base].set_index("deviceid")
    post = sw[sw.sensitivity != base]
    keep = ["scale_type", "sensitivity", "t_start", "t_end", "days", "n_tm", "n_aok", "n_tf", "n_mok",
            "n_auto", "n_rows"] + KPIS
    p = post[["deviceid"] + keep].join(pre[keep].add_suffix("_pre"), on="deviceid", how="inner")
    wins = _stayer_windows(dw, main.assign(device_role=main.device_role), cfg)
    parts = []
    for scale, g in p.groupby("scale_type"):
        g = g.copy()
        if scale in wins:
            c_pre = _kpis_from_sums(_window_sums(wins[scale], g.t_start_pre.values, g.t_end_pre.values))
            c_post = _kpis_from_sums(_window_sums(wins[scale], g.t_start.values, g.t_end.values))
            ctrl = (c_post.values - c_pre.values)
            ctrl = pd.DataFrame(ctrl, columns=c_pre.columns, index=g.index)
        else:
            ctrl = pd.DataFrame(0.0, index=g.index, columns=KPIS + [f"lograte_{k}" for k in RATES])
        for k in KPIS:
            g[f"delta_{k}"] = g[k] - g[f"{k}_pre"]
            g[f"did_{k}"] = g[f"delta_{k}"] - ctrl[k].fillna(0)
        for k in RATES:
            g[f"ctrl_lograte_{k}"] = ctrl[f"lograte_{k}"].fillna(0)
        parts.append(g)
    p = pd.concat(parts, ignore_index=True)

    rows = []
    reps, seed = cfg.analysis.bootstrap_reps, cfg.analysis.random_seed
    rng = np.random.default_rng(seed)
    for (scale, k), g in p.groupby(["scale_type", "sensitivity"]):
        row = dict(scale_type=scale, sensitivity=k, n_devices=len(g))
        for kpi in KPIS:
            row[f"delta_{kpi}"] = g[f"delta_{kpi}"].median()
            row[f"did_{kpi}"] = g[f"did_{kpi}"].median()
            row[f"did_{kpi}_lo"], row[f"did_{kpi}_hi"] = boot_ci(g[f"did_{kpi}"], reps=reps, seed=seed)
        for rk, col in RATES.items():
            a, b = g[f"{col}_pre"].values.astype(float), g[col].values.astype(float)
            e0 = g["days_pre"].values.astype(float)
            e1 = g["days"].values.astype(float) * np.exp(g[f"ctrl_lograte_{rk}"].values)  # stayer trend offset
            row[f"beta_{rk}"] = _fe_poisson_rr(a, e0, b, e1)
            if rk in ("tm", "tf") and len(g) >= 10:
                bs = []
                for _ in range(min(reps, 200)):
                    i = rng.integers(0, len(g), len(g))
                    bs.append(_fe_poisson_rr(a[i], e0[i], b[i], e1[i]))
                row[f"beta_{rk}_lo"], row[f"beta_{rk}_hi"] = np.nanquantile(bs, [0.025, 0.975])
        rows.append(row)
    eff = pd.DataFrame(rows)
    base_rows = pd.DataFrame([dict(scale_type=s, sensitivity=base, n_devices=np.nan,
                                   **{f"beta_{k}": 0.0 for k in RATES}) for s in eff.scale_type.unique()])
    eff = pd.concat([eff, base_rows], ignore_index=True).sort_values(["scale_type", "sensitivity"])
    return p, eff.reset_index(drop=True)


# ------------------------------------------------------------------------------ event study
def _band(k):
    if k <= 2:
        return "to 1-2 (low)"
    if k <= 5:
        return "to 4-5 (mid)"
    return "to 6-7 (high)"


def _event_study(dw, ds, cfg):
    base = float(cfg.analysis.baseline_sensitivity)
    ids = ds.loc[ds.device_role == "switcher", "deviceid"].unique()
    d = dw[dw.deviceid.isin(ids)].copy()
    if d.empty:
        return pd.DataFrame()
    d["week"] = pd.to_datetime(d.week)
    other = d[d.sensitivity != base].sort_values("week")
    first_week = other.groupby("deviceid").week.min()
    target = other.groupby("deviceid").sensitivity.first()
    d["rel_week"] = ((d.week - d.deviceid.map(first_week)).dt.days // 7).astype(int)
    d["band"] = d.deviceid.map(target).map(_band)
    # clean timeline: before = still on the baseline, after = still on the FIRST new level (a later
    # switch back to 8 or to a third level would otherwise blur the picture)
    d = d[((d.rel_week < 0) & (d.sensitivity == base)) |
          ((d.rel_week >= 0) & (d.sensitivity == d.deviceid.map(target)))]
    d = d[d.rel_week.between(-5, 8)]
    g = d.groupby(["scale_type", "band", "rel_week"])
    out = g[["n_auto", "n_tm", "n_tf", "n_aok", "n_rows", "active_days"]].sum()
    out["devices"] = g.deviceid.nunique()
    out = out.reset_index()
    out["frac_too_many"] = out.n_tm / out.n_auto.replace(0, np.nan)
    out["too_few_per_week"] = out.n_tf / out.active_days.replace(0, np.nan) * 7
    out["meas_per_day"] = out.n_rows / out.active_days.replace(0, np.nan)
    return out[out.devices >= 10]


# ------------------------------------------------------------------------------ anomaly
def _anomaly(ds, effects, cfg):
    base = float(cfg.analysis.baseline_sensitivity)
    lv = [float(x) for x in cfg.cleaning.anomaly_levels]
    ids = ds.loc[ds.sensitivity.isin(lv), "deviceid"].unique()
    if len(ids) == 0:
        return pd.DataFrame(), {}
    at8 = ds[(ds.sensitivity == base) & ds.main_analysis]
    rows, verdicts = [], {}
    for scale in sorted(ds.scale_type.dropna().unique()):
        a8 = at8[(at8.scale_type == scale) & at8.deviceid.isin(ids)]
        o8 = at8[(at8.scale_type == scale) & ~at8.deviceid.isin(ids)]
        a3 = ds[(ds.scale_type == scale) & ds.sensitivity.isin(lv)]
        if len(a3) < 5:
            continue
        paired = a3.merge(a8, on="deviceid", suffixes=("_anom", "_8"))
        p = st.mannwhitneyu(a8.frac_too_many.dropna(), o8.frac_too_many.dropna()).pvalue if len(a8) > 5 else np.nan
        row = dict(scale_type=scale, n_devices_anomaly=len(a3), n_with_baseline=len(a8),
                   frac_tm_at8_anomaly_devices=a8.frac_too_many.median(),
                   frac_tm_at8_other_devices=o8.frac_too_many.median(), p_selection=p,
                   meas_day_at8_anomaly_devices=a8.meas_per_day.median(),
                   meas_day_at8_other_devices=o8.meas_per_day.median(),
                   frac_tm_at_anomaly=a3.frac_too_many.median(), meas_day_at_anomaly=a3.meas_per_day.median(),
                   miss_rate_at_anomaly=a3.miss_rate.median(),
                   paired_delta_frac_tm=(paired.frac_too_many_anom - paired.frac_too_many_8).median()
                   if len(paired) else np.nan,
                   paired_delta_meas_day=(paired.meas_per_day_anom - paired.meas_per_day_8).median()
                   if len(paired) else np.nan)
        rows.append(row)
        sel = (row["frac_tm_at8_anomaly_devices"] - row["frac_tm_at8_other_devices"] > 0.05) and (p < 0.01)
        e = effects[(effects.scale_type == scale) & effects.sensitivity.isin([2.0, 4.0])] if len(effects) else []
        neighbours_lower = len(e) and np.nanmedian(e.did_frac_too_many) < 0
        unlike = (row["paired_delta_frac_tm"] > 0) and bool(neighbours_lower)
        parts = []
        if sel:
            parts.append("SELECTION: these devices were already noisier than others while on level 8")
        if unlike:
            parts.append("SETTING ANOMALY: moving to this level INCREASED empty measurements although its "
                         "neighbouring levels decrease them")
        verdicts[scale] = "; ".join(parts) or "no clear anomaly signal"
    return pd.DataFrame(rows), verdicts


# ------------------------------------------------------------------------------ figures
def _tradeoff(curve, cfg, r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ymax = max(cfg.kpi.tf_target * 1.6, float(curve.too_few_per_week_hi.max()) * 1.1)
    ax.add_patch(plt.Rectangle((0, 0), cfg.kpi.tm_target, cfg.kpi.tf_target, color=viz.GOOD, alpha=0.08, lw=0))
    for scale, g in curve.groupby("scale_type"):
        g = g.sort_values("sensitivity")
        c = viz.SCALE_COLOR.get(scale, viz.CAT[2])
        ax.errorbar(g.frac_too_many_median, g.too_few_per_week_median,
                    xerr=[g.frac_too_many_median - g.frac_too_many_lo, g.frac_too_many_hi - g.frac_too_many_median],
                    yerr=[g.too_few_per_week_median - g.too_few_per_week_lo,
                          g.too_few_per_week_hi - g.too_few_per_week_median],
                    fmt="none", ecolor=c, alpha=0.35, lw=1, zorder=2)
        ax.plot(g.frac_too_many_median, g.too_few_per_week_median, "-o", color=c, label=f"{scale} scales",
                mec=viz.SURFACE, mew=1.5, zorder=3)
        for _, row in g.iterrows():
            if int(row.sensitivity) in cfg.analysis.story_levels:
                ax.annotate(f"{int(row.sensitivity)}", (row.frac_too_many_median, row.too_few_per_week_median),
                            xytext=(6, 4), textcoords="offset points", fontsize=8, color=viz.INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    viz.target_line(ax, cfg.kpi.tm_target, "x")
    viz.target_line(ax, cfg.kpi.tf_target, "y")
    ax.set_xlabel("Fraction TOO_MANY (median)  ->  battery & cloud load")
    ax.set_ylabel("TOO_FEW per week (median)  ->  supply risk")
    ax.set_title("The trade-off: every sensitivity level is a point on this curve")
    viz.subtitle(ax, "numbers = sensitivity level · green box = target zone · bars = 95% CI · "
                     "population view (different devices per level)")
    ax.legend(loc="upper right")
    return viz.save(fig, r.fig_path(cfg.paths.figures, "tradeoff_population"))


def _kpi_small_multiples(curve, cfg, r):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    spec = [("frac_too_many", "Fraction TOO_MANY", cfg.kpi.tm_target),
            ("too_few_per_week", "TOO_FEW per week", cfg.kpi.tf_target),
            ("meas_per_day", "Measurements per day", None)]
    for ax, (k, lab, tgt) in zip(axes, spec):
        for scale, g in curve.groupby("scale_type"):
            g = g.sort_values("sensitivity")
            c = viz.SCALE_COLOR.get(scale, viz.CAT[2])
            ax.fill_between(g.sensitivity, g[f"{k}_lo"], g[f"{k}_hi"], color=c, alpha=0.10, lw=0)
            ax.plot(g.sensitivity, g[f"{k}_median"], "-o", color=c, label=scale, mec=viz.SURFACE, mew=1.5)
        if tgt is not None:
            viz.target_line(ax, tgt, "y", "target")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("Sensitivity (higher = more sensitive)")
        ax.set_ylim(bottom=0)
    axes[0].legend()
    fig.suptitle("KPIs by sensitivity level (median per device, 95% CI)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return viz.save(fig, r.fig_path(cfg.paths.figures, "kpi_vs_sensitivity"))


def _effects_chart(eff, cfg, r):
    import matplotlib.pyplot as plt
    e = eff[eff.n_devices.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    scales = sorted(e.scale_type.unique())
    levels = sorted(e.sensitivity.unique())
    x = np.arange(len(levels))
    wbar = 0.38
    for i, scale in enumerate(scales):
        g = e[e.scale_type == scale].set_index("sensitivity").reindex(levels)
        c = viz.SCALE_COLOR.get(scale, viz.CAT[2])
        off = (i - (len(scales) - 1) / 2) * wbar
        axes[0].bar(x + off, g.did_frac_too_many, width=wbar * 0.9, color=c, label=scale)
        axes[0].errorbar(x + off, g.did_frac_too_many,
                         yerr=[g.did_frac_too_many - g.did_frac_too_many_lo,
                               g.did_frac_too_many_hi - g.did_frac_too_many],
                         fmt="none", ecolor=viz.INK2, lw=1)
        axes[1].bar(x + off, g.did_too_few_per_week, width=wbar * 0.9, color=c, label=scale)
        axes[1].errorbar(x + off, g.did_too_few_per_week,
                         yerr=[g.did_too_few_per_week - g.did_too_few_per_week_lo,
                               g.did_too_few_per_week_hi - g.did_too_few_per_week],
                         fmt="none", ecolor=viz.INK2, lw=1)
    for ax in axes:
        ax.axhline(0, color=viz.AXIS, lw=1)
        ax.set_xticks(x, [f"8 -> {int(k)}" for k in levels])
        ax.set_xlabel("Same device moved from sensitivity 8 to ...")
    axes[0].set_title("Change in fraction TOO_MANY (points)", fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    axes[1].set_title("Change in TOO_FEW per week (absolute)", fontsize=10)
    viz.target_line(axes[1], cfg.kpi.tf_target, "y", f"target {cfg.kpi.tf_target:g}/week")
    fig.legend(h, l, loc="upper right", ncol=2)
    fig.suptitle("Causal effect of the setting: same devices before vs after, minus the trend of stayers",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return viz.save(fig, r.fig_path(cfg.paths.figures, "within_device_effects"))


def _event_chart(es, scale, cfg, r):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    bands = ["to 1-2 (low)", "to 4-5 (mid)", "to 6-7 (high)"]
    colors = dict(zip(bands, [viz.BLUE_RAMP[1], viz.BLUE_RAMP[5], viz.BLUE_RAMP[9]]))
    for band in bands:
        g = es[es.band == band].sort_values("rel_week")
        if g.empty:
            continue
        axes[0].plot(g.rel_week, g.frac_too_many, "-o", color=colors[band], label=f"switched {band}",
                     ms=4, mec=viz.SURFACE)
        axes[1].plot(g.rel_week, g.too_few_per_week, "-o", color=colors[band], ms=4, mec=viz.SURFACE)
    for ax in axes:
        ax.axvline(-0.5, color=viz.MUTED, lw=1)
        ax.set_xlabel("Weeks relative to the switch away from sensitivity 8")
        ax.set_ylim(bottom=0)
    viz.target_line(axes[0], cfg.kpi.tm_target, "y", "target")
    viz.target_line(axes[1], cfg.kpi.tf_target, "y", "target")
    axes[0].set_title("Fraction TOO_MANY", fontsize=10)
    axes[1].set_title("TOO_FEW per week", fontsize=10)
    axes[0].legend(loc="lower left")
    fig.suptitle(f"{scale}: what happens to a scale the week its sensitivity is lowered",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return viz.save(fig, r.fig_path(cfg.paths.figures, "event_study", scale))


def _anomaly_chart(anom, cfg, r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    lv = ",".join(str(x) for x in cfg.cleaning.anomaly_levels)
    groups = [("frac_tm_at8_other_devices", "other devices\non level 8"),
              ("frac_tm_at8_anomaly_devices", f"level-{lv} devices,\nwhile still on 8"),
              ("frac_tm_at_anomaly", f"level-{lv} devices\non level {lv}")]
    x = np.arange(len(groups))
    w = 0.38
    for i, (_, row) in enumerate(anom.iterrows()):
        c = viz.SCALE_COLOR.get(row.scale_type, viz.CAT[2])
        vals = [row[g] for g, _ in groups]
        ax.bar(x + (i - (len(anom) - 1) / 2) * w, vals, width=w * 0.9, color=c, label=row.scale_type)
    ax.set_xticks(x, [lab for _, lab in groups])
    ax.set_ylabel("Fraction TOO_MANY (median)")
    ax.set_ylim(0, 1)
    viz.target_line(ax, cfg.kpi.tm_target, "y", "target")
    ax.legend()
    ax.set_title(f"Why sensitivity {lv} is studied separately")
    return viz.save(fig, r.fig_path(cfg.paths.figures, "anomaly"))


def loop_flags(con, cfg) -> pd.DataFrame:
    """
    Trigger-loop detector, one row per scale and level. A scale-level is flagged when more than
    `flags.loop_min_share` of its automatic wake-ups follow the previous one within `flags.loop_max_gap_s`
    seconds AND it wakes up more than `flags.loop_min_per_day` times a day: the level-3 signature, on any level.
    """
    f = cfg.flags
    d = con.execute(f"""
        WITH auto AS (
            SELECT deviceid, scale_type, sensitivity, location_label, day,
                   ts - lag(ts) OVER (PARTITION BY deviceid ORDER BY ts) AS gap
            FROM events WHERE is_auto AND scale_type IS NOT NULL AND sensitivity IS NOT NULL)
        SELECT deviceid, scale_type, sensitivity, any_value(location_label) AS location_label,
               count(*) AS n_auto, count(DISTINCT day) AS days,
               avg(CASE WHEN gap <= {float(f.loop_max_gap_s)} THEN 1.0 ELSE 0 END) AS share_retrigger
        FROM auto GROUP BY ALL HAVING count(DISTINCT day) >= {int(f.loop_min_days)}
    """).df()
    d["auto_per_day"] = d.n_auto / d.days
    d["loop"] = (d.share_retrigger > f.loop_min_share) & (d.auto_per_day > f.loop_min_per_day)
    return d


def _anomaly_signature(con, cfg):
    """
    Bug test for the anomaly level: does the sensor on that level fire in rapid loops or at night
    (nobody working) much more than the SAME devices did on the baseline level? That pattern points
    to a firmware/config problem rather than to busy bins.
    """
    lv = ", ".join(str(float(x)) for x in cfg.cleaning.anomaly_levels)
    b = float(cfg.analysis.baseline_sensitivity)
    sig = con.execute(f"""
        WITH a AS (SELECT DISTINCT deviceid FROM events WHERE sensitivity IN ({lv})),
        auto AS (
            SELECT deviceid, scale_type, sensitivity, cls, day, hour,
                   ts - lag(ts) OVER (PARTITION BY deviceid ORDER BY ts) AS gap_auto
            FROM events WHERE is_auto AND sensitivity IS NOT NULL AND scale_type IS NOT NULL),
        g AS (
            SELECT *, CASE WHEN sensitivity IN ({lv}) THEN 'anomaly devices on anomaly level'
                           WHEN sensitivity = {b} AND deviceid IN (SELECT deviceid FROM a) THEN 'same devices on baseline'
                           WHEN sensitivity = {b} THEN 'other devices on baseline' END AS grp
            FROM auto)
        SELECT scale_type, grp, count(DISTINCT deviceid) AS devices, count(*) AS n_auto,
               count(*) * 1.0 / count(DISTINCT deviceid || '|' || CAST(day AS VARCHAR)) AS auto_per_device_day,
               median(gap_auto) AS median_gap_s,
               avg(CASE WHEN gap_auto <= 120 THEN 1.0 ELSE 0 END) AS share_retrigger_120s,
               avg(CASE WHEN hour < 6 OR hour >= 22 THEN 1.0 ELSE 0 END) AS share_night,
               avg(CASE WHEN cls = 'TOO_MANY' THEN 1.0 ELSE 0 END) AS frac_too_many
        FROM g WHERE grp IS NOT NULL GROUP BY ALL ORDER BY 1, 2
    """).df()
    bug = {}
    for scale, g in sig.groupby("scale_type"):
        g = g.set_index("grp")
        if {"anomaly devices on anomaly level", "same devices on baseline"} <= set(g.index):
            on, off = g.loc["anomaly devices on anomaly level"], g.loc["same devices on baseline"]
            flags = []
            if on.share_retrigger_120s > max(2 * off.share_retrigger_120s, off.share_retrigger_120s + 0.10):
                flags.append(f"re-trigger loop ({on.share_retrigger_120s:.0%} of triggers within 2 min vs "
                             f"{off.share_retrigger_120s:.0%} on the baseline)")
            if on.share_night > max(2 * off.share_night, off.share_night + 0.05):
                flags.append(f"fires at night ({on.share_night:.0%} vs {off.share_night:.0%})")
            if on.auto_per_device_day > 2 * off.auto_per_device_day:
                flags.append(f"{on.auto_per_device_day / off.auto_per_device_day:.1f}x more triggers per day")
            bug[scale] = ("BUG SIGNATURE: " + ", ".join(flags)) if flags else ""
    return sig, bug


def _bug_chart(sig, cfg, r):
    """The fingerprint of the anomaly level: how often, how fast and when the sensor fires."""
    import matplotlib.pyplot as plt
    lv = ",".join(str(x) for x in cfg.cleaning.anomaly_levels)
    groups = [("other devices on baseline", "other scales\non 8"), ("same devices on baseline", f"level-{lv} scales\nwhile on 8"),
              ("anomaly devices on anomaly level", f"level-{lv} scales\non {lv}")]
    panels = [("auto_per_device_day", "Automatic triggers per scale per day", "{:.0f}"),
              ("share_retrigger_120s", "Triggers within 2 min of the previous one", "{:.0%}"),
              ("share_night", "Triggers at night (22-06 h)", "{:.0%}")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    scales = sorted(sig.scale_type.unique())
    x = np.arange(len(groups))
    w = 0.38
    for ax, (col, title, fmt) in zip(axes, panels):
        for i, scale in enumerate(scales):
            g = sig[sig.scale_type == scale].set_index("grp").reindex([k for k, _ in groups])
            c = viz.SCALE_COLOR.get(scale, viz.CAT[2])
            off = (i - (len(scales) - 1) / 2) * w
            ax.bar(x + off, g[col], width=w * 0.9, color=c, label=scale)
            for xi, v in zip(x + off, g[col]):
                if np.isfinite(v):
                    ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=viz.INK2)
        ax.set_xticks(x, [lab for _, lab in groups], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
    axes[0].legend()
    fig.suptitle(f"Sensitivity {lv} is not a setting, it is a bug: the sensor fires in a loop, day and night",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return viz.save(fig, r.fig_path(cfg.paths.figures, "anomaly_bug"))
