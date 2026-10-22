"""
M7 - Recommendation: which sensitivity policy should Bossard run, and what does it buy?

Counterfactual ("what-if") simulator:
  every scale's own baseline rates (TOO_MANY, detected picks, TOO_FEW, manual checks per week at level 8)
  x the causal rate ratios of moving 8 -> k (from M2, same-device + stayer-trend corrected)
  = that scale's expected KPIs at every level k.

Policies compared on the SAME scales:
  P0 status quo       everybody on 8
  P1 one setting      best single level per scale type
  P2 segment table    best level per scale type x pick-size class (or item-weight class)
  P3 adaptive rule    every 2 weeks: too many misses -> +1 ; too many empty measurements AND
                      misses comfortably low -> -1 ; else hold   (simulated with real-world noise)
  P4 oracle           best level per scale with perfect knowledge (upper bound, not deployable)
  P5 one setting      best single level per scale type for KPI 2.0
  P6 adaptive rule    2-weekly rule for KPI 2.0, every scale starts on 8
  P7 start 7 + rule   our recommendation: factory default 7, then the KPI 2.0 rule (SET + TUNE)
  P8 fraction rule    daily, event-driven rule steered by the TOO_MANY fraction: each TOO_FEW -> +1,
                      median daily fraction > 50 % over the last 7 days (>= 3 days) -> -1, start on 6.
                      A fraction-driven feedback rule of the kind used in practice, for comparison.

Business translation: measurements/week, battery-life proxy, fleet-wide extrapolation.
Backtest: predicted vs actually observed KPIs of the switchers (does the simulator tell the truth?).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import viz
from ..kpi import battery_life_ratio
from . import ModuleResult

RATE_COLS = {"tm": "n_tm", "aok": "n_aok", "tf": "n_tf", "mok": "n_mok"}
POLICIES = ["P0 status quo (8)", "P1 one setting", "P2 segment table", "P3 adaptive rule", "P4 oracle",
            "P5 one setting (KPI 2.0)", "P6 adaptive rule (KPI 2.0)", "P7 start 7 + rule (KPI 2.0)",
            "P8 fraction rule (start 6)"]


def _baseline_rates(ds, cfg):
    b = float(cfg.analysis.baseline_sensitivity)
    d = ds[ds.main_analysis & (ds.sensitivity == b)].copy()
    weeks = d.days / 7
    for k, col in RATE_COLS.items():
        rate = d[col] / weeks
        prior = rate.groupby(d.scale_type).transform("mean")
        # empirical-Bayes shrink (1 pseudo-week) toward the scale-type MEAN: a scale with 0 observed misses
        # at level 8 must still be allowed to miss at level 1 (a median prior of 0 would forbid that)
        d[f"r_{k}"] = (d[col] + 1.0 * prior) / (weeks + 1.0)
    return d


def _betas(effects, cfg):
    e = effects.copy()
    base = float(cfg.analysis.baseline_sensitivity)
    ok = (e.sensitivity == base) | (e.n_devices >= cfg.policy.min_switchers)
    e = e[ok & e[["beta_tm", "beta_aok", "beta_tf"]].notna().all(axis=1)]
    e = e[e.sensitivity.isin([float(x) for x in cfg.policy.levels])]
    e["beta_mok"] = e.beta_mok.fillna(0.0)
    return e


def _predict(rates: pd.DataFrame, beta_row) -> pd.DataFrame:
    tm = rates.r_tm * np.exp(beta_row.beta_tm)
    aok = rates.r_aok * np.exp(beta_row.beta_aok)
    tf = rates.r_tf * np.exp(beta_row.beta_tf)
    mok = rates.r_mok * np.exp(beta_row.beta_mok)
    return pd.DataFrame({"frac": tm / (tm + aok), "tfw": tf, "mpd": (tm + aok + tf + mok) / 7,
                         "tm_w": tm, "aok_w": aok}, index=rates.index)


def run(ds, effects, pairs, pick_profile, cfg, loops=None) -> ModuleResult:
    r = ModuleResult("m7_policy", "Recommendation - which setting policy, and what does it buy?",
                     "What is the best sensitivity policy and how much does it save?")
    if effects is None or len(effects) == 0:
        r.status = "skipped"
        r.notes.append("No same-device effects available (no switchers) -> no causal what-if possible.")
        return r
    t_tm, t_tf = cfg.kpi.tm_target, cfg.kpi.tf_target
    base = float(cfg.analysis.baseline_sensitivity)
    rates = _baseline_rates(ds, cfg)
    k2 = cfg.kpi2
    if pick_profile is not None and len(pick_profile):
        rates = rates.merge(pick_profile[["deviceid", "pick_class"]], on="deviceid", how="left")
        rates["segment"] = rates.pick_class.astype(object).where(rates.pick_class.notna(), "unknown").astype(str)
        seg_name = "pick-size class"
    else:
        seg = pd.qcut(rates.item_weight.rank(method="first"), 3, labels=["light items", "medium items", "heavy items"])
        rates["segment"] = seg.astype(object).where(seg.notna(), "unknown").astype(str)
        seg_name = "item-weight class"
    betas = _betas(effects, cfg)
    # scales caught in a trigger loop on the baseline level: their load is a defect (FLAG), not a setting
    # question, so the savings of every policy are also reported without them
    loop_ids = set()
    if loops is not None and len(loops):
        lf = loops[loops["loop"].astype(bool) & (loops.sensitivity == base)]
        loop_ids = set(lf.deviceid)
    reps = int(getattr(cfg.policy, "replications", 5))

    def rng_for(scale_i, pol):
        # one independent random stream per scale type and policy: adding a policy never shifts the others
        return np.random.default_rng([int(cfg.policy.seed), scale_i, POLICIES.index(pol)])

    pred_rows, choice_rows, summary, frontier, recs, globals_ = [], [], [], [], [], {}
    for scale_i, (scale, R) in enumerate(rates.groupby("scale_type")):
        B = betas[betas.scale_type == scale].set_index("sensitivity").sort_index()
        if base not in B.index or len(B) < 2:
            continue
        levels = list(B.index)
        P = {k: _predict(R, B.loc[k]) for k in levels}
        frac = pd.DataFrame({k: P[k].frac for k in levels})
        tfw = pd.DataFrame({k: P[k].tfw for k in levels})
        mpd = pd.DataFrame({k: P[k].mpd for k in levels})
        tmw = pd.DataFrame({k: P[k].tm_w for k in levels})
        miss = pd.DataFrame({k: P[k].tfw / (P[k].tfw + P[k].aok_w) for k in levels})
        ok = (frac <= t_tm) & (tfw <= t_tf)
        # KPI 2.0 (our proposal): absolute load + share of real changes missed, both independent of how busy a bin is
        ok2 = (tmw <= k2.tm_per_week) & (miss <= k2.miss_rate) & (tfw <= t_tf)
        viol = (np.maximum(frac - t_tm, 0) / t_tm) + (np.maximum(tfw - t_tf, 0) / t_tf)

        for k in levels:
            frontier.append(dict(scale_type=scale, sensitivity=k, median_frac=frac[k].median(),
                                 median_tfw=tfw[k].median(), share_in_target=ok[k].mean(),
                                 mean_meas_day=mpd[k].mean(), median_tm_week=tmw[k].median(),
                                 median_miss_rate=miss[k].median(), share_in_target_kpi2=ok2[k].mean()))
        choice = pd.DataFrame(index=R.index)
        choice["P0 status quo (8)"] = base
        g_score = pd.DataFrame({k: [ok[k].mean(), -mpd[k].mean()] for k in levels}).T
        best_global = g_score.sort_values([0, 1], ascending=False).index[0]
        choice["P1 one setting"] = best_global
        globals_[scale] = float(best_global)
        seg_best = {}
        for seg, idx in R.groupby("segment").groups.items():
            sc = pd.DataFrame({k: [ok.loc[idx, k].mean(), -mpd.loc[idx, k].mean()] for k in levels}).T
            seg_best[seg] = sc.sort_values([0, 1], ascending=False).index[0]
            recs.append(dict(scale_type=scale, segment=seg, segment_type=seg_name, devices=len(idx),
                             recommended_sensitivity=seg_best[seg],
                             share_in_target_now=ok.loc[idx, base].mean(),
                             share_in_target_recommended=ok.loc[idx, seg_best[seg]].mean(),
                             median_frac_now=frac.loc[idx, base].median(),
                             median_frac_recommended=frac.loc[idx, seg_best[seg]].median(),
                             median_tfw_now=tfw.loc[idx, base].median(),
                             median_tfw_recommended=tfw.loc[idx, seg_best[seg]].median(),
                             meas_change_pct=(mpd.loc[idx, seg_best[seg]].sum() / mpd.loc[idx, base].sum() - 1) * 100))
        choice["P2 segment table"] = R.segment.map(seg_best)
        # simulated rules: `reps` independent replications of every scale (tiled), so the summary is an
        # average over replications instead of one random path
        tit, changes, tiled = {}, {}, {}
        Rt = pd.concat([R] * reps, ignore_index=True)
        for pol, kw in [("P3 adaptive rule", {}), ("P6 adaptive rule (KPI 2.0)", {"mode": "kpi2"}),
                        ("P7 start 7 + rule (KPI 2.0)", {"mode": "kpi2", "start": 7.0})]:
            tiled[pol], tit[pol], changes[pol] = _adaptive(Rt, B, levels, base, cfg, rng_for(scale_i, pol), **kw)
        pol = "P8 fraction rule (start 6)"
        tiled[pol], changes[pol] = _daily_fraction_rule(Rt, B, levels, cfg, rng_for(scale_i, pol), start=6.0)
        for pol, k in tiled.items():
            choice[pol] = k[:len(R)]            # first replication = the per-scale choice that is exported
        # oracle: in target with fewest measurements, else smallest violation
        score = viol.values * 1e6 + mpd.values
        choice["P4 oracle"] = np.array(levels)[np.argmin(np.where(ok.values, mpd.values, score), axis=1)]
        g2 = pd.DataFrame({k: [ok2[k].mean(), -mpd[k].mean()] for k in levels}).T
        choice["P5 one setting (KPI 2.0)"] = g2.sort_values([0, 1], ascending=False).index[0]
        globals_[f"{scale}_kpi2"] = float(choice["P5 one setting (KPI 2.0)"].iloc[0])

        for pol in POLICIES:
            k = np.asarray(tiled.get(pol, choice[pol]))
            rr = len(k) // len(R)               # 1 for fixed policies, `reps` for simulated rules
            col = np.array([levels.index(x) for x in k])
            pick = lambda T: np.tile(T.values, (rr, 1))[np.arange(len(k)), col]
            f, t, m, tw, ms = pick(frac), pick(tfw), pick(mpd), pick(tmw), pick(miss)
            m0 = np.tile(mpd[base].values, rr)
            nl = ~np.tile(R.deviceid.isin(loop_ids).values, rr)        # scales without a trigger loop
            row = dict(scale_type=scale, policy=pol, devices=len(R),
                       share_in_target=np.mean((f <= t_tm) & (t <= t_tf)), share_meets_tm=np.mean(f <= t_tm),
                       share_meets_tf=np.mean(t <= t_tf), median_frac=np.median(f), median_tfw=np.median(t),
                       meas_per_week_total=m.sum() * 7 / rr, meas_change_pct=(m.sum() / m0.sum() - 1) * 100,
                       battery_life_change_pct=(np.median(battery_life_ratio(m0, m, cfg)) - 1) * 100,
                       meas_saved_per_scale_week=(m0 - m).mean() * 7,
                       share_in_target_kpi2=np.mean((tw <= k2.tm_per_week) & (ms <= k2.miss_rate) & (t <= t_tf)),
                       median_tm_week=np.median(tw), median_miss_rate=np.nanmedian(ms),
                       loop_scales=int((~nl).sum() // rr), loop_share_of_today=1 - m0[nl].sum() / m0.sum(),
                       meas_change_pct_without_loops=(m[nl].sum() / m0[nl].sum() - 1) * 100,
                       meas_saved_per_scale_week_without_loops=(m0[nl] - m[nl]).mean() * 7)
            if pol in tit:
                row["time_in_target"] = tit[pol]
            if pol in changes:
                row["setting_changes_per_scale_month"] = changes[pol]
            summary.append(row)
        for pol in POLICIES:
            choice_rows.append(pd.DataFrame({"deviceid": R.deviceid.values, "scale_type": scale, "policy": pol,
                                             "sensitivity": choice[pol].values}))
        pred_rows.append(pd.concat([frac.add_prefix("frac_"), tfw.add_prefix("tfw_")], axis=1)
                         .assign(deviceid=R.deviceid.values, scale_type=scale))

    if not summary:
        r.status = "skipped"
        r.notes.append("Effects available for too few levels.")
        return r
    summ = pd.DataFrame(summary)
    n_dev = summ.groupby("scale_type").devices.first().sum()
    inst = cfg.fleet.installed_scales
    summ["fleet_meas_saved_per_week"] = summ.meas_saved_per_scale_week * inst
    r.tables["policy_comparison"] = summ
    r.tables["frontier"] = pd.DataFrame(frontier)
    r.tables["recommendation"] = pd.DataFrame(recs)
    r.tables["device_choices"] = pd.concat(choice_rows, ignore_index=True)
    r.tables["backtest"] = bt = _backtest(pairs, rates, betas)

    for scale, g in summ.groupby("scale_type"):
        g = g.set_index("policy")
        r.facts[scale] = {
            "share_in_target": g.share_in_target.round(3).to_dict(),
            "meas_change_pct": g.meas_change_pct.round(1).to_dict(),
            "battery_life_change_pct": g.battery_life_change_pct.round(1).to_dict(),
            "median_frac": g.median_frac.round(3).to_dict(), "median_tfw": g.median_tfw.round(2).to_dict(),
            "one_setting": globals_.get(scale),
            "one_setting_kpi2": globals_.get(f"{scale}_kpi2"),
            "share_in_target_kpi2": g.share_in_target_kpi2.round(3).to_dict(),
            "median_miss_rate": g.median_miss_rate.round(3).to_dict(),
            "median_tm_week": g.median_tm_week.round(2).to_dict(),
        }
    fr = pd.DataFrame(frontier)
    best = []
    for scale, g in fr.groupby("scale_type"):
        feasible = g[g.median_tfw <= t_tf]
        best.append(dict(scale_type=scale,
                         best_achievable_median_frac=float(feasible.median_frac.min()) if len(feasible) else np.nan,
                         at_level=float(feasible.loc[feasible.median_frac.idxmin(), "sensitivity"]) if len(feasible)
                         else np.nan))
    r.tables["achievable_target"] = pd.DataFrame(best)
    r.facts["achievable_target"] = pd.DataFrame(best).to_dict(orient="records")
    r.facts["fleet_extrapolation_scales"] = inst
    r.facts["devices_simulated"] = int(n_dev)
    if len(bt):
        r.facts["backtest"] = bt.to_dict(orient="records")
    r.figures["policy_comparison"] = _policy_chart(summ, cfg, r)
    for scale in fr.scale_type.unique():
        r.figures[f"frontier_{scale}"] = _frontier_chart(fr[fr.scale_type == scale], summ[summ.scale_type == scale],
                                                         scale, cfg, r)
    r.notes += [
        "Expected KPIs come from each scale's own level-8 history x causal rate ratios from same-device switches. "
        "Assumes a scale reacts to a setting change like the average switcher of its scale type.",
        f"Battery: measurement assumed to take {cfg.battery.measurement_seconds} s at "
        f"{cfg.battery.active_power_factor}x sleep power (case slide); change the assumption in config.yaml.",
        f"Fleet numbers extrapolate the per-scale average to {inst:,} installed scales.",
    ]
    return r


def _adaptive(R, B, levels, base, cfg, rng, mode="bossard", start=None, min_tf=0, down_miss=None, min_picks_down=0):
    """
    Simulate the 2-weekly feedback rule with Poisson noise.
    Returns final levels, mean time-in-target (second half) and setting changes per scale and month.
    mode="bossard": +1 if TOO_FEW/week > 2 ; -1 if TOO_MANY fraction > 0.5 and TOO_FEW comfortably low
    mode="kpi2"   : +1 if miss rate > 10% ; -1 if TOO_MANY/week > 7 and miss rate <= 5%
    start         : first level of every scale (default: the baseline level 8)
    min_tf        : kpi2 only - a step up also needs at least this many misses in the cycle (0 = off)
    down_miss     : kpi2 only - highest miss share that still allows a step down (default: half the miss target)
    min_picks_down: kpi2 only - a step down also needs at least this many picks seen in the cycle (0 = off)
    """
    t_tm, t_tf = cfg.kpi.tm_target, cfg.kpi.tf_target
    k2 = cfg.kpi2
    weeks = cfg.policy.cycle_days / 7
    first = start if start in levels else base
    idx = np.full(len(R), levels.index(first))
    rates = {k: _predict(R, B.loc[k]) for k in levels}
    in_target, n_changes = [], 0.0
    for _ in range(cfg.policy.n_cycles):
        cur = np.array(levels)[idx]
        tm = np.array([rates[k].tm_w.values[i] for i, k in enumerate(cur)])
        aok = np.array([rates[k].aok_w.values[i] for i, k in enumerate(cur)])
        tf = np.array([rates[k].tfw.values[i] for i, k in enumerate(cur)])
        miss_true = tf / np.maximum(tf + aok, 1e-9)
        if mode == "kpi2":
            in_target.append(np.mean((tm <= k2.tm_per_week) & (miss_true <= k2.miss_rate) & (tf <= t_tf)))
        else:
            in_target.append(np.mean((tm / (tm + aok) <= t_tm) & (tf <= t_tf)))
        c_tm, c_aok, c_tf = rng.poisson(tm * weeks), rng.poisson(aok * weeks), rng.poisson(tf * weeks)
        frac_obs = np.where(c_tm + c_aok > 0, c_tm / np.maximum(c_tm + c_aok, 1), 0.0)
        tf_obs = c_tf / weeks
        if mode == "kpi2":
            miss_obs = np.where(c_tf + c_aok > 0, c_tf / np.maximum(c_tf + c_aok, 1), 0.0)
            up = ((miss_obs > k2.miss_rate) | (tf_obs > t_tf)) & (c_tf >= min_tf)
            dm = k2.miss_rate / 2 if down_miss is None else down_miss
            down = (~up) & (c_tm / weeks > k2.tm_per_week) & (miss_obs <= dm) & (c_aok + c_tf >= min_picks_down)
        else:
            up = tf_obs > t_tf
            down = (~up) & (frac_obs > t_tm) & (tf_obs <= cfg.policy.tf_safety * t_tf)
        new_idx = np.clip(idx + up.astype(int) - down.astype(int), 0, len(levels) - 1)
        n_changes += float(np.mean(new_idx != idx))
        idx = new_idx
    months = cfg.policy.n_cycles * cfg.policy.cycle_days / 30.4
    return np.array(levels)[idx], float(np.mean(in_target[len(in_target) // 2:])), n_changes / months


def _daily_fraction_rule(R, B, levels, cfg, rng, start=6.0, window=7, min_days=3, threshold=0.5):
    """
    Event-driven rule evaluated once per day with Poisson noise (an approximation of a rule that reacts to
    every incoming weighing):
      * every TOO_FEW -> one level more sensitive;
      * otherwise, once at least `min_days` days have passed since the last down-step, the median of the
        daily TOO_MANY fractions over the last `window` days (or since that step) above `threshold`
        -> one level less sensitive.
    Level 3 is not in `levels`, i.e. it is blocked. Returns final levels and setting changes per scale and month.
    """
    import warnings
    days = int(cfg.policy.cycle_days * cfg.policy.n_cycles)
    L, n = len(levels), len(R)
    rows = np.arange(n)
    idx = np.full(n, levels.index(start) if start in levels else L - 1)
    rates = {k: _predict(R, B.loc[k]) for k in levels}
    TM = np.column_stack([rates[k].tm_w.values for k in levels]) / 7.0
    AOK = np.column_stack([rates[k].aok_w.values for k in levels]) / 7.0
    TF = np.column_stack([rates[k].tfw.values for k in levels]) / 7.0
    hist = np.full((n, window), np.nan)
    since = np.zeros(n, dtype=int)
    n_changes = 0.0
    for _ in range(days):
        c_tm = rng.poisson(TM[rows, idx])
        c_aok = rng.poisson(AOK[rows, idx])
        c_tf = rng.poisson(TF[rows, idx])
        auto = c_tm + c_aok
        hist = np.roll(hist, -1, axis=1)
        hist[:, -1] = np.where(auto > 0, c_tm / np.maximum(auto, 1), np.nan)
        since += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(hist, axis=1)
        down = (c_tf == 0) & (since >= min_days) & np.isfinite(med) & (med > threshold)
        new_idx = np.clip(idx + c_tf - down.astype(int), 0, L - 1)
        hist[down] = np.nan            # the next decision only looks at days after this step
        since[down] = 0
        n_changes += float(np.mean(new_idx != idx))
        idx = new_idx
    return np.array(levels)[idx], n_changes / (days / 30.4)


def _backtest(pairs, rates, betas):
    """Switchers: predict their post-switch KPIs from their level-8 history; compare to what really happened."""
    if pairs is None or len(pairs) == 0:
        return pd.DataFrame()
    rows = []
    rr = rates.set_index("deviceid")
    for (scale, k), g in pairs.groupby(["scale_type", "sensitivity"]):
        B = betas[(betas.scale_type == scale) & (betas.sensitivity == k)]
        g = g[g.deviceid.isin(rr.index)]
        if B.empty or len(g) < 10:
            continue
        pred = _predict(rr.loc[g.deviceid], B.iloc[0])
        obs = g.set_index("deviceid").frac_too_many
        err = (pred.frac - obs).dropna()
        rows.append(dict(scale_type=scale, sensitivity=k, devices=len(err), mae_frac=err.abs().mean(),
                         bias_frac=err.mean(), corr=np.corrcoef(pred.frac.loc[err.index], obs.loc[err.index])[0, 1]
                         if len(err) > 2 else np.nan))
    return pd.DataFrame(rows)


def _policy_chart(summ, cfg, r):
    """Four views of the same policies: Bossard KPI, KPI 2.0, how many real changes are missed, and the load."""
    import matplotlib.pyplot as plt
    pols = [p for p in POLICIES if p in set(summ.policy)]
    panels = [("share_in_target", "In target - Bossard KPI", "% of scales", 100, "{:.0f}%"),
              ("share_in_target_kpi2", f"In target - KPI 2.0 (<= {cfg.kpi2.tm_per_week:g} empty/week, "
                                       f"<= {cfg.kpi2.miss_rate:.0%} missed)", "% of scales", 100, "{:.0f}%"),
              ("median_miss_rate", "Real stock changes missed (median scale)", "% of changes", 100, "{:.0f}%"),
              ("meas_change_pct", "Measurements per week vs today", "change", 1, "{:+.0f}%")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharey=True)
    y = np.arange(len(pols))
    scales = sorted(summ.scale_type.unique())
    h = 0.38
    for ax, (col, title, xl, mult, fmt) in zip(axes.flat, panels):
        if col not in summ:
            ax.set_visible(False)
            continue
        vmax = 0
        for i, scale in enumerate(scales):
            g = summ[summ.scale_type == scale].set_index("policy").reindex(pols)
            c = viz.SCALE_COLOR.get(scale, viz.CAT[2])
            off = (i - (len(scales) - 1) / 2) * h
            v = g[col].values * mult
            ax.barh(y + off, v, height=h * 0.9, color=c, label=scale)
            for yi, vi in zip(y + off, v):
                if np.isfinite(vi):
                    ax.text(vi + (1 if vi >= 0 else -1), yi, fmt.format(vi), va="center",
                            ha="left" if vi >= 0 else "right", fontsize=7.5, color=viz.INK2)
            vmax = max(vmax, np.nanmax(np.abs(v)) if np.isfinite(v).any() else 0)
        if col == "meas_change_pct":
            ax.axvline(0, color=viz.AXIS, lw=1)
            ax.set_xlim(-max(10, vmax * 1.3), max(10, vmax * 0.35))
        else:
            ax.set_xlim(0, max(10, vmax * 1.25))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(y, pols)
    axes[0][0].invert_yaxis()
    axes[0][0].legend(loc="lower right")
    fig.suptitle("Same scales, seven policies: Bossard's KPI rewards settings that miss half of the real changes",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return viz.save(fig, r.fig_path(cfg.paths.figures, "policy_comparison"))


def _frontier_chart(fr, summ, scale, cfg, r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ymax = max(cfg.kpi.tf_target * 1.8, float(fr.median_tfw.max()) * 1.1)
    ax.add_patch(plt.Rectangle((0, 0), cfg.kpi.tm_target, cfg.kpi.tf_target, color=viz.GOOD, alpha=0.08, lw=0))
    c = viz.SCALE_COLOR.get(scale, viz.CAT[0])
    fr = fr.sort_values("sensitivity")
    ax.plot(fr.median_frac, fr.median_tfw, "-o", color=c, mec=viz.SURFACE, mew=1.5,
            label="everyone on one level (numbers = level)")
    for _, row in fr.iterrows():
        ax.annotate(f"{int(row.sensitivity)}", (row.median_frac, row.median_tfw), xytext=(6, 3),
                    textcoords="offset points", fontsize=8, color=viz.INK2)
    marks = {"P2 segment table": ("s", viz.CAT[2]), "P3 adaptive rule": ("D", viz.CAT[6]),
             "P4 oracle": ("*", viz.INK)}
    for pol, (m, col) in marks.items():
        g = summ[summ.policy == pol]
        if len(g):
            ax.scatter(g.median_frac, g.median_tfw, marker=m, s=90 if m != "*" else 160, color=col,
                       edgecolor=viz.SURFACE, linewidth=1.5, zorder=4, label=pol)
    viz.target_line(ax, cfg.kpi.tm_target, "x")
    viz.target_line(ax, cfg.kpi.tf_target, "y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Median fraction TOO_MANY (expected)")
    ax.set_ylabel("Median TOO_FEW per week (expected)")
    ax.set_title(f"{scale}: what is achievable - and which policy gets closest to the target")
    viz.subtitle(ax, "causal what-if on the same scales · green box = Bossard target zone")
    ax.legend(fontsize=8, loc="upper right")
    return viz.save(fig, r.fig_path(cfg.paths.figures, "frontier", scale))
