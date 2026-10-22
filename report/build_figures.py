"""
Figures and aggregate tables for report/report.qmd.

Run from the repository root after the pipeline has run on the full data:

    python report/build_figures.py --reports reports/bossard --db data/interim/smartbin.duckdb

Step 1 (extract) reads the pipeline outputs and the DuckDB database and writes ONLY aggregated
numbers to report/data/ (no row-level data). Step 2 (draw) builds report/figures/*.png from
report/data/ alone, so the figures can be redrawn without the confidential raw data:

    python report/build_figures.py --figures-only
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figures"

# ---------------------------------------------------------------------------------------------
# style: one quiet look for every figure (white page, hairline grid, two scale-type colours)
# ---------------------------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C = {"6kg": "#2a78d6", "20kg": "#eb6834"}          # scale type = identity, same in every figure
GRAY, GRAY_D = "#b9b8b0", "#7d7c76"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"  # status colours, only for detection zones
CLASS = {                                          # diverging around the 0.5 target
    "1) <= 0.25": "#1c5cab", "2) 0.25-0.50": "#86b6ef", "3) 0.50-0.75": "#f0a19b",
    "4) > 0.75": "#c23431", "5) no automatic weighing": "#dddcd6"}
SCALES = ["20kg", "6kg"]
LEVELS = [1, 2, 4, 5, 6, 7, 8]
W = 6.3                                            # inches = 16 cm, the text width of the report


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.6, "axes.titlesize": 9.2, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 6, "axes.labelsize": 8.6, "axes.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.fontsize": 8, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 110, "savefig.dpi": 220, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "xtick.major.size": 0, "ytick.major.size": 0,
    })
    return plt


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", pad_inches=0.04)
    import matplotlib.pyplot as plt
    plt.close(fig)


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


# =============================================================================================
# STEP 1 - extract aggregates
# =============================================================================================
COPY = {  # pipeline table -> report/data name (all are aggregates)
    "cleaning_funnel": "cleaning_funnel", "m1_baseline__activity_quintiles": "usage_quintiles",
    "m1_baseline__kpis": "baseline_kpis", "m6_crosstalk__correlogram": "coupling_correlogram",
    "m6_crosstalk__root_causes_baseline": "coupling_root_causes",
    "m6_crosstalk__attributable_by_level": "coupling_by_level",
    "m6_crosstalk__location_explained": "coupling_location_explained",
    "m5_pick_size__detection_zones": "zones", "m5_pick_size__zone_summary": "zone_summary",
    "m5_pick_size__min_sensitivity_by_pick_size": "min_sensitivity_by_pick", "m5_pick_size__label_audit": "label_audit",
    "m4_location__locations": "locations", "m4_location__tests": "location_tests",
    "m2_sensitivity__effects": "knob_effects", "m2_sensitivity__anomaly_signature": "sens3_signature",
    "m2_sensitivity__anomaly": "sens3_selection", "m2_sensitivity__event_study": "event_study",
    "m7_policy__policy_comparison": "policies", "m7_policy__backtest": "policy_backtest",
    "m3_importance__shapley": "shapley", "m3_importance__weight_check": "weight_check",
}


def tm_class(f: pd.Series) -> pd.Series:
    return pd.Series(np.select([f.isna(), f <= .25, f <= .5, f <= .75],
                               list(CLASS)[4:5] + list(CLASS)[0:3], list(CLASS)[3]), index=f.index)


def extract(reports: Path, processed: Path, db: Path | None):
    DATA.mkdir(parents=True, exist_ok=True)
    T = reports / "tables"
    for src, dst in COPY.items():
        p = T / f"{src}.csv"
        if p.exists():
            shutil.copy(p, DATA / f"{dst}.csv")
    res = json.loads((reports / "results.json").read_text())
    facts = {k: m.get("facts", {}) for k, m in res["modules"].items()}

    # one row = one scale on one sensitivity level (pipeline's per-scale table) -> medians and shares
    f = pd.read_csv(reports / "per_scale" / "fact_device_sensitivity.csv")
    f = f[f.analysis_group == "main"].copy()
    f["real_picks_per_week"] = f.n_auto_ok / f.weeks_observed
    f["auto_per_week"] = f.n_auto / f.weeks_observed
    f["cls"] = tm_class(f.too_many_fraction)
    g = f.groupby(["scale_type", "sensitivity"])
    k = g.agg(scales=("device_id", "size"), tm_fraction=("too_many_fraction", "median"),
              tf_per_week=("too_few_per_week", "median"), empty_per_week=("too_many_per_week", "median"),
              real_picks_per_week=("real_picks_per_week", "median"), auto_per_week=("auto_per_week", "median"),
              share_in_target=("in_target", "mean")).reset_index()
    k["share_tm_le_05"] = g.too_many_fraction.apply(lambda s: (s <= .5).mean()).values
    k["share_tf_gt_2"] = g.too_few_per_week.apply(lambda s: (s > 2).mean()).values
    k["share_no_auto"] = g.n_auto.apply(lambda s: (s == 0).mean()).values
    k.to_csv(DATA / "kpi_by_sensitivity.csv", index=False)
    cls = (f.groupby(["scale_type", "sensitivity"]).cls.value_counts(normalize=True)
           .rename("share").reset_index())
    cls.to_csv(DATA / "classes_by_sensitivity.csv", index=False)

    b = f[f.sensitivity == 8]
    tm_bands = np.r_[np.arange(0, 1.0, .1), 1.0001]
    tf_bands = [0, .25, .5, 1, 2, 4, np.inf]
    rows = []
    for st, x in b.groupby("scale_type"):
        h = pd.cut(x.too_many_fraction, tm_bands, right=False).value_counts(normalize=True, sort=False)
        rows += [dict(scale_type=st, kpi="too_many_fraction", band=f"{iv.left:.1f}-{min(iv.right, 1):.1f}",
                      share=v) for iv, v in h.items()]
        h = pd.cut(x.too_few_per_week, tf_bands, right=False).value_counts(normalize=True, sort=False)
        lab = ["0-0.25", "0.25-0.5", "0.5-1", "1-2", "2-4", "4+"]
        rows += [dict(scale_type=st, kpi="too_few_per_week", band=lab[i], share=v) for i, v in enumerate(h.values)]
    pd.DataFrame(rows).to_csv(DATA / "baseline_distribution.csv", index=False)

    parameter_spread(processed).to_csv(DATA / "parameter_spread.csv", index=False)

    solution_tables(reports, f)

    nums = {"rows_raw": res["meta"]["rows"], "scales_raw": res["meta"]["devices"],
            "rows_main": res["meta"]["rows_main"], "facts": facts}
    if db is not None and Path(db).exists():
        nums.update(from_database(Path(db)))
        nums["levers"] = levers_tested(Path(db), reports)
    elif (DATA / "numbers.json").exists():        # no database at hand: keep what an earlier run read from it
        old = json.loads((DATA / "numbers.json").read_text())
        nums.update({k: v for k, v in old.items() if k not in nums})
    (DATA / "numbers.json").write_text(json.dumps(nums, indent=1, default=float))


def solution_tables(reports: Path, fact: pd.DataFrame):
    """Aggregates behind the recommendation: trigger loops (FLAG), where the rules leave the scales (TUNE)
    and how missed changes are spread over the scales (WATCH). No row-level data leaves the pipeline."""
    T = reports / "tables"
    lf_path = T / "m2_sensitivity__loop_flags.csv"
    if lf_path.exists():
        lf = pd.read_csv(lf_path)
        lf["group"] = np.where(lf.sensitivity == 3, "level 3", "other levels")
        tot_auto, tot_scales = lf.n_auto.sum(), lf.deviceid.nunique()
        rows = []
        for (grp, loop), g in lf.groupby(["group", "loop"]):
            rows.append(dict(group=grp, loop=bool(loop), scale_levels=len(g), scales=g.deviceid.nunique(),
                             n_auto=int(g.n_auto.sum()), share_of_automatic=g.n_auto.sum() / tot_auto,
                             median_auto_per_day=g.auto_per_day.median(), median_share_retrigger=g.share_retrigger.median(),
                             sites=g.location_label.nunique()))
        out = pd.DataFrame(rows)
        out["scales_with_automatic_weighings"] = tot_scales
        out["loop_scales_any_level"] = lf[lf.loop].deviceid.nunique()
        out.to_csv(DATA / "loops_summary.csv", index=False)
        by_type = (lf[lf.loop].groupby(["group", "scale_type"])
                   .agg(scales=("deviceid", "nunique"), n_auto=("n_auto", "sum"), median_auto_per_day=("auto_per_day", "median"))
                   .reset_index())
        by_type.to_csv(DATA / "loops_by_scale_type.csv", index=False)
        (lf[lf.loop].groupby(["scale_type", "sensitivity"])
         .agg(scale_levels=("deviceid", "size"), n_auto=("n_auto", "sum"), sites=("location_label", "nunique"))
         .reset_index().to_csv(DATA / "loops_by_level.csv", index=False))
    loc_path = T / "m6_crosstalk__by_location.csv"
    if loc_path.exists():
        lc = pd.read_csv(loc_path)
        lc = lc[lc.cls == "TOO_MANY"].copy()
        lc["coupled"] = lc.n * lc.attributable.clip(lower=0)
        rows = []
        for st, g in lc.groupby("scale_type"):
            g = g.sort_values("coupled", ascending=False)
            rows.append(dict(scale_type=st, sites=len(g), share_coupled_top10_sites=g.coupled.head(10).sum() / g.coupled.sum(),
                             attributable_p10=g.attributable.quantile(.1), attributable_median=g.attributable.median(),
                             attributable_p90=g.attributable.quantile(.9)))
        pd.DataFrame(rows).to_csv(DATA / "coupling_top_sites.csv", index=False)
    pp_path = T / "m5_pick_size__device_pick_profile.csv"
    if pp_path.exists():
        pp = pd.read_csv(pp_path)
        pp = pp[pp.picks >= 5]
        (pp.groupby("scale_type").agg(scales=("deviceid", "size"),
                                      share_median_pick_below_25g=("median_pick_g", lambda v: (v < 25).mean()),
                                      share_median_pick_below_10g=("median_pick_g", lambda v: (v < 10).mean()),
                                      median_of_median_pick_g=("median_pick_g", "median"))
         .reset_index().to_csv(DATA / "pick_profile_summary.csv", index=False))
    ch_path = T / "m7_policy__device_choices.csv"
    if ch_path.exists():
        ch = pd.read_csv(ch_path)
        keep = ["P6 adaptive rule (KPI 2.0)", "P7 start 7 + rule (KPI 2.0)", "P8 fraction rule (start 6)"]
        fl = (ch[ch.policy.isin(keep)].groupby(["policy", "scale_type"]).sensitivity.value_counts(normalize=True)
              .rename("share").reset_index())
        fl.to_csv(DATA / "final_levels.csv", index=False)
    b = fact[fact.sensitivity == 8]
    rows = []
    for st, g in b.groupby("scale_type"):
        tf = g.n_too_few.fillna(0).sort_values(ascending=False)
        rows.append(dict(scale_type=st, scales=len(g), share_scales_without_too_few=(tf == 0).mean(),
                         share_of_too_few_from_top10pct=tf.head(int(len(tf) * 0.1)).sum() / tf.sum(),
                         manual_checks_per_scale_week=(g.n_manual / g.weeks_observed).median(),
                         share_manual_checks_without_change=g.n_manual_ok.sum() / g.n_manual.sum()))
    pd.DataFrame(rows).to_csv(DATA / "watch_numbers.csv", index=False)


def parameter_spread(processed: Path) -> pd.DataFrame:
    """Option A of the decision log: how far apart are the best and the worst group of each parameter?
    Non-sensitivity parameters are compared at the default level 8 only, so the setting cannot mix in."""
    ds = pd.read_parquet(processed / "device_settings.parquet")
    sec = pd.read_parquet(processed / "sections.parquet")
    ds = ds[ds.main_analysis & (ds.n_auto >= 5)]
    sec = sec[sec.main_analysis & (sec.n_auto >= 5)]
    out = []

    def add(st, param, med, method, fmt=str):
        med = med.dropna()
        if len(med) < 2:
            return
        if method == "p10-p90":
            lo, hi = med.quantile(.1), med.quantile(.9)
            best, worst = "10th percentile site", "90th percentile site"
        else:
            lo, hi = med.min(), med.max()
            best, worst = fmt(med.idxmin()), fmt(med.idxmax())
        out.append(dict(scale_type=st, parameter=param, groups=len(med), best_group=best, best=lo,
                        worst_group=worst, worst=hi, spread=hi - lo, method=method))

    for st in SCALES:
        d = ds[ds.scale_type == st]
        s = d[d.sensitivity != 3].groupby("sensitivity").frac_too_many.agg(["median", "size"])
        add(st, "Sensitivity level", s[s["size"] >= 30]["median"], "max-min", lambda v: f"level {int(v)}")
        d8 = d[d.sensitivity == 8]
        s = d8.groupby("location_label").frac_too_many.agg(["median", "size"])
        add(st, "Site (location)", s[s["size"] >= 20]["median"], "p10-p90")
        s = d8.groupby("box_name").frac_too_many.agg(["median", "size"])
        add(st, "Box type", s[s["size"] >= 30]["median"], "max-min")
        q = pd.qcut(d8.item_weight.rank(method="first"), 4, labels=["Q1 lightest", "Q2", "Q3", "Q4 heaviest"])
        add(st, "Item weight (quartile)", d8.groupby(q, observed=True).frac_too_many.median(), "max-min")
        s8 = sec[(sec.scale_type == st) & (sec.sensitivity == 8)]
        s = s8.groupby("weight_bin").frac_too_many.agg(["median", "size"])
        add(st, "Fill level (weight bin)", s[s["size"] >= 30]["median"], "max-min", lambda v: f"{v} g")
    return pd.DataFrame(out)


def from_database(db: Path) -> dict:
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    q = lambda sql: con.execute(sql).df()
    out = {}
    p = q("SELECT min(weighttime) AS a, max(weighttime) AS b FROM events")
    out["period"] = [str(p.a[0])[:10], str(p.b[0])[:10]]
    lab = q("""SELECT initiatortype, coalesce(weightclassification, 'missing') AS cls, count(*) AS n
               FROM events GROUP BY ALL ORDER BY ALL""")
    lab.to_csv(DATA / "label_counts.csv", index=False)
    # weight change behind each label: |grams| since the previous weighing (main analysis rows)
    w = q("""SELECT scale_type,
                    CASE WHEN is_auto AND cls = 'OK' THEN 'automatic OK'
                         WHEN is_auto AND cls = 'TOO_MANY' THEN 'TOO_MANY'
                         WHEN NOT is_auto AND cls = 'TOO_FEW' THEN 'TOO_FEW' ELSE 'manual OK' END AS label,
                    CAST(least(round(abs(dw)), 100000) AS INTEGER) AS grams, count(*) AS n
             FROM events WHERE exclusion_reason IS NULL AND dw IS NOT NULL AND scale_type IS NOT NULL
             GROUP BY ALL""")
    edges = [0, 1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 130, 200, 500, np.inf]
    names = ["0", "1", "2", "3-4", "5-7", "8-12", "13-19", "20-29", "30-49", "50-79", "80-129",
             "130-199", "200-499", "500+"]
    w["bin"] = pd.cut(w.grams, edges, right=False, labels=names)
    h = w.groupby(["scale_type", "label", "bin"], observed=False).n.sum().reset_index()
    h["share"] = h.n / h.groupby(["scale_type", "label"]).n.transform("sum")
    h.to_csv(DATA / "weight_change_hist.csv", index=False)
    # when do manual weighings happen? (24 h no-activity checks vs. person-triggered)
    t = q("""SELECT hour, gap_s >= 86400 AS after_24h_idle, count(*) AS n
             FROM events WHERE exclusion_reason IS NULL AND NOT is_auto AND gap_s IS NOT NULL
             GROUP BY ALL ORDER BY ALL""")
    t.to_csv(DATA / "manual_check_hours.csv", index=False)
    g = q("""SELECT quantile_cont(gap_s / 3600, [0.05, 0.25, 0.5, 0.75, 0.95]) AS q,
                    avg(CASE WHEN gap_s >= 86400 THEN 1.0 ELSE 0 END) AS share_idle
             FROM events WHERE exclusion_reason IS NULL AND NOT is_auto AND gap_s >= 86400""")
    out["idle_check_gap_hours_q05_q25_q50_q75_q95"] = [round(v, 1) for v in g.q[0]]
    s = q("""SELECT avg(CASE WHEN gap_s >= 86400 THEN 1.0 ELSE 0 END) AS a,
                    avg(CASE WHEN cls = 'TOO_FEW' AND gap_s >= 86400 THEN 1.0
                             WHEN cls = 'TOO_FEW' THEN 0.0 END) AS b
             FROM events WHERE exclusion_reason IS NULL AND NOT is_auto""")
    out["share_manual_after_24h_idle"], out["share_too_few_after_24h_idle"] = float(s.a[0]), float(s.b[0])
    # the natural experiment: which levels are active, week by week
    wk = q("""SELECT week, scale_type,
                     CASE WHEN sensitivity IS NULL THEN 'unknown' WHEN sensitivity = 8 THEN 'level 8 (default)'
                          WHEN sensitivity = 3 THEN 'level 3 (anomaly)' ELSE 'levels 1-7 (switched)' END AS grp,
                     count(DISTINCT deviceid) AS scales
              FROM events WHERE scale_type IS NOT NULL GROUP BY ALL ORDER BY ALL""")
    wk.to_csv(DATA / "weekly_levels.csv", index=False)
    return out


def levers_tested(db: Path, reports: Path) -> dict:
    """Levers tested next to the sensitivity rule (aggregates only, scales caught in a trigger loop left out):
    - hourly_profile.csv    when empty wake-ups and real picks happen (hour of day, weekend)
    - quiet_hours_test.csv  'quiet hours' learned per site on March-April, tested on May-June
    - burst_merge.csv       one trailing weighing per W-minute window instead of one weighing per wake-up
    - minwait_check.csv     KPIs by minimum wait time / delay (shows why their effect cannot be measured)"""
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    lf = pd.read_csv(reports / "tables" / "m2_sensitivity__loop_flags.csv")
    con.register("loopdev", pd.DataFrame({"deviceid": sorted(set(lf.loc[lf.loop, "deviceid"]))}))
    where = """in_kpi_analysis AND is_auto AND cls IN ('OK', 'TOO_MANY') AND scale_type IN ('6kg', '20kg')
               AND deviceid NOT IN (SELECT deviceid FROM loopdev)"""
    out = {}

    # 1) when do empty wake-ups happen compared with real picks?
    h = con.execute(f"""SELECT scale_type, cls, hour(weighttime) AS hour, isodow(weighttime) >= 6 AS weekend,
                               count(*) AS n FROM events WHERE {where} GROUP BY ALL ORDER BY ALL""").df()
    h.to_csv(DATA / "hourly_profile.csv", index=False)

    # 2) quiet hours: per site, the hours of the week that held at most x% of its real picks in March-April
    d = con.execute(f"""SELECT location_label AS site, scale_type, cls,
                               (isodow(weighttime) - 1) * 24 + hour(weighttime) AS how,
                               weighttime < TIMESTAMP '2022-05-01' AS train, count(*) AS n
                        FROM events WHERE {where} AND location_label IS NOT NULL GROUP BY ALL""").df()
    rows = []
    for thr in (0.01, 0.02, 0.05):
        for (site, st), g in d.groupby(["site", "scale_type"]):
            tr = g[g.train & (g.cls == "OK")].groupby("how").n.sum().reindex(range(168), fill_value=0)
            if tr.sum() < 200:
                continue
            cum = tr.sort_values(kind="stable").cumsum() / tr.sum()
            quiet = set(cum[cum <= thr].index)
            te = g[~g.train]
            ok, tm = te[te.cls == "OK"], te[te.cls == "TOO_MANY"]
            if ok.n.sum() == 0 or tm.n.sum() == 0:
                continue
            rows.append(dict(threshold=thr, scale_type=st, quiet_hours=len(quiet),
                             picks_in_quiet=ok[ok.how.isin(quiet)].n.sum(), picks=ok.n.sum(),
                             empty_in_quiet=tm[tm.how.isin(quiet)].n.sum(), empty=tm.n.sum()))
    r = pd.DataFrame(rows)
    qh = (r.groupby(["threshold", "scale_type"])
          .agg(sites=("quiet_hours", "size"), quiet_hours_per_week=("quiet_hours", "median"),
               picks_in_quiet=("picks_in_quiet", "sum"), picks=("picks", "sum"),
               empty_in_quiet=("empty_in_quiet", "sum"), empty=("empty", "sum")).reset_index())
    qh["share_picks_in_quiet_hours"] = qh["picks_in_quiet"] / qh["picks"]
    qh["share_empty_avoided"] = qh["empty_in_quiet"] / qh["empty"]
    qh.to_csv(DATA / "quiet_hours_test.csv", index=False)

    # 3) burst merge: after a weighing, wake-ups within W minutes are collected and weighed once at the end
    #    of the window (a change is never lost, only reported later). Optionally on a thinned sequence that
    #    mimics level 7 (causal rate ratios 8 -> 7 for empty wake-ups and caught picks).
    s = con.execute(f"""SELECT deviceid, scale_type, epoch(weighttime) AS t, cls = 'OK' AS ok
                        FROM events WHERE {where} ORDER BY deviceid, t""").df()
    dev, T, OK, ST = s.deviceid.values, s.t.values.astype(float), s.ok.values.astype(bool), s.scale_type.values
    cuts = np.flatnonzero(np.r_[True, dev[1:] != dev[:-1], True])
    ke = pd.read_csv(DATA / "knob_effects.csv")
    k7 = ke[ke.sensitivity == 7].set_index("scale_type")
    rng = np.random.default_rng(7)

    def merge(W, level7=False):
        acc = {st: dict(n=0, n_ok=0, perf=0, perf_ok=0, delays=[]) for st in SCALES}
        for a, b in zip(cuts[:-1], cuts[1:]):
            st, t, ok = ST[a], T[a:b], OK[a:b]
            if level7:
                keep = rng.random(b - a) < np.where(ok, np.exp(k7.loc[st, "beta_aok"]), np.exp(k7.loc[st, "beta_tm"]))
                t, ok = t[keep], ok[keep]
            o, n = acc[st], len(t)
            o["n"] += n
            o["n_ok"] += int(ok.sum())
            i = 0
            while i < n:
                o["perf"] += 1
                o["perf_ok"] += int(ok[i])
                end, j = t[i] + W, i + 1
                while j < n and t[j] <= end:          # wake-ups inside the window -> one trailing weighing
                    k = j
                    while k < n and t[k] <= end:
                        k += 1
                    o["perf"] += 1
                    o["perf_ok"] += int(ok[j:k].any())
                    o["delays"].extend((end - t[j:k][ok[j:k]]).tolist())
                    j, end = k, end + W
                i = j
        res = []
        for st, o in acc.items():
            dl = np.array(o["delays"]) / 60 if o["delays"] else np.zeros(1)
            res.append(dict(window_min=W / 60, scale_type=st, sequence="level 7 (simulated)" if level7 else "today",
                            weighings_before=o["n"], weighings_after=o["perf"], saved=1 - o["perf"] / o["n"],
                            empty_saved=1 - (o["perf"] - o["perf_ok"]) / (o["n"] - o["n_ok"]),
                            picks_reported_later=len(o["delays"]) / max(o["n_ok"], 1),
                            delay_median_min=float(np.median(dl)), delay_p95_min=float(np.percentile(dl, 95))))
        return res

    bm = []
    for W in (300, 600, 900):
        bm += merge(W)
    bm += merge(600, level7=True)
    pd.DataFrame(bm).to_csv(DATA / "burst_merge.csv", index=False)
    gaps = con.execute(f"""SELECT scale_type, cls, count(*) AS n, avg((gap_s <= 600)::INT) AS share_within_10min,
                                  median(gap_s) / 60 AS median_gap_min
                           FROM events WHERE {where} GROUP BY ALL ORDER BY ALL""").df()
    out["gap_to_previous_weighing"] = gaps.to_dict(orient="records")

    # 4) minimum wait time / delay: KPIs per setting and whether any scale ran on two wait times at one level
    s_sql = """
        WITH s AS (
          SELECT deviceid, scale_type, location_label, coalesce(minwaittime, 1.0) AS minwait,
                 coalesce(delay, 30.0) AS delay, sensitivity,
                 count(*) FILTER (WHERE is_auto AND cls = 'TOO_MANY') AS tm, count(*) FILTER (WHERE is_auto AND cls = 'OK') AS aok,
                 count(*) FILTER (WHERE cls = 'TOO_FEW') AS tf,
                 (epoch(max(weighttime)) - epoch(min(weighttime))) / 604800.0 AS wks
          FROM events WHERE scale_type IN ('6kg', '20kg') AND sensitivity IS NOT NULL AND sensitivity <> 3
          GROUP BY ALL HAVING wks >= 1 AND count(*) >= 20)"""
    sl = con.execute(s_sql + " SELECT * FROM s").df()      # one row per scale and setting, kept in memory only
    sl["empty_per_week"], sl["caught_per_week"] = sl.tm / sl.wks, sl.aok / sl.wks
    sl["missed_share"] = sl.tf / (sl.tf + sl.aok).replace(0, np.nan)
    rows = []
    for (st, m, dl), g in sl.groupby(["scale_type", "minwait", "delay"]):
        if len(g) < 8:
            continue
        per_site = g.groupby("location_label").deviceid.nunique().sort_values(ascending=False)
        rows.append(dict(scale_type=st, minwait=m, delay=dl, scale_levels=len(g), scales=g.deviceid.nunique(),
                         sites=len(per_site), share_scales_top2_sites=per_site.head(2).sum() / g.deviceid.nunique(),
                         share_on_level_1=(g.sensitivity == 1).mean(), empty_per_week=g.empty_per_week.median(),
                         caught_per_week=g.caught_per_week.median(), missed_share=g.missed_share.median()))
    pd.DataFrame(rows).sort_values(["scale_type", "scale_levels"], ascending=[True, False]).to_csv(
        DATA / "minwait_check.csv", index=False)
    both = con.execute(s_sql + """
        SELECT count(*) FROM (SELECT deviceid, sensitivity FROM s WHERE minwait IN (1, 60)
                              GROUP BY ALL HAVING count(DISTINCT minwait) = 2)""").fetchone()[0]
    out["scale_levels_with_both_1_and_60_min_wait"] = int(both)   # same scale, same level, both wait times
    return out


# =============================================================================================
# STEP 2 - draw
# =============================================================================================
def rd(name):
    return pd.read_csv(DATA / f"{name}.csv")


def legend_scales(ax, loc="upper right", **kw):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C[s], lw=6, solid_capstyle="butt") for s in SCALES]
    ax.legend(h, ["20 kg scales", "6 kg scales"], loc=loc, **kw)


def fig_design(plt):
    w = rd("weekly_levels")
    w["week"] = pd.to_datetime(w.week)
    order = ["level 8 (default)", "levels 1-7 (switched)", "level 3 (anomaly)", "unknown"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), sharey=True)
    for ax, st in zip(axes, SCALES):
        p = w[w.scale_type == st].pivot_table(index="week", columns="grp", values="scales", fill_value=0)
        p = p.reindex(columns=order, fill_value=0)
        p = p[p.sum(axis=1) >= 0.2 * p.sum(axis=1).max()]          # drop the two partial edge weeks
        share = p.div(p.sum(axis=1), axis=0)
        cols = {"level 8 (default)": GRAY, "levels 1-7 (switched)": C[st], "level 3 (anomaly)": "#4a3aa7",
                "unknown": "#ecebe6"}
        bottom = np.zeros(len(share))
        x = np.arange(len(share))
        for grp in order:
            ax.bar(x, share[grp].values, bottom=bottom, width=0.86, color=cols[grp], label=grp, linewidth=0)
            bottom += share[grp].values
        ticks = x[::4]
        ax.set_xticks(ticks, [d.strftime("%d %b") for d in share.index[::4]])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
        ax.set_title(f"{st.replace('kg', ' kg')} scales: active level per week")
        ax.grid(False)
    h, l = axes[0].get_legend_handles_labels()
    axes[1].legend(h[:4], ["level 8 (factory default)", "levels 1, 2, 4-7 (switched)", "level 3", "level unknown"],
                   loc="upper center", bbox_to_anchor=(-0.08, -0.13), ncol=4)
    axes[0].set_ylabel("share of active scales")
    save(fig, "fig00_design")


def fig_cleaning(plt):
    f = rd("cleaning_funnel")
    f = f[(f.step > 0) & (f.rows_removed > 0)].copy()
    names = {"A2_duplicate": "Duplicate copy (same scale, same ms)", "A5_weight_invalid": "Impossible weight",
             "B0_duplicate_label_conflict": "Duplicate with conflicting label", "B1_label_missing": "Label missing",
             "B3_sensitivity_missing": "Sensitivity missing", "B4_trigger_config_nonstandard":
             "Non-standard min-wait / delay", "B5_device_metadata_missing": "Scale type / site unknown",
             "B6_section_not_relevant": "Period too short (< 20 weighings or < 5 days)",
             "B7_sensitivity_anomaly": "Sensitivity 3 (studied separately)", "B8_sensitivity_low_support":
             "Level used by < 30 scales"}
    f["name"] = f.rule.map(names)
    f["stage"] = np.where(f.step < 10, "deleted", "flagged, kept in the event chain")
    f = f.iloc[::-1]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    cols = np.where(f.stage == "deleted", INK2, GRAY)
    ax.barh(f.name, f.rows_removed, color=cols, height=0.62)
    for y, v in enumerate(f.rows_removed):
        ax.text(v * 1.15, y, f"{v:,}", va="center", fontsize=7.2, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e7)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("rows (log scale)")
    from matplotlib.patches import Patch
    ax.legend([Patch(color=INK2), Patch(color=GRAY)], ["deleted (Stage A)", "flagged, kept for weight changes (Stage B)"],
              loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    ax.set_title("Rows removed or flagged per rule")
    save(fig, "fig01_cleaning")


def fig_one_sided(plt):
    d = rd("baseline_distribution")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"width_ratios": [1.45, 1]})
    for ax, kpi, title, target in [(axes[0], "too_many_fraction", "TOO_MANY fraction per scale", 4.5),
                                   (axes[1], "too_few_per_week", "TOO_FEW per week per scale", 3.5)]:
        x0 = d[(d.kpi == kpi) & (d.scale_type == "20kg")]
        x = np.arange(len(x0))
        for i, st in enumerate(SCALES):
            v = d[(d.kpi == kpi) & (d.scale_type == st)].share.values
            ax.bar(x + (i - .5) * 0.38, v, width=0.36, color=C[st])
        ax.set_xticks(x, x0.band.values, rotation=45, ha="right")
        ax.axvline(target, color=INK, lw=1, ls=(0, (3, 2)))
        ax.text(target + 0.12, ax.get_ylim()[1] * 0.92, "target", fontsize=7.2, color=INK)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
        ax.set_title(title)
    axes[0].set_ylabel("share of scales (level 8)")
    legend_scales(axes[0], loc="upper left")
    save(fig, "fig02_one_sided")


def fig_classes(plt):
    c = rd("classes_by_sensitivity")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    for ax, st in zip(axes, SCALES):
        p = c[c.scale_type == st].pivot_table(index="sensitivity", columns="cls", values="share", fill_value=0)
        p = p.reindex(index=[l for l in LEVELS[::-1] if l in p.index], columns=list(CLASS), fill_value=0)
        left = np.zeros(len(p))
        y = np.arange(len(p))
        for cl in CLASS:
            v = p[cl].values
            ax.barh(y, v, left=left, color=CLASS[cl], height=0.7, edgecolor="white", linewidth=1)
            for yi, (vi, li) in enumerate(zip(v, left)):
                if vi >= 0.09:
                    dark = cl in ("1) <= 0.25", "4) > 0.75")
                    ax.text(li + vi / 2, yi, f"{vi * 100:.0f}", ha="center", va="center", fontsize=6.6,
                            color="white" if dark else INK)
            left += v
        ax.set_yticks(y, [f"level {l}" for l in p.index])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
        ax.grid(False)
        ax.set_title(f"{st.replace('kg', ' kg')} scales")
    from matplotlib.patches import Patch
    labels = ["TOO_MANY ≤ 0.25", "0.25–0.50 (in target)", "0.50–0.75", "> 0.75",
              "no automatic weighing"]
    fig.legend([Patch(color=v) for v in CLASS.values()], labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.52, -0.06))
    save(fig, "fig03_classes")


def fig_fraction_vs_counts(plt):
    k = rd("kpi_by_sensitivity")
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw={"wspace": 0.38})
    spec = [("empty_per_week", "Empty wake-ups / week"), ("real_picks_per_week", "Picks caught / week"),
            ("tm_fraction", "TOO_MANY fraction")]
    for ax, (col, title) in zip(axes, spec):
        for st in SCALES:
            x = k[k.scale_type == st].set_index("sensitivity").reindex(LEVELS)
            ax.plot(range(len(LEVELS)), x[col].values, color=C[st], marker="o", ms=4,
                    markeredgecolor="white", markeredgewidth=1)
            # end values only on the count panels (start values are tiny and sit on the axis);
            # both ends on the fraction panel, start labels below the point so they clear the lines
            for i in ((0, len(LEVELS) - 1) if col == "tm_fraction" else (len(LEVELS) - 1,)):
                v = x[col].values[i]
                last = i == len(LEVELS) - 1
                if last:
                    xy = (6, 2) if st == "20kg" else (6, -4)
                    ha = "left"
                else:
                    xy, ha = (2, -12), "center"
                ax.annotate(f"{v:.2f}" if col == "tm_fraction" else f"{v:.1f}", (i, v),
                            textcoords="offset points", xytext=xy, ha=ha, fontsize=6.8, color=INK2)
        ax.set_xlim(-0.4, len(LEVELS) - 0.2)
        ax.set_xticks(range(len(LEVELS)), LEVELS)
        ax.set_xlabel("sensitivity level")
        ax.set_title(title)
    axes[2].axhline(0.5, color=INK, lw=1, ls=(0, (3, 2)))
    axes[2].text(3.2, 0.42, "target 0.5", fontsize=6.8, color=INK)
    axes[2].set_ylim(0, 1.05)
    axes[0].set_ylabel("median scale")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=C[s], lw=2, marker="o", ms=4) for s in SCALES],
               ["20 kg scales", "6 kg scales"], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
               fontsize=7.6)
    save(fig, "fig04_fraction_vs_counts")


def fig_usage(plt):
    q = rd("usage_quintiles")
    k = rd("kpi_by_sensitivity")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for st in SCALES:
        x = q[q.scale_type == st]
        ax.plot(range(5), x.frac_too_many_median.values, color=C[st], marker="o", ms=4, markeredgecolor="white")
    ax.set_xticks(range(5), ["Q1\nleast used", "Q2", "Q3", "Q4", "Q5\nmost used"])
    ax.axhline(0.5, color=INK, lw=1, ls=(0, (3, 2)))
    ax.text(0, 0.52, "target 0.5", fontsize=6.8, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("TOO_MANY fraction (median)")
    ax.set_title("Same setting (8), different usage")
    legend_scales(ax, loc="lower left")
    ax = axes[1]
    b = k[k.sensitivity == 8].set_index("scale_type")
    metrics = [("real_picks_per_week", "real picks\nper week"), ("empty_per_week", "empty wake-ups\nper week")]
    for i, st in enumerate(SCALES):
        vals = [b.loc[st, m] for m, _ in metrics]
        xs = np.arange(2) + (i - .5) * 0.36
        ax.bar(xs, vals, width=0.34, color=C[st])
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.25, f"{v:.1f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(2), [m[1] for m in metrics])
    ax.set_title("Typical scale at level 8")
    save(fig, "fig05_usage")


def fig_coupling(plt):
    c = rd("coupling_correlogram")
    c = c[c.lag_s.abs() < 118]                                    # the outermost bins are cut by the window
    r = rd("coupling_root_causes").set_index("scale_type")
    fig = plt.figure(figsize=(W, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.82], wspace=0.34)
    axes = [fig.add_subplot(gs[0, 0]), None, fig.add_subplot(gs[0, 2])]
    axes[1] = fig.add_subplot(gs[0, 1], sharey=axes[0])
    near = 14
    for ax, st in zip(axes[:2], SCALES):
        ax.axvspan(-near, near, color="#efeee9", zorder=0, lw=0)
        ax.axvspan(60, 60 + 2 * near, color="#f6f5f1", zorder=0, lw=0)
        for cls, col, lab in [("OK", GRAY, "real picks (reference)"), ("TOO_MANY", C[st], "empty wake-ups")]:
            x = c[(c.scale_type == st) & (c.cls == cls)].sort_values("lag_s")
            ax.plot(x.lag_s + 1, x.per_min, color=col, lw=1.6, label=lab)
        base = c[(c.scale_type == st) & (c.cls == "TOO_MANY")].baseline.iloc[0]
        ax.axhline(base, color=INK2, lw=0.8)
        ax.text(-114, base + 1.1, "chance", fontsize=6.4, color=INK2)
        top = 13.2
        ax.text(0, top * 0.97, "±14 s", ha="center", va="top", fontsize=6.4, color=INK2)
        ax.text(74, top * 0.97, "control", ha="center", va="top", fontsize=6.4, color=INK2)
        ax.set_xlim(-118, 118)
        ax.set_ylim(0, top)
        ax.set_xticks([-120, -60, 0, 60, 120])
        ax.set_title(f"{st.replace('kg', ' kg')} scales")
    axes[0].set_ylabel("picks at other scales\nof the site, per minute")
    plt.setp(axes[1].get_yticklabels(), visible=False)
    fig.text(0.36, -0.04, "seconds before (−) / after (+) the wake-up of the scale", ha="center", fontsize=8,
             color=INK2)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=GRAY, lw=2), Line2D([], [], color=C["20kg"], lw=2),
                Line2D([], [], color=C["6kg"], lw=2)],
               ["around real picks (reference)", "around empty wake-ups, 20 kg", "around empty wake-ups, 6 kg"],
               loc="upper center", bbox_to_anchor=(0.36, -0.1), ncol=3, fontsize=6.8)
    ax = axes[2]
    for i, st in enumerate(SCALES):
        row = r.loc[st]
        xs = np.array([0, 1]) + (i - .5) * 0.38
        ax.bar(xs[0], row.p_near, width=0.36, color=C[st])
        ax.bar(xs[1], row.p_chance, width=0.36, color="white", edgecolor=C[st], linewidth=1.2, hatch="////")
        for xx, v in zip(xs, [row.p_near, row.p_chance]):
            ax.text(xx, v + 0.015, pct(v), ha="center", fontsize=6.8, color=INK2)
    ax.set_xticks([0, 1], ["±14 s", "control"])
    ax.set_ylim(0, 0.75)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Empty wake-ups with a pick nearby")
    legend_scales(ax, loc="upper right", fontsize=6.4)
    save(fig, "fig06_coupling")


def fig_blind_zone(plt):
    h = rd("weight_change_hist")
    floor = {"20kg": 23, "6kg": 5}
    order = ["0", "1", "2", "3-4", "5-7", "8-12", "13-19", "20-29", "30-49", "50-79", "80-129", "130-199",
             "200-499", "500+"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.45), sharey=True)
    for ax, st in zip(axes, SCALES):
        x = np.arange(len(order))
        for j, (lab, col) in enumerate([("TOO_MANY", C[st]), ("automatic OK", GRAY)]):
            v = (h[(h.scale_type == st) & (h.label == lab)].set_index("bin").share.reindex(order).fillna(0))
            ax.bar(x + (j - .5) * 0.4, v.values, width=0.38, color=col)
        pos = {"20kg": 6.75, "6kg": 3.5}[st]                         # between bins 13-19|20-29 and 3-4|5-7
        ax.axvline(pos, color=INK, lw=1, ls=(0, (3, 2)))
        ax.text(pos + 0.2, 0.7, f"threshold\n≈ {floor[st]} g", fontsize=7.4, color=INK, va="top")
        ax.set_xticks(x, order, rotation=60, ha="right")
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
        ax.set_title(f"{st.replace('kg', ' kg')} scales")
        ax.set_ylim(0, 0.8)
    axes[0].set_ylabel("share of weighings of that label")
    fig.text(0.5, -0.16, "weight change since the previous weighing (g)", ha="center", fontsize=8.6, color=INK2)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=C["20kg"]), Patch(color=C["6kg"]), Patch(color=GRAY)],
               ["empty wake-up (TOO_MANY), 20 kg", "empty wake-up (TOO_MANY), 6 kg", "real pick caught (automatic OK)"],
               loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7.6)
    save(fig, "fig07_blind_zone")


def fig_zones(plt):
    from matplotlib.patches import Rectangle, Patch
    z = rd("zones")
    col = {"green": GOOD, "amber": WARN, "red": CRIT, "no data": "#ecebe6"}
    bins = ["0-2", "2-5", "5-10", "10-20", "20-50", "50-100", "100-200", "200-500", ">500"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.75), sharey=True)
    for ax, st in zip(axes, ["6kg", "20kg"]):
        levels = [l for l in LEVELS[::-1]]
        for i, k in enumerate(levels):
            for j, b in enumerate(bins):
                r = z[(z.scale_type == st) & (z.sensitivity == k) & (z.pick_bin == b)]
                zone = r.zone.iloc[0] if len(r) else "no data"
                y = len(levels) - 1 - i
                if zone == "below threshold":
                    ax.add_patch(Rectangle((j, y), .94, .88, facecolor="white", edgecolor=AXIS, hatch="////", lw=0))
                    continue
                ax.add_patch(Rectangle((j, y), .94, .88, color=col[zone], lw=0))
                if zone != "no data":
                    ax.text(j + .47, y + .44, f"{int(r.detection.iloc[0] * 100)}", ha="center", va="center",  # floor: label agrees with colour
                            fontsize=6.4, color="white" if zone in ("green", "red") else INK)
        ax.set_xlim(0, len(bins))
        ax.set_ylim(0, len(levels))
        ax.set_xticks([j + .47 for j in range(len(bins))], bins, rotation=50, ha="right")
        ax.set_yticks([len(levels) - 1 - i + .44 for i in range(len(levels))], [f"level {l}" for l in levels])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{st.replace('kg', ' kg')} scales: % of picks detected")
    fig.text(0.5, -0.1, "pick size (grams removed in one go)", ha="center", fontsize=8.6, color=INK2)
    fig.legend([Patch(color=GOOD), Patch(color=WARN), Patch(color=CRIT),
                Patch(facecolor="white", edgecolor=AXIS, hatch="////"), Patch(color="#ecebe6")],
               ["≥ 95% detected", "80–95%", "< 80%", "below the scale's threshold (not measurable)",
                "< 30 picks"], loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.13), fontsize=7.4)
    save(fig, "fig08_zones")


def fig_location(plt):
    loc = rd("locations")
    t = rd("location_tests").set_index("scale_type")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
    for ax, st in zip(axes, SCALES):
        x = loc[(loc.scale_type == st) & (loc.devices >= 10)].sort_values("frac_too_many_median")
        i = np.arange(len(x))
        ax.vlines(i, x.lo, x.hi, color=C[st], alpha=0.35, lw=1)
        ax.scatter(i, x.frac_too_many_median, s=9, color=C[st], zorder=3, edgecolor="white", linewidth=0.4)
        ax.axhline(0.5, color=INK, lw=1, ls=(0, (3, 2)))
        ax.text(len(x) * 0.72, 0.43, "target 0.5", fontsize=6.8, color=INK)
        ax.set_xticks([])
        ax.set_xlabel(f"{len(x)} sites with ≥ 10 scales, sorted")
        r = t.loc[st]
        ax.set_title(f"{st.replace('kg', ' kg')}: ICC {r.icc_frac_too_many:.2f}, split-half r "
                     f"{r.split_half_r_frac_too_many:.2f}")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("TOO_MANY fraction\n(median of the site, level 8)")
    save(fig, "fig09_location")


def fig_parameters(plt):
    p = rd("parameter_spread")
    order = ["Site (location)", "Sensitivity level", "Fill level (weight bin)", "Item weight (quartile)",
             "Box type"]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    y = np.arange(len(order))[::-1]
    for i, st in enumerate(SCALES):
        x = p[p.scale_type == st].set_index("parameter").reindex(order)
        yy = y + (0.5 - i) * 0.36
        ax.barh(yy, x.spread.values, height=0.34, color=C[st])
        for yi, v, lo, hi, n, par in zip(yy, x.spread.values, x.best.values, x.worst.values, x.groups.values,
                                         order):
            if np.isfinite(v):
                unit = {"Site (location)": "sites", "Box type": "box types"}.get(par)
                extra = f", {int(n)} {unit}" if unit else ""
                ax.text(v + 0.005, yi, f"{v:.2f}  ({lo:.2f} → {hi:.2f}{extra})", va="center", fontsize=6.7,
                        color=INK2)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 0.68)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("difference in TOO_MANY fraction between the best and the worst group (median scale)")
    legend_scales(ax, loc="lower right")
    save(fig, "fig10_parameters")


def fig_knob(plt):
    e = rd("knob_effects")
    e = e[e.sensitivity < 8].copy()
    lv = [7, 6, 5, 4, 2, 1]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"wspace": 0.3})
    for i, st in enumerate(SCALES):
        x = e[e.scale_type == st].set_index("sensitivity").reindex(lv)
        xs = np.arange(len(lv)) + (i - .5) * 0.38
        ch = (np.exp(x.beta_tm) - 1) * 100
        lo, hi = (np.exp(x.beta_tm_lo) - 1) * 100, (np.exp(x.beta_tm_hi) - 1) * 100
        axes[0].bar(xs, ch, width=0.36, color=C[st])
        axes[0].vlines(xs, lo, hi, color=INK2, lw=0.8)
        m = x.did_miss_rate * 100
        axes[1].bar(xs, m, width=0.36, color=C[st])
        axes[1].vlines(xs, x.did_miss_rate_lo * 100, x.did_miss_rate_hi * 100, color=INK2, lw=0.8)
        for xx, v, top in zip(xs, m, x.did_miss_rate_hi * 100):
            # 20 kg labels lean left so they never run into the taller 6 kg bar next to them
            axes[1].text(xx + (0.13 if st == "20kg" else 0), top + 1.2, f"+{v:.0f}",
                         ha="right" if st == "20kg" else "center", fontsize=6.2, color=INK2)
    for ax in axes:
        ax.set_xticks(range(len(lv)), [f"8→{l}" for l in lv])
        ax.set_xlabel("same scale moved from level 8 to ...")
    axes[0].set_ylim(-112, 0)
    axes[0].set_title("Empty wake-ups per week")
    axes[0].set_ylabel("change vs. level 8 (%)")
    axes[1].set_title("Share of real changes missed")
    axes[1].set_ylabel("change vs. level 8 (pp)")
    axes[1].set_ylim(0, 72)
    legend_scales(axes[1], loc="upper left")
    save(fig, "fig11_knob")


def fig_sens3(plt):
    s = rd("sens3_signature")
    groups = ["other devices on baseline", "same devices on baseline", "anomaly devices on anomaly level"]
    labels = ["other\nscales\non 8", "level-3\nscales\non 8", "level-3\nscales\non 3"]
    spec = [("auto_per_device_day", "Wake-ups per scale-day", 1, "{:.0f}"),
            ("share_retrigger_120s", "Re-trigger ≤ 2 min", 100, "{:.0f}"),   # axis carries the %; short labels
            ("share_night", "At night (22–06 h)", 100, "{:.0f}")]             # do not collide over paired bars
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw={"wspace": 0.45})
    for ax, (col, title, mult, fmt) in zip(axes, spec):
        for i, st in enumerate(SCALES):
            v = s[s.scale_type == st].set_index("grp").reindex(groups)[col].values * mult
            xs = np.arange(3) + (i - .5) * 0.38
            ax.bar(xs, v, width=0.36, color=C[st])
            for xx, vv in zip(xs, v):
                ax.text(xx, vv * 1.02 + (0.5 if mult == 1 else 0.6), fmt.format(vv), ha="center", fontsize=5.9,
                        color=INK2)
        ax.set_xticks(range(3), labels, fontsize=6.6)
        ax.set_title(title, fontsize=8)
        if mult == 100:
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(100, decimals=0))
    legend_scales(axes[0], loc="upper left", fontsize=6.4)
    save(fig, "fig12_sens3")


def fig_check_timing(plt):
    t = rd("manual_check_hours")
    fig, ax = plt.subplots(figsize=(W, 1.95))
    for flag, col, lab in [(True, INK, "after ≥ 24 h without any weighing (idle check)"),
                           (False, GRAY_D, "within 24 h of the previous weighing (person)")]:
        x = t[t.after_24h_idle == flag].set_index("hour").n.reindex(range(24), fill_value=0)
        ax.plot(range(24), x / x.sum(), color=col, marker="o", ms=3.2, label=lab, markeredgecolor="white",
                markeredgewidth=0.6)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("hour of day (as recorded)")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_ylabel("share of manual\nweighings")
    ax.legend(loc="upper left")
    save(fig, "fig13_check_timing")


def fig_policies(plt):
    p = rd("policies")
    show = {"P0 status quo (8)": "P0 today", "P1 one setting": "P1 one level (Bossard KPIs)",
            "P5 one setting (KPI 2.0)": "P5 one level (KPI 2.0)", "P7 start 7 + rule (KPI 2.0)": "P7 recommended",
            "P8 fraction rule (start 6)": "P8 fraction rule"}
    # label placement per scale type: (x offset, y offset, horizontal alignment) in points
    place = {"20kg": {"P0": (-7, 4, "right"), "P1": (8, 0, "left"), "P5": (8, 7, "left"),
                      "P7": (-9, 9, "right"), "P8": (8, 0, "left")},
             "6kg": {"P0": (-7, 4, "right"), "P1": (0, -16, "center", "P1 one level\n(Bossard KPIs)"),
                     "P5": (8, 0, "left"), "P7": (9, 2, "left"), "P8": (8, 0, "left")}}
    col = "meas_change_pct_without_loops" if "meas_change_pct_without_loops" in p else "meas_change_pct"
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
    for ax, st in zip(axes, SCALES):
        x = p[(p.scale_type == st) & p.policy.isin(show)]
        ax.axhspan(0, 10, color="#eef6ee", zorder=0)
        ax.text(-97, 1.6, "miss rate ≤ 10% (KPI 2.0)", fontsize=6.6, color="#2f6b2f")
        for _, r in x.iterrows():
            rec = r.policy.startswith("P7")
            ax.scatter(r[col], r.median_miss_rate * 100, s=70 if rec else 30, zorder=4,
                       color=C[st] if rec else "white", edgecolor=C[st], linewidth=1.6)
            dx, dy, ha, *txt = place[st][r.policy[:2]]
            ax.annotate(txt[0] if txt else show[r.policy], (r[col], r.median_miss_rate * 100), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va="center", fontsize=6.8, color=INK if rec else INK2,
                        fontweight="bold" if rec else "normal")
        ax.set_xlim(-100, 8)
        ax.set_ylim(0, 60)
        ax.set_title(f"{st.replace('kg', ' kg')} scales")
        ax.grid(axis="x", color=GRID)
    axes[0].set_ylabel("real changes missed (%)\nmedian scale")
    fig.text(0.5, -0.05, "weighings vs. today (%, scales without a trigger loop)", ha="center", fontsize=8.6,
             color=INK2)
    save(fig, "fig14_policies")


def fig_final_levels(plt):
    f = rd("final_levels")
    pols = [("P7 start 7 + rule (KPI 2.0)", "P7 recommended: start 7 + 2-weekly KPI 2.0 rule"),
            ("P8 fraction rule (start 6)", "P8 fraction rule: daily, steered by the TOO_MANY fraction")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True)
    for ax, st in zip(axes, SCALES):
        xs = np.arange(len(LEVELS))
        for i, (pol, lab) in enumerate(pols):
            v = (f[(f.policy == pol) & (f.scale_type == st)].set_index("sensitivity").share
                 .reindex([float(l) for l in LEVELS]).fillna(0).values * 100)
            ax.bar(xs + (i - .5) * 0.4, v, width=0.38, color=C[st] if i == 0 else GRAY, label=lab)
            for xx, vv in zip(xs + (i - .5) * 0.4, v):
                if vv >= 8:
                    ax.text(xx, vv + 1.2, f"{vv:.0f}", ha="center", fontsize=6.3, color=INK2)
        ax.set_xticks(xs, [str(l) for l in LEVELS])
        ax.set_xlabel("sensitivity level after 16 weeks")
        ax.set_title(f"{st.replace('kg', ' kg')} scales")
        ax.set_ylim(0, 55)
        ax.legend(["P7 recommended", "P8 fraction rule"], loc="upper right" if st == "20kg" else "upper left",
                  fontsize=7)
    axes[0].set_ylabel("share of scales (%)")
    save(fig, "fig15_final_levels")


def fig_loops(plt):
    """Share of the scales vs. share of the automatic weighings that are caught in a trigger loop.
    Denominators: all scales in the data and all automatic weighings (incl. those of scales whose level or type
    is unknown and that therefore cannot be checked) - so the loop shares are lower bounds."""
    s = rd("loops_summary")
    lab = rd("label_counts")
    nums = json.loads((DATA / "numbers.json").read_text())
    all_scales = nums["scales_raw"]
    all_auto = lab[lab.initiatortype == "AUTOMATIC"].n.sum()
    loop_scales = s.loop_scales_any_level.iloc[0]
    n = s.set_index(["group", "loop"]).n_auto
    l3 = n.get(("level 3", True), 0) / all_auto * 100
    lo = n.get(("other levels", True), 0) / all_auto * 100
    sc = loop_scales / all_scales * 100
    fig, ax = plt.subplots(figsize=(W, 1.55))
    rows = [("scales", [(sc, CRIT), (100 - sc, GRAY)]),
            ("automatic weighings", [(l3, CRIT), (lo, "#e9857f"), (100 - l3 - lo, GRAY)])]
    for y, (_, parts) in zip([1, 0], rows):
        left = 0
        for v, c in parts:
            ax.barh(y, v, left=left, color=c, height=0.56, edgecolor="white", linewidth=1.2)
            if v >= 4:
                ax.text(left + v / 2, y, f"{v:.1f}%", ha="center", va="center", fontsize=7.4,
                        color="white" if c != GRAY else INK)
            left += v
    ax.text(sc + 1, 1, f"{sc:.1f}% of the scales ({loop_scales:,})", va="center", fontsize=7.4, color=INK)
    ax.set_yticks([1, 0], ["scales", "automatic weighings"])
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(100, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=CRIT), Patch(color="#e9857f"), Patch(color=GRAY)],
               ["trigger loop on level 3", "trigger loop on another level", "no loop found"],
               loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7.4)
    save(fig, "fig16_loops")


def fig_shapley(plt):
    s = rd("shapley")
    s = s[(s.kpi == "frac_too_many") & s.scope.isin(SCALES)]
    order = ["Location (factory)", "Weight in box (fill level)", "Item weight (single part)",
             "Sensitivity setting", "Box type"]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    y = np.arange(len(order))[::-1]
    for i, st in enumerate(SCALES):
        x = s[s.scope == st].set_index("parameter").reindex(order)
        yy = y + (0.5 - i) * 0.36
        ax.barh(yy, x.shapley_r2.values * 100, height=0.34, color=C[st])
        for yi, v in zip(yy, x.shapley_r2.values * 100):
            ax.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=6.7, color=INK2)
    ax.set_yticks(y, ["Site (location)", "Fill level (weight bin)", "Item weight", "Sensitivity level", "Box type"])
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of the differences between periods explained (Shapley R², %)")
    legend_scales(ax, loc="lower right")
    save(fig, "figA1_shapley")


def fig_event_study(plt):
    """Empty wake-ups per scale-day of the switched scales, by week relative to their switch (raw, no stayer
    correction): shows the drop at the switch and the extra activity in the weeks before it."""
    e = rd("event_study")
    e["tm_day"] = e.n_tm / e.active_days
    bands = [("to 1-2 (low)", GRAY, "new level 1–2"), ("to 4-5 (mid)", GRAY_D, "new level 4–5"),
             ("to 6-7 (high)", INK, "new level 6–7")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), sharey=True)
    for ax, st in zip(axes, SCALES):
        for b, col, lab in bands:
            x = e[(e.scale_type == st) & (e.band == b)].sort_values("rel_week")
            ax.plot(x.rel_week, x.tm_day, color=col, lw=1.6, marker="o", ms=2.5, label=lab)
        ax.axvline(-0.5, color=C[st], lw=1)
        ax.set_xticks(range(-4, 9, 2))
        ax.set_xlabel("weeks relative to the switch")
        ax.set_title(f"{st.replace('kg', ' kg')} scales")
        ax.set_ylim(0, 16)
    axes[0].set_ylabel("empty wake-ups\nper scale-day")
    axes[1].legend(loc="upper right", fontsize=6.6)
    save(fig, "figA2_event_study")


def draw():
    plt = style()
    for fn in [fig_design, fig_cleaning, fig_one_sided, fig_classes, fig_fraction_vs_counts, fig_usage,
               fig_coupling, fig_blind_zone, fig_zones, fig_location, fig_parameters, fig_knob, fig_sens3,
               fig_check_timing, fig_policies, fig_final_levels, fig_loops, fig_shapley, fig_event_study]:
        try:
            fn(plt)
        except FileNotFoundError as e:
            print(f"skipped {fn.__name__}: {e.filename} missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports/bossard")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--db", default="data/interim/smartbin.duckdb")
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    if not a.figures_only:
        extract(Path(a.reports), Path(a.processed), Path(a.db))
    draw()


if __name__ == "__main__":
    main()
