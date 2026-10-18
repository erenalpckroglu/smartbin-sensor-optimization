"""
Robustness and evidence checks for the client deck.

1. Do the main findings hold when every scale caught in a trigger loop is left out?
   (baseline at level 8, the causal effect of the knob, the coupling share)
2. Is the coupling real? Per-scale view, and a placebo: picks at a *different* site of similar size.
3. How sensitive is the loop detector to its two thresholds?
4. How long does a missed change stay invisible?
5. Two small, anonymised examples for the slides: one coupling episode and one looping day.

Writes aggregated tables to report/data/ (no scale IDs, no timestamps).

    python report/robustness_loops.py --db <interim>/smartbin.duckdb --reports reports/bossard
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smartbin import tables  # noqa: E402
from smartbin.analysis import sensitivity as sens  # noqa: E402
from smartbin.config import load_config  # noqa: E402

OUT = ROOT / "report" / "data"
NEAR, CTRL0, W = 14.0, 60.0, 120


def baseline(ds, loop_ids, cfg):
    b = float(cfg.analysis.baseline_sensitivity)
    rows = []
    for label, d in [("all scales", ds), ("without loop scales", ds[~ds.deviceid.isin(loop_ids)])]:
        d = d[d.main_analysis & (d.sensitivity == b)]
        for s, g in d.groupby("scale_type"):
            wk = g.days / 7
            rows.append(dict(sample=label, scale_type=s, scales=len(g),
                             tm_per_week=(g.n_tm / wk).median(), picks_per_week=(g.n_aok / wk).median(),
                             frac_too_many=g.frac_too_many.median(), share_in_target=g.in_target.mean(),
                             share_meets_tf=g.meets_tf.mean(), share_meets_tm=g.meets_tm.mean()))
    return pd.DataFrame(rows)


def knob(ds, dw, loop_ids, cfg):
    levels = [float(x) for x in cfg.analysis.main_levels]
    main = ds[ds.main_analysis & ds.sensitivity.isin(levels)]
    out = []
    for label, m, w in [("all scales", main, dw),
                        ("without loop scales", main[~main.deviceid.isin(loop_ids)], dw[~dw.deviceid.isin(loop_ids)])]:
        _, eff = sens._within_device(m, w, cfg)
        eff = eff[eff.sensitivity != float(cfg.analysis.baseline_sensitivity)]
        out.append(pd.DataFrame(dict(sample=label, scale_type=eff.scale_type, sensitivity=eff.sensitivity,
                                     switchers=eff.n_devices, empty_wakeups_change=np.expm1(eff.beta_tm),
                                     picks_change=np.expm1(eff.beta_aok), missed_pp=eff.did_miss_rate * 100,
                                     fraction_change=eff.did_frac_too_many)))
    return pd.concat(out, ignore_index=True)


def policy_check(ds, dw, loops, loop_ids, cfg, tab):
    """Re-run the policy simulator with knob effects estimated WITHOUT the loop scales."""
    from smartbin.analysis import policy
    levels = [float(x) for x in cfg.analysis.main_levels]
    main = ds[ds.main_analysis & ds.sensitivity.isin(levels)]
    keep = ~main.deviceid.isin(loop_ids)
    pairs, eff = sens._within_device(main[keep], dw[~dw.deviceid.isin(loop_ids)], cfg)
    prof = pd.read_csv(tab / "m5_pick_size__device_pick_profile.csv")
    scratch = load_config("config/config.yaml", **{"paths.reports": str(Path(tab).parents[1] / "robust_tmp")})
    res = policy.run(ds, eff, pairs, prof, scratch, loops)       # figures go to a scratch folder
    cols = ["scale_type", "policy", "meas_change_pct_without_loops", "median_miss_rate", "share_in_target_kpi2",
            "median_tm_week", "battery_life_change_pct"]
    new = res.tables["policy_comparison"][cols].assign(effects="estimated without loop scales")
    old = pd.read_csv(tab / "m7_policy__policy_comparison.csv")[cols].assign(effects="estimated on all scales")
    return pd.concat([old, new], ignore_index=True)


def coupling(con, loop_ids, cfg):
    std = ", ".join(str(float(x)) for x in cfg.cleaning.standard_delay)
    d0 = float(cfg.cleaning.standard_delay[0])
    b = float(cfg.analysis.baseline_sensitivity)
    con.execute("CREATE OR REPLACE TEMP TABLE loopdev AS SELECT unnest(?::VARCHAR[]) AS deviceid", [sorted(loop_ids)])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE src AS
        SELECT location_id, deviceid AS src_dev, ts AS src_ts, CAST(floor(ts / {W}) AS BIGINT) + o AS b
        FROM events, (SELECT unnest([-1, 0, 1]) AS o)
        WHERE is_auto AND cls = 'OK' AND location_id IS NOT NULL AND coalesce(delay, {d0}) IN ({std})""")
    # placebo partner: sites ranked by number of picks, neighbours in the ranking are paired (1-2, 3-4, ...)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE partner AS
        WITH r AS (SELECT location_id, row_number() OVER (ORDER BY count(*) DESC, location_id) AS k
                   FROM src WHERE b IS NOT NULL GROUP BY location_id)
        SELECT a.location_id, p.location_id AS partner_id
        FROM r a JOIN r p ON p.k = CASE WHEN a.k % 2 = 1 THEN a.k + 1 ELSE a.k - 1 END""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tgt AS
        SELECT e.rid, e.deviceid, e.location_id, p.partner_id, e.scale_type, e.ts, e.hour,
               CAST(floor(e.ts / {W}) AS BIGINT) AS b,
               e.deviceid IN (SELECT deviceid FROM loopdev) AS is_loop
        FROM events e LEFT JOIN partner p USING (location_id)
        WHERE e.is_auto AND e.cls = 'TOO_MANY' AND e.exclusion_reason IS NULL AND e.sensitivity = {b}
          AND e.location_id IS NOT NULL AND coalesce(e.delay, {d0}) IN ({std})""")

    def flags(key):
        cond = "t.location_id = s.location_id AND s.src_dev <> t.deviceid" if key == "same" else "t.partner_id = s.location_id"
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE fl_{key} AS
            WITH f AS (SELECT t.rid, bool_or(abs(t.ts - s.src_ts) <= {NEAR}) AS near,
                              bool_or((s.src_ts - t.ts) BETWEEN {CTRL0} AND {CTRL0 + 2 * NEAR}) AS ctrl
                       FROM tgt t JOIN src s ON {cond} AND t.b = s.b
                       WHERE abs(t.ts - s.src_ts) <= {W} GROUP BY t.rid)
            SELECT t.*, coalesce(f.near, false) AS near, coalesce(f.ctrl, false) AS ctrl
            FROM tgt t LEFT JOIN f USING (rid)""")
    flags("same")
    flags("placebo")

    def attr(df):
        df["p_near"], df["p_chance"] = df.n_near / df.n, df.n_ctrl / df.n
        df["coupled_beyond_chance"] = (df.p_near - df.p_chance) / (1 - df.p_chance)
        return df

    rows = []
    for key, label in [("same", "picks at the same site"), ("placebo", "placebo: picks at another site of similar size")]:
        for samp, where in [("all scales", "TRUE"), ("without loop scales", "NOT is_loop")]:
            if key == "placebo" and samp == "all scales":
                continue
            q = f"""SELECT {{g}} count(*) AS n, sum(near::INT) AS n_near, sum(ctrl::INT) AS n_ctrl
                    FROM fl_{key} WHERE {where} {{gb}}"""
            a = con.execute(q.format(g="scale_type,", gb="GROUP BY scale_type")).df()
            p = con.execute(q.format(g="'both' AS scale_type,", gb="")).df()
            rows.append(attr(pd.concat([a, p])).assign(source=label, sample=samp))
    pooled = pd.concat(rows, ignore_index=True)

    # per scale: every scale counts once (scales with >= 30 empty wake-ups at level 8, no loop)
    ps = con.execute("""SELECT scale_type, deviceid, count(*) AS n, sum(near::INT) AS n_near, sum(ctrl::INT) AS n_ctrl
                        FROM fl_same WHERE NOT is_loop GROUP BY ALL HAVING count(*) >= 30""").df()
    ps = attr(ps)
    per_scale = (ps.groupby("scale_type").coupled_beyond_chance
                 .agg(scales="size", q25=lambda x: x.quantile(.25), median="median", q75=lambda x: x.quantile(.75))
                 .reset_index())
    per_scale["share_scales_above_chance"] = ps.assign(up=ps.p_near > ps.p_chance).groupby("scale_type").up.mean().values

    # dose-response: how often an empty wake-up has a pick within +-14 s, by how many picks the site had that hour
    dose = con.execute(f"""
        WITH act AS (SELECT location_id, CAST(floor(ts / 3600) AS BIGINT) AS h, count(*) AS picks
                     FROM events WHERE is_auto AND cls = 'OK' AND location_id IS NOT NULL GROUP BY ALL)
        SELECT f.scale_type,
               CASE WHEN a.picks IS NULL THEN '0' WHEN a.picks <= 5 THEN '1-5' WHEN a.picks <= 20 THEN '6-20'
                    WHEN a.picks <= 50 THEN '21-50' ELSE '>50' END AS site_picks_that_hour,
               count(*) AS n, sum(f.near::INT) AS n_near, sum(f.ctrl::INT) AS n_ctrl
        FROM fl_same f LEFT JOIN act a ON a.location_id = f.location_id AND a.h = CAST(floor(f.ts / 3600) AS BIGINT)
        WHERE NOT f.is_loop GROUP BY ALL""").df()
    dose = attr(dose)
    return pooled, per_scale, dose


def example_episode(con, cfg):
    """One clean coupling episode: one pick, >= 3 empty wake-ups at other scales of the site within +-14 s,
    nothing else at that site in the 5 minutes before and after. Times relative to the pick; scales anonymised."""
    b = float(cfg.analysis.baseline_sensitivity)
    cand = con.execute(f"""
        WITH p AS (SELECT rid, location_id, deviceid, ts FROM events
                   WHERE is_auto AND cls = 'OK' AND sensitivity = {b} AND exclusion_reason IS NULL
                     AND hash(rid) % 20 = 0),
        w AS (SELECT p.rid, p.location_id, p.ts,
                     count(*) FILTER (WHERE e.cls = 'TOO_MANY' AND abs(e.ts - p.ts) <= {NEAR} AND e.deviceid <> p.deviceid) AS n_tm_near,
                     count(DISTINCT e.deviceid) FILTER (WHERE e.cls = 'TOO_MANY' AND abs(e.ts - p.ts) <= {NEAR}) AS n_dev_near,
                     count(*) FILTER (WHERE abs(e.ts - p.ts) > {NEAR}) AS n_other,
                     count(*) FILTER (WHERE e.is_auto AND e.cls = 'OK' AND e.rid <> p.rid) AS n_other_picks
              FROM p JOIN events e ON e.location_id = p.location_id AND e.ts BETWEEN p.ts - 300 AND p.ts + 300
              GROUP BY ALL)
        SELECT * FROM w WHERE n_tm_near >= 3 AND n_dev_near >= 3 AND n_other = 0 AND n_other_picks = 0
        ORDER BY hash(rid) LIMIT 50""").df()
    if cand.empty:
        return pd.DataFrame(), 0
    c = cand.iloc[0]
    ev = con.execute(f"""SELECT deviceid, ts - {c.ts} AS t_rel_s, cls, initiatortype
                         FROM events WHERE location_id = ? AND ts BETWEEN {c.ts} - 300 AND {c.ts} + 300
                         ORDER BY ts""", [c.location_id]).df()
    order = list(dict.fromkeys(ev.sort_values(["cls", "t_rel_s"], ascending=[True, True]).deviceid))
    # the picked bin first
    picked = ev[ev.cls == "OK"].deviceid.iloc[0]
    order = [picked] + [d for d in order if d != picked]
    names = {d: f"Bin {chr(65 + i)}" for i, d in enumerate(order)}
    ev["bin"] = ev.deviceid.map(names)
    hour = con.execute(f"SELECT hour FROM events WHERE rid = {int(c.rid)}").fetchone()[0]
    ev["hour_of_day"] = hour
    return ev.drop(columns="deviceid"), len(cand)


def example_loop(con, loops, cfg):
    """Hourly wake-ups of one looping scale (outside level 3, close to the median looper) on its busiest
    full day, next to the median non-looping scale of the same site and level on that day."""
    lf = loops[loops.loop & (loops.sensitivity != 3.0)].copy()
    med = lf.auto_per_day.median()
    pick = lf.iloc[(lf.auto_per_day - med).abs().argsort().iloc[0]]
    dev, lvl = pick.deviceid, float(pick.sensitivity)
    day = con.execute("""SELECT day FROM events WHERE deviceid = ? AND sensitivity = ? AND is_auto
                         GROUP BY day ORDER BY count(*) DESC LIMIT 1 OFFSET 1""", [dev, lvl]).fetchone()[0]
    loc = con.execute("SELECT any_value(location_id) FROM events WHERE deviceid = ?", [dev]).fetchone()[0]
    h = con.execute("""SELECT deviceid, hour, count(*) AS wakeups FROM events
                       WHERE location_id = ? AND day = ? AND is_auto AND sensitivity = ? GROUP BY ALL""",
                    [loc, day, lvl]).df()
    tot = h.groupby("deviceid").wakeups.sum()
    others = tot.drop(dev, errors="ignore")
    others = others[~others.index.isin(set(loops[loops.loop].deviceid))]
    rows = [pd.DataFrame({"hour": range(24)}).merge(h[h.deviceid == dev][["hour", "wakeups"]], how="left")
            .fillna(0).assign(series="looping scale")]
    if len(others):
        normal = others.index[np.argsort(np.abs(others.values - others.median()))[0]]
        rows.append(pd.DataFrame({"hour": range(24)}).merge(h[h.deviceid == normal][["hour", "wakeups"]], how="left")
                    .fillna(0).assign(series="typical scale, same site"))
    out = pd.concat(rows, ignore_index=True)
    out["sensitivity"] = lvl
    out["share_retrigger"] = float(pick.share_retrigger)
    gaps = con.execute("""SELECT median(gap_s) FROM events WHERE deviceid = ? AND sensitivity = ? AND is_auto
                          AND prev_initiator IS NOT NULL""", [dev, lvl]).fetchone()[0]
    out["median_gap_s"] = gaps
    return out


def correlogram_no_loops(con, loop_ids, cfg):
    """Picks at other scales of the site per minute, around the empty wake-ups at level 8 of scales without a loop."""
    std = ", ".join(str(float(x)) for x in cfg.cleaning.standard_delay)
    d0 = float(cfg.cleaning.standard_delay[0])
    b = float(cfg.analysis.baseline_sensitivity)
    con.execute("CREATE OR REPLACE TEMP TABLE loopdev AS SELECT unnest(?::VARCHAR[]) AS deviceid", [sorted(loop_ids)])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE src AS
        SELECT location_id, deviceid AS src_dev, ts AS src_ts, CAST(floor(ts / {W}) AS BIGINT) + o AS b
        FROM events, (SELECT unnest([-1, 0, 1]) AS o)
        WHERE is_auto AND cls = 'OK' AND location_id IS NOT NULL AND coalesce(delay, {d0}) IN ({std})""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tgt2 AS
        SELECT rid, deviceid, location_id, scale_type, ts, CAST(floor(ts / {W}) AS BIGINT) AS b
        FROM events
        WHERE is_auto AND cls = 'TOO_MANY' AND exclusion_reason IS NULL AND sensitivity = {b}
          AND location_id IS NOT NULL AND coalesce(delay, {d0}) IN ({std})
          AND deviceid NOT IN (SELECT deviceid FROM loopdev)""")
    h = con.execute(f"""
        SELECT t.scale_type, CAST(floor((s.src_ts - t.ts) / 2) AS INTEGER) * 2 AS lag_s, count(*) AS n
        FROM tgt2 t JOIN src s ON t.location_id = s.location_id AND t.b = s.b
        WHERE s.src_dev <> t.deviceid AND abs(t.ts - s.src_ts) <= {W} GROUP BY ALL""").df()
    n = con.execute("SELECT scale_type, count(*) AS n_targets FROM tgt2 GROUP BY ALL").df()
    h = h.merge(n, on="scale_type")
    h["per_min"] = h.n / h.n_targets / 2 * 60
    base = h[h.lag_s.abs() >= W / 2].groupby("scale_type").per_min.mean().rename("baseline").reset_index()
    return h.merge(base, on="scale_type").sort_values(["scale_type", "lag_s"])


def loop_flag_distribution(loops):
    """How far apart are looping and normal scale-levels on the two detector inputs?"""
    q = [0.5, 0.9, 0.99]
    rows = []
    for lab, g in [("loop", loops[loops.loop]), ("no loop", loops[~loops.loop])]:
        for col in ["auto_per_day", "share_retrigger"]:
            for qq in q:
                rows.append(dict(group=lab, measure=col, quantile=qq, value=g[col].quantile(qq)))
    rows.append(dict(group="no loop", measure="share above 20 per day", quantile=None,
                     value=(loops[~loops.loop].auto_per_day > 20).mean()))
    rows.append(dict(group="no loop", measure="share above 50% retrigger", quantile=None,
                     value=(loops[~loops.loop].share_retrigger > 0.5).mean()))
    rows.append(dict(group="all", measure="scale-levels judged", quantile=None, value=len(loops)))
    return pd.DataFrame(rows)


def loop_detector_bins(loops):
    """2-D histogram of the two detector inputs over all judged scale-levels (for the FLAG chart)."""
    d = loops.copy()
    d["x"] = np.floor(np.log10(d.auto_per_day.clip(lower=0.1)) * 10) / 10        # 0.1-decade bins
    d["y"] = np.floor(d.share_retrigger.clip(0, 0.9999) * 25) / 25               # 4-point bins
    return (d.groupby(["x", "y", "loop"]).agg(scale_levels=("deviceid", "size"), n_auto=("n_auto", "sum"))
            .reset_index())


def loop_evidence(con, loops, cfg):
    """Why we call a scale-level a trigger loop: timing, night share, emptiness, persistence, behaviour on its
    other levels, and whether its wake-ups coincide with picks nearby. Aggregated, no IDs."""
    lv = loops[["deviceid", "sensitivity", "loop"]].copy()
    con.register("lf_df", lv)
    con.execute("CREATE OR REPLACE TEMP TABLE lf AS SELECT * FROM lf_df")
    prof = con.execute("""
        SELECT CASE WHEN lf.loop THEN 'trigger loop' ELSE 'normal' END AS grp,
               count(*) AS wakeups,
               avg((e.cls = 'TOO_MANY')::INT) AS share_empty,
               avg((e.hour < 5)::INT) AS share_night_0_5h,
               avg((e.isodow >= 6)::INT) AS share_weekend,
               avg((e.gap_s BETWEEN 85 AND 95)::INT) FILTER (WHERE e.prev_initiator = 'AUTOMATIC') AS share_gap_85_95s,
               median(e.gap_s) FILTER (WHERE e.prev_initiator = 'AUTOMATIC') AS median_gap_s
        FROM events e JOIN lf ON lf.deviceid = e.deviceid AND lf.sensitivity = e.sensitivity
        WHERE e.is_auto GROUP BY 1""").df()
    picks = con.execute("""SELECT 'real picks (all scales)' AS grp, count(*) AS wakeups, 0.0 AS share_empty,
                                  avg((hour < 5)::INT) AS share_night_0_5h, avg((isodow >= 6)::INT) AS share_weekend
                           FROM events WHERE is_auto AND cls = 'OK'""").df()
    prof = pd.concat([prof, picks], ignore_index=True)
    # persistence: share of observed days on which a looping scale-level woke up more than 20 times
    pers = con.execute("""
        WITH d AS (SELECT e.deviceid, e.sensitivity, e.day, count(*) AS n
                   FROM events e JOIN lf ON lf.deviceid = e.deviceid AND lf.sensitivity = e.sensitivity AND lf.loop
                   WHERE e.is_auto GROUP BY ALL),
        s AS (SELECT deviceid, sensitivity, count(*) AS days, avg((n > 20)::INT) AS share_days_over_20 FROM d GROUP BY ALL)
        SELECT count(*) AS loop_scale_levels, median(days) AS median_days_observed,
               median(share_days_over_20) AS median_share_days_over_20,
               avg((share_days_over_20 >= 0.8)::INT) AS share_looping_on_80pct_of_days FROM s""").df()
    # the same scales on their other levels
    ids = set(loops[loops.loop].deviceid)
    other = loops[loops.deviceid.isin(ids)]
    rows = []
    for grp, sel in [("loop on level 3", other[(other.sensitivity == 3.0) & other.loop].deviceid),
                     ("loop on another level", other[(other.sensitivity != 3.0) & other.loop].deviceid)]:
        sel = set(sel)
        o = other[other.deviceid.isin(sel)]
        looplv = o[o.loop].groupby("deviceid").sensitivity.apply(set)
        oth = o[o.apply(lambda r: r.sensitivity not in looplv.get(r.deviceid, set()), axis=1)]
        rows.append(dict(group=grp, scales=len(sel), scales_seen_on_other_level=oth.deviceid.nunique(),
                         of_which_normal_there=oth[~oth.loop].deviceid.nunique(),
                         median_wakeups_per_day_other_level=oth.auto_per_day.median(),
                         median_wakeups_per_day_loop_level=o[o.loop].auto_per_day.median()))
    other_tab = pd.DataFrame(rows)
    # coupling of loop wake-ups vs normal wake-ups at level 8
    std = ", ".join(str(float(x)) for x in cfg.cleaning.standard_delay)
    d0 = float(cfg.cleaning.standard_delay[0])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE src AS
        SELECT location_id, deviceid AS src_dev, ts AS src_ts, CAST(floor(ts / {W}) AS BIGINT) + o AS b
        FROM events, (SELECT unnest([-1, 0, 1]) AS o)
        WHERE is_auto AND cls = 'OK' AND location_id IS NOT NULL AND coalesce(delay, {d0}) IN ({std})""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tgt3 AS
        SELECT e.rid, e.deviceid, e.location_id, e.ts, CAST(floor(e.ts / {W}) AS BIGINT) AS b, lf.loop
        FROM events e JOIN lf ON lf.deviceid = e.deviceid AND lf.sensitivity = e.sensitivity
        WHERE e.is_auto AND e.cls = 'TOO_MANY' AND e.sensitivity = 8 AND e.location_id IS NOT NULL
          AND coalesce(e.delay, {d0}) IN ({std})""")
    cp = con.execute(f"""
        WITH f AS (SELECT t.rid, bool_or(abs(t.ts - s.src_ts) <= {NEAR}) AS near,
                          bool_or((s.src_ts - t.ts) BETWEEN {CTRL0} AND {CTRL0 + 2 * NEAR}) AS ctrl
                   FROM tgt3 t JOIN src s ON t.location_id = s.location_id AND t.b = s.b
                   WHERE s.src_dev <> t.deviceid AND abs(t.ts - s.src_ts) <= {W} GROUP BY t.rid)
        SELECT CASE WHEN t.loop THEN 'trigger loop' ELSE 'normal' END AS grp, count(*) AS n,
               avg(coalesce(f.near, false)::INT) AS p_near, avg(coalesce(f.ctrl, false)::INT) AS p_chance
        FROM tgt3 t LEFT JOIN f USING (rid) GROUP BY 1""").df()
    cp["coupled_beyond_chance"] = (cp.p_near - cp.p_chance) / (1 - cp.p_chance)
    hourly = con.execute("""
        SELECT CASE WHEN lf.loop THEN 'trigger loop' ELSE 'normal' END AS grp, e.hour, count(*) AS n
        FROM events e JOIN lf ON lf.deviceid = e.deviceid AND lf.sensitivity = e.sensitivity
        WHERE e.is_auto GROUP BY ALL""").df()
    return prof, pers, other_tab, cp, hourly


def watch_tables(con, ds, cfg):
    """How long a missed change stays invisible (buckets), and how concentrated the misses are across scales."""
    delay = con.execute("""
        SELECT scale_type,
               CASE WHEN gap_s < 24 * 3600 THEN '< 24 h' WHEN gap_s < 48 * 3600 THEN '24–48 h'
                    WHEN gap_s < 72 * 3600 THEN '48–72 h' ELSE '> 72 h' END AS found_after,
               count(*) AS too_few
        FROM events WHERE cls = 'TOO_FEW' AND exclusion_reason IS NULL AND gap_s IS NOT NULL GROUP BY ALL""").df()
    b = float(cfg.analysis.baseline_sensitivity)
    d = ds[ds.main_analysis & (ds.sensitivity == b)]
    rows = []
    for st, g in d.groupby("scale_type"):
        tf = np.sort(g.n_tf.values)[::-1]
        cum = np.cumsum(tf) / max(tf.sum(), 1)
        for q in np.linspace(0, 1, 101):
            k = int(round(q * len(tf)))
            rows.append(dict(scale_type=st, share_of_scales=q, share_of_too_few=float(cum[k - 1]) if k > 0 else 0.0))
    return delay, pd.DataFrame(rows)


def loop_grid(loops, total_auto):
    rows = []
    for share in (0.3, 0.4, 0.5, 0.6, 0.7):
        for per_day in (10, 20, 40, 80):
            f = loops[(loops.share_retrigger > share) & (loops.auto_per_day > per_day)]
            rows.append(dict(min_share_retrigger=share, min_wakeups_per_day=per_day,
                             scales=f.deviceid.nunique(), share_of_automatic=f.n_auto.sum() / total_auto))
    return pd.DataFrame(rows)


def miss_delay(con):
    return con.execute("""
        SELECT scale_type, count(*) AS too_few,
               quantile_cont(gap_s / 3600, 0.5) AS hours_median, quantile_cont(gap_s / 3600, 0.9) AS hours_p90,
               quantile_cont(gap_s / 3600, 0.99) AS hours_p99, avg((gap_s <= 48 * 3600)::INT) AS share_within_48h
        FROM events WHERE cls = 'TOO_FEW' AND exclusion_reason IS NULL AND gap_s IS NOT NULL GROUP BY ALL""").df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--reports", default="reports/bossard")
    ap.add_argument("--parts", default="knob,coupling,examples")
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    cfg = load_config("config/config.yaml", **{"paths.reports": a.reports})
    tab = Path(a.reports) / "tables"
    ds = pd.read_parquet(ROOT / "data/processed/device_settings.parquet")
    loops = pd.read_csv(tab / "m2_sensitivity__loop_flags.csv")
    loops["loop"] = loops["loop"].astype(bool)
    loop_ids = set(loops[loops.loop].deviceid)
    con = duckdb.connect(a.db, read_only=True)
    con.execute("SET memory_limit='1400MB'; SET threads=4")
    total_auto = con.execute("SELECT count(*) FROM events WHERE is_auto").fetchone()[0]

    res = {}
    if "knob" in parts:
        res["robust_baseline"] = baseline(ds, loop_ids, cfg)
        print(res["robust_baseline"].round(3).to_string(), flush=True)
        dw = tables.build_device_weeks(con, cfg)
        res["robust_knob"] = knob(ds, dw, loop_ids, cfg)
        print(res["robust_knob"].round(3).to_string(), flush=True)
        res["loop_threshold_grid"] = loop_grid(loops, total_auto)
        res["miss_delay"] = miss_delay(con)
        print(res["loop_threshold_grid"].round(3).to_string(), res["miss_delay"].round(2).to_string(), sep="\n")
    if "extra" in parts:
        res["coupling_correlogram_no_loops"] = correlogram_no_loops(con, loop_ids, cfg)
        res["loop_flag_distribution"] = loop_flag_distribution(loops)
        c = res["coupling_correlogram_no_loops"]
        print(c[c.lag_s.abs() <= 4].round(2).to_string(), res["loop_flag_distribution"].round(3).to_string(), sep="\n")
    if "loopevidence" in parts:
        prof, pers, oth, cp, hourly = loop_evidence(con, loops, cfg)
        res["loop_evidence_profile"], res["loop_evidence_persistence"] = prof, pers
        res["loop_evidence_other_levels"], res["loop_evidence_coupling"], res["loop_evidence_hourly"] = oth, cp, hourly
        print(prof.round(3).to_string(), pers.round(3).to_string(), oth.round(2).to_string(), cp.round(3).to_string(),
              sep="\n")
    if "watch" in parts:
        res["miss_delay_buckets"], res["too_few_concentration"] = watch_tables(con, ds, cfg)
        print(res["miss_delay_buckets"].to_string())
    if "bins" in parts:
        res["loop_detector_bins"] = loop_detector_bins(loops)
        print(res["loop_detector_bins"].describe().round(2).to_string())
    if "policy" in parts:
        dw = tables.build_device_weeks(con, cfg)
        res["robust_policy"] = policy_check(ds, dw, loops, loop_ids, cfg, tab)
        print(res["robust_policy"].round(3).to_string(), flush=True)
    if "coupling" in parts:
        pooled, per_scale, dose = coupling(con, loop_ids, cfg)
        res["robust_coupling"], res["coupling_per_scale"], res["coupling_dose"] = pooled, per_scale, dose
        print(pooled.round(3).to_string(), per_scale.round(3).to_string(), dose.round(3).to_string(), sep="\n")
    n_ep = None
    if "examples" in parts:
        ep, n_ep = example_episode(con, cfg)
        res["example_coupling_episode"] = ep
        res["example_loop_day"] = example_loop(con, loops, cfg)
        print(ep.to_string(), n_ep, res["example_loop_day"].to_string(), sep="\n")
    for k, v in res.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    mp = OUT / "robustness_meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    meta.update({"loop_scales_excluded": len(loop_ids), "total_automatic": int(total_auto)})
    if n_ep is not None:
        meta["example_episode_candidates_in_5pct_sample"] = int(n_ep)
    mp.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
