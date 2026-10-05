"""
Analysis tables + trust-but-verify checks on D ONE's pre-computed fields.

Units of analysis
-----------------
section        = one device, constant sensitivity, constant weight bin, consecutive in time
                 ("one uninterrupted experiment run"). Used for the weight-in-box analysis.
device-setting = one device at one sensitivity, all its relevant sections pooled
                 (lecture hint: "groups of measurements with constant deviceid & sensitivity").
                 Main unit: every device counts once per setting, chatty devices can't dominate.
device-week    = one device, one calendar week, one sensitivity. Used for before/after timelines.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .kpi import add_kpis

log = logging.getLogger("smartbin")

# rows that enter KPI tables: clean rows + the two "separate/low support" groups (flagged)
KPI_ROWS = ("(exclusion_reason IS NULL OR exclusion_reason IN "
            "('B7_sensitivity_anomaly', 'B8_sensitivity_low_support'))")

COUNTS = """
    count(*)                                              AS n_rows,
    sum(CASE WHEN is_auto THEN 1 ELSE 0 END)              AS n_auto,
    sum(CASE WHEN NOT is_auto THEN 1 ELSE 0 END)          AS n_manual,
    sum(CASE WHEN cls = 'TOO_MANY' THEN 1 ELSE 0 END)     AS n_tm,
    sum(CASE WHEN cls = 'TOO_FEW' THEN 1 ELSE 0 END)      AS n_tf,
    sum(CASE WHEN is_auto AND cls = 'OK' THEN 1 ELSE 0 END)     AS n_aok,
    sum(CASE WHEN NOT is_auto AND cls = 'OK' THEN 1 ELSE 0 END) AS n_mok
"""


def verify_dataset(con, cfg) -> pd.DataFrame:
    """Checks we can make ourselves instead of trusting the delivered flags blindly."""
    checks = []

    def add(name, value, expectation, ok, note=""):
        checks.append(dict(check=name, value=value, expectation=expectation, status="PASS" if ok else "WARN",
                           note=note))

    v = con.execute("""
        SELECT count(*) FILTER (WHERE n_dev > 1), count(*) FILTER (WHERE n_sens > 1),
               count(*) FILTER (WHERE n_bin > 1), count(*) FILTER (WHERE n_rel > 1), count(*)
        FROM (SELECT section_id, count(DISTINCT deviceid) n_dev, count(DISTINCT sensitivity) n_sens,
                     count(DISTINCT weight_bin) n_bin, count(DISTINCT is_relevant_section) n_rel
              FROM events WHERE section_id IS NOT NULL GROUP BY 1)
    """).fetchone()
    add("section_id = one device / sensitivity / weight bin", f"{v[0] + v[1] + v[2]} violations of {v[4]:,}",
        "0", v[0] + v[1] + v[2] == 0)
    add("is_relevant_section constant within section", f"{v[3]} violations", "0", v[3] == 0)

    r = con.execute("""
        SELECT avg(CASE WHEN flag = recomputed THEN 1.0 ELSE 0 END), count(*)
        FROM (SELECT section_id, any_value(is_relevant_section) AS flag,
                     (count(*) >= 20 AND (max(ts) - min(ts)) >= 5 * 86400) AS recomputed
              FROM events WHERE section_id IS NOT NULL GROUP BY 1)
    """).fetchone()
    add("is_relevant_section = (>=20 meas AND >=5 days), recomputed", f"{(r[0] or 0):.1%} agreement",
        ">= 95%", (r[0] or 0) >= 0.95,
        "Low agreement on a random-row sample is expected (rows were thinned).")

    c = con.execute("""
        SELECT count(*) FILTER (WHERE initiatortype='AUTOMATIC' AND cls='TOO_FEW'),
               count(*) FILTER (WHERE initiatortype='MANUAL' AND cls='TOO_MANY') FROM events
    """).fetchone()
    add("label logic (TOO_FEW only MANUAL, TOO_MANY only AUTOMATIC)", f"{c[0] + c[1]} violations", "0",
        c[0] + c[1] == 0)

    p = con.execute("""
        WITH rel AS (SELECT deviceid, count(DISTINCT sensitivity) ns, count(DISTINCT weight_bin) nb
                     FROM events WHERE is_relevant_section GROUP BY 1)
        SELECT avg(CASE WHEN coalesce(p.device_has_relevant_sections, false) = (rel.deviceid IS NOT NULL)
                        THEN 1.0 ELSE 0 END),
               avg(CASE WHEN rel.deviceid IS NULL OR p.device_changes_sensitivity = (rel.ns > 1)
                        THEN 1.0 ELSE 0 END),
               avg(CASE WHEN rel.deviceid IS NULL OR p.device_changes_weight_bin = (rel.nb > 1)
                        THEN 1.0 ELSE 0 END)
        FROM pre p LEFT JOIN rel USING (deviceid)
        WHERE p.deviceid IN (SELECT DISTINCT deviceid FROM events)
    """).fetchone()
    for name, val in zip(["device_has_relevant_sections", "device_changes_sensitivity",
                          "device_changes_weight_bin"], p):
        add(f"preprocessing flag {name}, recomputed", f"{(val or 0):.1%} agreement", ">= 95%",
            (val or 0) >= 0.95, "Sample files under-count changes (rows thinned).")

    t = con.execute(f"""
        SELECT count(*) FILTER (WHERE sensitivity = {cfg.analysis.baseline_sensitivity}) * 1.0 /
               nullif(count(*) FILTER (WHERE sensitivity IS NOT NULL), 0)
        FROM events WHERE weighttime < TIMESTAMP '{cfg.analysis.baseline_end}'
    """).fetchone()[0]
    add(f"baseline: share of rows on sensitivity {cfg.analysis.baseline_sensitivity} before "
        f"{cfg.analysis.baseline_end}", f"{(t or 0):.1%}", ">= 95% (natural experiment design)",
        (t or 0) >= 0.95)

    m = con.execute("""SELECT avg(CASE WHEN scale_type IS NULL THEN 1.0 ELSE 0 END) FROM
                       (SELECT DISTINCT deviceid, scale_type FROM events)""").fetchone()[0]
    add("devices without metadata", f"{(m or 0):.1%}", "< 10%", (m or 0) < 0.10)
    return pd.DataFrame(checks)


def build_sections(con, cfg) -> pd.DataFrame:
    con.execute(f"""
        CREATE OR REPLACE TABLE sections AS
        SELECT deviceid, section_id,
               any_value(sensitivity) AS sensitivity, any_value(weight_bin) AS weight_bin,
               any_value(scale_type) AS scale_type, any_value(box_name) AS box_name,
               any_value(location_id) AS location_id, any_value(location_label) AS location_label,
               any_value(location_size) AS location_size, any_value(item_weight) AS item_weight,
               any_value(is_reserve) AS is_reserve,
               bool_or(relevant_for_weight_analysis) AS rel_weight,
               bool_or(config_unknown) AS config_unknown,
               bool_and(exclusion_reason IS NULL) AS main_analysis,
               any_value(exclusion_reason) AS exclusion_reason,
               min(ts) AS t_start, max(ts) AS t_end,
               {COUNTS}
        FROM events
        WHERE {KPI_ROWS} AND section_id IS NOT NULL
        GROUP BY deviceid, section_id
    """)
    s = con.execute("SELECT * FROM sections").df()
    s["days"] = ((s.t_end - s.t_start) / 86400).clip(lower=1)
    return add_kpis(s, cfg)


def device_roles(con, cfg) -> pd.Series:
    """stayer = every logged sensitivity is the baseline; switcher = baseline + another main level."""
    base = float(cfg.analysis.baseline_sensitivity)
    anomaly = {float(x) for x in cfg.cleaning.anomaly_levels}
    lv = con.execute("""SELECT deviceid, list(DISTINCT sensitivity) AS all_levels FROM events
                        WHERE sensitivity IS NOT NULL GROUP BY 1""").df()
    main = con.execute("""SELECT deviceid, list(DISTINCT sensitivity) AS main_levels FROM events
                          WHERE exclusion_reason IS NULL GROUP BY 1""").df()
    lv = lv.merge(main, on="deviceid", how="left")

    def role(r):
        a = set(r.all_levels)
        m = set(r.main_levels) if isinstance(r.main_levels, (list, tuple, np.ndarray)) else set()
        if a == {base}:
            return "stayer"
        if base in m and len(m) > 1:
            return "switcher"
        if a & anomaly:
            return "anomaly"
        return "other"

    return lv.set_index("deviceid").apply(role, axis=1)


def build_device_settings(sections: pd.DataFrame, roles: pd.Series, cfg) -> pd.DataFrame:
    keys = ["deviceid", "sensitivity"]
    static = ["scale_type", "box_name", "location_id", "location_label", "location_size", "item_weight",
              "is_reserve"]
    agg = {c: "sum" for c in ["n_rows", "n_auto", "n_manual", "n_tm", "n_tf", "n_aok", "n_mok", "days"]}
    agg.update({c: "first" for c in static})
    agg.update({"t_start": "min", "t_end": "max", "section_id": "count", "main_analysis": "all",
                "config_unknown": "any"})
    ds = sections.groupby(keys, as_index=False).agg(agg).rename(columns={"section_id": "n_sections"})
    ds["device_role"] = ds.deviceid.map(roles).fillna("other")
    return add_kpis(ds, cfg)


def build_device_weeks(con, cfg) -> pd.DataFrame:
    return con.execute(f"""
        SELECT deviceid, week, sensitivity, any_value(scale_type) AS scale_type,
               any_value(location_id) AS location_id, count(DISTINCT day) AS active_days, {COUNTS}
        FROM events WHERE exclusion_reason IS NULL
        GROUP BY deviceid, week, sensitivity
    """).df()


def sensitivity_by_month(con) -> pd.DataFrame:
    return con.execute("""
        SELECT strftime(date_trunc('month', weighttime), '%Y-%m') AS month, sensitivity,
               count(DISTINCT deviceid) AS devices, count(*) AS rows
        FROM events WHERE sensitivity IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
