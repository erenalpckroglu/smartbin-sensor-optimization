"""
Data cleaning as an explicit, ordered rule book.

Two stages, because "delete" and "exclude from the KPI analysis" are different things:

Stage A - ROW VALIDITY (rows are deleted)
    A row that is technically broken can't be used anywhere.

Stage B - ANALYSIS ELIGIBILITY (rows are kept but flagged with the FIRST rule they fail)
    A row with unknown sensitivity is useless for "what does sensitivity do", but it is
    still a real weighing: it is needed to compute the weight change of the NEXT row,
    and it is a real pick that can shake a neighbour (crosstalk). Deleting it would
    silently break both analyses. So Stage B only flags.

Every rule logs rows/devices removed -> reports/tables/cleaning_funnel.csv
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("smartbin")


def _sql_list(values) -> str:
    return ", ".join(str(float(v)) for v in values)


def stage_a_rules(cfg):
    end = f"OR weighttime > TIMESTAMP '{cfg.cleaning.end_date}'" if cfg.cleaning.end_date else ""
    return [
        ("A1_mandatory_missing", "Mandatory",
         "deviceid IS NULL OR weighttime IS NULL OR weight IS NULL OR initiatortype IS NULL",
         "No device, time, weight or trigger type = not a usable measurement"),
        ("A2_duplicate", "Mandatory", "dup_rank > 1",
         "Same device + same millisecond twice = logging duplicate; keep one copy "
         "(copies with different labels are flagged by B0)"),
        ("A3_initiator_invalid", "Mandatory", "initiatortype NOT IN ('AUTOMATIC', 'MANUAL')",
         "Trigger type must be AUTOMATIC (vibration) or MANUAL (person / 24h)"),
        ("A4_out_of_window", "Mandatory",
         f"weighttime < TIMESTAMP '{cfg.cleaning.start_date}' {end}",
         "D ONE: data before March 2022 used sub-optimal settings"),
        ("A5_weight_invalid", "Mandatory", f"weight < 0 OR weight > {cfg.cleaning.max_weight_g}",
         "Physically impossible weights (scale error)"),
    ]


def stage_b_rules(cfg):
    return [
        ("B0_duplicate_label_conflict", "Mandatory", "dup_label_conflict",
         "Same device + same millisecond with two different labels (e.g. OK and TOO_FEW): outcome is "
         "ambiguous; the kept copy stays in the weight chain but not in the KPI"),
        ("B1_label_missing", "Mandatory", "weightclassification IS NULL",
         "No OK / TOO_MANY / TOO_FEW label = the outcome is unknown"),
        ("B2_label_inconsistent", "Mandatory",
         "weightclassification NOT IN ('OK','TOO_MANY','TOO_FEW') "
         "OR (initiatortype='AUTOMATIC' AND weightclassification='TOO_FEW') "
         "OR (initiatortype='MANUAL' AND weightclassification='TOO_MANY')",
         "TOO_FEW can only come from a MANUAL check, TOO_MANY only from an AUTOMATIC trigger"),
        ("B3_sensitivity_missing", "Mandatory", "sensitivity IS NULL",
         "The setting we want to evaluate is unknown (row kept for weight deltas & crosstalk)"),
        ("B4_trigger_config_nonstandard", "Analysis",
         f"(minwaittime IS NOT NULL AND minwaittime NOT IN ({_sql_list(cfg.cleaning.standard_minwaittime)})) "
         f"OR (delay IS NOT NULL AND delay NOT IN ({_sql_list(cfg.cleaning.standard_delay)}))",
         "Min-wait / delay also change triggering; keep one standard configuration so they can't confound"),
        ("B5_device_metadata_missing", "Analysis", "scale_type IS NULL",
         "Scale type / box / location unknown -> cannot be assigned to the 6kg or 20kg analysis"),
        ("B6_section_not_relevant", "Analysis", "is_relevant_section IS DISTINCT FROM TRUE",
         "Section with < 20 measurements or < 5 days -> KPI per week would be noise"),
        ("B7_sensitivity_anomaly", "Separate",
         f"sensitivity IN ({_sql_list(cfg.cleaning.anomaly_levels)})",
         "Lecture: ignore sensitivity 3 - behaves unlike its neighbours; studied separately, not deleted"),
        ("B8_sensitivity_low_support", "Analysis", None,
         f"Level used by < {cfg.cleaning.min_devices_per_level} devices of that scale type -> no stable estimate"),
    ]


def apply_stage_a(con, cfg) -> pd.DataFrame:
    rules = stage_a_rules(cfg)
    case = " ".join(f"WHEN {cond} THEN '{rid}'" for rid, _, cond, _ in rules)
    con.execute(f"""
        CREATE OR REPLACE TABLE raw_flagged AS
        SELECT *, CASE {case} ELSE NULL END AS a_reason
        FROM (SELECT *,
                     row_number() OVER (PARTITION BY deviceid, weighttime ORDER BY rid) AS dup_rank,
                     min(coalesce(weightclassification, '-')) OVER k
                       <> max(coalesce(weightclassification, '-')) OVER k AS dup_label_conflict
              FROM raw WINDOW k AS (PARTITION BY deviceid, weighttime))
    """)
    con.execute("CREATE OR REPLACE TABLE base AS SELECT * EXCLUDE (a_reason, dup_rank) "
                "FROM raw_flagged WHERE a_reason IS NULL")
    total = con.execute("SELECT count(*), count(DISTINCT deviceid) FROM raw").fetchone()
    counts = dict(con.execute("SELECT a_reason, count(*) FROM raw_flagged GROUP BY 1").fetchall())
    rows = [dict(step=0, rule="raw file", tier="", action="", rows_removed=0, rows_remaining=total[0],
                 devices_remaining=total[1], rationale="measurements.csv as delivered")]
    remaining = total[0]
    for i, (rid, tier, _, why) in enumerate(rules, start=1):
        removed = counts.get(rid, 0)
        remaining -= removed
        dev = con.execute(f"""SELECT count(DISTINCT deviceid) FROM raw_flagged
                              WHERE a_reason IS NULL OR a_reason > '{rid}'""").fetchone()[0]
        rows.append(dict(step=i, rule=rid, tier=tier, action="delete row", rows_removed=removed,
                         rows_remaining=remaining, devices_remaining=dev, rationale=why))
    # keep what Stage A removed, so it can be inspected (reports -> data/processed/filtered_out/)
    con.execute("""
        CREATE OR REPLACE TABLE stage_a_removed AS
        SELECT * EXCLUDE (dup_rank) FROM raw_flagged WHERE a_reason IS NOT NULL AND a_reason <> 'A2_duplicate'
    """)
    con.execute("""
        CREATE OR REPLACE TABLE duplicate_groups AS
        WITH g AS (
            SELECT deviceid, weighttime, count(*) AS copies,
                   count(DISTINCT (minwaittime, sensitivity, delay, weight, initiatortype, weightclassification,
                                   weight_bin, section_id, is_relevant_section, relevant_for_weight_analysis)) AS variants
            FROM raw_flagged GROUP BY 1, 2 HAVING count(*) > 1)
        SELECT r.* EXCLUDE (dup_rank, a_reason), g.copies,
               CASE WHEN g.variants = 1 THEN 'identical copy' ELSE 'conflicting copy' END AS duplicate_type,
               r.dup_rank = 1 AS kept
        FROM raw_flagged r JOIN g USING (deviceid, weighttime)
    """)
    con.execute("DROP TABLE raw_flagged")
    return pd.DataFrame(rows)


ORIGINAL_COLUMNS = ("deviceid, weighttime, minwaittime, sensitivity, delay, weight, initiatortype, "
                    "weightclassification, weight_bin, section_id, is_relevant_section, relevant_for_weight_analysis")


def export_filtered_out(con, cfg, funnel: pd.DataFrame, sample_per_reason: int = 200) -> None:
    """Everything that did NOT enter the main KPI analysis, in one folder, with the reason per row.

    deleted_stage_a.csv              rows deleted as technically broken (not duplicates)
    duplicates_all_copies.csv        every copy of every duplicate, incl. the one that was kept
    excluded_stage_b.parquet         all flagged rows (kept in the event chain, not in the KPI)
    excluded_sample_by_reason.csv    up to N rows per reason - open in Excel to eyeball them
    filtered_out_summary.csv         the funnel: rule, rows, devices, why
    """
    out = cfg.paths.processed / "filtered_out"
    out.mkdir(parents=True, exist_ok=True)
    q = lambda sql, f, fmt="HEADER": con.execute(f"COPY ({sql}) TO '{(out / f).as_posix()}' ({fmt})")
    q("SELECT * EXCLUDE (rid) FROM stage_a_removed ORDER BY deviceid, weighttime", "deleted_stage_a.csv")
    q("SELECT * EXCLUDE (rid) FROM duplicate_groups ORDER BY deviceid, weighttime, kept DESC",
      "duplicates_all_copies.csv")
    cols = f"{ORIGINAL_COLUMNS}, scale_type, location_label, exclusion_reason"
    q(f"SELECT {cols} FROM events WHERE exclusion_reason IS NOT NULL ORDER BY exclusion_reason, deviceid, weighttime",
      "excluded_stage_b.parquet", "FORMAT parquet")
    q(f"""SELECT {cols} FROM (
              SELECT *, row_number() OVER (PARTITION BY exclusion_reason ORDER BY hash(rid)) AS k
              FROM events WHERE exclusion_reason IS NOT NULL)
          WHERE k <= {int(sample_per_reason)} ORDER BY exclusion_reason, deviceid, weighttime""",
      "excluded_sample_by_reason.csv")
    funnel.to_csv(out / "filtered_out_summary.csv", index=False)
    con.execute("DROP TABLE stage_a_removed; DROP TABLE duplicate_groups")


def apply_stage_b(con, cfg) -> pd.DataFrame:
    """Adds `exclusion_reason` to table `events` (first failed rule)."""
    rules = stage_b_rules(cfg)
    case = " ".join(f"WHEN {cond} THEN '{rid}'" for rid, _, cond, _ in rules if cond)
    con.execute(f"""CREATE OR REPLACE TABLE events_b AS
                    SELECT *, CASE {case} ELSE NULL END AS exclusion_reason FROM events_pre""")
    # B8: support is judged on rows that survived B1-B7
    con.execute("""
        CREATE OR REPLACE TABLE level_support AS
        SELECT scale_type, sensitivity, count(DISTINCT deviceid) AS n_devices, count(*) AS n_rows
        FROM events_b WHERE exclusion_reason IS NULL GROUP BY 1, 2
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE events AS
        SELECT e.* REPLACE (
            CASE WHEN e.exclusion_reason IS NULL AND ls.n_devices < {int(cfg.cleaning.min_devices_per_level)}
                 THEN 'B8_sensitivity_low_support' ELSE e.exclusion_reason END AS exclusion_reason)
        FROM events_b e LEFT JOIN level_support ls
          ON e.scale_type = ls.scale_type AND e.sensitivity = ls.sensitivity
    """)
    # D ONE-style flag column next to the reason: TRUE = this weighing enters the KPI analysis
    con.execute("CREATE OR REPLACE TABLE events AS SELECT *, exclusion_reason IS NULL AS in_kpi_analysis FROM events")
    con.execute("DROP TABLE events_b; DROP TABLE events_pre")
    start = con.execute("SELECT count(*), count(DISTINCT deviceid) FROM events").fetchone()
    counts = dict(con.execute("SELECT exclusion_reason, count(*) FROM events GROUP BY 1").fetchall())
    rows, remaining = [], start[0]
    for i, (rid, tier, _, why) in enumerate(rules, start=1):
        removed = counts.get(rid, 0)
        remaining -= removed
        dev = con.execute(f"""SELECT count(DISTINCT deviceid) FROM events
                              WHERE exclusion_reason IS NULL OR exclusion_reason > '{rid}'""").fetchone()[0]
        action = "separate study" if tier == "Separate" else "exclude from KPI analysis (kept for deltas)"
        rows.append(dict(step=10 + i, rule=rid, tier=tier, action=action, rows_removed=removed,
                         rows_remaining=remaining, devices_remaining=dev, rationale=why))
    return pd.DataFrame(rows)


COLUMN_DECISIONS = [
    # column, file, tier, decision, why
    ("deviceid", "all three", "Key", "keep", "Join key between the three files"),
    ("weighttime", "measurements", "Mandatory", "keep + derive", "Rates per week, weight deltas, crosstalk timing, before/after"),
    ("weightclassification", "measurements", "Mandatory", "keep", "THE outcome: OK / TOO_MANY / TOO_FEW"),
    ("initiatortype", "measurements", "Mandatory", "keep", "Automatic (sensor) vs manual (person / 24h check)"),
    ("sensitivity", "measurements", "Mandatory", "keep", "The control knob we optimise"),
    ("weight", "measurements", "Analysis", "derive", "Only differences are used: pick size = weight change"),
    ("weight_bin", "measurements", "Analysis", "keep", "Box fill level - one of the tested parameters"),
    ("section_id", "measurements", "Analysis", "keep", "Defines the analysis unit (stable period)"),
    ("is_relevant_section", "measurements", "Filter", "use as filter", "Keeps only periods long enough for a KPI"),
    ("relevant_for_weight_analysis", "measurements", "Filter", "use as filter", "Weight-effect subset (lecture hint) - outcome-selected, robustness check added"),
    ("minwaittime", "measurements", "Config", "filter, then drop", "Out of scope, but non-standard values are filtered so they can't confound"),
    ("delay", "measurements", "Config", "filter, then drop", "Same as min-wait; also aligns timing for crosstalk (30 s)"),
    ("SCALE_NAME", "scale_metadata", "Analysis", "keep -> scale_type", "6kg vs 20kg: separate analyses / dashboards"),
    ("BOX_NAME", "scale_metadata", "Analysis", "keep", "Tested parameter; nested in scale type -> compared within scale only"),
    ("LOCATIONID", "scale_metadata", "Analysis", "keep -> string label", "Factory/site level; carries environment + rack layout"),
    ("ITEM_WEIGHT", "scale_metadata", "Analysis", "keep", "Tested parameter; proxy for pick size"),
    ("CUSTOMER_NUMBER", "scale_metadata", "Dropped", "drop", "Every location belongs to exactly one customer -> redundant"),
    ("DESCRIPTION", "scale_metadata", "Dropped", "drop (keep 'reserve' flag)", "Free text in 5 languages; only 'Reserve' carries signal"),
    ("DIMENSION", "scale_metadata", "Dropped", "drop", "Free-text size (e.g. 'M 3 x 5'), 23% empty; item weight already captures size"),
    ("SUPPLIER_ITEM_NUMBER", "scale_metadata", "Dropped", "drop", "Product code; 10k levels, no physical meaning for the sensor"),
    ("device_has_relevant_sections", "scale_preprocessing", "Filter", "verify, then use", "Device usable at all?"),
    ("device_changes_sensitivity", "scale_preprocessing", "Filter", "verify, then use", "Device usable for before/after (same device, two settings)"),
    ("device_changes_weight_bin", "scale_preprocessing", "Filter", "verify, then use", "Device usable for the weight-in-box analysis"),
]


def column_decisions() -> pd.DataFrame:
    return pd.DataFrame(COLUMN_DECISIONS, columns=["column", "file", "tier", "decision", "why"])
