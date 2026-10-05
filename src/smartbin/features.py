"""
Feature layer: everything the analyses need, computed ONCE on the full event chain.

Order matters: weight deltas are computed BEFORE analysis filters, otherwise removing
one row (e.g. unknown sensitivity) would glue two unrelated weighings together.
"""
from __future__ import annotations

import logging

log = logging.getLogger("smartbin")


def build_events(con, meta, pre) -> None:
    con.register("meta_df", meta)
    con.register("pre_df", pre)
    con.execute("CREATE OR REPLACE TABLE meta AS SELECT * FROM meta_df")
    con.execute("CREATE OR REPLACE TABLE pre AS SELECT * FROM pre_df")
    con.execute("""
        CREATE OR REPLACE TABLE events_pre AS
        WITH b AS (
            SELECT *, epoch_ms(weighttime) / 1000.0 AS ts FROM base
        ), w AS (
            SELECT *,
                   lag(ts)                   OVER d AS prev_ts,
                   lag(weight)               OVER d AS prev_weight,
                   lag(initiatortype)        OVER d AS prev_initiator,
                   lag(weightclassification) OVER d AS prev_cls,
                   lead(ts)                  OVER d AS next_ts
            FROM b
            WINDOW d AS (PARTITION BY deviceid ORDER BY weighttime, rid)
        )
        SELECT w.*,
               weight - prev_weight                AS dw,          -- signed weight change since previous weighing
               ts - prev_ts                        AS gap_s,       -- seconds since previous weighing (any type)
               initiatortype = 'AUTOMATIC'         AS is_auto,
               weightclassification                AS cls,
               hour(weighttime)                    AS hour,
               isodow(weighttime)                  AS isodow,      -- 1 = Monday ... 7 = Sunday
               date_trunc('week', weighttime)      AS week,
               CAST(weighttime AS DATE)            AS day,
               m.scale_type, m.box_name, m.location_id, m.location_label, m.location_size,
               m.customer, m.item_weight, m.is_reserve,
               p.device_changes_sensitivity, p.device_changes_weight_bin, p.device_has_relevant_sections,
               (minwaittime IS NULL OR delay IS NULL) AS config_unknown
        FROM w
        LEFT JOIN meta m USING (deviceid)
        LEFT JOIN pre  p USING (deviceid)
    """)
    n = con.execute("SELECT count(*) FROM events_pre").fetchone()[0]
    log.info("feature table built: %s rows", f"{n:,}")


def is_sample(con, cfg) -> bool:
    """A random-row sample breaks event chains (deltas, crosstalk). Detect it."""
    mode = cfg.runtime.sample_mode
    if isinstance(mode, bool):
        return mode
    if str(mode).lower() in ("true", "false"):
        return str(mode).lower() == "true"
    n, med_gap_h = con.execute("""
        SELECT count(*), median(gap_s) / 3600 FROM events WHERE gap_s IS NOT NULL
    """).fetchone()
    sample = n < 1_000_000 and (med_gap_h or 0) > 12
    if sample:
        log.warning("SAMPLE MODE: %s rows, median gap between weighings %.1f h -> rows are a random sample. "
                    "Chain-based modules (pick size from weight deltas, crosstalk) are skipped; "
                    "rates per week are deflated by the sampling fraction.", f"{n:,}", med_gap_h or 0)
    return sample
