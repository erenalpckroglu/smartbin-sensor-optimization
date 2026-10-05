"""Loading: DuckDB for the 7.5M-row measurement file, pandas for the two small device files."""
from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd

log = logging.getLogger("smartbin")

MEASUREMENT_TYPES = {
    "deviceid": "VARCHAR", "weighttime": "TIMESTAMP", "minwaittime": "DOUBLE", "sensitivity": "DOUBLE",
    "delay": "DOUBLE", "weight": "DOUBLE", "initiatortype": "VARCHAR", "weightclassification": "VARCHAR",
    "weight_bin": "VARCHAR", "section_id": "VARCHAR", "is_relevant_section": "BOOLEAN",
    "relevant_for_weight_analysis": "BOOLEAN",
}
WEIGHT_BIN_ORDER = ["0 - 803", "803 - 1877", "1877 - 3859", "3859 - 6673", "6673 - 100000"]


def connect(cfg) -> duckdb.DuckDBPyConnection:
    db = cfg.paths.interim / "smartbin.duckdb"
    for f in (db, db.with_suffix(".duckdb.wal")):
        if f.exists():
            f.unlink()
    con = duckdb.connect(str(db))   # on-disk database: 7.5M rows never have to fit in RAM
    tmp = cfg.paths.interim / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    if getattr(cfg.runtime, "duckdb_memory_limit", None):
        con.execute(f"SET memory_limit='{cfg.runtime.duckdb_memory_limit}'")
    if getattr(cfg.runtime, "threads", None):
        con.execute(f"SET threads={int(cfg.runtime.threads)}")
    con.execute("SET preserve_insertion_order=false")
    return con


def load_measurements(con, path) -> int:
    types = ", ".join(f"'{k}': '{v}'" for k, v in MEASUREMENT_TYPES.items())
    con.execute(f"""
        CREATE OR REPLACE TABLE raw AS
        SELECT row_number() OVER () AS rid, *
        FROM read_csv('{path.as_posix()}', header=true, types={{{types}}})
    """)
    cols = {r[0] for r in con.execute("DESCRIBE raw").fetchall()}
    missing = set(MEASUREMENT_TYPES) - cols
    if missing:
        raise ValueError(f"measurements file is missing columns: {sorted(missing)}")
    n = con.execute("SELECT count(*) FROM raw").fetchone()[0]
    log.info("loaded %s measurement rows", f"{n:,}")
    return n


def load_metadata(path) -> pd.DataFrame:
    """One row per device. Normalises IDs (LOCATIONID arrives as float text '1234567.0')."""
    m = pd.read_csv(path, dtype=str)
    m.columns = [c.strip() for c in m.columns]
    out = pd.DataFrame({"deviceid": m["deviceid"].str.strip()})
    kg = m["SCALE_NAME"].str.extract(r"(\d+)\s*kg", flags=2)[0]
    out["scale_type"] = np.where(kg.notna(), kg + "kg", None)
    out["box_name"] = m["BOX_NAME"].str.strip()
    loc = pd.to_numeric(m["LOCATIONID"], errors="coerce").round().astype("Int64")
    out["location_id"] = loc.astype(str).where(loc.notna(), None)
    out["customer"] = m["CUSTOMER_NUMBER"].str.strip()
    out["item_weight"] = pd.to_numeric(m["ITEM_WEIGHT"], errors="coerce")
    out["is_reserve"] = m["DESCRIPTION"].str.contains("reserve", case=False, na=False)
    # readable, anonymous location labels: L01 = largest location
    sizes = out.dropna(subset=["location_id"]).location_id.value_counts()
    labels = {loc_id: f"L{i + 1:03d}" for i, loc_id in enumerate(sizes.index)}
    out["location_label"] = out.location_id.map(labels)
    out["location_size"] = out.location_id.map(sizes)
    return out.drop_duplicates("deviceid")


def load_preprocessing(path) -> pd.DataFrame:
    p = pd.read_csv(path)
    for c in ["device_changes_sensitivity", "device_changes_weight_bin", "device_has_relevant_sections"]:
        p[c] = p[c].astype(str).str.lower().map({"true": True, "false": False})
    return p.drop_duplicates("deviceid")
