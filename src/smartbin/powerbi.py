"""
Power BI pack: small, analysis-ready tables for a self-built Power BI report.

Why not load measurements.csv into Power BI directly?
  * 7.6M rows make every visual slow, and
  * a ratio such as "TOO_MANY / automatic" must be computed PER SCALE first and only then
    summarised (median over scales). Summing raw rows lets the noisiest scales dominate
    (sensitivity 3 fires ~150x a day and would swamp every pooled number).

So the pipeline does the per-scale arithmetic once, and Power BI only filters and summarises.

reports/<run>/powerbi/
    fact_device_sensitivity.csv   one row = one scale on one sensitivity level (grain of every KPI page)
    cleaning_funnel.csv           what was removed or flagged, in order, and why
    README.md                     column dictionary
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BANDS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0000001]
BAND_LABELS = [f"{BANDS[i]:.1f}-{min(BANDS[i + 1], 1):.1f}" for i in range(len(BANDS) - 1)]

COLUMNS = [
    # name in the CSV, meaning
    ("device_id", "Scale ID (pseudonymous)"),
    ("scale_type", "6kg or 20kg"),
    ("sensitivity", "Sensitivity level the scale was on (1 = least sensitive, 8 = default)"),
    ("analysis_group", "main = enters the KPI analysis; sens 3 (anomaly) = studied separately; "
                       "low support = level used by < 30 scales"),
    ("device_role", "stayer = always on 8; switcher = moved from 8 to another level; anomaly = was on 3"),
    ("location", "Anonymous site label (L001 = largest site)"),
    ("box_type", "Box on the scale"),
    ("item_weight", "Weight of one item in the box (unit as delivered in scale_metadata.csv)"),
    ("period_start", "First weighing of this scale on this level"),
    ("period_end", "Last weighing of this scale on this level"),
    ("days_observed", "Days covered (sum over the scale's relevant sections on this level)"),
    ("weeks_observed", "days_observed / 7"),
    ("n_measurements", "All weighings"),
    ("n_auto", "Automatic weighings (vibration sensor woke up)"),
    ("n_too_many", "Automatic weighings without a weight change (empty wake-ups)"),
    ("n_auto_ok", "Automatic weighings with a real weight change"),
    ("n_manual", "Manual / 24 h check weighings"),
    ("n_too_few", "Manual / 24 h checks that found a change the sensor had missed"),
    ("n_manual_ok", "Manual / 24 h checks without a change"),
    ("too_many_fraction", "n_too_many / n_auto  (Bossard KPI, target <= 0.5)"),
    ("too_many_fraction_band", "too_many_fraction in steps of 0.1 (for distribution charts)"),
    ("too_few_per_week", "n_too_few / weeks_observed  (Bossard KPI, target <= 2)"),
    ("too_many_per_week", "n_too_many / weeks_observed  (absolute empty wake-ups)"),
    ("measurements_per_day", "n_measurements / days_observed  (battery / cloud load)"),
    ("miss_rate", "n_too_few / (n_auto_ok + n_too_few): share of real changes the sensor slept through"),
    ("meets_too_many_target", "too_many_fraction <= 0.5"),
    ("meets_too_few_target", "too_few_per_week <= 2"),
    ("in_target", "both Bossard targets met"),
]


def _band(x: pd.Series) -> pd.Series:
    return pd.cut(x.clip(0, 1), BANDS, labels=BAND_LABELS, right=False).astype(str).replace("nan", "")


def fact_device_sensitivity(ds: pd.DataFrame) -> pd.DataFrame:
    d = ds.copy()
    group = np.where(d.main_analysis, "main", np.where(d.sensitivity == 3, "sens 3 (anomaly)", "low support"))
    out = pd.DataFrame({
        "device_id": d.deviceid,
        "scale_type": d.scale_type,
        "sensitivity": d.sensitivity.round().astype("Int64"),
        "analysis_group": group,
        "device_role": d.device_role,
        "location": d.location_label,
        "box_type": d.box_name,
        "item_weight": d.item_weight,
        "period_start": pd.to_datetime(d.t_start, unit="s").dt.strftime("%Y-%m-%d"),
        "period_end": pd.to_datetime(d.t_end, unit="s").dt.strftime("%Y-%m-%d"),
        "days_observed": d.days,
        "weeks_observed": d.days / 7,
        "n_measurements": d.n_rows,
        "n_auto": d.n_auto,
        "n_too_many": d.n_tm,
        "n_auto_ok": d.n_aok,
        "n_manual": d.n_manual,
        "n_too_few": d.n_tf,
        "n_manual_ok": d.n_mok,
        "too_many_fraction": d.frac_too_many,
        "too_many_fraction_band": _band(d.frac_too_many),
        "too_few_per_week": d.too_few_per_week,
        "too_many_per_week": d.too_many_per_week,
        "measurements_per_day": d.meas_per_day,
        "miss_rate": d.miss_rate,
        "meets_too_many_target": d.meets_tm,
        "meets_too_few_target": d.meets_tf,
        "in_target": d.in_target,
    })
    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(4)
    for c in [c for c in out.columns if c.startswith("n_")]:
        out[c] = out[c].round().astype("Int64")
    return out.sort_values(["scale_type", "sensitivity", "device_id"]).reset_index(drop=True)


def export_powerbi(ds: pd.DataFrame, funnel: pd.DataFrame, cfg) -> Path:
    out = cfg.paths.reports / "powerbi"
    out.mkdir(parents=True, exist_ok=True)
    fact_device_sensitivity(ds).to_csv(out / "fact_device_sensitivity.csv", index=False)
    f = funnel.copy()
    f.insert(0, "stage", np.where(f.step < 10, "A - delete broken rows", "B - flag (kept for weight changes)"))
    f.to_csv(out / "cleaning_funnel.csv", index=False)
    lines = ["# Power BI pack", "",
             "Import with *File > Options > Current file > Regional settings > Locale for import = "
             "English (United States)* so that `0.92` is read as a decimal.", "",
             "## fact_device_sensitivity.csv", "One row = one scale on one sensitivity level.", "",
             "| column | meaning |", "|---|---|"]
    lines += [f"| `{c}` | {m} |" for c, m in COLUMNS]
    lines += ["", "## cleaning_funnel.csv", "Every cleaning rule in the order it is applied: rows removed / "
              "flagged, rows remaining, scales remaining, rationale."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
