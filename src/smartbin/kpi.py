"""
KPI definitions - pure functions, unit-tested (tests/test_kpi.py).

Bossard's two KPIs (case slides):
    frac_too_many    = TOO_MANY / automatic measurements     target <= 0.5, alarm > 0.8
    too_few_per_week = TOO_FEW / weeks                       target <= 2

Supporting KPIs (ours):
    miss_rate        = TOO_FEW / (detected changes + TOO_FEW)  share of real stock changes the sensor slept through
    meas_per_day     = all measurements / day                  battery + radio + cloud load proxy
    too_many_per_week= absolute wasted measurements            what the battery actually pays for

Zones = the three situations from the case (R1 / R2 / R3) + "both" (violates both targets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONES = ["R2 target", "R1 too sensitive", "R3 too insensitive", "R1+R3 unstable"]


def add_kpis(df: pd.DataFrame, cfg) -> pd.DataFrame:
    d = df.copy()
    weeks = d["days"] / 7.0
    d["frac_too_many"] = d.n_tm / d.n_auto.where(d.n_auto > 0)
    d["too_few_per_week"] = d.n_tf / weeks
    d["too_many_per_week"] = d.n_tm / weeks
    d["meas_per_day"] = d.n_rows / d["days"]
    d["miss_rate"] = d.n_tf / (d.n_aok + d.n_tf).where((d.n_aok + d.n_tf) > 0)
    d["meets_tm"] = d.frac_too_many <= cfg.kpi.tm_target
    d["meets_tf"] = d.too_few_per_week <= cfg.kpi.tf_target
    d["in_target"] = d.meets_tm & d.meets_tf
    d["zone"] = np.select(
        [d.meets_tm & d.meets_tf, ~d.meets_tm & d.meets_tf, d.meets_tm & ~d.meets_tf],
        ZONES[:3], default=ZONES[3])
    return d


def pooled(df: pd.DataFrame) -> pd.Series:
    """KPIs of a group computed from summed counts (used for weekly timelines)."""
    n_auto, n_tm, n_tf, n_aok = df.n_auto.sum(), df.n_tm.sum(), df.n_tf.sum(), df.n_aok.sum()
    dev_days = df.active_days.sum() if "active_days" in df else df["days"].sum()
    return pd.Series({
        "frac_too_many": n_tm / n_auto if n_auto else np.nan,
        "too_few_per_week": n_tf / dev_days * 7 if dev_days else np.nan,
        "miss_rate": n_tf / (n_aok + n_tf) if (n_aok + n_tf) else np.nan,
        "meas_per_day": df.n_rows.sum() / dev_days if dev_days else np.nan,
        "devices": df.deviceid.nunique() if "deviceid" in df else np.nan,
    })


def battery_life_ratio(meas_per_day_old, meas_per_day_new, cfg):
    """Relative battery life new/old. Energy/day = sleep*86400 + 3000*sleep*t_meas*n."""
    k = cfg.battery.active_power_factor * cfg.battery.measurement_seconds
    return (86400 + k * np.asarray(meas_per_day_old)) / (86400 + k * np.asarray(meas_per_day_new))
