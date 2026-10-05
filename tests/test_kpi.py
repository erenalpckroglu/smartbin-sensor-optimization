from types import SimpleNamespace

import numpy as np
import pandas as pd

from smartbin.kpi import add_kpis, battery_life_ratio

CFG = SimpleNamespace(kpi=SimpleNamespace(tm_target=0.5, tm_alarm=0.8, tf_target=2.0),
                      battery=SimpleNamespace(measurement_seconds=2.0, active_power_factor=3000))


def _df(**kw):
    base = dict(n_rows=100, n_auto=80, n_manual=20, n_tm=40, n_tf=2, n_aok=40, n_mok=18, days=14.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_bossard_kpis():
    d = add_kpis(_df(), CFG).iloc[0]
    assert d.frac_too_many == 0.5            # 40 / 80
    assert d.too_few_per_week == 1.0         # 2 per 2 weeks
    assert d.meas_per_day == 100 / 14
    assert abs(d.miss_rate - 2 / 42) < 1e-12
    assert d.in_target and d.zone == "R2 target"


def test_zones():
    assert add_kpis(_df(n_tm=70, n_aok=10), CFG).zone.iloc[0] == "R1 too sensitive"
    assert add_kpis(_df(n_tf=10), CFG).zone.iloc[0] == "R3 too insensitive"
    assert add_kpis(_df(n_tm=70, n_aok=10, n_tf=10), CFG).zone.iloc[0] == "R1+R3 unstable"


def test_no_automatic_measurements_gives_nan_not_crash():
    d = add_kpis(_df(n_auto=0, n_tm=0, n_aok=0), CFG).iloc[0]
    assert np.isnan(d.frac_too_many)


def test_battery_fewer_measurements_live_longer():
    r = battery_life_ratio(10, 5, CFG)
    assert r > 1
    assert battery_life_ratio(5, 5, CFG) == 1
