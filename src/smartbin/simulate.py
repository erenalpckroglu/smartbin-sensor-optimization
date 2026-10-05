"""
Synthetic SmartBin fleet simulator with KNOWN ground truth.

Why this exists
---------------
1. Validation: every analysis in the pipeline is first run on data where we *know*
   the answer (planted effects). If the pipeline cannot recover a planted effect,
   we do not trust it on real data.
2. Portfolio: the repository runs end-to-end (`make demo`) without the confidential
   Bossard data. Real data goes into data/raw/ and is never committed.

Output files mirror the real schema 1:1 (column names, formats, null patterns):
    measurements.csv, scale_metadata.csv, scale_preprocessing.csv, ground_truth.json

Physics of the simulation (the "truth" the pipeline should rediscover)
--------------------------------------------------------------------
* A pick (part removal) shakes the bin. Vibration intensity grows with pick mass
  (log grams) and is damped by a heavy box.
* The sensor fires when intensity > threshold(sensitivity). Higher sensitivity =
  lower threshold.
* CROSSTALK: a pick also shakes the other bins on the same rack (attenuated by
  distance). If a neighbour's sensor fires, it measures, finds no weight change
  -> TOO_MANY. Racks are hidden; only LOCATIONID is visible.
* Location = factory: ambient vibration level, rack stiffness, rack size, pick volume.
* 20kg scales pick up more crosstalk and feel small picks less.
* Box type: NO effect (planted null).
* Echo: after a pick the bin is often touched again -> a second, empty measurement.
* Misses: undetected picks are only found by a MANUAL measurement (person or 24h
  heartbeat) -> TOO_FEW.
* Experiment: March everybody on sensitivity 8. From April ~46% of devices are
  switched to another level. Sensitivity 3 is given preferentially to NOISY devices
  (selection bias - mirrors the anomaly seen in the real sample).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHT_BINS = [(0, 803), (803, 1877), (1877, 3859), (3859, 6673), (6673, 100000)]
BIN_LABELS = [f"{lo} - {hi}" for lo, hi in WEIGHT_BINS]
HOUR_PROFILE = np.array([0.013, 0.011, 0.012, 0.016, 0.031, 0.049, 0.071, 0.073, 0.073, 0.072, 0.057, 0.070,
                         0.084, 0.080, 0.065, 0.046, 0.030, 0.032, 0.032, 0.031, 0.023, 0.011, 0.011, 0.009])
HOUR_PROFILE = HOUR_PROFILE / HOUR_PROFILE.sum()
DOW_FACTOR = np.array([1.0, 1.05, 1.05, 1.0, 0.9, 0.4, 0.32])  # Mon..Sun
SWITCH_LEVELS = np.array([1, 2, 4, 5, 6, 7])
SWITCH_PROBS = np.array([0.12, 0.14, 0.14, 0.17, 0.21, 0.22])

# ---- ground-truth physics parameters -------------------------------------------------
TRUTH = dict(
    threshold_intercept=3.3,      # theta(s) = 3.3 - 0.27*s
    threshold_slope=0.27,
    own_intercept=1.0,            # V_own = 1.0 + 0.55*ln(pick_g) - damping + noise
    own_log_slope=0.55,
    own_noise=0.7,
    damping_per_log_kg=0.25,      # heavier box -> weaker vibration
    own_penalty_20kg=0.35,        # 20kg scale feels small picks less
    crosstalk_gain_20kg=0.60,     # 20kg platform picks up more neighbour vibration
    attenuation_mean=1.55,        # rack stiffness (location level)
    attenuation_per_slot=0.35,    # extra attenuation per rack slot distance
    crosstalk_noise=0.9,
    echo_prob=0.40,
    handling_rate_per_day=0.6,    # bin touched/bumped without removing parts
    handling_mu=2.3,
    handling_sd=1.0,
    box_effect=0.0,               # planted NULL effect
    level3_behaves_like=10.5,     # sensitivity-3 anomaly (selection + setting quirk)
    significance_g="max(2, 0.5*item_weight)",
)


def _theta(sens: np.ndarray) -> np.ndarray:
    # firmware quirk planted on purpose: level 3 behaves like a hyper-sensitive level (~10.5)
    eff = np.where(np.asarray(sens) == 3, TRUTH["level3_behaves_like"], sens)
    return TRUTH["threshold_intercept"] - TRUTH["threshold_slope"] * eff


def _hexid(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:20]


def _sample_times(rng, day_starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Epoch seconds for `counts[i]` events on day i, following the factory hour profile."""
    n = int(counts.sum())
    if n == 0:
        return np.empty(0)
    day = np.repeat(day_starts, counts)
    hour = rng.choice(24, size=n, p=HOUR_PROFILE)
    return day + hour * 3600 + rng.uniform(0, 3600, n)


def simulate(n_devices: int = 2000, out_dir: str | Path = "data/synthetic", seed: int = 7,
             start: str = "2022-03-01", end: str = "2022-06-20 15:00") -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = pd.Timestamp(start).value / 1e9
    t_end = pd.Timestamp(end).value / 1e9
    days = pd.date_range(start, end, freq="D")
    day_starts = (days.values.astype("datetime64[s]").astype(np.int64)).astype(float)
    dow = days.dayofweek.values
    n_days = len(days)

    # ------------------------------------------------------------------ locations
    n_loc = max(8, n_devices // 65)
    sizes = rng.lognormal(np.log(60), 0.7, n_loc)
    sizes = np.maximum(5, np.round(sizes / sizes.sum() * n_devices)).astype(int)
    while sizes.sum() > n_devices:
        sizes[np.argmax(sizes)] -= 1
    while sizes.sum() < n_devices:
        sizes[rng.integers(n_loc)] += 1
    loc_ids = np.unique(rng.integers(1_000_000, 12_999_999_999, n_loc * 2))[:n_loc]
    rng.shuffle(loc_ids)
    loc = pd.DataFrame({
        "loc_idx": np.arange(n_loc),
        "LOCATIONID": loc_ids,
        "amb_mu": rng.normal(0.0, 0.65, n_loc),             # ambient vibration level
        "amb_rate": rng.lognormal(np.log(0.8), 0.8, n_loc), # ambient shocks per working day per rack
        "attenuation": np.clip(rng.normal(TRUTH["attenuation_mean"], 0.40, n_loc), 0.6, 2.8),
        "rack_size": rng.choice([6, 8, 10, 12, 16], n_loc, p=[.15, .25, .3, .2, .1]),
        "pick_mult": rng.lognormal(0, 0.30, n_loc),
        "n_devices": sizes,
    })
    cust_of_loc = rng.integers(0, max(2, int(n_loc * 0.6)), n_loc)  # some customers own several sites
    loc["CUSTOMER_NUMBER"] = ["S" + str(200000 + 37 * c) for c in cust_of_loc]

    # ------------------------------------------------------------------ devices
    dev_loc = np.repeat(loc.loc_idx.values, sizes)
    is20 = rng.random(n_devices) < 0.42
    box = np.where(is20,
                   rng.choice(["SK 3521", "LF 321", "BOD XL 4327 (Cloud)", "SS 5032-3", "SK 5032", "SK 5033"],
                              n_devices, p=[.91, .04, .016, .016, .009, .009]),
                   rng.choice(["SK 2311", "LF 221"], n_devices, p=[.93, .07]))
    item_w = np.where(is20, rng.lognormal(np.log(10), 1.7, n_devices), rng.lognormal(np.log(3), 1.6, n_devices))
    item_w = np.round(np.clip(item_w, 0.01, 4222), 3)
    reserve = rng.random(n_devices) < 0.03
    dev = pd.DataFrame({
        "deviceid": [f"A{v:07X}" for v in rng.permutation(np.arange(0x10, 0x10 + 40 * n_devices, 37))[:n_devices]],
        "loc_idx": dev_loc, "is20": is20, "BOX_NAME": box, "ITEM_WEIGHT": item_w, "reserve": reserve,
    }).merge(loc, on="loc_idx", how="left")
    # hidden racks: shuffle within location, chunk by rack size
    dev["rack"] = -1
    dev["pos"] = -1
    rack_counter = 0
    for li, g in dev.groupby("loc_idx"):
        idx = rng.permutation(g.index.values)
        rs = int(g.rack_size.iloc[0])
        for k in range(0, len(idx), rs):
            chunk = idx[k:k + rs]
            dev.loc[chunk, "rack"] = rack_counter
            dev.loc[chunk, "pos"] = np.arange(len(chunk))
            rack_counter += 1
    dev["pick_rate"] = rng.lognormal(np.log(1.6), 0.55, n_devices) * dev.pick_mult * np.where(reserve, 0.05, 1)
    dev["cap"] = np.where(is20, rng.uniform(3000, 16000, n_devices), rng.uniform(1500, 5200, n_devices))
    dev["tare"] = np.where(is20, 900.0, 350.0)
    # configuration (minwait / delay) - mostly standard, some unknown (NaN) or non-standard
    u = rng.random(n_devices)
    dev["minwait"] = np.select([u < 0.08, u < 0.086, u < 0.088], [np.nan, 60.0, 10.0], 1.0)
    dev["delay"] = np.where((u < 0.025), np.nan, np.where(rng.random(n_devices) < 0.004, 5.0, 30.0))
    # devices whose sensitivity is not logged during the first weeks
    dev["sens_unknown_until"] = np.where(rng.random(n_devices) < 0.08,
                                         t0 + rng.uniform(5, 30, n_devices) * 86400, t0)

    # ------------------------------------------------------------------ sensitivity schedule
    # noisiness score drives the sens-3 selection (selection bias)
    noisy = dev.amb_mu.values - dev.attenuation.values + 0.08 * dev.rack_size.values + 0.4 * is20
    z = (noisy - noisy.mean()) / noisy.std()
    p3 = np.exp(1.6 * z)
    p3 = p3 / p3.sum() * 0.045 * n_devices
    gets3 = rng.random(n_devices) < np.clip(p3, 0, 0.9)
    switcher = (~gets3) & (rng.random(n_devices) < 0.46)
    sched = []  # list of (t_from, sens)
    apr1 = pd.Timestamp("2022-04-01").value / 1e9
    for i in range(n_devices):
        s = [(t0, 8)]
        if gets3[i]:
            s.append((pd.Timestamp("2022-04-07").value / 1e9 + rng.uniform(0, 3) * 86400, 3))
        elif switcher[i]:
            t_sw = apr1 + rng.uniform(0, 25) * 86400
            first = int(rng.choice(SWITCH_LEVELS, p=SWITCH_PROBS))
            s.append((t_sw, first))
            if rng.random() < 0.2:
                t2 = pd.Timestamp("2022-05-10").value / 1e9 + rng.uniform(0, 12) * 86400
                second = 8 if rng.random() < 0.15 else int(rng.choice(SWITCH_LEVELS[SWITCH_LEVELS != first]))
                s.append((t2, second))
        elif rng.random() < 0.002:
            s.append((apr1 + rng.uniform(10, 20) * 86400, 10))
        sched.append(s)

    # ------------------------------------------------------------------ own picks & refills
    picks_t, picks_dev, picks_g = [], [], []
    for i in range(n_devices):
        lam = dev.pick_rate.iat[i] * DOW_FACTOR[dow]
        cnt = rng.poisson(lam)
        t = np.sort(_sample_times(rng, day_starts, cnt))
        if len(t) == 0:
            continue
        iw = dev.ITEM_WEIGHT.iat[i]
        k = 1 + rng.poisson(np.clip(6.0 / np.sqrt(iw), 0, 40), len(t))
        picks_t.append(t); picks_dev.append(np.full(len(t), i)); picks_g.append(k * iw)
    picks = pd.DataFrame({"t": np.concatenate(picks_t), "dev": np.concatenate(picks_dev),
                          "g": np.concatenate(picks_g)}).sort_values(["dev", "t"]).reset_index(drop=True)

    # stock trajectory with refills (sequential per device, cheap: ~120 picks/device)
    stock_events = []   # (dev, t, delta_g, kind, level_after) kind: 0 pick, 1 refill
    start_level = {}
    for i, g in picks.groupby("dev", sort=False):
        cap = dev.cap.iat[i]
        level = cap * rng.uniform(0.4, 1.0)
        start_level[i] = level
        pending_refill = None
        for t, grams in zip(g.t.values, g.g.values):
            if pending_refill is not None and t > pending_refill:
                stock_events.append((i, pending_refill, cap - level, 1, cap))
                level = cap
                pending_refill = None
            take = min(grams, max(level - 1, 0))
            if take <= 0:
                continue
            level -= take
            stock_events.append((i, t, -take, 0, level))
            if level < 0.2 * cap and pending_refill is None:
                pending_refill = t + rng.uniform(0.5, 2.5) * 86400
    ev = (pd.DataFrame(stock_events, columns=["dev", "t", "dg", "kind", "level_after"])
          .sort_values(["dev", "t"]).reset_index(drop=True))
    ev["own_V"] = np.where(
        ev.kind == 1, 4.0 + rng.normal(0, 0.3, len(ev)),
        TRUTH["own_intercept"] + TRUTH["own_log_slope"] * np.log(np.maximum(-ev.dg, 0.05))
        - TRUTH["damping_per_log_kg"] * np.log1p(ev.level_after / 1000)
        - TRUTH["own_penalty_20kg"] * dev.is20.values[ev.dev.values]
        + rng.normal(0, TRUTH["own_noise"], len(ev)))

    # ------------------------------------------------------------------ candidate vibrations per device
    cand = [pd.DataFrame({"dev": ev.dev, "t": ev.t, "V": ev.own_V, "src": np.where(ev.kind == 1, 1, 0)})]
    # echoes (bin touched again after a pick)
    pk = ev[ev.kind == 0]
    e = rng.random(len(pk)) < TRUTH["echo_prob"]
    cand.append(pd.DataFrame({"dev": pk.dev.values[e], "t": pk.t.values[e] + rng.uniform(65, 240, e.sum()),
                              "V": pk.own_V.values[e] - 0.1 + rng.normal(0, 0.7, e.sum()), "src": 2}))
    # crosstalk to rack neighbours
    rack_of = dev.rack.values
    pos_of = dev.pos.values
    members = dev.groupby("rack").apply(lambda g: g.index.values, include_groups=False).to_dict()
    src_dev = ev.dev.values
    rows_dev, rows_t, rows_V = [], [], []
    for r, mem in members.items():
        sel = np.isin(src_dev, mem)
        if not sel.any() or len(mem) < 2:
            continue
        s_dev, s_t, s_V = src_dev[sel], ev.t.values[sel], ev.own_V.values[sel]
        for j in mem:
            m = s_dev != j
            if not m.any():
                continue
            dist = np.abs(pos_of[s_dev[m]] - pos_of[j])
            att = dev.attenuation.iat[j] + TRUTH["attenuation_per_slot"] * (dist - 1)
            V = (s_V[m] - att + TRUTH["crosstalk_gain_20kg"] * dev.is20.iat[j]
                 - TRUTH["damping_per_log_kg"] * 0.5 + rng.normal(0, TRUTH["crosstalk_noise"], m.sum()))
            rows_dev.append(np.full(m.sum(), j)); rows_t.append(s_t[m] + rng.normal(0, 0.8, m.sum()))
            rows_V.append(V)
    cand.append(pd.DataFrame({"dev": np.concatenate(rows_dev), "t": np.concatenate(rows_t),
                              "V": np.concatenate(rows_V), "src": 3}))
    # ambient shocks (forklift, machines): hit a whole rack at once
    amb = []
    for r, mem in members.items():
        li = dev.loc_idx.iat[mem[0]]
        lam = loc.amb_rate.iat[li] * DOW_FACTOR[dow]
        t = _sample_times(rng, day_starts, rng.poisson(lam))
        for j in mem:
            amb.append(pd.DataFrame({"dev": j, "t": t + rng.normal(0, 1.0, len(t)),
                                     "V": loc.amb_mu.iat[li] + 1.0 + 0.3 * dev.is20.iat[j]
                                     + rng.normal(0, 0.9, len(t)), "src": 4}))
    # independent device noise
    for j in range(n_devices):
        t = _sample_times(rng, day_starts, rng.poisson(0.25 * DOW_FACTOR[dow]))
        amb.append(pd.DataFrame({"dev": j, "t": t, "V": dev.amb_mu.iat[j] + 0.8 + rng.normal(0, 0.9, len(t)),
                                 "src": 5}))
        # handling: somebody touches / pulls the bin without taking parts (strong, weight unchanged)
        th = _sample_times(rng, day_starts, rng.poisson(TRUTH["handling_rate_per_day"] * DOW_FACTOR[dow]
                                                        * dev.pick_mult.iat[j]))
        amb.append(pd.DataFrame({"dev": j, "t": th, "V": rng.normal(TRUTH["handling_mu"], TRUTH["handling_sd"], len(th)),
                                 "src": 6}))
    cand.append(pd.concat(amb, ignore_index=True))
    cand = pd.concat(cand, ignore_index=True)
    cand = cand[(cand.t > t0) & (cand.t < t_end)].sort_values(["dev", "t"]).reset_index(drop=True)

    # ------------------------------------------------------------------ trigger -> measurements
    out_rows = []
    cand_groups = dict(tuple(cand.groupby("dev", sort=False)))
    ev_groups = dict(tuple(ev.groupby("dev", sort=False)))
    for i in range(n_devices):
        c = cand_groups.get(i)
        s = sched[i]
        s_t = np.array([x[0] for x in s]); s_v = np.array([x[1] for x in s], dtype=float)
        minwait = 60.0 * (1.0 if np.isnan(dev.minwait.iat[i]) else dev.minwait.iat[i])
        delay = 30.0 if np.isnan(dev.delay.iat[i]) else dev.delay.iat[i]
        auto_t = []
        if c is not None and len(c):
            sens_at = s_v[np.searchsorted(s_t, c.t.values, side="right") - 1]
            trig = c.V.values > _theta(sens_at)
            tt = c.t.values[trig]
            nxt = -np.inf
            for t in tt:
                if t >= nxt:
                    mt = t + delay
                    auto_t.append(mt)
                    nxt = mt + minwait
        auto_t = np.array(auto_t)
        # manual: person-initiated + 24h heartbeat
        person = _sample_times(rng, day_starts, rng.poisson(0.10 * DOW_FACTOR[dow]))
        allm = np.sort(np.concatenate([auto_t, person]))
        kind = np.isin(allm, auto_t)
        times, kinds = [], []
        last = t0 + rng.uniform(0, 6 * 3600)
        for t, k in zip(allm, kind):
            while t - last > 86400:
                last = last + 86400
                times.append(last); kinds.append(False)
            times.append(t); kinds.append(bool(k))
            last = t
        while t_end - last > 86400:
            last += 86400
            times.append(last); kinds.append(False)
        times = np.array(times); kinds = np.array(kinds)
        # weight at each measurement
        g = ev_groups.get(i)
        if g is not None and len(g):
            idx = np.searchsorted(g.t.values, times, side="right") - 1
            lvl0 = start_level.get(i, dev.cap.iat[i] * 0.7)
            level = np.where(idx >= 0, g.level_after.values[np.maximum(idx, 0)], lvl0)
        else:
            level = np.full(len(times), dev.cap.iat[i] * 0.6)
        w = np.round(level + dev.tare.iat[i] + rng.normal(0, 0.5, len(times)))
        w = np.maximum(w, 0)
        thr = max(2.0, 0.5 * dev.ITEM_WEIGHT.iat[i])
        dw = np.abs(np.diff(w, prepend=w[0] if len(w) else 0))
        sig = dw >= thr
        if len(sig):
            sig[0] = False
        cls = np.where(kinds, np.where(sig, "OK", "TOO_MANY"), np.where(sig, "TOO_FEW", "OK"))
        sens_at = s_v[np.searchsorted(s_t, times, side="right") - 1]
        out_rows.append(pd.DataFrame({"dev": i, "ts": times, "auto": kinds, "weight": w, "cls": cls,
                                      "sens_true": sens_at}))
    meas = pd.concat(out_rows, ignore_index=True)

    # ------------------------------------------------------------------ logging artefacts (nulls, labels)
    meas["deviceid"] = dev.deviceid.values[meas.dev.values]
    unknown = meas.ts.values < dev.sens_unknown_until.values[meas.dev.values]
    meas["sensitivity"] = np.where(unknown, np.nan, meas.sens_true)
    meas["minwaittime"] = np.where(unknown, np.nan, dev.minwait.values[meas.dev.values])
    meas["delay"] = np.where(unknown, np.nan, dev.delay.values[meas.dev.values])
    meas.loc[meas.auto & (rng.random(len(meas)) < 0.0015), "cls"] = None
    meas["initiatortype"] = np.where(meas.auto, "AUTOMATIC", "MANUAL")
    meas["weight_bin"] = pd.cut(meas.weight, [-1, 803, 1877, 3859, 6673, 100000], labels=BIN_LABELS).astype(str)

    # sections: consecutive rows with constant (sensitivity, weight_bin) per device
    meas = meas.sort_values(["dev", "ts"]).reset_index(drop=True)
    key = meas.sensitivity.astype(str) + "|" + meas.weight_bin
    new_run = (meas.dev != meas.dev.shift()) | (key != key.shift())
    run = new_run.cumsum()
    run_ids = {r: _hexid(seed, r) for r in run.unique()}
    meas["section_id"] = run.map(run_ids)
    meas.loc[meas.sensitivity.isna(), "section_id"] = None
    sec = meas[meas.section_id.notna()].groupby("section_id").agg(n=("ts", "size"), t_min=("ts", "min"),
                                                                   t_max=("ts", "max"))
    sec["rel"] = (sec.n >= 20) & ((sec.t_max - sec.t_min) >= 5 * 86400)
    meas["is_relevant_section"] = meas.section_id.map(sec.rel)

    # relevant_for_weight_analysis: weight-bin move at constant sensitivity AND "something changes"
    rs = meas[meas.is_relevant_section == True].copy()
    rs["tm"] = rs.auto & (rs.cls == "TOO_MANY")
    st = rs.groupby(["dev", "sensitivity", "weight_bin", "section_id"]).agg(
        n_auto=("auto", "sum"), n_tm=("tm", "sum")).reset_index()
    st["frac"] = st.n_tm / st.n_auto.clip(lower=1)
    flagged = set()
    for (d, s_), g in st.groupby(["dev", "sensitivity"]):
        if g.weight_bin.nunique() >= 2:
            byb = g.groupby("weight_bin").frac.mean()
            if byb.max() - byb.min() > 0.10:
                flagged.update(g.section_id)
    meas["relevant_for_weight_analysis"] = meas.section_id.isin(flagged)

    # ------------------------------------------------------------------ write files in REAL format
    m_out = pd.DataFrame({
        "deviceid": meas.deviceid,
        "weighttime": pd.to_datetime(meas.ts, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3],
        "minwaittime": meas.minwaittime, "sensitivity": meas.sensitivity, "delay": meas.delay,
        "weight": meas.weight.astype(float), "initiatortype": meas.initiatortype,
        "weightclassification": meas.cls, "weight_bin": meas.weight_bin, "section_id": meas.section_id,
        "is_relevant_section": meas.is_relevant_section.map({True: "True", False: "False"}),
        "relevant_for_weight_analysis": meas.relevant_for_weight_analysis.map({True: "True", False: "False"}),
    }).sample(frac=1.0, random_state=seed)  # real file is not sorted
    m_out.to_csv(out / "measurements.csv", index=False)

    rel = meas[meas.is_relevant_section == True]
    pre = pd.DataFrame({"deviceid": dev.deviceid})
    pre["device_changes_sensitivity"] = pre.deviceid.map(rel.groupby("deviceid").sensitivity.nunique() > 1).fillna(False)
    pre["device_changes_weight_bin"] = pre.deviceid.map(rel.groupby("deviceid").weight_bin.nunique() > 1).fillna(False)
    pre["device_has_relevant_sections"] = pre.deviceid.isin(rel.deviceid.unique())
    pre.to_csv(out / "scale_preprocessing.csv", index=False)

    words = ["Sechskantschraube", "Washer A2", "Hex nut M8", "Socket screw", "Rondelle", "Spring pin", "Rivet"]
    meta = pd.DataFrame({
        "deviceid": dev.deviceid,
        "SCALE_NAME": np.where(dev.is20, "Cloud-V1.00-20kg", "Cloud-V1.00-6kg"),
        "BOX_NAME": dev.BOX_NAME,
        "CUSTOMER_NUMBER": dev.CUSTOMER_NUMBER,
        "LOCATIONID": dev.LOCATIONID.astype(float),
        "SUPPLIER_ITEM_NUMBER": rng.integers(1_000_000, 9_999_999, n_devices),
        "DESCRIPTION": np.where(dev.reserve, "Reserve", rng.choice(words, n_devices)),
        "DIMENSION": [f"M {rng.integers(2, 13)} x {rng.integers(4, 60)}" for _ in range(n_devices)],
        "ITEM_WEIGHT": dev.ITEM_WEIGHT,
    })
    missing = rng.random(n_devices) < 0.053
    meta.loc[missing, meta.columns[1:]] = np.nan
    meta.to_csv(out / "scale_metadata.csv", index=False)

    truth = {
        "planted": {
            "sensitivity": "strong: threshold = 3.4 - 0.30*sens",
            "location": "strong: ambient level, rack stiffness, rack size, pick volume",
            "crosstalk": "picks shake rack neighbours within ~1s -> TOO_MANY at neighbour",
            "scale_type": "20kg: +0.60 crosstalk pickup, -0.35 own-pick intensity",
            "weight_in_box": "small: damping 0.25 per ln(kg)",
            "item_weight": "acts through pick size (light picks missed / below 1g resolution)",
            "box_type": "NONE (planted null effect)",
            "sens3_anomaly": "sens 3 given preferentially to noisy devices AND behaves like level ~10.5",
        },
        "params": TRUTH,
        "n_devices": int(n_devices), "n_locations": int(n_loc), "n_racks": int(rack_counter),
        "rows": int(len(m_out)),
        "rack_of_device": dict(zip(dev.deviceid, dev.rack.astype(int).tolist())),
        "location_effects": loc[["LOCATIONID", "amb_mu", "amb_rate", "attenuation", "rack_size", "pick_mult"]]
        .assign(LOCATIONID=lambda d: d.LOCATIONID.astype(str)).to_dict(orient="records"),
    }
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=1))
    return truth


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate a synthetic SmartBin dataset with known ground truth")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--devices", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    t = simulate(a.devices, a.out, a.seed)
    print(f"synthetic data -> {a.out}: {t['rows']:,} rows, {t['n_devices']} devices, "
          f"{t['n_locations']} locations, {t['n_racks']} hidden racks")


if __name__ == "__main__":
    main()
