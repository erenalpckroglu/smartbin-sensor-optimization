"""
End-to-end run:  load -> clean (A) -> features -> flag (B) -> verify -> tables -> M1..M7 -> reports.
Every analysis module is isolated: if one fails on the real data, the others still run and the
failure is written to the report instead of killing the pipeline.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
import traceback
from pathlib import Path

import pandas as pd

from . import cleaning, features, io, powerbi, report, tables, viz
from .analysis import ModuleResult, baseline, crosstalk, importance, location, pick_size, policy, sensitivity, validation
from .config import load_config

log = logging.getLogger("smartbin")


def _setup_logging(cfg):
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(cfg.paths.reports / "run.log", mode="w")):
        h.setFormatter(fmt)
        log.addHandler(h)


def _safe(name, fn, *a, **k):
    t = time.time()
    try:
        res = fn(*a, **k)
        log.info("%-16s done in %.1fs", name, time.time() - t)
        return res
    except Exception as e:  # keep going, report the failure
        log.error("%s FAILED: %s\n%s", name, e, traceback.format_exc())
        r = ModuleResult(name, status="failed")
        r.notes.append(f"{type(e).__name__}: {e}")
        return r


def _funnel_chart(funnel: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    f = funnel[funnel.rows_removed.gt(0) | funnel.step.eq(0)].copy()
    f = pd.concat([f, funnel.tail(1).assign(rule="= main KPI analysis", rows_removed=0)])
    labels = f.rule.str.replace(r"^[AB]\d+_", "", regex=True).str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(f) + 1.2))
    y = range(len(f))
    colors = [viz.CAT[0] if i in (0, len(f) - 1) else viz.GRAY_BAR for i in range(len(f))]
    ax.barh(list(y), f.rows_remaining / 1e6, color=colors, height=0.6)
    for yi, (rem, rm) in enumerate(zip(f.rows_remaining, f.rows_removed)):
        txt = f"{rem / 1e6:.2f}M" + (f"   (-{rm:,})" if rm else "")
        ax.text(rem / 1e6, yi, "  " + txt, va="center", fontsize=8, color=viz.INK2)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, f.rows_remaining.max() / 1e6 * 1.35)
    ax.set_xlabel("Rows remaining (millions)")
    ax.set_title("Data cleaning funnel: every rule, in order")
    ax.grid(axis="y", visible=False)
    return viz.save(fig, path)


def run(config="config/config.yaml", measurements=None, metadata=None, preprocessing=None, reports=None,
        interim=None, memory_limit=None, threads=None):
    t0 = time.time()
    cfg = load_config(config, **{"paths.measurements": measurements, "paths.metadata": metadata,
                                 "paths.preprocessing": preprocessing, "paths.reports": reports,
                                 "paths.interim": interim, "runtime.duckdb_memory_limit": memory_limit,
                                 "runtime.threads": threads})
    _setup_logging(cfg)
    log.info("input: %s", cfg.paths.measurements)
    con = io.connect(cfg)
    io.load_measurements(con, cfg.paths.measurements)
    n_rows, n_dev = con.execute("SELECT count(*), count(DISTINCT deviceid) FROM raw").fetchone()
    funnel_a = cleaning.apply_stage_a(con, cfg)
    con.execute("DROP TABLE raw")
    meta_df = io.load_metadata(cfg.paths.metadata)
    pre_df = io.load_preprocessing(cfg.paths.preprocessing)
    features.build_events(con, meta_df, pre_df)
    con.execute("DROP TABLE base")
    funnel_b = cleaning.apply_stage_b(con, cfg)
    funnel = pd.concat([funnel_a, funnel_b], ignore_index=True)
    cleaning.export_filtered_out(con, cfg, funnel)
    sample = features.is_sample(con, cfg)
    checks = tables.verify_dataset(con, cfg)
    sections = tables.build_sections(con, cfg)
    roles = tables.device_roles(con, cfg)
    ds = tables.build_device_settings(sections, roles, cfg)
    dw = tables.build_device_weeks(con, cfg)
    con.execute(f"COPY (SELECT * FROM events) TO '{(cfg.paths.interim / 'events.parquet').as_posix()}' (FORMAT parquet)")
    # slim per-row flag file (the "use this row for the KPI?" column, D ONE style) + analysis tables
    con.execute(f"""COPY (SELECT deviceid, weighttime, in_kpi_analysis, exclusion_reason, dw AS weight_change_g
                          FROM events) TO '{(cfg.paths.processed / 'row_flags.parquet').as_posix()}' (FORMAT parquet)""")
    sections.to_parquet(cfg.paths.processed / "sections.parquet", index=False)
    ds.to_parquet(cfg.paths.processed / "device_settings.parquet", index=False)
    powerbi.export_powerbi(ds, funnel, cfg)
    log.info("tables: %s sections, %s device-settings, roles %s", f"{len(sections):,}", f"{len(ds):,}",
             roles.value_counts().to_dict())

    R = {}
    R["m1_baseline"] = _safe("m1_baseline", baseline.run, ds, cfg)
    R["m2_sensitivity"] = _safe("m2_sensitivity", sensitivity.run, ds, dw, cfg, con, sample)
    R["m3_importance"] = _safe("m3_importance", importance.run, sections, cfg)
    R["m4_location"] = _safe("m4_location", location.run, ds, cfg)
    R["m5_pick_size"] = _safe("m5_pick_size", pick_size.run, con, ds, cfg, sample)
    loc_tab = R["m4_location"].tables.get("locations") if R["m4_location"].status == "ok" else None
    R["m6_crosstalk"] = _safe("m6_crosstalk", crosstalk.run, con, cfg, sample, loc_tab)
    eff = R["m2_sensitivity"].tables.get("effects")
    pairs = R["m2_sensitivity"].tables.get("within_device_pairs")
    prof = R["m5_pick_size"].tables.get("device_pick_profile")
    if sample:
        R["m7_policy"] = ModuleResult("m7_policy", status="skipped",
                                      notes=["Sample mode: per-scale rates are not estimable from random rows -> "
                                             "no what-if simulation. Run on the full file."])
    else:
        R["m7_policy"] = _safe("m7_policy", policy.run, ds, eff, pairs, prof, cfg,
                               R["m2_sensitivity"].tables.get("loop_flags"))
    val = _safe("m0_validation", validation.run, R, cfg.paths.measurements, cfg)
    if val is not None:
        R = {"m0_validation": val, **R}
    for res in R.values():
        if res:
            res.save_tables(cfg.paths.tables)
    funnel.to_csv(cfg.paths.tables / "cleaning_funnel.csv", index=False)
    checks.to_csv(cfg.paths.tables / "data_checks.csv", index=False)
    cleaning.column_decisions().to_csv(cfg.paths.tables / "column_decisions.csv", index=False)

    meta = dict(run_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), measurements=str(cfg.paths.measurements),
                rows=n_rows, devices=n_dev, sample_mode=sample,
                synthetic=(Path(cfg.paths.measurements).parent / "ground_truth.json").exists(),
                rows_main=con.execute("SELECT count(*) FROM events WHERE exclusion_reason IS NULL").fetchone()[0],
                scales=sorted(ds.scale_type.dropna().unique().tolist()),
                funnel_fig=_funnel_chart(funnel, cfg.paths.figures / "cleaning_funnel.png"),
                sens_by_month=tables.sensitivity_by_month(con))
    report.build(R, funnel, checks, cleaning.column_decisions(), meta, cfg)
    log.info("finished in %.0fs -> open %s", time.time() - t0, cfg.paths.reports / "index.html")
    return R

def run_policy(config="config/config.yaml", reports=None):
    """
    Re-run only the recommendation module (M7) from the saved outputs of a previous full run:
    data/processed/device_settings.parquet + the effect, pair and pick-profile tables in <reports>/tables.
    Overwrites the m7_policy tables and figures and the m7 entry of results.json. Takes seconds, not minutes.
    """
    import json
    from .report import _Enc
    cfg = load_config(config, **{"paths.reports": reports})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    tab = cfg.paths.tables
    ds = pd.read_parquet(cfg.paths.processed / "device_settings.parquet")
    eff = pd.read_csv(tab / "m2_sensitivity__effects.csv")
    pairs = pd.read_csv(tab / "m2_sensitivity__within_device_pairs.csv")
    prof = pd.read_csv(tab / "m5_pick_size__device_pick_profile.csv")
    lf_path = tab / "m2_sensitivity__loop_flags.csv"
    loops = pd.read_csv(lf_path) if lf_path.exists() else None
    res = _safe("m7_policy", policy.run, ds, eff, pairs, prof, cfg, loops)
    res.save_tables(tab)
    rj = cfg.paths.reports / "results.json"
    if rj.exists():
        out = json.loads(rj.read_text())
        out.setdefault("modules", {})["m7_policy"] = {"status": res.status, "facts": res.facts, "notes": res.notes,
                                                      "figures": [Path(f).name for f in res.figures.values()]}
        rj.write_text(json.dumps(out, cls=_Enc, indent=1))
    log.info("m7_policy re-run: %s", res.status)
    return res
