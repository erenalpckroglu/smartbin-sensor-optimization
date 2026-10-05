"""
Command line:
    python -m smartbin run                       # real data in data/raw/ (paths in config/config.yaml)
    python -m smartbin run --measurements data/raw/measurements_10mb.csv --reports reports/smoke
    python -m smartbin simulate --out data/synthetic --devices 2000
    python -m smartbin demo                      # simulate + run on synthetic data -> reports/demo
"""
from __future__ import annotations

import argparse

from . import simulate as sim


def main(argv=None):
    p = argparse.ArgumentParser(prog="smartbin", description="SmartBin sensor sensitivity optimisation")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline")
    r.add_argument("--config", default="config/config.yaml")
    r.add_argument("--measurements")
    r.add_argument("--metadata")
    r.add_argument("--preprocessing")
    r.add_argument("--reports")
    r.add_argument("--interim", help="working folder for the DuckDB database (fast local disk)")
    r.add_argument("--memory-limit", help='DuckDB RAM cap, e.g. "1200MB" on a small machine')
    r.add_argument("--threads", type=int)
    s = sub.add_parser("simulate", help="generate synthetic data with known ground truth")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--devices", type=int, default=2000)
    s.add_argument("--seed", type=int, default=7)
    d = sub.add_parser("demo", help="simulate + run (no confidential data needed)")
    d.add_argument("--devices", type=int, default=2000)
    a = p.parse_args(argv)

    if a.cmd == "simulate":
        sim.main(["--out", a.out, "--devices", str(a.devices), "--seed", str(a.seed)])
        return
    from .pipeline import run
    if a.cmd == "run":
        run(a.config, a.measurements, a.metadata, a.preprocessing, a.reports, a.interim, a.memory_limit, a.threads)
    elif a.cmd == "demo":
        sim.main(["--out", "data/synthetic", "--devices", str(a.devices)])
        run("config/config.yaml", "data/synthetic/measurements.csv", "data/synthetic/scale_metadata.csv",
            "data/synthetic/scale_preprocessing.csv", "reports/demo")


if __name__ == "__main__":
    main()
