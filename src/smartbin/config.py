"""Load config.yaml into attribute-style objects and resolve paths relative to the repo root."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def load_config(path: str | Path = "config/config.yaml", **overrides) -> SimpleNamespace:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = yaml.safe_load(path.read_text())
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = raw
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    cfg = _ns(raw)
    for k, v in vars(cfg.paths).items():
        p = Path(v)
        setattr(cfg.paths, k, p if p.is_absolute() else ROOT / p)
    cfg.paths.figures = cfg.paths.reports / "figures"
    cfg.paths.tables = cfg.paths.reports / "tables"
    if not hasattr(cfg.paths, "processed"):
        cfg.paths.processed = ROOT / "data" / "processed"
    for p in (cfg.paths.interim, cfg.paths.processed, cfg.paths.reports, cfg.paths.figures, cfg.paths.tables):
        p.mkdir(parents=True, exist_ok=True)
    return cfg
