from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ModuleResult:
    """What every analysis module returns: headline facts, tables, figures, caveats."""
    name: str
    title: str = ""
    question: str = ""
    facts: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    figures: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    status: str = "ok"          # ok | skipped | failed

    def save_tables(self, tabdir: Path):
        for k, df in self.tables.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(tabdir / f"{self.name}__{k}.csv", index=False)

    def fig_path(self, figdir: Path, key: str, scale: str | None = None) -> Path:
        return figdir / f"{self.name}__{key}{'_' + scale if scale else ''}.png"
