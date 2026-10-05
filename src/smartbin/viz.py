"""
One visual language for every figure (validated reference palette, light surface).

Rules applied everywhere: one y-axis per chart, thin marks, hairline solid grid,
legend whenever >= 2 series, colour follows the entity (6kg is always blue,
20kg always orange), ordinal sensitivity levels use a single-hue ramp,
status colours only for good/bad zones (always with a text label).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SCALE_COLOR = {"6kg": CAT[0], "20kg": CAT[1]}
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
ZONE_COLOR = {"R2 target": GOOD, "R1 too sensitive": SERIOUS, "R3 too insensitive": CRITICAL,
              "R1+R3 unstable": MUTED}
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
             "#104281", "#0d366b"]
GRAY_BAR = "#c3c2b7"


def level_color(level, levels):
    """Ordinal ramp: low sensitivity = light blue, high = dark blue."""
    levels = sorted(levels)
    i = levels.index(level) if level in levels else 0
    pos = int(round(i * (len(BLUE_RAMP) - 1) / max(1, len(levels) - 1)))
    return BLUE_RAMP[pos]


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "lines.markersize": 7, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def subtitle(ax, text):
    ax.text(0, 1.0, text, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom", ha="left")


def target_line(ax, value, axis="x", label=None):
    """Thin reference line; the axis is extended so the line and its label are always inside."""
    if axis == "x" and ax.get_xlim()[1] < value * 1.08:
        ax.set_xlim(right=value * 1.08)
    if axis == "y" and ax.get_ylim()[1] < value * 1.12:
        ax.set_ylim(top=value * 1.12)
    if axis == "x":
        ax.axvline(value, color=MUTED, lw=1, zorder=1)
        if label:
            ax.text(value, ax.get_ylim()[1], f" {label}", color=INK2, fontsize=8, va="top", ha="left")
    else:
        ax.axhline(value, color=MUTED, lw=1, zorder=1)
        if label:
            ax.text(ax.get_xlim()[1], value, f"{label} ", color=INK2, fontsize=8, va="bottom", ha="right")


def note(fig, text):
    fig.text(0.01, -0.02, text, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def scatter_points(ax, x, y, color, n_max=4000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) > n_max:
        i = np.random.default_rng(seed).choice(len(x), n_max, replace=False)
        x, y = x[i], y[i]
    ax.scatter(x, y, s=10, color=color, alpha=0.35, linewidths=0, zorder=3)


setup()
