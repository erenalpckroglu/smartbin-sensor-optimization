"""
Client deck for the CTO: SmartBin Cloud sensor sensitivity. 16:9 PowerPoint with speaker notes.

    python presentation/build_cto_deck.py

Story: the problem -> what we found -> what we recommend (flag, tune, watch) -> result -> next steps.
Plain language, one message per slide, charts instead of text. Reads only the aggregated tables in report/data/
(report/build_figures.py and report/robustness_loops.py), draws the charts into presentation/charts_cto/ and
writes the deck. Every number on a slide is computed from those tables.

Look: dark navy with mint and periwinkle accents, Space Grotesk for headings and numbers, DM Sans for text
(both free Google fonts). All design tokens sit at the top of this file.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_deck import REFS, facts, mi, n0, pc, rd, sp  # noqa: E402  (numbers and sources shared with the long deck)

CHARTS = HERE / "charts_cto"
DECK = HERE / "SmartBin – CTO presentation.pptx"

# ---------------------------------------------------------------------------------------------
# design tokens
# ---------------------------------------------------------------------------------------------
BG, BG2 = "0A1020", "0E1628"
WHITE, TXT, MUTED, SOFT = "FFFFFF", "B8C4D9", "8D99B0", "D7DEFF"
MINT, PERI, CORAL, AMBER = "5EEAD4", "8EA2FF", "FF8FA3", "FFC56E"
C = {"20kg": MINT, "6kg": PERI}
HEAD, BODY = "Space Grotesk", "DM Sans"
SCALES = ["20kg", "6kg"]
LABEL = {"20kg": "20 kg", "6kg": "6 kg"}
SW, SH = 13.333, 7.5
LM, CW = 0.89, 11.56
OWN = "Source: own analysis of the Bossard SmartBin data, 1 Mar – 20 Jun 2022"


# ---------------------------------------------------------------------------------------------
# facts from the robustness and evidence tables (scales caught in a trigger loop left out)
# ---------------------------------------------------------------------------------------------
def robust_facts() -> dict:
    R: dict = {}
    rc = rd("robust_coupling")
    R["coup"] = rc[(rc.source == "picks at the same site") & (rc["sample"] == "without loop scales")].set_index("scale_type")
    R["coup_all"] = rc[(rc.source == "picks at the same site") & (rc["sample"] == "all scales")].set_index("scale_type")
    R["placebo"] = rc[rc.source.str.startswith("placebo")].set_index("scale_type")
    R["per_scale"] = rd("coupling_per_scale").set_index("scale_type")
    R["dose"] = rd("coupling_dose")
    cor = rd("coupling_correlogram_no_loops")
    R["cor"] = cor
    z = cor[cor.lag_s == 0].set_index("scale_type")
    R["peak"], R["chance_rate"] = z.per_min, z.baseline
    k = rd("robust_knob")
    R["knob"] = k[k["sample"] == "without loop scales"].set_index(["scale_type", "sensitivity"])
    R["knob_all"] = k[k["sample"] == "all scales"].set_index(["scale_type", "sensitivity"])
    R["base"] = rd("robust_baseline").set_index(["sample", "scale_type"])
    g = rd("loop_threshold_grid")
    gg = g[g.min_share_retrigger.between(0.4, 0.6) & g.min_wakeups_per_day.between(10, 80)]
    R["grid_share"] = (gg.share_of_automatic.min(), gg.share_of_automatic.max())
    R["grid_scales"] = (int(gg.scales.min()), int(gg.scales.max()))
    dist = rd("loop_flag_distribution")
    q = lambda grp, m, qq: float(dist[(dist.group == grp) & (dist.measure == m) & (dist["quantile"] == qq)].value.iloc[0])
    R["loop_n_med"], R["loop_r_med"] = q("loop", "auto_per_day", 0.5), q("loop", "share_retrigger", 0.5)
    R["norm_n_med"], R["norm_r_med"] = q("no loop", "auto_per_day", 0.5), q("no loop", "share_retrigger", 0.5)
    R["norm_n_p99"], R["norm_r_p99"] = q("no loop", "auto_per_day", 0.99), q("no loop", "share_retrigger", 0.99)
    R["bins"] = rd("loop_detector_bins")
    R["delay"] = rd("miss_delay").set_index("scale_type")
    R["delay_b"] = rd("miss_delay_buckets")
    R["conc"] = rd("too_few_concentration")
    p = rd("robust_policy")
    R["pol_nl"] = p[p.effects == "estimated without loop scales"].set_index(["scale_type", "policy"])
    R["episode"] = rd("example_coupling_episode")
    R["loopday"] = rd("example_loop_day")
    R["lev_prof"] = rd("loop_evidence_profile").set_index("grp")
    R["lev_pers"] = rd("loop_evidence_persistence").iloc[0]
    R["lev_other"] = rd("loop_evidence_other_levels").set_index("group")
    R["lev_coup"] = rd("loop_evidence_coupling").set_index("grp")
    R["kpi_lv"] = rd("kpi_by_sensitivity")
    R["sites"] = rd("locations")
    return R


# ---------------------------------------------------------------------------------------------
# charts (dark, transparent, drawn at slide size)
# ---------------------------------------------------------------------------------------------
GRID = (1, 1, 1, 0.08)
AXIS = (1, 1, 1, 0.22)


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": BODY, "font.size": 12.5, "axes.labelsize": 12.5, "axes.labelcolor": "#" + TXT,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "xtick.color": "#" + TXT, "ytick.color": "#" + TXT,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 12, "legend.labelcolor": "#" + TXT,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2.6, "lines.solid_capstyle": "round",
        "axes.titlesize": 13.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": "#" + WHITE,
        "text.color": "#" + TXT, "axes.unicode_minus": True,
    })
    return plt


def hx(c, a=None):
    from matplotlib.colors import to_rgba
    return to_rgba("#" + c, a) if a is not None else "#" + c


def save(fig, name):
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / f"{name}.png", dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return CHARTS / f"{name}.png"


def patches(colors, labels, where, **kw):
    from matplotlib.patches import Patch
    return where.legend([Patch(color=hx(c)) for c in colors], labels, **kw)


def scale_legend(fig, **kw):
    kw.setdefault("ncol", 2)
    kw.setdefault("handlelength", 1.0)
    return patches([C["20kg"], C["6kg"]], ["20 kg scales", "6 kg scales"], fig, **kw)


def ch_donut(plt, N, R, w, h):
    v = N["share_empty_auto"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.pie([v, 1 - v], colors=[hx(CORAL), hx(MINT)], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.28, edgecolor=hx(BG2), linewidth=3))
    ax.text(0, 0.1, pc(v), ha="center", va="center", fontsize=40, fontweight="bold", color=hx(WHITE), family=HEAD)
    ax.text(0, -0.24, "found nothing", ha="center", va="center", fontsize=14, color=hx(TXT))
    ax.set_aspect("equal")
    return save(fig, "k01_donut")


def ch_one_sided(plt, N, R, w, h):
    b = N["base"]
    rows = [("Wakes up rarely enough\n(Bossard target)", "share_meets_tm"),
            ("Misses few picks\n(Bossard target)", "share_meets_tf")]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.34, right=0.95, top=0.84, bottom=0.14)
    ys = np.arange(len(rows))[::-1] * 1.0
    for i, st in enumerate(SCALES):
        vals = [b.loc[st, k] * 100 for _, k in rows]
        yy = ys + (0.19 if i == 0 else -0.19)
        ax.barh(yy, vals, height=0.34, color=hx(C[st]))
        for y, v in zip(yy, vals):
            ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=13, fontweight="bold", color=hx(C[st]))
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=13, colors=hx(WHITE))
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("share of scales on the default setting (8)", fontsize=12)
    scale_legend(fig, loc="upper left", bbox_to_anchor=(0.32, 1.0))
    return save(fig, "k02_one_sided")


def ch_kpi_levels(plt, N, R, w, h):
    k = R["kpi_lv"]
    lv = [8, 7, 6, 5, 4, 2, 1]
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.16, wspace=0.28)
    xs = np.arange(len(lv))
    for i, st in enumerate(SCALES):
        x = k[k.scale_type == st].set_index("sensitivity").reindex(lv)
        off = -0.19 if i == 0 else 0.19
        axes[0].bar(xs + off, x.empty_per_week, width=0.36, color=hx(C[st]))
        axes[1].bar(xs + off, x.tm_fraction, width=0.36, color=hx(C[st]))
    axes[0].set_title("Empty wake-ups per week")
    axes[1].set_title("Bossard KPI · target 0.5 (dashed)")
    axes[1].axhline(0.5, color=hx(WHITE), lw=1.2, ls=(0, (4, 3)))
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    for ax in axes:
        ax.set_xticks(xs, [str(l) for l in lv])
        ax.set_xlabel("sensor setting (8 = default)", fontsize=11.5)
    scale_legend(fig, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    return save(fig, "k03_kpi_levels")


def ch_episode(plt, N, R, w, h):
    e = R["episode"].copy()
    bins = sorted(e.bin.unique())
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.2)
    ax.axvspan(-14, 14, color=hx(MINT, 0.1), lw=0)
    for i, b in enumerate(bins):
        y = len(bins) - 1 - i
        ax.axhline(y, color=GRID, lw=1, zorder=0)
        for _, r in e[e.bin == b].iterrows():
            pick = r.cls == "OK"
            ax.scatter(r.t_rel_s, y, s=260 if pick else 190, zorder=3, color=hx(MINT) if pick else hx(BG2),
                       edgecolor=hx(MINT) if pick else hx(CORAL), linewidth=2.6)
            ax.annotate("real pick" if pick else "woke up, found nothing", (r.t_rel_s, y), textcoords="offset points",
                        xytext=(16, 0), ha="left", va="center", fontsize=12, color=hx(MINT) if pick else hx(CORAL))
    ax.set_yticks(range(len(bins)), bins[::-1])
    ax.tick_params(axis="y", labelsize=13, colors=hx(WHITE))
    ax.set_ylim(-0.6, len(bins) - 0.4)
    ax.set_xlim(-60, 60)
    ax.set_xticks([-60, -30, 0, 30, 60], ["−60 s", "−30 s", "0", "+30 s", "+60 s"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("seconds around the pick at Bin A", fontsize=12)
    return save(fig, "k04_episode")


def ch_correlogram(plt, N, R, w, h):
    cor = R["cor"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.18)
    for st in SCALES:
        x = cor[(cor.scale_type == st) & (cor.lag_s.abs() <= 116)].sort_values("lag_s")
        ax.plot(x.lag_s, x.per_min, color=hx(C[st]), lw=2.4)
    ax.axhline(R["chance_rate"].mean(), color=hx(WHITE, 0.6), lw=1, ls=(0, (3, 3)))
    ax.text(114, R["chance_rate"].mean() + 0.4, "normal level", fontsize=11.5, ha="right", color=hx(TXT))
    ax.set_xlim(-116, 116)
    ax.set_ylim(0, 13.5)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xlabel("seconds before / after an empty wake-up", fontsize=12)
    ax.set_ylabel("picks per minute nearby", fontsize=12)
    scale_legend(fig, loc="upper left", bbox_to_anchor=(0.12, 1.0))
    return save(fig, "k05_correlogram")


def ch_placebo(plt, N, R, w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.22)
    xs = np.arange(2)
    for i, st in enumerate(SCALES):
        v = [R["coup"].loc[st, "coupled_beyond_chance"] * 100, R["placebo"].loc[st, "coupled_beyond_chance"] * 100]
        off = -0.19 if i == 0 else 0.19
        ax.bar(xs + off, v, width=0.36, color=hx(C[st]))
        for xx, vv in zip(xs + off, v):
            ax.text(xx, vv + 1.5, f"{vv:.0f}%" if vv >= 1 else f"{vv:.1f}%", ha="center", fontsize=13,
                    fontweight="bold", color=hx(C[st]))
    ax.set_xticks(xs, ["picks in the\nsame factory", "picks in another\nfactory (check)"])
    ax.tick_params(axis="x", labelsize=12.5, colors=hx(WHITE))
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 20, 40, 60], ["0%", "20%", "40%", "60%"])
    ax.set_ylabel("empty wake-ups linked to a pick", fontsize=12)
    return save(fig, "k06_placebo")


def ch_loopday(plt, N, R, w, h):
    ld = R["loopday"]
    x = ld[ld.series == "looping scale"].sort_values("hour")
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.17)
    ax.bar(x.hour, x.wakeups, color=hx(CORAL), width=0.72)
    ax.set_xticks([0, 6, 12, 18, 23], ["0 h", "6 h", "12 h", "18 h", "23 h"])
    ax.set_xlim(-0.7, 23.7)
    ax.set_ylim(0, 62)
    ax.set_ylabel("wake-ups per hour", fontsize=12)
    ax.text(0, 60, f"one looping scale, one day: {int(x.wakeups.sum())} wake-ups", fontsize=12.5, color=hx(WHITE),
            fontweight="bold", va="top")
    ax.text(0, 54.5, f"a normal scale: about {R['norm_n_med']:.0f} a day", fontsize=12, color=hx(MINT), va="top")
    return save(fig, "k07_loopday")


def ch_loop_evidence(plt, N, R, w, h):
    p, cp = R["lev_prof"], R["lev_coup"]
    rows = [("wake-ups at night (0–5 h)", p.loc["trigger loop", "share_night_0_5h"], p.loc["normal", "share_night_0_5h"]),
            ("wake-ups at the weekend", p.loc["trigger loop", "share_weekend"], p.loc["normal", "share_weekend"]),
            ("exactly 90 s after the last one", p.loc["trigger loop", "share_gap_85_95s"], p.loc["normal", "share_gap_85_95s"]),
            ("linked to a pick nearby", max(cp.loc["trigger loop", "coupled_beyond_chance"], 0),
             cp.loc["normal", "coupled_beyond_chance"])]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.45, right=0.95, top=0.86, bottom=0.04)
    ys = np.arange(len(rows))[::-1].astype(float)
    for j, (col, off) in enumerate([(CORAL, 0.18), (MINT, -0.18)]):
        vals = [r[1 + j] * 100 for r in rows]
        ax.barh(ys + off, vals, height=0.32, color=hx(col))
        for y, v in zip(ys + off, vals):
            ax.text(v + 1.2, y, f"{v:.0f}%", va="center", fontsize=12, fontweight="bold", color=hx(col))
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=12.5, colors=hx(WHITE))
    ax.set_xlim(0, 70)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    patches([CORAL, MINT], ["looping scales", "normal scales"], fig, loc="upper left", ncol=2,
            bbox_to_anchor=(0.43, 1.0), handlelength=1.0)
    return save(fig, "k08_loop_evidence")


def ch_knob(plt, N, R, w, h):
    k = R["knob"]
    lv = [7, 6, 5, 4, 2, 1]
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.8, bottom=0.16, wspace=0.3)
    xs = np.arange(len(lv))
    for i, st in enumerate(SCALES):
        x = k.loc[st].reindex([float(l) for l in lv])
        off = -0.19 if i == 0 else 0.19
        axes[0].bar(xs + off, -x.empty_wakeups_change * 100, width=0.36, color=hx(C[st]))
        axes[1].bar(xs + off, x.missed_pp, width=0.36, color=hx(C[st]))
    axes[0].set_title("Fewer empty wake-ups")
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    axes[1].set_title("More picks missed (points)")
    axes[1].set_ylim(0, 70)
    for ax in axes:
        ax.set_xticks(xs, [f"8→{l}" for l in lv])
        ax.axvspan(-0.5, 0.5, color=hx(MINT, 0.12), lw=0)
        ax.set_xlabel("change of the setting (same scales)", fontsize=11.5)
    scale_legend(fig, loc="upper left", bbox_to_anchor=(0.08, 1.0))
    return save(fig, "k09_knob")


def ch_sites(plt, N, R, w, h):
    s = R["sites"]
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.12, wspace=0.08)
    for ax, st in zip(axes, SCALES):
        x = s[(s.scale_type == st) & (s.devices >= 20)].sort_values("frac_too_many_median")
        xs = np.arange(len(x))
        ax.bar(xs, x.frac_too_many_median, color=hx(C[st]), width=0.75)
        ax.axhline(0.5, color=hx(WHITE), lw=1.2, ls=(0, (4, 3)))
        ax.set_title(f"{LABEL[st]} scales · {len(x)} factories", color=hx(C[st]))
        ax.set_xticks([])
        ax.set_xlabel("factories, best to worst", fontsize=12)
        ax.set_ylim(0, 1)
    axes[0].text(0, 0.52, "target", fontsize=11.5, color=hx(WHITE), va="bottom")
    axes[0].set_ylabel("share of empty wake-ups (setting 8)", fontsize=12)
    return save(fig, "k10_sites")


def ch_drivers(plt, N, R, w, h):
    s = N["spread"]
    order = ["Site (location)", "usage", "Sensitivity level", "Fill level (weight bin)", "Box type",
             "Item weight (quartile)"]
    names = ["Factory", "How often the bin is used", "Sensor setting", "How full the box is", "Box type",
             "Item weight"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.47, right=0.97, top=0.88, bottom=0.04)
    ys = np.arange(len(order))[::-1].astype(float)
    for i, st in enumerate(SCALES):
        v = s[s.scale_type == st].set_index("parameter").spread.to_dict()
        lo_, hi_ = N["usage"][st]
        v["usage"] = lo_ - hi_
        vals = [v[o] for o in order]
        yy = ys + (0.18 if i == 0 else -0.18)
        ax.barh(yy, vals, height=0.32, color=hx(C[st]))
    ax.set_yticks(ys, names)
    ax.tick_params(axis="y", labelsize=12.5, colors=hx(WHITE))
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("What makes the biggest difference", color=hx(WHITE), loc="right")
    return save(fig, "k11_drivers")


def ch_detector(plt, N, R, w, h):
    b = R["bins"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.15)
    ax.axvspan(20, 1500, ymin=0.5, ymax=1, color=hx(CORAL, 0.08), lw=0)
    for loop, col in [(False, MINT), (True, CORAL)]:
        x = b[b.loop == loop]
        ax.scatter(10 ** (x.x + 0.05), (x.y + 0.02) * 100, s=np.clip(np.sqrt(x.scale_levels) * 9, 6, 260),
                   color=hx(col), alpha=0.6, lw=0, label="looping" if loop else "normal")
    ax.axvline(20, color=hx(WHITE, 0.7), lw=1.1, ls=(0, (4, 3)))
    ax.axhline(50, color=hx(WHITE, 0.7), lw=1.1, ls=(0, (4, 3)))
    ax.text(26, 96, "looping", fontsize=13, color=hx(CORAL), fontweight="bold", va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1500)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("wake-ups per day", fontsize=12)
    ax.set_ylabel("share within 2 minutes\nof the last wake-up", fontsize=12)
    ax.legend(loc="upper left", fontsize=12, markerscale=0.8)
    return save(fig, "k12_detector")


def ch_tune_example(plt, N, R, w, h):
    weeks = [0, 2, 4, 6, 8]
    level = [7, 6, 5, 6, 6]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.17)
    ax.step(weeks, level, where="post", color=hx(MINT), lw=3)
    ax.scatter(weeks[:-1], level[:-1], s=90, color=hx(MINT), zorder=3)
    notes_ = [(0, 7, "too many empty\nwake-ups → down"), (2, 6, "still too many\n→ down"),
              (4, 5, "missed a pick\n→ up"), (6, 6, "fine → stays")]
    for x, y, t in notes_:
        ax.annotate(t, (x + 1, y), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=11.5,
                    color=hx(WHITE), va="bottom", linespacing=1.05)
    ax.set_xlim(0, 8)
    ax.set_ylim(4.3, 8.4)
    ax.set_yticks([5, 6, 7, 8])
    ax.set_xticks(weeks, ["week 0", "2", "4", "6", "8"])
    ax.set_ylabel("sensor setting", fontsize=12)
    return save(fig, "k13_tune_example")


def ch_delay(plt, N, R, w, h):
    d = R["delay_b"]
    order = ["< 24 h", "24–48 h", "48–72 h", "> 72 h"]
    cols = [MINT, "3FB9A8", AMBER, CORAL]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.76, bottom=0.08)
    for yi, st in enumerate(SCALES[::-1]):
        x = d[d.scale_type == st].set_index("found_after").reindex(order).too_few
        x = x / x.sum() * 100
        left = 0
        for o, c in zip(order, cols):
            v = x.loc[o]
            ax.barh(yi, v, left=left, color=hx(c), height=0.55, edgecolor=hx(BG2), linewidth=1.5)
            if v >= 6:
                ax.text(left + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold",
                        color=hx(BG))
            left += v
    ax.set_yticks([0, 1], [LABEL[s] for s in SCALES[::-1]])
    ax.tick_params(axis="y", labelsize=13, colors=hx(WHITE))
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    patches(cols, ["within 1 day", "1–2 days", "2–3 days", "over 3 days"], fig, loc="upper left", ncol=4,
            bbox_to_anchor=(0.12, 1.0), handlelength=1.0, fontsize=11.5, columnspacing=1.0)
    return save(fig, "k14_delay")


def ch_concentration(plt, N, R, w, h):
    c = R["conc"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.18)
    for st in SCALES:
        x = c[c.scale_type == st]
        ax.plot(x.share_of_scales * 100, x.share_of_too_few * 100, color=hx(C[st]), lw=2.6)
        y10 = float(x[x.share_of_scales.round(2) == 0.10].share_of_too_few.iloc[0]) * 100
        ax.scatter([10], [y10], color=hx(C[st]), s=60, zorder=3)
        ax.annotate(f"{y10:.0f}%", (10, y10), textcoords="offset points", xytext=(10, -4), fontsize=12.5,
                    fontweight="bold", color=hx(C[st]), va="top")
    ax.axvline(10, color=hx(WHITE, 0.5), lw=1, ls=(0, (3, 3)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 10, 25, 50, 75, 100], ["0%", "10%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of scales (most misses first)", fontsize=12)
    ax.set_ylabel("share of all missed picks", fontsize=12)
    scale_legend(fig, loc="lower right", bbox_to_anchor=(0.96, 0.2), ncol=1)
    return save(fig, "k15_concentration")


def ch_result(plt, N, R, w, h):
    pol, P = N["pol"], N["P"]
    opts = [("P0", "Today"), ("P1", "One for\nall"), ("P8", "Steer by\nKPI"), ("P7", "Our\nrule")]
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.17, wspace=0.25)
    xs = np.arange(len(opts))
    for i, st in enumerate(SCALES):
        off = -0.19 if i == 0 else 0.19
        sv = [-pol.loc[(st, P[k]), "meas_change_pct_without_loops"] for k, _ in opts]
        ms = [pol.loc[(st, P[k]), "median_miss_rate"] * 100 for k, _ in opts]
        axes[0].bar(xs + off, sv, width=0.36, color=hx(C[st]))
        axes[1].bar(xs + off, ms, width=0.36, color=hx(C[st]))
        for xx, v in zip(xs + off, sv):
            if v > 1:
                axes[0].text(xx, v + 1.5, f"{v:.0f}", ha="center", fontsize=11, color=hx(C[st]))
        for xx, v in zip(xs + off, ms):
            axes[1].text(xx, v + 1.0, f"{v:.0f}", ha="center", fontsize=11, color=hx(C[st]))
    axes[0].set_title("Weighings saved (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_title("Picks missed, typical scale (%)")
    axes[1].set_ylim(0, 56)
    axes[1].axhline(10, color=hx(WHITE), lw=1.1, ls=(0, (4, 3)))
    axes[1].text(-0.45, 11, "target 10%", fontsize=11, color=hx(WHITE), va="bottom")
    for ax in axes:
        ax.set_xticks(xs, [o[1] for o in opts])
        ax.tick_params(axis="x", labelsize=11.5, colors=hx(WHITE))
        ax.axvspan(2.5, 3.5, color=hx(MINT, 0.1), lw=0)
    scale_legend(fig, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    return save(fig, "k16_result")


def ch_cleaning(plt, N, R, w, h):
    fu = N["funnel"]
    names = {"A2_duplicate": "Duplicate rows", "A5_weight_invalid": "Impossible weights",
             "B0_duplicate_label_conflict": "Duplicates with different labels", "B1_label_missing": "Label missing",
             "B3_sensitivity_missing": "Setting missing",
             "B4_trigger_config_nonstandard": "Other wait time / delay",
             "B5_device_metadata_missing": "Scale type or site unknown",
             "B6_section_not_relevant": "Too short periods",
             "B7_sensitivity_anomaly": "Setting 3 (studied apart)", "B8_sensitivity_low_support": "Rare settings"}
    x = fu[fu.rule.isin(names) & (fu.rows_removed > 0)].copy()
    x["name"] = x.rule.map(names)
    x = x.sort_values("rows_removed")
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.4, right=0.95, top=0.98, bottom=0.24)
    col = [CORAL if a == "delete row" else (AMBER if r.startswith("B7") else PERI) for a, r in zip(x.action, x.rule)]
    ax.barh(x.name, x.rows_removed, color=[hx(c) for c in col], height=0.62)
    for yi, v in enumerate(x.rows_removed):
        ax.text(v * 1.3, yi, n0(v), va="center", fontsize=11.5, color=hx(TXT))
    ax.set_xscale("log")
    ax.set_xlim(2, 3e7)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    ax.tick_params(axis="y", labelsize=12, colors=hx(WHITE))
    ax.set_xticks([10, 1e3, 1e5, 1e7], ["10", "1,000", "100,000", "10 M"])
    ax.set_xlabel("rows (log scale)", fontsize=12)
    patches([CORAL, PERI, AMBER], ["deleted", "kept, not used for the KPIs", "setting 3"], fig,
            loc="lower center", ncol=3, fontsize=11.5, bbox_to_anchor=(0.6, 0.0), handlelength=1.0)
    return save(fig, "k17_cleaning")


def ch_pick_heat(plt, N, R, w, h):
    z = N["zones"]
    order = z.drop_duplicates("pick_bin").sort_values("bin_idx").pick_bin.tolist()
    lv = [8, 7, 6, 5, 4, 2, 1]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("d", [hx(CORAL), hx(AMBER), hx(MINT)])
    cmap.set_bad((1, 1, 1, 0.06))
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.25, wspace=0.06)
    for ax, st in zip(axes, SCALES):
        x = z[z.scale_type == st]
        M = np.full((len(lv), len(order)), np.nan)
        for i, l in enumerate(lv):
            for j, o in enumerate(order):
                r = x[(x.sensitivity == l) & (x.pick_bin == o) & x.measurable]
                if len(r):
                    M[i, j] = r.detection.iloc[0] * 100
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=40, vmax=100, aspect="auto")
        for i in range(len(lv)):
            for j in range(len(order)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9.5, color=hx(BG))
        ax.set_xticks(range(len(order)), [o.replace(">500", "500+") for o in order], rotation=45, ha="right",
                      fontsize=11)
        ax.set_yticks(range(len(lv)), [str(l) for l in lv])
        ax.set_title(f"{LABEL[st]} scales", color=hx(C[st]))
        ax.set_xlabel("size of the pick (g)", fontsize=12)
        ax.grid(False)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
    axes[0].set_ylabel("sensor setting", fontsize=12)
    return save(fig, "k18_pick_heat")


# ---------------------------------------------------------------------------------------------
# PowerPoint helpers
# ---------------------------------------------------------------------------------------------
from pptx import Presentation                                   # noqa: E402
from pptx.dml.color import RGBColor                             # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE           # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN                 # noqa: E402
from pptx.oxml.ns import qn                                     # noqa: E402
from pptx.util import Emu, Inches, Pt                           # noqa: E402
from lxml import etree                                          # noqa: E402

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_FONTS: dict = {}


def _font(family, bold, size):
    key = (family, bold, size)
    if key not in _FONTS:
        try:
            from matplotlib import font_manager as fm
            from PIL import ImageFont
            path = fm.findfont(fm.FontProperties(family=family, weight="bold" if bold else "normal"),
                               fallback_to_default=False)
            _FONTS[key] = ImageFont.truetype(path, size=int(round(size * 10)))
        except Exception:
            _FONTS[key] = None
    return _FONTS[key]


def n_lines(text, width_in, size, family=BODY, bold=False):
    f = _font(family, bold, size)
    total = 0
    for para in text.split("\n"):
        words = para.replace("**", "").split()
        if not words:
            total += 1
            continue
        if f is None:
            total += max(1, int(np.ceil(len(" ".join(words)) * size * 0.55 / (width_in * 72))))
            continue
        maxw, line_, n = width_in * 720 * 0.97, "", 1
        for w in words:
            t = (line_ + " " + w).strip()
            if f.getlength(t) <= maxw:
                line_ = t
            else:
                n, line_ = n + 1, w
        total += n
    return total


def text_h(paras, width_in, size, line=1.12, family=BODY, bold=False, space_after=0, bullet_indent=0.22):
    if isinstance(paras, str):
        paras = paras.split("\n")
    h = 0.0
    for p in paras:
        w = width_in - (bullet_indent if p.startswith("• ") else 0)
        h += n_lines(p[2:] if p.startswith("• ") else p, w, size, family, bold) * size * 1.2 * line / 72
        h += space_after / 72
    return h


def rgb(c):
    return RGBColor.from_string(c)


def _clean(shape_):
    st = shape_._element.find(qn("p:style"))
    if st is not None:
        shape_._element.remove(st)


def _runs(text):
    parts = text.split("**")
    return [(p, i % 2 == 1) for i, p in enumerate(parts) if p]


def _solid(color, alpha=None):
    sf = etree.Element(qn("a:solidFill"))
    c = etree.SubElement(sf, qn("a:srgbClr"))
    c.set("val", color)
    if alpha is not None:
        etree.SubElement(c, qn("a:alpha")).set("val", str(int(alpha * 100000)))
    return sf


def style_shape(sh, fill=None, fill_a=None, line=None, line_a=None, lw=0.5):
    spPr = sh._element.spPr
    for tag in ("a:solidFill", "a:noFill", "a:gradFill", "a:ln"):
        for el in spPr.findall(qn(tag)):
            spPr.remove(el)
    geom = spPr.find(qn("a:prstGeom"))
    idx = list(spPr).index(geom) + 1
    spPr.insert(idx, _solid(fill, fill_a) if fill else etree.Element(qn("a:noFill")))
    ln = etree.SubElement(spPr, qn("a:ln"))
    if line:
        ln.set("w", str(int(Pt(lw))))
        ln.append(_solid(line, line_a))
    else:
        etree.SubElement(ln, qn("a:noFill"))


def shape(s, x, y, w, h, kind=MSO_SHAPE.RECTANGLE, radius=None, **style):
    sh = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    _clean(sh)
    if radius is not None and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    style_shape(sh, **style)
    return sh


def card(s, x, y, w, h, accent=None, strong=False):
    """rounded card as in the reference deck: tinted fill and a thin tinted border"""
    r = min(0.5, 0.12 / min(w, h))
    if accent:
        return shape(s, x, y, w, h, MSO_SHAPE.ROUNDED_RECTANGLE, r, fill=accent, fill_a=0.13 if strong else 0.08,
                     line=accent, line_a=0.5)
    return shape(s, x, y, w, h, MSO_SHAPE.ROUNDED_RECTANGLE, r, fill=WHITE, fill_a=0.05, line=WHITE, line_a=0.12)


def tb(s, x, y, w, h, paras, size=14, color=TXT, family=BODY, bold=False, align="l", anchor="t", line=1.12,
       space_after=0, spc=None, bullet_indent=0.22, bold_color=WHITE, bullet_color=MINT):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(max(h, 0.2)))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, str):
        paras = paras.split("\n")
    for i, ptxt in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        p.line_spacing = line
        if space_after:
            p.space_after = Pt(space_after)
        if ptxt.startswith("• "):
            ptxt = ptxt[2:]
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(bullet_indent))))
            pPr.set("indent", str(-int(Inches(bullet_indent))))
            bc = etree.SubElement(pPr, qn("a:buClr"))
            etree.SubElement(bc, qn("a:srgbClr")).set("val", bullet_color)
            etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
            etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
        for t, b in _runs(ptxt) or [("", False)]:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name, f.size, f.bold = family, Pt(size), (bold or b)
            f.color.rgb = rgb(bold_color if (b and not bold) else color)
            if spc:
                r._r.get_or_add_rPr().set("spc", str(spc))
    return y + text_h(paras, w, size, line, family, bold, space_after, bullet_indent)


def big(s, x, y, w, value, caption, color=MINT, size=36, cap_size=13):
    nh = size / 72 * 1.1
    tb(s, x, y, w, nh, value, size=size, family=HEAD, bold=True, color=color, line=0.95)
    return tb(s, x, y + nh + 0.04, w, 0.3, caption, size=cap_size, color=TXT, line=1.12)


def head(s, eyebrow, title, size=28):
    tb(s, LM, 0.55, CW, 0.25, eyebrow, size=11, bold=True, color=MINT, spc=150)
    lines = n_lines(title, CW, size, HEAD, True)
    h = size / 72 * 1.15 * lines
    tb(s, LM, 0.85, CW, h, title, size=size, family=HEAD, bold=True, color=WHITE, line=1.0)
    return 0.85 + h + 0.35


def footer(s, text, number):
    tb(s, LM, 7.02, 10.8, 0.25, text, size=9, color=MUTED, line=1.0)
    tb(s, 11.95, 7.02, 0.5, 0.25, str(number), size=9, color=MUTED, align="r")


def pic(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def arrow(s, x1, y1, x2, y2, color=MINT, lw=1.75):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(lw)
    te = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"))
    te.set("type", "triangle")
    te.set("w", "med")
    te.set("len", "med")
    return c


def badge(s, x, y, d, text, fill=MINT, color=BG, size=13):
    sh = shape(s, x, y, d, d, MSO_SHAPE.OVAL, fill=fill)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.name, r.font.size, r.font.bold = HEAD, Pt(size), True
    r.font.color.rgb = rgb(color)
    return sh


def pill(s, x, y, text, color=MINT, size=12, h=0.34):
    f = _font(BODY, True, size)
    wtxt = (f.getlength(text) / 720) if f else len(text) * size * 0.55 / 72
    w = wtxt + 0.4
    sh = shape(s, x, y, w, h, MSO_SHAPE.ROUNDED_RECTANGLE, 0.5, fill=color, fill_a=0.16, line=color, line_a=0.45)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.name, r.font.size, r.font.bold = BODY, Pt(size), True
    r.font.color.rgb = rgb(color)
    return x + w


def glow(s):
    """soft radial glow in the top-right corner, as in the reference deck"""
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.64), 0, Inches(5.69), Inches(4.44))
    _clean(sh)
    spPr = sh._element.spPr
    for tag in ("a:solidFill", "a:noFill", "a:ln"):
        for el in spPr.findall(qn(tag)):
            spPr.remove(el)
    grad = etree.fromstring(
        f'<a:gradFill xmlns:a="{A_NS}" rotWithShape="1"><a:gsLst>'
        f'<a:gs pos="0"><a:srgbClr val="{PERI}"><a:alpha val="18000"/></a:srgbClr></a:gs>'
        f'<a:gs pos="65000"><a:srgbClr val="{PERI}"><a:alpha val="0"/></a:srgbClr></a:gs></a:gsLst>'
        f'<a:path path="circle"><a:fillToRect l="70000" t="25000" r="30000" b="75000"/></a:path></a:gradFill>')
    geom = spPr.find(qn("a:prstGeom"))
    spPr.insert(list(spPr).index(geom) + 1, grad)
    ln = etree.SubElement(spPr, qn("a:ln"))
    etree.SubElement(ln, qn("a:noFill"))
    return sh


def table(s, x, y, w, col_w, rows, size=12.5, row_h=0.42, align=None, col_colors=None, hl_rows=()):
    nrow, ncol = len(rows), len(rows[0])
    gt = s.shapes.add_table(nrow, ncol, Inches(x), Inches(y), Inches(w), Inches(row_h * nrow))
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    for a in ["firstRow", "bandRow"]:
        tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        tblPr.remove(sid)
    tot = sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Emu(int(Inches(w) * cw / tot))
    for i in range(nrow):
        tbl.rows[i].height = Inches(row_h)
        for j in range(ncol):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.1)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = {"l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT, "c": PP_ALIGN.CENTER}[(align or "l" * ncol)[j]]
            is_head = i == 0
            for t, b in _runs(str(rows[i][j])) or [("", False)]:
                r = p.add_run()
                r.text = t
                r.font.name, r.font.size = (HEAD if is_head else BODY), Pt(size)
                r.font.bold = b or is_head
                c = MINT if is_head else (col_colors[j] if (col_colors and col_colors[j]) else TXT)
                if b and not is_head:
                    c = WHITE
                r.font.color.rgb = rgb(c)
            tcPr = cell._tc.get_or_add_tcPr()
            for side, show in [("a:lnL", False), ("a:lnR", False), ("a:lnT", False), ("a:lnB", True)]:
                ln = etree.SubElement(tcPr, qn(side))
                if show:
                    ln.set("w", "6350")
                    ln.append(_solid(WHITE, 0.14 if not is_head else 0.4))
                else:
                    ln.set("w", "0")
                    etree.SubElement(ln, qn("a:noFill"))
            if i in hl_rows:
                tcPr.append(_solid(MINT, 0.1))
            else:
                etree.SubElement(tcPr, qn("a:noFill"))
    return gt


def notes(s, text):
    s.notes_slide.notes_text_frame.text = "\n".join(t.strip() for t in text.strip().split("\n"))


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(SW), Inches(SH)
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, alt=False, glow_=True):
        s = self.prs.slides.add_slide(self.blank)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(BG2 if alt else BG)
        if glow_:
            glow(s)
        self.n += 1
        return s

    def save(self, path):
        cp = self.prs.core_properties
        cp.author = cp.last_modified_by = "Erenalp Çakıroğlu"
        cp.title = "SmartBin Cloud – wake up for picks, not for noise"
        cp.subject = "HSLU · Data-driven Supply Chain Management and Logistics · Smart Logistics Service Optimisation"
        cp.keywords = "SmartBin; IoT; sensitivity; Bossard; case study"
        cp.comments = ""
        cp.created = cp.modified = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0, tzinfo=None)
        cp.revision = 1
        self.prs.save(str(path))


# ---------------------------------------------------------------------------------------------
# the story
# ---------------------------------------------------------------------------------------------
def build():
    N = facts()
    R = robust_facts()
    plt = mpl()
    b = N["base"]
    pol, P = N["pol"], N["P"]
    rk, coup = R["knob"], R["coup"]
    d = Deck()

    pct0 = lambda v: mi(f"{v:.0f}") + "%"
    c_both = coup.loc["both", "coupled_beyond_chance"]
    c_both_all = R["coup_all"].loc["both", "coupled_beyond_chance"]
    p7 = {st: pol.loc[(st, P["P7"])] for st in SCALES}
    p7_nl = {st: R["pol_nl"].loc[(st, P["P7"])] for st in SCALES}
    save20, save6 = -p7["20kg"].meas_change_pct_without_loops, -p7["6kg"].meas_change_pct_without_loops
    miss20, miss6 = p7["20kg"].median_miss_rate, p7["6kg"].median_miss_rate
    loop_share = N["loop_share_auto"]
    picks20 = R["base"].loc[("all scales", "20kg"), "picks_per_week"]
    fleet = N["fleet_saved_week"]
    lo3 = R["lev_other"].loc["loop on level 3"]
    e7 = {st: -rk.loc[(st, 7.0), "empty_wakeups_change"] for st in SCALES}
    m7 = {st: rk.loc[(st, 7.0), "missed_pp"] for st in SCALES}

    # ---- 1 title --------------------------------------------------------------------------------
    s = d.slide()
    tb(s, LM, 0.9, 7, 0.3, "SMARTBIN CLOUD · BOSSARD × D ONE · HSLU CASE STUDY", size=11, bold=True, color=MINT,
       spc=150)
    tb(s, LM, 1.35, 7.2, 2.2, ["Wake up for picks,", "not for noise"], size=46, family=HEAD, bold=True,
       color=WHITE, line=1.0)
    tb(s, LM, 3.15, 6.6, 0.9, "How each SmartBin scale can find its own sensor setting", size=19, color=TXT,
       line=1.15)
    tb(s, LM, 6.3, 7, 0.3, "Erenalp Çakıroğlu", size=15, family=HEAD, bold=True, color=WHITE)
    tb(s, LM, 6.62, 7, 0.3, "MSc Applied Information and Data Science · HSLU Lucerne", size=12.5, color=TXT)
    stats = [(f"{pct0(-save20)} / {pct0(-save6)}", "fewer weighings with our rule", MINT),
             (pc(c_both), "of empty wake-ups happen when someone picks nearby", PERI),
             (f"{N['loop_share_scales'] * 100:.1f}% → {loop_share * 100:.0f}%",
              "1.5% of the scales cause 27% of all weighings", CORAL)]
    y = 1.35
    for v, cap, col in stats:
        card(s, 8.45, y, 4.0, 1.42, accent=col)
        tb(s, 8.7, y + 0.18, 3.6, 0.6, v, size=30, family=HEAD, bold=True, color=col)
        tb(s, 8.7, y + 0.82, 3.6, 0.5, cap, size=12.5, color=SOFT, line=1.1)
        y += 1.62
    notes(s, """(30 seconds)
Thank you for having us. This talk is about the SmartBin scales.
Today most of their weighings find nothing.
We looked at 7.6 million weighings and found out why.
We also found a simple way for every scale to find its own setting.
The three numbers on the right tell the whole story.
We can cut the weighings by more than half.
More than half of the empty wake-ups come from picking nearby.
And a very small group of scales causes a quarter of the load.""")

    # ---- 2 problem ------------------------------------------------------------------------------
    s = d.slide(alt=True)
    y = head(s, "THE PROBLEM", "3 out of 4 automatic weighings find nothing")
    steps = [("1", "Pick", "A worker takes parts from a bin."), ("2", "Wake up", "A vibration sensor wakes the scale."),
             ("3", "Weigh", "The scale weighs the bin."), ("4", "Reorder", "ARIMS gets the new stock and reorders.")]
    bw, gap = 1.72, 0.22
    for i, (n_, t, body) in enumerate(steps):
        x = LM + i * (bw + gap)
        col = MINT if i < 3 else PERI
        card(s, x, y, bw, 1.35, accent=col)
        badge(s, x + 0.15, y + 0.16, 0.3, n_, fill=col, size=11)
        tb(s, x + 0.55, y + 0.17, bw - 0.6, 0.3, t, size=14, family=HEAD, bold=True, color=WHITE)
        tb(s, x + 0.15, y + 0.6, bw - 0.3, 0.7, body, size=12, color=TXT, line=1.1)
        if i < 3:
            arrow(s, x + bw + 0.03, y + 0.67, x + bw + gap - 0.03, y + 0.67)
    yy = y + 1.65
    cw2 = (4 * bw + 3 * gap - 0.25) / 2
    for i, (t, body, col) in enumerate([("Too sensitive", "The scale wakes up, but nothing was taken. This wastes "
                                          "battery, network and cloud capacity.", CORAL),
                                         ("Not sensitive enough", "The scale sleeps through a pick. The stock level is "
                                          "wrong until the next weighing.", AMBER)]):
        x = LM + i * (cw2 + 0.25)
        card(s, x, yy, cw2, 1.55, accent=col)
        tb(s, x + 0.2, yy + 0.18, cw2 - 0.4, 0.3, t, size=15, family=HEAD, bold=True, color=col)
        tb(s, x + 0.2, yy + 0.58, cw2 - 0.4, 0.9, body, size=12.5, color=TXT, line=1.12)
    tb(s, LM, yy + 1.85, 7.5, 0.6, "Every weighing costs energy. It uses about 3,000 times more power than sleeping.",
       size=12.5, color=SOFT)
    pic(s, ch_donut(plt, N, R, 3.6, 3.6), 8.95, y - 0.05, w=3.6)
    tb(s, 8.6, y + 3.7, 4.3, 0.6, f"{N['auto_all'] / 1e6:.1f} million automatic weighings, March–June 2022",
       size=12, color=TXT, align="c")
    footer(s, OWN + " · case description Bossard / D ONE (2026) · Report §2, §4.1", d.n)
    notes(s, """(1 minute)
A SmartBin is a scale under a storage bin.
When a worker takes parts, a sensor feels the vibration and wakes the scale up.
The scale weighs the bin and sends the new stock level to ARIMS. ARIMS then reorders.
The problem is that three out of four of these weighings find nothing.
Each one costs battery, network and cloud capacity.
If we make the sensor less sensitive, it may sleep through a real pick.
Then the stock level is wrong for a while.
So the question is simple. How should the sensor be set?""")

    # ---- 3 data and approach --------------------------------------------------------------------
    s = d.slide()
    y = head(s, "DATA AND APPROACH", "We cleaned 7.6 million weighings and looked at every scale on its own")
    stats = [(f"{N['rows'] / 1e6:.1f} M", "weighings"), (f"{N['scales']:,}", "scales"), ("16 weeks", "March – June 2022"),
             (f"{N['switchers']:,}", "scales got a new setting in April")]
    sw_ = CW / 4
    for i, (v, cap) in enumerate(stats):
        big(s, LM + i * sw_, y - 0.1, sw_ - 0.2, v, cap, size=34, cap_size=13)
    y2 = y + 1.3
    steps = [("1", "Clean", f"Removed {N['dupes']:,} duplicate rows and {N['bad_weight']} impossible weights. "
              f"{pc(N['share_main'])} of the rows are used for the KPIs."),
             ("2", "Measure", "One result per scale and setting. We use the typical scale (the median), so a few "
              "noisy scales cannot distort the result."),
             ("3", "Compare", f"In April, {N['switchers']:,} scales got a new setting. We compare each one with "
              "itself before the change."),
             ("4", "Simulate", "We tested 9 ways to set the sensor on every scale and compared the results.")]
    cw4, g4 = (CW - 3 * 0.22) / 4, 0.22
    for i, (num, t, body) in enumerate(steps):
        x = LM + i * (cw4 + g4)
        col = MINT if i < 3 else PERI
        card(s, x, y2, cw4, 2.45, accent=col)
        badge(s, x + 0.2, y2 + 0.2, 0.36, num, fill=col)
        tb(s, x + 0.68, y2 + 0.22, cw4 - 0.8, 0.35, t, size=16, family=HEAD, bold=True, color=WHITE)
        tb(s, x + 0.2, y2 + 0.8, cw4 - 0.4, 1.6, body, size=12.5, color=TXT, line=1.13)
        if i < 3:
            arrow(s, x + cw4 + 0.02, y2 + 0.38, x + cw4 + g4 - 0.02, y2 + 0.38)
    x = LM
    for t in ["Python", "DuckDB", "Power BI", "Quarto", "Tested on simulated data first"]:
        x = pill(s, x, y2 + 2.75, t, color=PERI if t.startswith("Tested") else MINT) + 0.12
    footer(s, OWN + " · Report §2–3, Appendix A", d.n)
    notes(s, f"""(1 minute)
We had 7.6 million weighings from {N['scales']:,} scales over 16 weeks.
First we cleaned the data. We removed duplicate rows and a few impossible weights.
Then we measured each scale on its own. We use the typical scale, so a few noisy scales cannot distort the picture.
The most useful part of the data is April. Bossard gave {N['switchers']:,} scales a new setting. So we can compare the same scale before and after.
Finally we simulated nine ways to set the sensor.
The whole analysis can be run again in about one minute.""")

    # ---- 4 fleet today --------------------------------------------------------------------------
    s = d.slide(alt=True)
    y = head(s, "TODAY", "Almost no scale misses picks – but almost every scale wakes up too often")
    pic(s, ch_one_sided(plt, N, R, 7.4, 3.6), LM - 0.15, y - 0.05, w=7.4)
    xr = 8.6
    card(s, xr, y - 0.05, 3.85, 3.6)
    tb(s, xr + 0.25, y + 0.15, 3.4, 0.3, "A TYPICAL 20 KG SCALE, PER WEEK", size=10.5, bold=True, color=MINT, spc=100)
    ty = big(s, xr + 0.25, y + 0.5, 3.4, f"{b.loc['20kg', 'too_many_per_week_median']:.1f}", "empty wake-ups",
             color=CORAL, size=44)
    big(s, xr + 0.25, ty + 0.25, 3.4, f"{picks20:.1f}", "real picks", color=MINT, size=44)
    tb(s, LM, y + 3.85, CW, 0.6, "So the setting can safely go down. The question is how far.", size=16,
       family=HEAD, bold=True, color=WHITE)
    footer(s, OWN + " · scales on the default setting (8) · Report §4.1", d.n)
    notes(s, f"""(1 minute)
This is the fleet today, on the factory setting.
Almost every scale meets Bossard's target for missed picks.
But almost no scale meets the target for empty wake-ups.
A typical 20 kg scale wakes up for nothing about {b.loc['20kg', 'too_many_per_week_median']:.0f} times a week. It sees only one or two real picks.
So the sensor is set too sensitive. There is room to turn it down.
The real question is how far we can go before we start missing picks.
If the Power BI report is open, this page can be shown live here.""")

    # ---- 5 KPI misleads -------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "FINDING 1 · THE KPI", "The current KPI looks better when the sensor is switched off")
    pic(s, ch_kpi_levels(plt, N, R, 7.7, 3.7), LM - 0.15, y - 0.05, w=7.7)
    tb(s, LM, y + 3.75, 7.4, 0.6, "Lower settings remove almost all empty wake-ups, but the KPI hardly moves. "
       "It only improves when the scale also stops seeing real picks.", size=13, color=TXT, line=1.12)
    xr = 8.85
    card(s, xr, y - 0.05, 3.6, 4.45, accent=MINT)
    tb(s, xr + 0.25, y + 0.15, 3.2, 0.3, "WE SUGGEST: KPI 2.0", size=10.5, bold=True, color=MINT, spc=100)
    tb(s, xr + 0.25, y + 0.48, 3.2, 0.5, "Count what really costs money", size=16, family=HEAD, bold=True,
       color=WHITE, line=1.05)
    items = [("≤ 7", "empty wake-ups per week"), ("≤ 10%", "of picks missed"), ("≤ 2", "missed picks per week")]
    ty = y + 1.3
    for v, t in items:
        tb(s, xr + 0.25, ty, 1.2, 0.5, v, size=24, family=HEAD, bold=True, color=MINT)
        tb(s, xr + 1.45, ty + 0.08, 2.0, 0.5, t, size=12.5, color=SOFT, line=1.1)
        ty += 0.85
    tb(s, xr + 0.25, ty + 0.05, 3.1, 0.6, "Slow-moving bins are no longer punished.", size=12, color=TXT)
    footer(s, OWN + " · typical scale per setting · Report §4.3, §4.4, §6.1", d.n)
    notes(s, """(1 minute)
Bossard's KPI is the share of wake-ups that find nothing.
Look at what happens when the setting goes down.
On the left, the empty wake-ups almost disappear.
On the right, the KPI barely changes. Only on setting 1 does it touch the target, and there the sensor is almost off.
Why? A less sensitive scale also misses real picks. Both numbers fall together, so the share stays high.
The KPI also punishes bins that are rarely used.
So we suggest to count what really costs money. Empty wake-ups per week, and the share of picks we miss.""")

    # ---- 6 coupling insight ---------------------------------------------------------------------
    s = d.slide(alt=True)
    y = head(s, "FINDING 2 · KEY INSIGHT", "Over half of the empty wake-ups happen when someone picks nearby")
    tb(s, LM, y - 0.12, 7, 0.3, "A REAL EXAMPLE FROM THE DATA", size=10.5, bold=True, color=MINT, spc=120)
    pic(s, ch_episode(plt, N, R, 7.5, 3.2), LM - 0.1, y + 0.2, w=7.5)
    tb(s, LM, y + 3.5, 7.2, 0.6, "A worker picks from Bin A. Three bins next to it wake up within 14 seconds and "
       "find nothing.", size=13, color=TXT, line=1.12)
    xr = 8.7
    card(s, xr, y - 0.05, 3.75, 2.15, accent=PERI, strong=True)
    tb(s, xr + 0.25, y + 0.1, 3.3, 0.9, pc(c_both), size=54, family=HEAD, bold=True, color=PERI)
    tb(s, xr + 0.25, y + 1.05, 3.3, 1.0, "of empty wake-ups come within 14 seconds of a real pick in the same "
       "factory", size=12.5, color=SOFT, line=1.12)
    card(s, xr, y + 2.3, 3.75, 2.0)
    tb(s, xr + 0.25, y + 2.45, 3.3, 0.3, "WHAT IT MEANS", size=10.5, bold=True, color=MINT, spc=100)
    tb(s, xr + 0.25, y + 2.8, 3.3, 1.5, ["• The sensor feels the whole rack, not only its own bin.",
                                         "• A lower setting filters part of it.",
                                         "• Damping the rack may do the rest."], size=12.5, color=TXT,
       space_after=4)
    footer(s, OWN + " · above the chance level · looping scales left out · Report §4.5, Appendix C", d.n)
    notes(s, """(1.5 minutes)
This is our key finding.
Here is one real moment from the data. A worker takes parts from Bin A. In the same seconds, three bins next to it wake up. They weigh and find nothing.
This is not a rare story. It happens all the time.
More than half of all empty wake-ups come within 14 seconds of a real pick in the same factory. That is after we remove what would happen by chance.
So the sensor does not only feel its own bin. It feels the whole rack.
A lower setting filters part of this. Better damping of the rack could do the rest.""")

    # ---- 7 proof --------------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "FINDING 2 · PROOF", "It is not chance – four checks agree")
    pic(s, ch_correlogram(plt, N, R, 6.0, 3.3), LM - 0.15, y - 0.05, w=6.0)
    pic(s, ch_placebo(plt, N, R, 5.4, 3.3), 7.1, y - 0.05, w=5.4)
    checks = [("Peak at zero", "Picks nearby jump about 8 times at the moment of an empty wake-up."),
              ("Not just a busy hour", "One minute later, in the same hour, the link is gone."),
              ("Not in another factory", "Picks in a different factory at the same time show no link."),
              ("Busier site, stronger link", f"And it holds for {pc(R['per_scale'].share_scales_above_chance.min(), 1)} "
               "of all scales.")]
    cw4 = (CW - 3 * 0.2) / 4
    ty = y + 3.45
    for i, (t, body) in enumerate(checks):
        x = LM + i * (cw4 + 0.2)
        card(s, x, ty, cw4, 1.3)
        badge(s, x + 0.15, ty + 0.15, 0.28, "✓", size=11)
        tb(s, x + 0.52, ty + 0.15, cw4 - 0.62, 0.3, t, size=13, family=HEAD, bold=True, color=WHITE)
        tb(s, x + 0.15, ty + 0.55, cw4 - 0.3, 0.75, body, size=11.5, color=TXT, line=1.1)
    footer(s, OWN + " · event coincidence analysis (Donges et al., 2016) · Report §4.5, Appendix C", d.n)
    notes(s, """(1 minute)
A claim like this needs proof. We checked it in four ways.
First, the left chart. Around an empty wake-up, picks nearby jump to about eight times the normal level. Exactly at zero seconds.
Second, we look one minute later, in the same busy hour. The link is gone. So it is not just a busy time of day.
Third, the right chart. We take picks from a different factory at the same moment. There is no link at all.
Fourth, the busier the factory, the stronger the link. And it holds for almost every scale.
What the data cannot tell us is how the vibration travels. A short test on site with a vibration sensor would show that.""")

    # ---- 8 loops --------------------------------------------------------------------------------
    s = d.slide(alt=True)
    y = head(s, "FINDING 3 · LOOPING SCALES", f"{N['loop_share_scales'] * 100:.1f}% of the scales wake themselves "
             f"up – and cause {loop_share * 100:.0f}% of all weighings")
    pic(s, ch_loopday(plt, N, R, 6.4, 3.0), LM - 0.15, y - 0.05, w=6.4)
    pic(s, ch_loop_evidence(plt, N, R, 5.7, 3.0), 6.85, y - 0.05, w=5.7)
    tb(s, LM, y + 3.05, 6.0, 0.6, "About every 90 seconds, day and night. That is the shortest wait the scale allows "
       "(60 s) plus its measuring delay (30 s).", size=12.5, color=TXT, line=1.12)
    tb(s, 7.05, y + 3.05, 5.4, 0.6, f"The same scales act normally on another setting ({int(lo3.of_which_normal_there)} "
       f"of {int(lo3.scales_seen_on_other_level)}). This points to the firmware, not to broken hardware.",
       size=12.5, color=TXT, line=1.12)
    card(s, LM, 5.9, CW, 0.8, accent=MINT, strong=True)
    tb(s, LM + 0.25, 5.9, CW - 0.5, 0.8, f"**We took these scales out and repeated every analysis.** The results "
       f"hold. The link to picks nearby gets even clearer ({pc(c_both_all)} → {pc(c_both)}).", size=13.5,
       color=SOFT, anchor="m")
    footer(s, OWN + " · looping = over half of the wake-ups within 2 min and over 20 a day · Report §5.3, Appendix C",
           d.n)
    notes(s, """(1.5 minutes)
Our third finding was the biggest surprise.
A small group of scales wakes itself up again and again. On the left you see one of them. About forty wake-ups every hour, day and night.
The gap is usually 90 seconds. That is the shortest wait the scale allows, plus its measuring delay. So the scale triggers itself.
On the right you see how different these scales are. They wake up at night and at weekends. And they have no link to picks nearby.
Only 1.5 percent of the scales do this. But they cause 27 percent of all weighings. Most of them during a short test of setting 3 in April.
The same scales behave normally on another setting. So the hardware is fine. This looks like a firmware problem.
We removed these scales and repeated all our analyses. The results did not change.""")

    # ---- 9 knob ---------------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "FINDING 4 · THE SETTING", "One step down halves the empty wake-ups – and costs almost nothing")
    pic(s, ch_knob(plt, N, R, 7.8, 3.7), LM - 0.15, y - 0.05, w=7.8)
    xr = 8.95
    ty = big(s, xr, y + 0.1, 3.5, f"{pct0(-e7['20kg'] * 100)} / {pct0(-e7['6kg'] * 100)}", "empty wake-ups when the "
             "setting goes from 8 to 7 (20 kg / 6 kg)", size=34, color=MINT) + 0.35
    ty = big(s, xr, ty, 3.5, f"+{m7['20kg']:.0f} / +{m7['6kg']:.0f}", "points more missed picks", size=34,
             color=PERI) + 0.35
    tb(s, xr, ty, 3.5, 0.8, "Below setting 6, the scale starts to miss real picks.", size=14, family=HEAD,
       bold=True, color=WHITE, line=1.1)
    footer(s, OWN + " · same scales before and after their change, minus the trend of unchanged scales · Report §5.1",
           d.n)
    notes(s, """(1 minute)
Now, what does the setting really do?
In April Bossard changed the setting of thousands of scales. We compare each scale with itself, before and after.
The first step, from 8 to 7, removes about half of the empty wake-ups. It costs only two or three points of missed picks.
The steps below 6 are different. There the scale starts to miss real picks.
So the first steps are cheap. The deep steps are expensive.""")

    # ---- 10 sites -------------------------------------------------------------------------------
    s = d.slide(alt=True)
    y = head(s, "FINDING 5 · NO ONE-SIZE-FITS-ALL", "The same setting works differently in every factory")
    pic(s, ch_sites(plt, N, R, 7.4, 3.6), LM - 0.15, y - 0.05, w=7.4)
    pic(s, ch_drivers(plt, N, R, 4.6, 3.6), 7.9, y - 0.05, w=4.6)
    card(s, LM, 5.9, CW, 0.8, accent=MINT, strong=True)
    tb(s, LM + 0.25, 5.9, CW - 0.5, 0.8, "**What decides the best setting only shows once a scale is running.** "
       "So each scale should learn its own setting.", size=13.5, color=SOFT, anchor="m")
    footer(s, OWN + " · factories with at least 20 scales, default setting · Report §4.7, §4.8", d.n)
    notes(s, """(1 minute)
Can we not just pick one good setting for everyone?
No. On the left, every bar is a factory. Same setting, very different results.
On the right is what makes the biggest difference. The factory, and how often a bin is used.
Box type and item weight matter much less. But those are the only things Bossard knows before a scale is installed.
We also tested a table by pick size. It chose the same setting for almost every scale.
So the best setting cannot be looked up. Each scale has to learn it.
If the Power BI report is open, the factory page fits here.""")

    # ---- 11 solution ----------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "OUR RECOMMENDATION", "Flag, tune, watch")
    cols = [("F", "Flag", "Find looping scales every day and send them to service.", f"{N['loop_scales']} scales",
             CORAL),
            ("T", "Tune", "Start every scale on 7. Every two weeks it moves one step – or stays.",
             f"{pct0(-save20)} / {pct0(-save6)} weighings", MINT),
            ("W", "Watch", "Track KPI 2.0. Keep the daily check. Watch the few scales that miss most.",
             "misses stay visible", PERI)]
    cw3 = (CW - 2 * 0.3) / 3
    for i, (l, t, body, eff, col) in enumerate(cols):
        x = LM + i * (cw3 + 0.3)
        card(s, x, y, cw3, 2.75, accent=col)
        badge(s, x + 0.25, y + 0.25, 0.55, l, fill=col, size=20)
        tb(s, x + 0.95, y + 0.3, cw3 - 1.1, 0.5, t, size=24, family=HEAD, bold=True, color=WHITE)
        tb(s, x + 0.25, y + 1.0, cw3 - 0.5, 1.1, body, size=13.5, color=SOFT, line=1.15)
        pill(s, x + 0.25, y + 2.2, eff, color=col, size=12)
    ty = y + 3.2
    phases = [("NOW", "Start the loop check · default 8 → 7"), ("WEEKS 1–8", "Test the rule in about 10 factories"),
              ("THEN", "Roll out to all scales")]
    pw = CW / 3
    for i, (t, body) in enumerate(phases):
        x = LM + i * pw
        shape(s, x, ty + 0.13, pw - 0.05, 0.04, fill=MINT if i == 0 else WHITE, fill_a=None if i == 0 else 0.2)
        badge(s, x, ty, 0.3, "", fill=MINT if i == 0 else PERI)
        tb(s, x, ty + 0.45, pw - 0.3, 0.3, t, size=11, bold=True, color=MINT, spc=120)
        tb(s, x, ty + 0.75, pw - 0.3, 0.6, body, size=13, color=WHITE, line=1.1)
    footer(s, OWN + " · simulation over 16 weeks · Report §6.3–6.4", d.n)
    notes(s, """(1 minute)
Our recommendation has three steps. None of them needs new hardware. Everything runs on data the cloud already has.
Flag. Find the looping scales every day and send them to service.
Tune. Every scale starts on setting 7. Every two weeks it moves one step, or it stays.
Watch. Track the new KPIs. Keep the daily safety check. Look closer at the few scales that miss most picks.
We suggest to start the first part now, and to test the rule in about ten factories for eight weeks.""")

    # ---- 12 flag --------------------------------------------------------------------------------
    s = d.slide(alt=True)
    y = head(s, "STEP 1 · FLAG", "A simple daily check finds the looping scales")
    card(s, LM, y - 0.05, 5.6, 2.3, accent=CORAL, strong=True)
    tb(s, LM + 0.25, y + 0.12, 5.1, 0.3, "A SCALE IS LOOPING IF, OVER THE LAST WEEK", size=10.5, bold=True,
       color=CORAL, spc=100)
    tb(s, LM + 0.25, y + 0.5, 5.1, 1.6, ["• more than **half** of its wake-ups come within **2 minutes** of the "
                                         "last one, and",
                                         "• it wakes up more than **20 times a day**."], size=15, color=SOFT,
       space_after=8, bullet_color=CORAL)
    tb(s, LM, y + 2.5, 5.6, 0.3, "THEN", size=10.5, bold=True, color=MINT, spc=120)
    acts = ["Set it back to 7 from the cloud", "Still looping after 2 days → service visit",
            "Send the list to the firmware team"]
    ty = y + 2.85
    for i, t in enumerate(acts):
        badge(s, LM, ty, 0.34, str(i + 1), size=12)
        tb(s, LM + 0.5, ty + 0.03, 5.0, 0.4, t, size=13.5, color=WHITE)
        ty += 0.5
    pic(s, ch_detector(plt, N, R, 6.2, 3.8), 6.4, y - 0.1, w=6.2)
    tb(s, 6.6, y + 3.8, 5.8, 0.5, "Looping and normal scales are far apart. Other thresholds give almost the same "
       "list.", size=12.5, color=TXT, line=1.12)
    footer(s, OWN + " · each dot = scales with similar values · Report §5.3, Appendix C", d.n)
    notes(s, """(1 minute)
How do we find a looping scale? With two simple numbers.
How often does it wake up again within two minutes? And how many times a day does it wake up?
If more than half of the wake-ups come that fast, and it wakes up more than 20 times a day, it is looping.
The chart shows why this is easy. Normal scales sit bottom left. Looping scales sit top right. There is a big gap between them.
Then three steps. Set the scale back to 7 from the cloud. If it still loops after two days, send service. And give the list to the firmware team.
We deliver the check as a short script, together with today's list.""")

    # ---- 13 tune --------------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "STEP 2 · TUNE", "Every two weeks, each scale moves one step – or stays")
    rules = [("Missed more than 10% of its picks?", "one step MORE sensitive", AMBER),
             ("More than 7 empty wake-ups a week, and almost no misses?", "one step LESS sensitive", MINT),
             ("Neither?", "stay", PERI)]
    ty = y - 0.05
    tb(s, LM, ty, 5.6, 0.3, "EVERY 2 WEEKS, FOR EACH SCALE", size=10.5, bold=True, color=MINT, spc=120)
    ty += 0.4
    for q_, a, col in rules:
        card(s, LM, ty, 5.6, 0.95, accent=col)
        tb(s, LM + 0.2, ty + 0.1, 3.35, 0.75, q_, size=12.5, color=WHITE, line=1.1, anchor="m")
        arrow(s, LM + 3.6, ty + 0.47, LM + 3.95, ty + 0.47, color=col)
        tb(s, LM + 4.0, ty + 0.1, 1.5, 0.75, a, size=12.5, family=HEAD, bold=True, color=col, line=1.05,
           anchor="m")
        ty += 1.1
    tb(s, LM, ty + 0.05, 5.6, 0.4, "Start on setting 7. Never below 1, never setting 3.", size=12.5, color=TXT)
    xr = 6.85
    tb(s, xr, y - 0.05, 5.6, 0.3, "EXAMPLE · A BUSY 20 KG SCALE", size=10.5, bold=True, color=MINT, spc=120)
    pic(s, ch_tune_example(plt, N, R, 5.7, 3.3), xr - 0.1, y + 0.3, w=5.7)
    tb(s, xr, y + 3.65, 5.5, 0.6, "It settles on setting 6 with about two thirds fewer empty wake-ups.",
       size=13, color=TXT, line=1.12)
    footer(s, OWN + " · rule simulated on every scale for 16 weeks; the example is illustrative · Report §6.2–6.4", d.n)
    notes(s, """(1.5 minutes)
This is the heart of the solution, and it is simple.
Every two weeks the cloud looks at each scale.
Did it miss more than 10 percent of its picks? Then it becomes one step more sensitive.
Did it wake up for nothing more than seven times a week, with almost no misses? Then it becomes one step less sensitive.
Otherwise it stays where it is.
On the right is an example. A busy scale starts on 7 and steps down twice. On 5 it misses a pick, so it goes back up to 6. There it stays.
It ends with about two thirds fewer empty wake-ups.
On average a scale changes its setting about once a month.""")

    # ---- 14 watch -------------------------------------------------------------------------------
    s = d.slide(alt=True)
    y = head(s, "STEP 3 · WATCH", "Missed picks are still found – usually within two days")
    tb(s, LM, y - 0.12, 6, 0.3, "WHEN IS A MISSED PICK FOUND?", size=10.5, bold=True, color=MINT, spc=120)
    pic(s, ch_delay(plt, N, R, 6.4, 2.2), LM - 0.15, y + 0.2, w=6.4)
    tb(s, LM, y + 2.55, 6.0, 0.9, f"Bossard's daily check finds {pc(N['idle_share_tf'])} of them. We keep it. A "
       "missed pick only matters if it should have started a reorder.", size=12.5, color=TXT, line=1.12)
    tb(s, 6.95, y - 0.12, 6, 0.3, "WHO MISSES THE PICKS?", size=10.5, bold=True, color=MINT, spc=120)
    pic(s, ch_concentration(plt, N, R, 5.6, 3.3), 6.85, y + 0.2, w=5.6)
    ty = 5.95
    x = LM
    for t, col in [("Keep the daily check", MINT), ("Extra check for the worst 10%", MINT),
                   ("Alarm if a scale is silent for 48 h", AMBER), ("No extra night check for all", CORAL)]:
        x = pill(s, x, ty, t, color=col, size=12.5, h=0.4) + 0.15
    footer(s, OWN + " · time since the last weighing of every missed pick · Report §3, §6.4", d.n)
    notes(s, """(1 minute)
A fair question is this. If the sensor sleeps more, does the stock stay wrong for days?
The data says no. Most missed picks are found within one or two days. Bossard already runs a daily check on every scale that was quiet for 24 hours. It finds most of the misses, and we keep it.
A few take longer. We suggest an alarm for any scale that has been silent for two days.
On the right you see that misses are concentrated. Ten percent of the scales cause about half of them. Those scales get an extra check.
We do not recommend an extra night check for all scales. It would eat a third of the savings.""")

    # ---- 15 result ------------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "RESULT", "More than half fewer weighings – and misses stay under control")
    pic(s, ch_result(plt, N, R, 8.0, 3.9), LM - 0.15, y - 0.05, w=8.0)
    xr = 9.15
    ty = big(s, xr, y, 3.3, f"≈ {fleet / 1e6:.1f} M", f"fewer weighings per week on {N['installed']:,} scales",
             size=38, color=MINT) + 0.4
    ty = big(s, xr, ty, 3.3, f"+{p7['20kg'].battery_life_change_pct:.0f}% / +{p7['6kg'].battery_life_change_pct:.0f}%",
             "battery life – the big gain is network and cloud", size=30, color=PERI) + 0.4
    tb(s, xr, ty, 3.3, 1.0, f"To be honest: the typical scale misses {pc(miss20)}–{pc(miss6)}, just above the 10% "
       "target. The pilot fine-tunes this.", size=12.5, color=TXT, line=1.12)
    footer(s, OWN + " · simulation with each scale's own behaviour, looping scales left out; the fleet number is an "
           "estimate, not a forecast · Report §6.3", d.n)
    notes(s, f"""(1 minute)
We simulated every option on every scale.
One setting for all saves a lot. But the typical scale then misses close to half of its picks. Steering by Bossard's KPI ends in the same place.
Our rule saves {save20:.0f} and {save6:.0f} percent of the weighings. The typical scale misses around 11 to 12 percent of its picks, and the daily check finds them.
On the whole fleet that is about {fleet / 1e6:.0f} million fewer weighings every week. That is an estimate, not a promise.
To be honest, 11 to 12 percent is just above the 10 percent target. The pilot will fine-tune this.""")

    # ---- 16 alternatives ------------------------------------------------------------------------
    s = d.slide(alt=True)
    y = head(s, "WHAT WE RULED OUT", "We tested the shortcuts – none of them works")
    q = N["quiet"][N["quiet"].threshold == 0.01].set_index("scale_type")
    p1 = {st: pol.loc[(st, P["P1"])] for st in SCALES}
    rows = [["Idea", "What happened", "Verdict"],
            ["One setting for all scales", f"Misses {pc(p1['20kg'].median_miss_rate)}–{pc(p1['6kg'].median_miss_rate)} "
             "of the picks", "no"],
            ["Steer by Bossard's KPI", "Pushes scales to setting 1 – the sensor is practically off", "no"],
            ["A table by pick size or item weight", "Chooses the same setting for almost every scale", "no gain"],
            ["Sensor off at quiet hours", f"Saves only {pc(q.loc['20kg', 'share_empty_avoided'])} – and real picks "
             "happen then too", "no"],
            ["Extra night check for every scale", "Eats a third of the savings", "no"],
            ["Longer wait time between weighings", "Only used in two factories – cannot be measured", "open"],
            ["Bundle wake-ups into one weighing", "Saves a few % more – needs a firmware change", "maybe later"]]
    table(s, LM, y, CW, [3.7, 6.2, 1.66], rows, size=13, row_h=0.52, align="lll",
          col_colors=[WHITE, TXT, CORAL])
    footer(s, OWN + " · Report §6.3, §6.5", d.n)
    notes(s, """(1 minute, or keep it for questions)
We tested the obvious shortcuts.
One setting for all looks good on Bossard's KPI, but it misses many real picks.
Switching the sensor off at quiet hours does not help. Empty wake-ups follow the working day, just like real picks.
An extra night check for every scale costs too much.
A longer wait time was only used in two factories, so we cannot measure it.
Bundling wake-ups is a nice firmware idea for later, but it adds only a little.""")

    # ---- 17 next steps --------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "NEXT STEPS", "Start the safe steps now – let a pilot decide the rest")
    phases = [("NOW", MINT, ["Run the daily loop check", "Remove setting 3 from the menu", "Default setting 8 → 7",
                             "Show KPI 2.0 in the dashboard"]),
              ("PILOT · 8 WEEKS", MINT, ["Rule in about 10 factories", "Compare with similar factories without it",
                                         "Fine-tune the step-down threshold"]),
              ("IN PARALLEL", PERI, ["Vibration test in the 10 most affected factories", "Firmware fix for the loops",
                                     "Confirm how the daily check works"])]
    pw = (CW - 2 * 0.25 - 2.9 - 0.25) / 3
    for i, (t, col, bullets) in enumerate(phases):
        x = LM + i * (pw + 0.25)
        card(s, x, y, pw, 3.3, accent=col)
        tb(s, x + 0.2, y + 0.2, pw - 0.4, 0.3, t, size=11, bold=True, color=col, spc=120)
        tb(s, x + 0.2, y + 0.6, pw - 0.4, 2.6, ["• " + bb for bb in bullets], size=13, color=WHITE, space_after=7,
           bullet_color=col)
    xr = LM + 3 * (pw + 0.25)
    card(s, xr, y, 2.9, 3.3)
    tb(s, xr + 0.2, y + 0.2, 2.5, 0.3, "LIMITS", size=11, bold=True, color=CORAL, spc=120)
    tb(s, xr + 0.2, y + 0.6, 2.55, 2.6, ["• Only one season of data", "• The April change was not random",
                                         "• Some missed picks are never seen", "• No rack positions in the data"],
       size=12, color=TXT, space_after=6, bullet_color=CORAL)
    card(s, LM, y + 3.6, CW, 0.85, accent=MINT, strong=True)
    tb(s, LM + 0.25, y + 3.6, CW - 0.5, 0.85, "**Our advice for today:** start the loop check and the new default 7 "
       "now. Both are low risk and measured. Let the pilot decide the details of the rule.", size=14, color=SOFT,
       anchor="m")
    footer(s, OWN + " · Report §7–8", d.n)
    notes(s, """(1 minute)
What should Bossard do on Monday?
Two things are safe and can start now. The daily loop check, and the new default setting of 7. We measured both on real scales.
The rule itself goes into an eight-week pilot in about ten factories. We compare them with similar factories that keep the old setting.
In parallel, a vibration test in the factories with the most empty wake-ups, and a firmware fix for the loops.
Our limits are simple. We have one season of data, and the April change was not random. That is why we want a pilot first.""")

    # ---- 18 thank you ---------------------------------------------------------------------------
    s = d.slide()
    tb(s, LM, 1.2, 8, 0.3, "THANK YOU", size=12, bold=True, color=MINT, spc=150)
    tb(s, LM, 1.6, 8, 1.0, "Questions?", size=52, family=HEAD, bold=True, color=WHITE)
    yy = 3.2
    for l, t, body, col in [("F", "Flag", "find the looping scales every day", CORAL),
                            ("T", "Tune", "every scale learns its own setting", MINT),
                            ("W", "Watch", "keep missed picks visible", PERI)]:
        badge(s, LM, yy, 0.5, l, fill=col, size=17)
        tb(s, LM + 0.7, yy + 0.05, 1.5, 0.4, t, size=20, family=HEAD, bold=True, color=WHITE)
        tb(s, LM + 2.2, yy + 0.1, 8, 0.4, body, size=17, color=SOFT)
        yy += 0.75
    tb(s, LM, 6.5, 11, 0.4, "Erenalp Çakıroğlu · code, report and pipeline in a private repository (case data under NDA)",
       size=11.5, color=MUTED)
    notes(s, "Leave this slide up during questions. The backup slides cover data cleaning, the results without the "
             "looping scales, the methods, pick size, the code and the sources.")

    # ================================= backup =================================
    s = d.slide(alt=True)
    y = head(s, "BACKUP · DATA CLEANING", "We delete only what is broken – everything else is kept and marked")
    pic(s, ch_cleaning(plt, N, R, 7.4, 4.4), LM - 0.15, y - 0.1, w=7.4)
    tb(s, 8.3, y, 4.2, 4.5, [
        f"• **{N['dupes']:,} duplicate rows** (same scale, same millisecond). In 1,300 cases the copies had different "
        "labels. We keep one copy but do not trust its label.",
        f"• **{N['bad_weight']} impossible weights** (144 kg to 50 t).",
        "• **Marked, not deleted.** Rows we cannot use for the KPIs stay in the data. The next weighing needs them "
        "to compute its weight change.",
        f"• **{pc(N['share_main'])} of the rows** are used for the KPIs."], size=12.5, color=TXT, space_after=8)
    footer(s, OWN + " · Report §3, Appendix A", d.n)
    notes(s, "Use this if someone asks about duplicates. Yes, we checked them. Not all duplicates were identical. Some "
             "copies had different labels, so we keep the row but do not use its label.")

    s = d.slide()
    y = head(s, "BACKUP · ROBUSTNESS", "Without the looping scales, every key result holds")
    bs, ka, kn = R["base"], R["knob_all"], R["knob"]
    rows = [["Result", "All scales", "Without looping scales"],
            ["Empty wake-ups per week, typical scale (20 kg / 6 kg)",
             f"{bs.loc[('all scales', '20kg'), 'tm_per_week']:.1f} / {bs.loc[('all scales', '6kg'), 'tm_per_week']:.1f}",
             f"{bs.loc[('without loop scales', '20kg'), 'tm_per_week']:.1f} / {bs.loc[('without loop scales', '6kg'), 'tm_per_week']:.1f}"],
            ["Empty wake-ups linked to picks nearby", pc(c_both_all), f"**{pc(c_both)}**"],
            ["Setting 8 → 7: fewer empty wake-ups",
             f"{pct0(-ka.loc[('20kg', 7.0), 'empty_wakeups_change'] * 100)} / {pct0(-ka.loc[('6kg', 7.0), 'empty_wakeups_change'] * 100)}",
             f"{pct0(-kn.loc[('20kg', 7.0), 'empty_wakeups_change'] * 100)} / {pct0(-kn.loc[('6kg', 7.0), 'empty_wakeups_change'] * 100)}"],
            ["Our rule: fewer weighings", f"{pct0(save20)} / {pct0(save6)}",
             f"{pct0(-p7_nl['20kg'].meas_change_pct_without_loops)} / {pct0(-p7_nl['6kg'].meas_change_pct_without_loops)}"],
            ["Our rule: picks missed, typical scale", f"{pc(miss20)} / {pc(miss6)}",
             f"{pc(p7_nl['20kg'].median_miss_rate)} / {pc(p7_nl['6kg'].median_miss_rate)}"]]
    table(s, LM, y, CW, [6.0, 2.8, 2.8], rows, size=13, row_h=0.55, align="lrr", hl_rows=(2,))
    footer(s, OWN + " · report/robustness_loops.py · Report Appendix C", d.n)
    notes(s, "Use this if someone asks whether the looping scales distort the results. We removed them and ran "
             "everything again. The key numbers stay the same, and the link to picks nearby gets stronger.")

    s = d.slide(alt=True)
    y = head(s, "BACKUP · METHODS", "Four methods, each answering one question")
    meth = [("Does the setting cause the change?", "Each scale before and after its change in April, minus the "
             "change of scales that kept their setting (difference-in-differences)."),
            ("Are empty wake-ups linked to picks?", "Count picks at other scales of the same factory within ±14 s, "
             "minus the chance level one minute later (event coincidence analysis)."),
            ("How small a pick can a scale see?", "The weight change behind each label. Below about 23 g (20 kg) or "
             "5 g (6 kg) a pick is invisible."),
            ("Which rule is best?", "Each scale's own behaviour on setting 8, times the measured effect of each "
             "setting. 9 rules, 16 weeks, 5 repeats.")]
    cw2 = (CW - 0.3) / 2
    for i, (t, body) in enumerate(meth):
        x, yy = LM + (i % 2) * (cw2 + 0.3), y + (i // 2) * 2.0
        card(s, x, yy, cw2, 1.8, accent=MINT if i % 2 == 0 else PERI)
        tb(s, x + 0.25, yy + 0.2, cw2 - 0.5, 0.35, t, size=15, family=HEAD, bold=True, color=WHITE)
        tb(s, x + 0.25, yy + 0.65, cw2 - 0.5, 1.1, body, size=12.5, color=TXT, line=1.12)
    footer(s, "Angrist & Pischke (2009) · Donges et al. (2016) · Green & Swets (1966) · Report Appendix A", d.n)
    notes(s, "Use only if someone asks about a method.")

    s = d.slide()
    y = head(s, "BACKUP · PICK SIZE", "Small picks need a sensitive setting – or a 6 kg scale")
    pic(s, ch_pick_heat(plt, N, R, 8.4, 4.1), LM - 0.15, y - 0.1, w=8.4)
    tb(s, 8.9, y, 3.6, 4.0, ["• Each cell shows the share of picks the scale caught (%).",
                             "• Settings 1–5 miss too many picks for every pick size.",
                             "• Picks below about 25 g belong on 6 kg scales."], size=13, color=TXT, space_after=8)
    footer(s, OWN + " · empty cells: too few picks, or below what the scale can see · Report §4.6", d.n)
    notes(s, "Use this if someone asks about pick size. Small picks are hard to see on a 20 kg scale. Settings 6 to 8 "
             "work for almost every pick size. That is why the rule starts on 7.")

    s = d.slide(alt=True)
    y = head(s, "BACKUP · CODE", "One pipeline from raw data to this deck")
    mods = [("Load and clean", "load 7.6 M rows into DuckDB, delete broken rows, mark the rest"),
            ("Build tables", "weight change per weighing; one record per scale and setting"),
            ("Analyse", "today, the setting, parameters, factories, pick size, picks nearby, simulation"),
            ("Validate", "a simulated fleet with known answers – the pipeline must find them first"),
            ("Report", "tables for Power BI, figures, the written report and both decks"),
            ("Robustness", "every key result again without the looping scales")]
    table(s, LM, y, 7.6, [2.0, 5.6], [["Step", "What it does"]] + [[a, bb] for a, bb in mods], size=12.5,
          row_h=0.52, align="ll")
    card(s, 8.8, y, 3.65, 3.65)
    tb(s, 9.0, y + 0.2, 3.3, 0.3, "RUN IT", size=10.5, bold=True, color=MINT, spc=120)
    tb(s, 9.0, y + 0.55, 3.3, 3.0, ["bash run_bossard.sh", "python report/build_figures.py",
                                    "python report/robustness_loops.py", "quarto render report/report.qmd",
                                    "python presentation/build_cto_deck.py"], size=11.5, color=SOFT, space_after=5)
    tb(s, LM, y + 3.95, CW, 0.4, "Raw data and Power BI files never go into the repository – only summary tables.",
       size=12.5, color=TXT)
    footer(s, "Source: project repository (private)", d.n)
    notes(s, "For the data scientists in the room. Everything can be rebuilt from the three raw files.")

    s = d.slide()
    y = head(s, "REFERENCES", "Sources")
    tb(s, LM, y - 0.1, CW, 4.9, REFS, size=11, color=TXT, line=1.08, space_after=5)
    footer(s, "", d.n)
    notes(s, "All sources cited on the slides. Case data are confidential and used under the module NDA.")

    d.save(DECK)
    return DECK, N, R


if __name__ == "__main__":
    path, _, _ = build()
    print(f"deck written: {path}")
