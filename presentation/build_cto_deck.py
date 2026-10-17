"""
Client deck for the CTO: SmartBin Cloud sensor sensitivity, 16:9 PowerPoint with speaker notes.

    python presentation/build_cto_deck.py

Story: problem -> what we found -> what we did -> what we recommend (flag, tune, watch) -> result -> next steps.
Reads only aggregated tables in report/data/ (report/build_figures.py and report/robustness_loops.py),
draws the charts into presentation/charts_cto/ and writes the deck. Every number on a slide is computed from
those tables. Power BI screenshots are inserted by hand on the POWER BI DASHBOARD A/B/C placeholder slides.

Look: corporate industrial palette (deep blue, petrol, light sand), Arial. All design tokens sit at the top.
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

ROOT = HERE.parent
CHARTS = HERE / "charts_cto"
DECK = HERE / "SmartBin – CTO presentation.pptx"

# ---------------------------------------------------------------------------------------------
# design tokens
# ---------------------------------------------------------------------------------------------
DEEP, INK2, MUTED = "000028", "45465E", "7D7F93"
WHITE, SAND, SAND2, LINE_C = "FFFFFF", "F3F3F0", "E6E6E1", "D4D4D9"
PETROL, PETROL_D, PETROL_L, GREEN = "009999", "007B80", "8FD6D6", "00FFB9"
NAVY6 = "3D3F8A"                         # 6 kg series
RED, RED_L, AMBER = "D72339", "F2A7B1", "E8A317"
C = {"20kg": PETROL, "6kg": NAVY6}
SANS = "Arial"
MPL_SANS = "Liberation Sans"             # metric twin of Arial for charts and text measurement
SCALES = ["20kg", "6kg"]
LABEL = {"20kg": "20 kg", "6kg": "6 kg"}
SW, SH = 13.333, 7.5
LM, CW = 0.62, 12.09
OWN = "Source: own analysis of the Bossard SmartBin data, 1 Mar – 20 Jun 2022"


# ---------------------------------------------------------------------------------------------
# extra facts: robustness tables (scales caught in a trigger loop left out)
# ---------------------------------------------------------------------------------------------
def robust_facts(N: dict) -> dict:
    R: dict = {}
    rc = rd("robust_coupling")
    same = rc[(rc.source == "picks at the same site") & (rc["sample"] == "without loop scales")].set_index("scale_type")
    alls = rc[(rc.source == "picks at the same site") & (rc["sample"] == "all scales")].set_index("scale_type")
    plac = rc[rc.source.str.startswith("placebo")].set_index("scale_type")
    R["coup"], R["coup_all"], R["placebo"] = same, alls, plac
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
    p = rd("robust_policy")
    R["pol_nl"] = p[p.effects == "estimated without loop scales"].set_index(["scale_type", "policy"])
    R["episode"] = rd("example_coupling_episode")
    R["loopday"] = rd("example_loop_day")
    return R


# ---------------------------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------------------------
def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": MPL_SANS, "font.size": 12, "axes.labelsize": 12, "axes.labelcolor": "#" + INK2,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "xtick.color": "#" + INK2, "ytick.color": "#" + INK2,
        "axes.edgecolor": "#" + LINE_C, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#" + SAND2, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 11.5,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2.4, "lines.solid_capstyle": "round",
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": "#" + DEEP,
        "text.color": "#" + DEEP, "axes.unicode_minus": True,
    })
    return plt


def hx(c):
    return "#" + c


def save(fig, name):
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / f"{name}.png", dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return CHARTS / f"{name}.png"


def patches(colors, labels, where, **kw):
    from matplotlib.patches import Patch
    return where.legend([Patch(color=hx(c)) for c in colors], labels, **kw)


def ch_one_sided(plt, N, R, w, h):
    b = N["base"]
    rows = [("Meets the TOO_MANY target\n(fraction ≤ 0.5)", "share_meets_tm"),
            ("Meets the TOO_FEW target\n(≤ 2 missed changes a week)", "share_meets_tf")]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.84, bottom=0.1)
    ys = np.arange(len(rows))[::-1] * 1.0
    for i, st in enumerate(SCALES):
        vals = [b.loc[st, k] * 100 for _, k in rows]
        yy = ys + (0.19 if i == 0 else -0.19)
        ax.barh(yy, vals, height=0.34, color=hx(C[st]))
        for y, v in zip(yy, vals):
            ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=12.5, fontweight="bold", color=hx(C[st]))
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=12, colors=hx(DEEP))
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=hx(SAND2))
    patches([C["20kg"], C["6kg"]], ["20 kg scales", "6 kg scales"], fig, loc="upper left", ncol=2,
            bbox_to_anchor=(0.34, 1.0), handlelength=1.0)
    return save(fig, "k01_one_sided")


def ch_episode(plt, N, R, w, h):
    e = R["episode"].copy()
    bins = sorted(e.bin.unique())
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.27)
    ax.axvspan(-14, 14, color=hx(PETROL), alpha=0.09, lw=0)
    ax.text(0, -0.5, "±14 s", ha="center", va="bottom", fontsize=11, color=hx(PETROL_D), fontweight="bold")
    for i, b in enumerate(bins):
        y = len(bins) - 1 - i
        ax.axhline(y, color=hx(SAND2), lw=1, zorder=0)
        for _, r in e[e.bin == b].iterrows():
            pick = r.cls == "OK"
            ax.scatter(r.t_rel_s, y, s=210 if pick else 150, zorder=3, color=hx(PETROL) if pick else hx(WHITE),
                       edgecolor=hx(PETROL_D) if pick else hx(RED), linewidth=2.4,
                       marker="o")
            ax.annotate("real pick" if pick else "empty wake-up", (r.t_rel_s, y), textcoords="offset points",
                        xytext=(0, 13), ha="center", fontsize=10.5, color=hx(PETROL_D) if pick else hx(RED))
    ax.set_yticks(range(len(bins)), bins[::-1])
    ax.tick_params(axis="y", labelsize=12, colors=hx(DEEP))
    ax.set_ylim(-0.6, len(bins) - 0.1)
    ax.set_xlim(-60, 60)
    ax.set_xticks([-60, -30, 0, 30, 60], ["−60 s", "−30 s", "0", "+30 s", "+60 s"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("time relative to the pick at Bin A (no other weighing at the site ±5 min)", fontsize=11)
    return save(fig, "k02_episode")


def ch_correlogram(plt, N, R, w, h):
    cor = R["cor"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.9, bottom=0.2)
    for st in SCALES:
        x = cor[(cor.scale_type == st) & (cor.lag_s.abs() <= 116)].sort_values("lag_s")
        ax.plot(x.lag_s, x.per_min, color=hx(C[st]), lw=2.2, label=f"{LABEL[st]} scales")
        ax.axhline(x.baseline.iloc[0], color=hx(C[st]), lw=1, ls=(0, (3, 3)), alpha=0.8)
    ax.axvspan(-14, 14, color=hx(PETROL), alpha=0.08, lw=0)
    ax.text(114, R["chance_rate"].max() + 0.35, "chance level", fontsize=10.5, ha="right", color=hx(INK2))
    ax.set_xlim(-116, 116)
    ax.set_ylim(0, 13.5)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xlabel("seconds from an empty wake-up", fontsize=11)
    ax.set_ylabel("real picks per minute at\nother bins of the same site", fontsize=11)
    ax.legend(loc="upper left", fontsize=11)
    return save(fig, "k03_correlogram")


def ch_proof(plt, N, R, w, h):
    fig, axes = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.25, wspace=0.32)
    ax = axes[0]
    xs = np.arange(2)
    for i, st in enumerate(SCALES):
        v = [R["coup"].loc[st, "coupled_beyond_chance"] * 100, R["placebo"].loc[st, "coupled_beyond_chance"] * 100]
        off = -0.19 if i == 0 else 0.19
        ax.bar(xs + off, v, width=0.36, color=hx(C[st]))
        for xx, vv in zip(xs + off, v):
            ax.text(xx, vv + 1.5, f"{vv:.0f}%" if vv >= 1 else f"{vv:.1f}%", ha="center", fontsize=11.5,
                    fontweight="bold", color=hx(C[st]))
    ax.set_xticks(xs, ["same site", "placebo:\nanother site"])
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 20, 40, 60], ["0%", "20%", "40%", "60%"])
    ax.set_title("Coupled beyond chance")
    ax = axes[1]
    order = ["1-5", "6-20", "21-50", ">50"]
    d = R["dose"]
    for st in SCALES:
        x = d[(d.scale_type == st)].set_index("site_picks_that_hour").reindex(order)
        ax.plot(range(4), x.coupled_beyond_chance * 100, color=hx(C[st]), marker="o", markersize=6)
    ax.set_xticks(range(4), ["1–5", "6–20", "21–50", "> 50"])
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80], ["0%", "20%", "40%", "60%", "80%"])
    ax.set_xlabel("picks at the site in that hour", fontsize=11)
    ax.set_title("Busier site, more coupling")
    patches([C["20kg"], C["6kg"]], ["20 kg scales", "6 kg scales"], fig, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, 0.0), handlelength=1.0)
    return save(fig, "k04_proof")


def ch_loopday(plt, N, R, w, h):
    ld = R["loopday"]
    x = ld[ld.series == "looping scale"].sort_values("hour")
    hp = N["hourly"]
    typ = hp[(hp.cls.isin(["OK", "TOO_MANY"]))].groupby("hour").n.sum()
    typ = typ / typ.sum() * R["norm_n_med"]                      # typical scale: fleet profile x median per day
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.87, bottom=0.22)
    ax.bar(x.hour, x.wakeups, color=hx(RED), width=0.75, label=f"looping scale: {int(x.wakeups.sum())} wake-ups that day")
    ax.bar(typ.index, typ.values, color=hx(PETROL), width=0.75)
    ax.annotate(f"typical scale: about {R['norm_n_med']:.0f} wake-ups a day – too small to see",
                (12, 0.6), xytext=(12, 8), ha="center", fontsize=11, color=hx(PETROL_D), fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none"),
                arrowprops=dict(arrowstyle="-", color=hx(PETROL_D), lw=1))
    ax.set_xticks([0, 6, 12, 18, 23], ["0 h", "6 h", "12 h", "18 h", "23 h"])
    ax.set_xlim(-0.7, 23.7)
    ax.set_ylim(0, 52)
    ax.set_ylabel("wake-ups per hour", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.16), ncol=2, fontsize=11, handlelength=1.0)
    return save(fig, "k05_loopday")


def ch_loops(plt, N, R, w, h):
    sc = N["loop_share_scales"] * 100
    l3 = N["loop_l3_auto"] / N["auto_all"] * 100
    lo = (N["loop_auto"] - N["loop_l3_auto"]) / N["auto_all"] * 100
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.22, right=0.95, top=0.97, bottom=0.4)
    rows = [("scales", [(sc, RED, None), (100 - sc, SAND2, f"{100 - sc:.1f}%")]),
            ("automatic\nweighings", [(l3, RED, f"{l3:.0f}%"), (lo, RED_L, f"{lo:.0f}%"),
                                      (100 - l3 - lo, SAND2, f"{100 - l3 - lo:.0f}%")])]
    for y, (_, parts) in zip([1, 0], rows):
        left = 0
        for v, c, lab in parts:
            ax.barh(y, v, left=left, color=hx(c), height=0.58, edgecolor="white", linewidth=1.5)
            if lab:
                ax.text(left + v / 2, y, lab, ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if c == RED else hx(DEEP))
            left += v
    ax.text(sc + 1.2, 1, f"{sc:.1f}% ({N['loop_scales']} scales)", va="center", fontsize=12, color=hx(RED),
            fontweight="bold")
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=12, colors=hx(DEEP))
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=hx(SAND2))
    patches([RED, RED_L, SAND2], ["loop on level 3 (April trial)", "loop on another level", "no loop"],
            fig, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.55, 0.0), handlelength=0.9,
            columnspacing=1.0)
    return save(fig, "k06_loops")


def ch_knob(plt, N, R, w, h):
    k = R["knob"]
    lv = [7, 6, 5, 4, 2, 1]
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.8, bottom=0.12, wspace=0.3)
    xs = np.arange(len(lv))
    for i, st in enumerate(SCALES):
        x = k.loc[st].reindex([float(l) for l in lv])
        off = -0.19 if i == 0 else 0.19
        axes[0].bar(xs + off, x.empty_wakeups_change * 100, width=0.36, color=hx(C[st]))
        axes[1].bar(xs + off, x.missed_pp, width=0.36, color=hx(C[st]))
        for xx, v in zip(xs + off, x.missed_pp):
            axes[1].text(xx, v + 1.2, f"+{v:.0f}", ha="center", fontsize=10, color=hx(C[st]))
    axes[0].set_title("Empty wake-ups vs. level 8")
    axes[0].set_ylim(-100, 0)
    axes[0].set_yticks([0, -25, -50, -75, -100], ["0%", "−25%", "−50%", "−75%", "−100%"])
    axes[1].set_title("Extra changes missed (points)")
    axes[1].set_ylim(0, 72)
    for ax in axes:
        ax.set_xticks(xs, [f"8→{l}" for l in lv])
        ax.tick_params(axis="x", labelsize=11)
    axes[1].axvspan(-0.5, 1.5, color=hx(PETROL), alpha=0.1, lw=0)
    axes[1].text(0.5, 64, "cheap", ha="center", fontsize=10.5, color=hx(PETROL_D), fontweight="bold")
    patches([C["20kg"], C["6kg"]], ["20 kg scales", "6 kg scales"], fig, loc="upper left", ncol=2,
            fontsize=11, bbox_to_anchor=(0.08, 1.0), handlelength=1.0)
    return save(fig, "k07_knob")


def ch_drivers(plt, N, R, w, h):
    s = N["spread"]
    order = ["Site (location)", "usage", "Sensitivity level", "Fill level (weight bin)", "Box type",
             "Item weight (quartile)"]
    names = ["Site", "Usage of the bin", "Sensitivity level", "Fill level", "Box type", "Item weight"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.29, right=0.96, top=0.86, bottom=0.15)
    ys = np.arange(len(order))[::-1].astype(float)
    for i, st in enumerate(SCALES):
        v = s[s.scale_type == st].set_index("parameter").spread.to_dict()
        lo_, hi_ = N["usage"][st]
        v["usage"] = lo_ - hi_                                   # least- vs. most-used fifth of the bins
        vals = [v[o] for o in order]
        yy = ys + (0.18 if i == 0 else -0.18)
        ax.barh(yy, vals, height=0.32, color=hx(C[st]))
        for y, val in zip(yy, vals):
            ax.text(val + 0.006, y, f"{val:.2f}", va="center", fontsize=10.5, color=hx(C[st]), fontweight="bold")
    ax.set_yticks(ys, names)
    ax.tick_params(axis="y", labelsize=12, colors=hx(DEEP))
    ax.set_xlim(0, 0.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=hx(SAND2))
    ax.set_xlabel("TOO_MANY fraction: best vs. worst group (level 8)", fontsize=11)
    patches([C["20kg"], C["6kg"]], ["20 kg scales", "6 kg scales"], fig, loc="upper left", ncol=2,
            bbox_to_anchor=(0.27, 1.0), handlelength=1.0, fontsize=11)
    return save(fig, "k08_drivers")


def ch_detector(plt, N, R, w, h):
    b = R["bins"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.9, bottom=0.17)
    for loop, col in [(False, PETROL), (True, RED)]:
        x = b[b.loop == loop]
        ax.scatter(10 ** (x.x + 0.05), (x.y + 0.02) * 100, s=np.clip(np.sqrt(x.scale_levels) * 9, 6, 260),
                   color=hx(col), alpha=0.55, lw=0, label="loop" if loop else "normal")
    ax.axvline(20, color=hx(DEEP), lw=1.1, ls=(0, (4, 3)))
    ax.axhline(50, color=hx(DEEP), lw=1.1, ls=(0, (4, 3)))
    ax.text(24, 97, "loop: > 50% and > 20 a day", fontsize=10.5, color=hx(RED), fontweight="bold", va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1500)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=hx(SAND2))
    ax.set_xlabel("n: automatic wake-ups per day (log scale)", fontsize=11)
    ax.set_ylabel("r: share within 2 min\nof the previous wake-up", fontsize=11)
    ax.legend(loc="upper left", fontsize=11, markerscale=0.8, bbox_to_anchor=(0.0, 1.0))
    return save(fig, "k09_detector")


def ch_policies(plt, N, R, w, h):
    pol, P = N["pol"], N["P"]
    lab = {"P0": "today", "P1": "one level for all\n(Bossard KPIs)", "P5": "one level for all (KPI 2.0)",
           "P8": "steer by the fraction", "P7": "recommended rule"}
    place = {"20kg": {"P0": (-9, 0, "right"), "P1": (9, 0, "left"), "P5": (9, -1, "left"),
                      "P8": (9, 2, "left"), "P7": (9, 12, "left")},
             "6kg": {"P0": (-9, 0, "right"), "P1": (0, -19, "center"), "P5": (9, 0, "left"),
                     "P8": (9, 7, "left"), "P7": (10, 3, "left")}}
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.9, bottom=0.17, wspace=0.07)
    for ax, st in zip(axes, SCALES):
        ax.axhspan(0, 10, color=hx(PETROL), alpha=0.1, lw=0)
        ax.text(-98, 1.6, "≤ 10% missed", fontsize=10, color=hx(PETROL_D))
        for key in ["P0", "P1", "P5", "P8", "P7"]:
            r = pol.loc[(st, P[key])]
            x, y = r.meas_change_pct_without_loops, r.median_miss_rate * 100
            rec = key == "P7"
            ax.scatter(x, y, s=170 if rec else 62, color=hx(C[st]) if rec else "white", edgecolor=hx(C[st]),
                       linewidth=2, zorder=4)
            dx, dy, ha = place[st][key]
            ax.annotate(lab[key], (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center",
                        fontsize=10.5, color=hx(DEEP) if rec else hx(INK2), fontweight="bold" if rec else "normal",
                        linespacing=1.0)
        ax.set_title(f"{LABEL[st]} scales", color=hx(C[st]))
        ax.set_xlim(-100, 10)
        ax.set_ylim(0, 56)
        ax.set_xticks([-100, -75, -50, -25, 0], ["−100%", "−75%", "−50%", "−25%", "0%"])
        ax.grid(axis="x", color=hx(SAND2))
        ax.set_xlabel("weighings vs. today", fontsize=11)
    axes[0].set_ylabel("changes missed, median scale (%)", fontsize=11)
    return save(fig, "k10_policies")


def ch_cleaning(plt, N, R, w, h):
    fu = N["funnel"]
    names = {"A2_duplicate": "Duplicate copies", "A5_weight_invalid": "Impossible weights",
             "B0_duplicate_label_conflict": "Duplicates, conflicting labels", "B1_label_missing": "Label missing",
             "B3_sensitivity_missing": "Sensitivity level missing",
             "B4_trigger_config_nonstandard": "Non-standard wait time / delay",
             "B5_device_metadata_missing": "Scale type or site unknown",
             "B6_section_not_relevant": "Periods too short (D ONE rule)",
             "B7_sensitivity_anomaly": "Level 3 (studied separately)", "B8_sensitivity_low_support": "Levels used by < 30 scales"}
    x = fu[fu.rule.isin(names) & (fu.rows_removed > 0)].copy()
    x["name"] = x.rule.map(names)
    x = x.sort_values("rows_removed")
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.38, right=0.95, top=0.98, bottom=0.25)
    col = [RED if a == "delete row" else (AMBER if r.startswith("B7") else "B9BAC4") for a, r in zip(x.action, x.rule)]
    ax.barh(x.name, x.rows_removed, color=[hx(c) for c in col], height=0.62)
    for yi, v in enumerate(x.rows_removed):
        ax.text(v * 1.3, yi, n0(v), va="center", fontsize=11, color=hx(INK2))
    ax.set_xscale("log")
    ax.set_xlim(2, 3e7)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=hx(SAND2))
    ax.tick_params(axis="y", labelsize=11.5, colors=hx(DEEP))
    ax.set_xticks([10, 1e3, 1e5, 1e7], ["10", "1,000", "100,000", "10 M"])
    ax.set_xlabel("rows (log scale)", fontsize=11)
    patches([RED, "B9BAC4", AMBER], ["deleted (stage A)", "flagged, kept (stage B)", "flagged: level 3"], fig,
            loc="lower center", ncol=3, fontsize=10.5, bbox_to_anchor=(0.6, 0.0), handlelength=1.0)
    return save(fig, "k11_cleaning")


def ch_blind_zone(plt, N, R, w, h):
    z = N["zones"]
    order = z.drop_duplicates("pick_bin").sort_values("bin_idx").pick_bin.tolist()
    lev = [(8, DEEP, 2.6), (6, "5B6FA0", 2.0), (4, "9AA6C4", 1.8), (1, "C9CFDE", 1.8)]
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.25, wspace=0.08)
    for ax, st in zip(axes, SCALES):
        x = z[z.scale_type == st]
        floor_idx = x[~x.measurable].bin_idx.max() if (~x.measurable).any() else -1
        ax.axvspan(-0.5, floor_idx + 0.5, color=hx(SAND2), lw=0, zorder=0)
        ax.text((floor_idx - 0.5) / 2 + 0.5 * (floor_idx < 2), 6, f"blind\nzone\n< {N['floor_g'][st]:.0f} g",
                ha="center", fontsize=10.5, color=hx(INK2))
        ax.axhline(95, color=hx(PETROL), lw=1.1, ls=(0, (4, 3)))
        ax.text(-0.35, 89, "95%", fontsize=10.5, color=hx(PETROL_D))
        for lv, col, lw in lev:
            y = x[(x.sensitivity == lv) & x.measurable].sort_values("bin_idx")
            ax.plot(y.bin_idx, y.detection * 100, color=hx(col), lw=lw, marker="o", markersize=4.5, label=f"level {lv}")
        ax.set_title(f"{LABEL[st]} scales", color=hx(C[st]))
        ax.set_xticks(range(len(order)), [o.replace(">500", "500+") for o in order], rotation=45, ha="right",
                      fontsize=10.5)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(0, 104)
        ax.set_xlabel("pick size (g)", fontsize=11)
    axes[0].set_ylabel("picks caught at once (%)")
    axes[1].legend(loc="lower right", fontsize=10.5, ncol=2)
    return save(fig, "k12_blind_zone")


# ---------------------------------------------------------------------------------------------
# PowerPoint helpers
# ---------------------------------------------------------------------------------------------
from pptx import Presentation                                   # noqa: E402
from pptx.dml.color import RGBColor                             # noqa: E402
from pptx.enum.dml import MSO_LINE_DASH_STYLE                   # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE           # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN                 # noqa: E402
from pptx.oxml.ns import qn                                     # noqa: E402
from pptx.util import Emu, Inches, Pt                           # noqa: E402
from lxml import etree                                          # noqa: E402

_FONTS: dict = {}


def _font(bold, size):
    key = (bold, size)
    if key not in _FONTS:
        try:
            from matplotlib import font_manager as fm
            from PIL import ImageFont
            path = fm.findfont(fm.FontProperties(family=MPL_SANS, weight="bold" if bold else "normal"),
                               fallback_to_default=False)
            _FONTS[key] = ImageFont.truetype(path, size=int(round(size * 10)))
        except Exception:
            _FONTS[key] = None
    return _FONTS[key]


def n_lines(text, width_in, size, bold=False):
    f = _font(bold, size)
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


def text_h(paras, width_in, size, line=1.12, bold=False, space_after=0, bullet_indent=0.2):
    if isinstance(paras, str):
        paras = paras.split("\n")
    h = 0.0
    for p in paras:
        w = width_in - (bullet_indent if p.startswith("• ") else 0)
        h += n_lines(p[2:] if p.startswith("• ") else p, w, size, bold) * size * 1.17 * line / 72
        h += space_after / 72
    return h


def rgb(c):
    return RGBColor.from_string(c)


def _clean(shape):
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def _runs(text):
    parts = text.split("**")
    return [(p, i % 2 == 1) for i, p in enumerate(parts) if p]


def tb(s, x, y, w, h, paras, size=14, color=DEEP, bold=False, italic=False, align="l", anchor="t", line=1.12,
       space_after=0, spc=None, bullet_indent=0.2, bold_color=None):
    """Text box. paras: str (\\n = new paragraph) or list. '• ' = bullet, **x** = bold. Returns measured bottom."""
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
            etree.SubElement(pPr, qn("a:buClr")).append(etree.fromstring(
                f'<a:srgbClr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" val="{PETROL}"/>'))
            etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
            etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
        for t, b in _runs(ptxt) or [("", False)]:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name, f.size, f.bold, f.italic = SANS, Pt(size), (bold or b), italic
            f.color.rgb = rgb(bold_color if (b and bold_color) else color)
            if spc:
                r._r.get_or_add_rPr().set("spc", str(spc))
    return y + text_h(paras, w, size, line, bold, space_after, bullet_indent)


def rect(s, x, y, w, h, fill=None, line=None, lw=0.75, shape=MSO_SHAPE.RECTANGLE, radius=None, dash=False):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    _clean(sh)
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(lw)
        if dash:
            sh.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    else:
        sh.line.fill.background()
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    return sh


def card(s, x, y, w, h, title=None, body=None, title_size=15, body_size=12.5, pad=0.2, title_color=DEEP,
         body_color=INK2, fill=SAND, accent=PETROL, gap=0.07, body_space=3):
    """flat card with a petrol accent bar on the left (industrial style)"""
    rect(s, x, y, w, h, fill=fill)
    if accent:
        rect(s, x, y, 0.06, h, fill=accent)
    cy = y + pad
    if title:
        cy = tb(s, x + pad + 0.06, cy, w - 2 * pad - 0.06, 0.3, title, size=title_size, bold=True, color=title_color,
                line=1.0) + gap
    if body:
        cy = tb(s, x + pad + 0.06, cy, w - 2 * pad - 0.06, y + h - cy, body, size=body_size, color=body_color,
                line=1.13, space_after=body_space)
    return cy


def callout(s, x, y, w, h, text, size=14, fill=DEEP, color=WHITE, bold_color=GREEN):
    rect(s, x, y, w, h, fill=fill)
    tb(s, x + 0.25, y, w - 0.45, h, text, size=size, color=color, anchor="m", line=1.12, bold_color=bold_color)


def bignum(s, x, y, w, value, caption, color=PETROL, size=40, cap_size=12.5, cap_color=INK2):
    nh = size / 72 * 1.05
    tb(s, x, y, w, nh, value, size=size, bold=True, color=color, line=0.95)
    return tb(s, x, y + nh + 0.03, w, 0.3, caption, size=cap_size, color=cap_color, line=1.1)


def head(s, eyebrow, title, dark=False, size=27):
    tb(s, LM, 0.42, CW, 0.25, eyebrow, size=11, bold=True, color=GREEN if dark else PETROL, spc=120)
    lines = n_lines(title, CW, size, True)
    h = size / 72 * 1.12 * lines
    tb(s, LM, 0.7, CW, h, title, size=size, bold=True, color=WHITE if dark else DEEP, line=1.0)
    return 0.7 + h + 0.32


def footer(s, text, dark=False, number=None):
    col = "9A9CB8" if dark else MUTED
    tb(s, LM, 7.0, 10.6, 0.25, text, size=9, color=col, line=1.0)
    if number is not None:
        tb(s, 12.2, 7.0, 0.5, 0.25, str(number), size=9, color=col, align="r")


def pic(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def arrow(s, x1, y1, x2, y2, color=PETROL, lw=1.75):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(lw)
    te = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"))
    te.set("type", "triangle")
    te.set("w", "med")
    te.set("len", "med")
    return c


def hline(s, x1, y, x2, color=LINE_C, lw=0.75):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y), Inches(x2), Inches(y))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(lw)
    return c


def table(s, x, y, w, col_w, rows, size=12, row_h=0.36, align=None, bold_rows=(), header=True, hl_rows=(),
          col_colors=None):
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
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.035)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = {"l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT, "c": PP_ALIGN.CENTER}[(align or "l" * ncol)[j]]
            is_head = header and i == 0
            for t, b in _runs(str(rows[i][j])) or [("", False)]:
                r = p.add_run()
                r.text = t
                r.font.name, r.font.size = SANS, Pt(size)
                r.font.bold = b or is_head or (i in bold_rows)
                c = WHITE if is_head else (col_colors[j] if (col_colors and col_colors[j]) else DEEP)
                r.font.color.rgb = rgb(c)
            tcPr = cell._tc.get_or_add_tcPr()
            for side, show in [("a:lnL", False), ("a:lnR", False), ("a:lnT", False), ("a:lnB", True)]:
                ln = etree.SubElement(tcPr, qn(side))
                if show:
                    ln.set("w", "9525")
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    etree.SubElement(sf, qn("a:srgbClr")).set("val", LINE_C)
                else:
                    ln.set("w", "0")
                    etree.SubElement(ln, qn("a:noFill"))
            fill = DEEP if is_head else ("E3F4F4" if i in hl_rows else None)
            if fill:
                sf = etree.SubElement(tcPr, qn("a:solidFill"))
                etree.SubElement(sf, qn("a:srgbClr")).set("val", fill)
            else:
                etree.SubElement(tcPr, qn("a:noFill"))
    return gt


def placeholder(s, x, y, w, h, letter, what, how):
    rect(s, x, y, w, h, fill=SAND, line=PETROL, lw=1.5, dash=True)
    tb(s, x, y + h * 0.3, w, 0.5, f"POWER BI DASHBOARD {letter}", size=24, bold=True, color=PETROL, align="c", spc=100)
    tb(s, x + 0.5, y + h * 0.3 + 0.62, w - 1.0, 0.9, what, size=14, color=DEEP, align="c", line=1.15)
    tb(s, x + 0.5, y + h - 0.75, w - 1.0, 0.5, how, size=11, color=MUTED, align="c", line=1.1)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = "\n".join(t.strip() for t in text.strip().split("\n"))


def badge(s, x, y, d, text, fill=PETROL, color=WHITE, size=14):
    sh = rect(s, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = p.add_run()
    r.text = text
    r.font.name, r.font.size, r.font.bold = SANS, Pt(size), True
    r.font.color.rgb = rgb(color)
    return sh


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(SW), Inches(SH)
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, dark=False):
        s = self.prs.slides.add_slide(self.blank)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(DEEP if dark else WHITE)
        self.n += 1
        if not dark:
            rect(s, 0, 0, SW, 0.09, fill=PETROL)            # thin petrol band on top
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


def src(s, d, text, dark=False):
    footer(s, text, dark, d.n)


# ---------------------------------------------------------------------------------------------
# the story
# ---------------------------------------------------------------------------------------------
def build():
    N = facts()
    R = robust_facts(N)
    plt = mpl()
    b = N["base"]
    pol, P = N["pol"], N["P"]
    rk = R["knob"]
    coup = R["coup"]
    d = Deck()

    # numbers used in several places
    c_both = coup.loc["both", "coupled_beyond_chance"]
    c_both_all = R["coup_all"].loc["both", "coupled_beyond_chance"]
    p7 = {st: pol.loc[(st, P["P7"])] for st in SCALES}
    p7_nl = {st: R["pol_nl"].loc[(st, P["P7"])] for st in SCALES}
    save20, save6 = -p7["20kg"].meas_change_pct_without_loops / 100, -p7["6kg"].meas_change_pct_without_loops / 100
    miss20, miss6 = p7["20kg"].median_miss_rate, p7["6kg"].median_miss_rate
    miss0 = {st: pol.loc[(st, P["P0"]), "median_miss_rate"] for st in SCALES}
    loop_share = N["loop_share_auto"]
    other_loop_share = N["loop_other_auto"] / N["auto_all"]
    W = N["watch"]
    dl = R["delay"]
    fleet = N["fleet_saved_week"]
    prange = lambda a, b_: f"{a * 100:.0f}–{b_ * 100:.0f}%"
    pct0 = lambda v: mi(f"{v:.0f}") + "%"                      # e.g. -61.4 -> '−61%'
    picks20 = R["base"].loc[("all scales", "20kg"), "picks_per_week"]

    # ---- 1 title --------------------------------------------------------------------------------
    s = d.slide(dark=True)
    rect(s, 0, 0, 0.22, SH, fill=PETROL)
    tb(s, 0.9, 0.85, 11, 0.3, "DATA-DRIVEN SUPPLY CHAIN MANAGEMENT AND LOGISTICS · CASE STUDY BOSSARD × D ONE",
       size=11, bold=True, color=GREEN, spc=120)
    ty = tb(s, 0.9, 1.35, 7.5, 1.8, ["SmartBin Cloud:", "wake up for picks,", "not for noise"], size=38, bold=True,
            color=WHITE, line=1.0)
    tb(s, 0.9, ty + 0.35, 7.0, 0.9, "How every scale can learn its own sensor sensitivity – from 7.6 million "
       "weighings", size=17, color="C9CBE0", line=1.15)
    tb(s, 0.9, 6.35, 8, 0.6, "Erenalp Çakıroğlu · MSc Applied Information and Data Science, HSLU", size=12,
       color="C9CBE0")
    x0 = 8.75
    tb(s, x0, 1.4, 4, 0.3, "THE ANSWER IN THREE NUMBERS", size=10.5, bold=True, color=GREEN, spc=120)
    items = [(f"{pct0(-save20 * 100)} / {pct0(-save6 * 100)}", "weighings with the recommended rule "
              "(20 kg / 6 kg scales)"),
             (pc(c_both), "of the empty wake-ups of normally working scales coincide with a pick at the same site – "
              "beyond chance"),
             (f"{N['loop_share_scales'] * 100:.1f}% → {loop_share * 100:.0f}%", "1.5% of the scales were stuck in a "
              "trigger loop and caused 27% of all automatic weighings")]
    y = 1.8
    for v, cap in items:
        hline(s, x0, y, x0 + 3.9, color="2E3156")
        y = bignum(s, x0, y + 0.15, 3.9, v, cap, color=WHITE, size=30, cap_size=12, cap_color="C9CBE0") + 0.25
    notes(s, f"""Time: 30 seconds.
Bossard's SmartBin scales weigh a bin every time a vibration sensor wakes them up. Today three out of four of these weighings find nothing.
The question of the case: how should the sensor sensitivity be set?
Our answer in one sentence: there is no single right setting – every scale has to learn its own, from data the cloud already has.
The three numbers on the right are the story: half the weighings can go, more than half of the empty wake-ups come from picking nearby, and a tiny group of broken scales caused a quarter of the load.
Key line: "Wake up for picks, not for noise." """)

    # ---- 2 problem ------------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "THE PROBLEM", "Three out of four automatic weighings find nothing")
    steps = ["A worker picks\nfrom a bin", "The vibration sensor\nwakes the scale", "The scale weighs\n(≈ 3,000× sleep\npower)",
             "New stock level\nto ARIMS →\nreplenishment"]
    bw, gap = 1.62, 0.3
    for i, t in enumerate(steps):
        x = LM + i * (bw + gap)
        rect(s, x, y, bw, 1.2, fill=SAND)
        rect(s, x, y, bw, 0.06, fill=PETROL)
        tb(s, x + 0.12, y + 0.18, bw - 0.24, 1.0, t, size=12, color=DEEP, line=1.1)
        if i < 3:
            arrow(s, x + bw + 0.04, y + 0.6, x + bw + gap - 0.04, y + 0.6)
    yy = y + 1.5
    cw2 = (4 * bw + 3 * gap - 0.25) / 2
    card(s, LM, yy, cw2, 1.75, "Too sensitive → empty wake-up",
         "Label TOO_MANY. Battery, radio network, IoT cloud and ARIMS do work for nothing.", accent=RED,
         title_size=14, body_size=12.5)
    card(s, LM + cw2 + 0.25, yy, cw2, 1.75, "Too insensitive → missed pick",
         "Label TOO_FEW. The stock level stays wrong until a later weighing finds the change.", accent=AMBER,
         title_size=14, body_size=12.5)
    xr = 8.45
    ty = y - 0.05
    ty = bignum(s, xr, ty, 4.2, pc(N["share_empty_auto"]), f"of {N['auto_all'] / 1e6:.1f} million automatic weighings "
                "were empty") + 0.28
    ty = bignum(s, xr, ty, 4.2, f"{b.loc['20kg', 'too_many_per_week_median']:.1f} vs. {picks20:.1f}",
                "empty wake-ups vs. real picks per week – typical 20 kg scale on the factory default (level 8)") + 0.28
    bignum(s, xr, ty, 4.2, f"{pc(b.loc['20kg', 'share_in_target'])} / {pc(b.loc['6kg', 'share_in_target'])}",
           "of the 20 kg / 6 kg scales meet both Bossard targets")
    callout(s, LM, yy + 2.05, CW, 0.75, "**The case question:** which parameters matter – and how should the sensitivity "
            "(1–10) be set so that a scale wakes up for picks, not for noise?", size=13.5)
    src(s, d, OWN + " · case description Bossard / D ONE (2026) · Report §2, §4.1")
    notes(s, f"""Time: 1 minute.
How a SmartBin works: a worker picks, the vibration sensor wakes the scale, the scale weighs and sends the new stock level to ARIMS, which reorders.
A weighing is expensive: about 3,000 times the sleep power, plus radio, cloud and ARIMS traffic.
Two ways to fail. Too sensitive: the scale wakes up and nothing changed – TOO_MANY. Too insensitive: it sleeps through a pick and the stock is wrong until a later weighing – TOO_FEW.
Today the balance is clearly off: {pc(N['share_empty_auto'])} of all automatic weighings are empty. The typical 20 kg scale wakes up empty {b.loc['20kg', 'too_many_per_week_median']:.1f} times a week for 1.5 real picks.
Only {pc(b.loc['20kg', 'share_in_target'])} of the 20 kg and {pc(b.loc['6kg', 'share_in_target'])} of the 6 kg scales meet both targets.
Key line: "The sensor is set for safety – and pays for it with noise." """)

    # ---- 3 data and approach --------------------------------------------------------------------
    s = d.slide()
    y = head(s, "DATA AND APPROACH", "7.6 million weighings, cleaned and then measured scale by scale")
    stats = [(f"{N['rows'] / 1e6:.1f} M", "weighings"), (f"{N['scales']:,}", "scales at customer sites"),
             ("16 weeks", "1 March – 20 June 2022"), (f"{N['switchers']:,}", "scales moved off level 8 in April: "
                                                      "a natural experiment")]
    sw_ = CW / 4
    for i, (v, cap) in enumerate(stats):
        bignum(s, LM + i * sw_, y - 0.05, sw_ - 0.25, v, cap, size=32, cap_size=12)
    y2 = y + 1.25
    steps = [("1", "Clean", f"Delete what is broken: {N['dupes']:,} duplicate copies, {N['bad_weight']} impossible "
              f"weights. Flag, don't delete, rows unfit for KPIs (no label, unknown level, short periods) – they stay "
              f"in the weight chain. {pc(N['share_main'])} of the rows enter the KPI analysis."),
             ("2", "Measure", "One record per scale and level; the median is the 'typical scale'. Pooled rows would "
              "let a few noisy scales dominate."),
             ("3", "Explain", "Timing of empty wake-ups against picks at the same site, detection by pick size, "
              "site tests, a loop detector."),
             ("4", "Decide", f"Same scale before vs. after the switch, minus the trend of scales that stayed on 8 "
              f"(difference-in-differences); a simulator for 9 policies.")]
    cw4, g4 = (CW - 3 * 0.22) / 4, 0.22
    for i, (num, t, body) in enumerate(steps):
        x = LM + i * (cw4 + g4)
        rect(s, x, y2, cw4, 3.0, fill=SAND)
        badge(s, x + 0.2, y2 + 0.2, 0.5, num)
        tb(s, x + 0.82, y2 + 0.28, cw4 - 1.0, 0.4, t, size=16, bold=True, color=DEEP)
        tb(s, x + 0.2, y2 + 0.9, cw4 - 0.4, 2.0, body, size=12, color=INK2, line=1.13)
        if i < 3:
            arrow(s, x + cw4 + 0.02, y2 + 0.45, x + cw4 + g4 - 0.02, y2 + 0.45)
    tb(s, LM, y2 + 3.2, CW, 0.4, "**Built to be rerun:** Python + DuckDB pipeline (about one minute on a laptop), "
       "tested on a synthetic fleet with known ground truth; Power BI reads only aggregated tables.", size=12,
       color=INK2)
    src(s, d, OWN + " · Report §2–3, Appendix A")
    notes(s, f"""Time: 1 minute.
The data: {N['rows'] / 1e6:.1f} million weighings of {N['scales']:,} scales over 16 weeks in 2022. Each weighing is labelled: real pick, empty wake-up, nothing missed, or a missed change.
Four steps. Clean: we delete only what is technically broken – {N['dupes']:,} duplicate copies and {N['bad_weight']} impossible weights. Everything else that is unfit for the KPIs is flagged but kept, because the next weighing needs it to compute the weight change.
Measure: one record per scale and level, summarised with the median. Otherwise a handful of noisy scales would drive every number.
Explain and decide: the strongest evidence comes from April, when Bossard moved {N['switchers']:,} scales away from level 8. We compare each of them with itself and with the scales that stayed on 8.
Key line: "We let the data run its own experiment." """)

    # ---- 4 Power BI A --------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "DASHBOARD", "Fleet overview: how far today's scales are from the targets")
    placeholder(s, LM, y, CW, 6.75 - y, "A", "Page 1 – fleet overview: KPIs by scale type and sensitivity level, "
                "consistency of every level, share of scales in target",
                "Insert a screenshot of Power BI page 1 here · data: fact_device_sensitivity.csv (one row per scale "
                "and level, built by the pipeline)")
    src(s, d, "Source: Power BI report on the aggregated pipeline output (fact_device_sensitivity.csv)")
    notes(s, """Time: 1 minute (live or screenshot).
Walk through the dashboard in three clicks: the KPI cards (fraction and TOO_FEW per scale type), the level filter, and the consistency chart that shows how wide the spread is within every level.
Point out: whatever level you pick, the spread inside the level is wider than the difference between levels – the first hint that one setting for all cannot work.
Key line: "The level matters – but where and how a bin is used matters more." """)

    # ---- 5 finding 1 ---------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "FINDING 1 · THE TARGETS", "The problem is one-sided – and the KPI points the wrong way")
    pic(s, ch_one_sided(plt, N, R, 6.3, 2.6), LM - 0.1, y - 0.05, w=6.3)
    kk = N["knob"].set_index(["scale_type", "sensitivity"])
    tb(s, LM, y + 2.75, 5.9, 2.4, [
        f"• **Misses are rare today:** {pc(b.loc['20kg', 'share_meets_tf'])} / {pc(b.loc['6kg', 'share_meets_tf'])} of "
        f"the scales meet the TOO_FEW target on level 8. There is room to lower the sensitivity.",
        f"• **The fraction hides progress:** level 8 → 1 removes {pc(-np.expm1(kk.loc[('20kg', 1.0), 'beta_tm']))} / "
        f"{pc(-np.expm1(kk.loc[('6kg', 1.0), 'beta_tm']))} of the empty wake-ups, but the fraction falls only by "
        f"{-kk.loc[('20kg', 1.0), 'did_frac_too_many']:.2f} / {-kk.loc[('6kg', 1.0), 'did_frac_too_many']:.2f} – real "
        f"picks vanish too.",
        f"• **It punishes slow movers:** least-used fifth of the 20 kg bins {N['uq'][N['uq'].scale_type == '20kg'].frac_too_many_median.iloc[0]:.2f}, "
        f"most-used fifth {N['uq'][N['uq'].scale_type == '20kg'].frac_too_many_median.iloc[-1]:.2f}."],
       size=12.5, color=DEEP, line=1.12, space_after=6)
    xr = 7.05
    rect(s, xr, y - 0.05, SW - xr - LM + 0.02, 4.95, fill=SAND)
    rect(s, xr, y - 0.05, SW - xr - LM + 0.02, 0.06, fill=PETROL)
    tb(s, xr + 0.3, y + 0.2, 5, 0.3, "WE PROPOSE: KPI 2.0", size=11, bold=True, color=PETROL, spc=120)
    tb(s, xr + 0.3, y + 0.5, 5.3, 0.6, "Measure the load and the service directly", size=17, bold=True, color=DEEP)
    table(s, xr + 0.3, y + 1.15, 5.1, [2.6, 0.9, 2.2], [
        ["KPI per scale", "Target", "Why"],
        ["Empty wake-ups per week", "≤ 7", "battery, radio and cloud pay per weighing"],
        ["Share of changes missed\n= TOO_FEW ÷ (picks + TOO_FEW)", "≤ 10%", "the service the customer feels"],
        ["TOO_FEW per week", "≤ 2", "Bossard's floor, kept"]], size=11.5, row_h=0.62, align="lcl")
    tb(s, xr + 0.3, y + 3.85, 5.1, 0.9, "Absolute counts instead of a ratio: a slow-moving bin is no longer "
       "punished, and switching the sensor off no longer looks like success.", size=12, color=INK2, line=1.12)
    src(s, d, OWN + " · same scales before and after the switch (difference-in-differences) · Report §4.1, §4.3, §4.4, §6.1")
    notes(s, f"""Time: 1.5 minutes.
First finding: the problem is one-sided. Almost every scale meets the TOO_FEW target, almost none the TOO_MANY target. So the real question is how far we can lower the sensitivity before we start missing picks.
Second: Bossard's KPI, the TOO_MANY fraction, is the wrong compass. It divides two numbers that both fall when you lower the sensitivity. On level 1 the sensor is practically off – yet the fraction looks better.
Third: the fraction punishes slow-moving parts. A bin that is picked once a week can never have a good fraction.
So we propose KPI 2.0: count empty wake-ups per week, and measure the share of real changes that were missed. That is what the battery and the customer actually feel.
Key line: "Count what costs money, not a ratio." """)

    # ---- 6 finding 2: key insight ---------------------------------------------------------------
    s = d.slide()
    y = head(s, "FINDING 2 · KEY INSIGHT", "Over half the empty wake-ups coincide with picking nearby")
    tb(s, LM, y - 0.12, 7.3, 0.3, "ONE EXAMPLE FROM THE DATA", size=10.5, bold=True, color=PETROL, spc=120)
    pic(s, ch_episode(plt, N, R, 7.4, 3.3), LM - 0.1, y + 0.15, w=7.4)
    ep = R["episode"]
    tm_ = ep[ep.cls == "TOO_MANY"].t_rel_s.abs()
    tb(s, LM, y + 3.55, 7.1, 0.6, f"One pick at Bin A. Three neighbouring bins of the same site wake up "
       f"{tm_.min():.0f}–{tm_.max():.0f} s before it and find nothing (level 8, {int(ep.hour_of_day.iloc[0])}:00).",
       size=12, color=INK2, line=1.12)
    xr = 8.3
    ty = bignum(s, xr, y - 0.1, 4.4, pc(c_both), "of the empty wake-ups of normally working scales come within ±14 s "
                f"of a real pick at another bin of the same site – beyond chance ({LABEL['20kg']}: "
                f"{pc(coup.loc['20kg', 'coupled_beyond_chance'])}, {LABEL['6kg']}: {pc(coup.loc['6kg', 'coupled_beyond_chance'])})",
                size=54, cap_size=12.5)
    card(s, xr, ty + 0.3, 4.4, 2.2, "What it means", [
        "• The sensor reacts to activity at the rack, not only at its own bin.",
        "• A lower sensitivity filters part of it; the rest needs mechanics (damping).",
        f"• The rest (about {pc(1 - c_both)}) has no link to picking nearby – it stays unexplained."], body_size=12, title_size=14)
    src(s, d, OWN + " · event coincidence analysis (Donges et al., 2016); scales in a trigger loop left out · Report §4.5")
    notes(s, f"""Time: 1.5 minutes. This is the key insight – slow down here.
Here is one real moment from the data: a worker picks from Bin A. Within seconds three neighbouring bins wake up, weigh, and find nothing.
This is not a rare story. For every empty wake-up we looked for a real pick at another bin of the same site in a window of plus or minus 14 seconds. After subtracting what you would expect by chance, {pc(c_both)} of the empty wake-ups of normally working scales are linked to picking nearby.
What does it mean? The sensor reacts to the rack, not only to its own bin. Lowering the sensitivity filters part of it; the rest is mechanical – damping or mounting.
Be honest about the rest: it has no link to picking nearby, and the data cannot tell us what it is.
Transition: "A claim like this needs proof. Here are four tests." """)

    # ---- 7 finding 2: evidence ------------------------------------------------------------------
    s = d.slide()
    y = head(s, "FINDING 2 · EVIDENCE", "Four tests show the coupling is not chance")
    pic(s, ch_correlogram(plt, N, R, 5.6, 3.25), LM - 0.1, y - 0.05, w=5.6)
    pic(s, ch_proof(plt, N, R, 6.6, 3.25), 6.25, y - 0.05, w=6.6)
    ps = R["per_scale"]
    dz = R["dose"]
    d20 = dz[dz.scale_type == "20kg"].set_index("site_picks_that_hour").coupled_beyond_chance
    tests = [("1 · Peak at zero", f"Picks at other bins jump from {R['chance_rate'].loc['20kg']:.1f} to "
              f"{R['peak'].loc['20kg']:.1f} per minute at the moment of an empty wake-up (20 kg)."),
             ("2 · Chance window", f"In a window 60 s later – same hour, same activity – only "
              f"{pc(coup.loc['both', 'p_chance'])} have a pick nearby, against {pc(coup.loc['both', 'p_near'])}."),
             ("3 · Placebo site", f"Picks at another site of similar size at the same second: "
              f"{R['placebo'].loc['both', 'coupled_beyond_chance'] * 100:.1f}% beyond chance. It is local, not the clock."),
             ("4 · Dose and spread", f"Coupling rises from {pc(d20.loc['1-5'])} to {pc(d20.loc['>50'])} with site "
              f"activity; {pc(ps.share_scales_above_chance.min(), 1)} of the scales are above chance.")]
    cw4 = (CW - 3 * 0.18) / 4
    ty = y + 3.35
    for i, (t, body) in enumerate(tests):
        card(s, LM + i * (cw4 + 0.18), ty, cw4, 1.75, t, body, title_size=13, body_size=11.5, pad=0.15)
    src(s, d, OWN + " · scales in a trigger loop left out; chance window 60–88 s after the wake-up; placebo = site "
        "paired by number of picks · Report §4.5, Appendix C")
    notes(s, f"""Time: 1.5 minutes.
Four tests, each closing a loophole.
One: around an empty wake-up, picks at other bins of the site pile up sharply at zero seconds – from about {R['chance_rate'].loc['20kg']:.1f} to {R['peak'].loc['20kg']:.1f} per minute. Both sensors wait the same 30 seconds before they weigh, so the peak sits exactly at zero.
Two: a busy site has picks all the time. So we compare with a window one minute later – same hour, same activity. That is the chance level, and we subtract it.
Three, the placebo: we take picks at a different site of similar size at the same second. If this were just the time of day, the placebo would show the same effect. It shows almost nothing – {R['placebo'].loc['both', 'coupled_beyond_chance'] * 100:.1f} percent.
Four: the busier the site in that hour, the stronger the coupling, and it holds for practically every scale, not for a few.
Limit: the data cannot separate vibration through the rack from a worker touching the bin. A short on-site test with an accelerometer or a damping pad settles that.
Key line: "It is the rack, not the clock." """)

    # ---- 8 finding 3: loops ---------------------------------------------------------------------
    s = d.slide()
    y = head(s, "FINDING 3 · A DEFECT, NOT A SETTING", f"{N['loop_share_scales'] * 100:.1f}% of the scales were stuck in "
             f"a loop – and caused {loop_share * 100:.0f}% of all weighings")
    ld = R["loopday"]
    tb(s, LM, y - 0.12, 6.6, 0.3, "ONE LOOPING SCALE, ONE DAY", size=10.5, bold=True, color=PETROL, spc=120)
    pic(s, ch_loopday(plt, N, R, 6.7, 3.0), LM - 0.1, y + 0.15, w=6.7)
    tb(s, LM, y + 3.25, 6.4, 0.7, f"A wake-up every {ld.median_gap_s.iloc[0]:.0f} s, day and night: exactly the minimum "
       f"wait time (60 s) plus the measurement delay (30 s). The scale triggers itself.", size=12, color=INK2,
       line=1.12)
    xr = 7.35
    pic(s, ch_loops(plt, N, R, 5.4, 1.9), xr - 0.1, y - 0.1, w=5.4)
    tb(s, xr, y + 1.9, 5.3, 2.3, [
        f"• **{N['loop_l3']} scales** looped during the four-week trial of level 3 (April 2022); **{N['loop_other']} on "
        f"other levels** still caused {pc(other_loop_share)} of all automatic weighings.",
        f"• A loop empties a battery designed for five years in **6–14 weeks**.",
        f"• Same picture for any threshold: {prange(*R['grid_share'])} of the weighings."],
       size=12.5, color=DEEP, space_after=6)
    callout(s, LM, 5.95, CW, 0.75, f"**We separated these scales and re-ran every analysis:** the findings hold, and the "
            f"coupling gets stronger ({pc(c_both_all)} → {pc(c_both)}). A different sensitivity is not the fix – service and "
            "firmware are.", size=13)
    src(s, d, OWN + " · battery: case assumption of ≈ 2 s per weighing at ≈ 3,000× sleep power · Report §5.3, Appendix C")
    notes(s, f"""Time: 1 minute.
Third finding, and the most surprising one: a small group of scales is simply broken. They wake up every {ld.median_gap_s.iloc[0]:.0f} seconds, around the clock – that is exactly the minimum wait time plus the measurement delay, so the scale keeps triggering itself.
{N['loop_scales']} scales, {N['loop_share_scales'] * 100:.1f} percent of the fleet, caused {loop_share * 100:.0f} percent of all automatic weighings. Most of them during the four-week trial of level 3 in April – that is why the case says "ignore level 3". But {N['loop_other']} scales looped on normal levels too.
A loop drains a five-year battery in a few weeks.
Important for trust: we took these scales out and re-ran everything. The results hold, and the coupling becomes even clearer, because a looping scale wakes up at night when nobody picks.
Key line: "You cannot tune a broken scale – you flag it." """)

    # ---- 9 finding 4: knob ----------------------------------------------------------------------
    s = d.slide()
    y = head(s, "FINDING 4 · THE KNOB", "The knob works – but the right level is different for every bin")
    pic(s, ch_knob(plt, N, R, 6.6, 3.3), LM - 0.1, y - 0.05, w=6.6)
    e7 = {st: -rk.loc[(st, 7.0), "empty_wakeups_change"] for st in SCALES}
    m7 = {st: rk.loc[(st, 7.0), "missed_pp"] for st in SCALES}
    tb(s, LM, y + 3.35, 6.3, 1.0, f"Same scales before and after the switch: 8 → 7 removes {pc(e7['20kg'])} / "
       f"{pc(e7['6kg'])} of the empty wake-ups for {m7['20kg']:.0f} / {m7['6kg']:.0f} points more misses. Below 6, misses "
       "climb steeply.", size=12.5, color=DEEP, line=1.12)
    xr = 7.15
    pic(s, ch_drivers(plt, N, R, 5.6, 3.3), xr - 0.1, y - 0.05, w=5.6)
    tb(s, xr, y + 3.35, 5.4, 1.0, "Site and usage spread the result most; box type and item weight – the facts known "
       "before installation – matter little.", size=12.5, color=DEEP, line=1.12)
    callout(s, LM, 5.95, CW, 0.75, "**So a lookup table cannot carry the setting:** a table by pick size picks the same "
            "level as 'one level for all' for 95–99% of the scales. **Each scale has to learn its level.**", size=13)
    src(s, d, OWN + " · difference-in-differences with a fixed-effects Poisson model, scales in a trigger loop left "
        "out · Report §4.8, §5.1, §6.2")
    notes(s, f"""Time: 1.5 minutes.
Left: what the knob really does, measured on the same scales before and after Bossard switched them, corrected for the trend of scales that stayed on 8. The first step, 8 to 7, removes about {pc(e7['20kg'])} to {pc(e7['6kg'])} of the empty wake-ups and costs only a few points of misses. Below level 6 the price rises steeply.
Right: what drives the result between scales. Site and usage of the bin dominate. Box type and item weight – the only things Bossard knows before installing a bin – hardly matter.
That is the logical turning point of the project. If the decisive facts only show once the scale runs, no master-data table can set the level. We tested it: a table by pick size ends up choosing the same level as one level for all.
Key line: "The right setting is not in the master data – it is in the scale's own history." """)

    # ---- 10 Power BI B --------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "DASHBOARD", "Sites and levels: the same setting behaves differently at every site")
    placeholder(s, LM, y, CW, 6.75 - y, "B", "Page 2 – sites × sensitivity level: median TOO_MANY fraction and empty "
                "wake-ups per week by site; filter by scale type",
                "Insert a screenshot of Power BI page 2 here · data: fact_device_sensitivity.csv (site, level, "
                "KPIs per scale)")
    src(s, d, "Source: Power BI report on the aggregated pipeline output (fact_device_sensitivity.csv)")
    notes(s, """Time: 1 minute.
Show one site with a good and one with a bad median on the same level. The differences between sites are large and reproducible: 40 to 42 percent of the variation lies between sites, and two random halves of each site agree closely.
Key line: "Same knob, different factory, different result." """)

    # ---- 11 solution overview -------------------------------------------------------------------
    s = d.slide()
    y = head(s, "OUR RECOMMENDATION", "Flag, tune, watch – three steps on data the cloud already has")
    cols = [("FLAG", "Find broken scales every day",
             [f"Daily loop detector on every scale and level", "Looping scale: back to 7, then service",
              "Level 3 off the menu until the firmware is fixed"],
             f"{N['loop_scales']} scales ({N['loop_share_scales'] * 100:.1f}%) in the data"),
            ("TUNE", "Every scale learns its level",
             ["Start every scale on level 7", "Every two weeks: one step up, one step down or hold",
              "Same rule for 6 kg and 20 kg, reported apart"],
             f"weighings {pct0(-save20 * 100)} / {pct0(-save6 * 100)}"),
            ("WATCH", "Measure what the customer feels",
             ["KPI 2.0 per scale", "Keep the existing 24-hour idle check",
              f"Watch list for the 10% of scales with most misses"],
             "most misses found within 2 days")]
    cw3 = (CW - 2 * 0.25) / 3
    for i, (tag, t, bullets, eff) in enumerate(cols):
        x = LM + i * (cw3 + 0.25)
        rect(s, x, y, cw3, 3.45, fill=SAND)
        rect(s, x, y, cw3, 0.62, fill=DEEP)
        tb(s, x + 0.22, y + 0.15, 1.2, 0.35, tag, size=16, bold=True, color=GREEN, spc=150)
        tb(s, x + 1.35, y + 0.17, cw3 - 1.5, 0.35, f"{i + 1} / 3", size=12, color="C9CBE0", align="r")
        tb(s, x + 0.22, y + 0.8, cw3 - 0.44, 0.4, t, size=15, bold=True, color=DEEP)
        tb(s, x + 0.22, y + 1.3, cw3 - 0.44, 1.5, ["• " + bb for bb in bullets], size=12, color=INK2, space_after=4)
        hline(s, x + 0.22, y + 2.85, x + cw3 - 0.22)
        tb(s, x + 0.22, y + 2.95, cw3 - 0.44, 0.4, eff, size=13, bold=True, color=PETROL_D)
    ty = y + 3.75
    phases = [("NOW", "Loop detector live · level 3 off the menu · factory default 8 → 7"),
              ("WEEKS 1–8", "Pilot the rule at about 10 sites against matched control sites"),
              ("THEN", "Roll out; tune the thresholds with the pilot results")]
    pw = CW / 3
    for i, (t, body) in enumerate(phases):
        x = LM + i * pw
        rect(s, x, ty + 0.12, pw - 0.05, 0.05, fill=PETROL if i == 0 else LINE_C)
        badge(s, x, ty, 0.3, "", fill=PETROL if i == 0 else LINE_C)
        tb(s, x, ty + 0.45, pw - 0.3, 0.3, t, size=11, bold=True, color=PETROL_D, spc=100)
        tb(s, x, ty + 0.75, pw - 0.3, 0.7, body, size=12, color=DEEP, line=1.1)
    src(s, d, OWN + " · simulation over 16 weeks, scales in a trigger loop left out of the savings · Report §6.3–6.4")
    notes(s, f"""Time: 1 minute.
Our recommendation has three parts, and none of them needs new hardware.
Flag: a daily rule finds the broken scales and takes them out of the tuning – they go to service.
Tune: every scale starts on level 7 and then moves one step at a time, every two weeks, based on its own empty wake-ups and misses.
Watch: KPI 2.0 per scale, the existing 24-hour check as the safety net, and extra attention for the few scales where misses concentrate.
Roll-out in three phases: the detector and the new default can start now; the rule goes into an eight-week pilot with control sites; then roll out.
Key line: "Flag what is broken, tune what works, watch what the customer feels." """)

    # ---- 12 FLAG --------------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "STEP 1 · FLAG", "A two-number rule finds the broken scales every day")
    rect(s, LM, y - 0.05, 6.2, 2.15, fill=DEEP)
    tb(s, LM + 0.3, y + 0.12, 5.6, 0.3, "FOR EVERY SCALE AND LEVEL · LAST 7 DAYS (≥ 5 WITH DATA)", size=10, bold=True,
       color=GREEN, spc=100)
    tb(s, LM + 0.3, y + 0.5, 5.7, 1.5, ["r = wake-ups within 2 min of the previous one ÷ all wake-ups",
                                        "n = automatic wake-ups per day",
                                        "**Loop  ⇔  r > 50%  and  n > 20**"], size=14, color=WHITE, line=1.2,
       space_after=6, bold_color=GREEN)
    table(s, LM, y + 2.4, 6.2, [2.9, 1.1, 0.9, 1.3], [
        ["Example (from the data)", "n per day", "r", "Result"],
        ["Typical looping scale", f"{R['loop_n_med']:.0f}", pc(R["loop_r_med"]), "**LOOP**"],
        ["Normal scales, 99th percentile", f"{R['norm_n_p99']:.0f}", pc(R["norm_r_p99"]), "normal"],
        ["Typical scale", f"{R['norm_n_med']:.0f}", pc(R["norm_r_med"]), "normal"]], size=12, row_h=0.4,
          align="lrrl", col_colors=[None, None, None, None])
    pic(s, ch_detector(plt, N, R, 5.7, 3.6), 7.0, y - 0.1, w=5.7)
    steps = [("1", "Set the scale back to 7 remotely – as Bossard did for the April switch"),
             ("2", "Still looping after 2 days → service ticket (sensor, battery)"),
             ("3", "List to the firmware team: the 90 s pattern is a firmware loop")]
    ty = 5.68
    tb(s, LM, ty - 0.02, 2.2, 0.3, "WHAT HAPPENS NEXT", size=10.5, bold=True, color=PETROL, spc=120)
    sw3 = (CW - 2.3) / 3
    for i, (n_, t) in enumerate(steps):
        x = LM + 2.3 + i * sw3
        badge(s, x, ty - 0.05, 0.36, n_, size=12)
        tb(s, x + 0.48, ty - 0.03, sw3 - 0.6, 0.8, t, size=12, color=DEEP, line=1.1)
    tb(s, LM, ty + 0.72, CW, 0.3, f"Robust: any threshold between 40–60% and 10–80 a day flags {R['grid_scales'][0]}–"
       f"{R['grid_scales'][1]} scales holding {prange(*R['grid_share'])} of the weighings. "
       "Delivered as about 20 lines of SQL/Python plus today's list.", size=11.5, color=INK2)
    src(s, d, OWN + " · each dot = scale-levels with similar values, size = number of scale-levels · Report §5.3, Appendix C")
    notes(s, f"""Time: 1 minute.
How do you find a broken scale? Two numbers per scale and level over the last week: r, the share of wake-ups that come within two minutes of the previous one, and n, wake-ups per day. If more than half come in rapid succession and there are more than 20 a day, it is a loop.
The chart shows why this is easy: normal scales and looping scales sit in completely different corners. The typical looping scale wakes up {R['loop_n_med']:.0f} times a day; even the 99th percentile of normal scales stays at about {R['norm_n_p99']:.0f} wake-ups a day and {pc(R['norm_r_p99'])} quick re-triggers.
What happens then? First, the scale is set back to 7 remotely. If it keeps looping for two days, a service ticket. And the list goes to the firmware team.
We do not send anyone to service ourselves: we deliver the rule as code and today's list; Bossard runs it daily in the cloud.
Key line: "Two numbers, one rule, every day." """)

    # ---- 13 TUNE --------------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "STEP 2 · TUNE", "Every two weeks, each scale moves one step – or stays")
    rect(s, LM, y - 0.05, 5.55, 3.95, fill=DEEP)
    tb(s, LM + 0.3, y + 0.12, 5.0, 0.3, "EVERY 14 DAYS, PER SCALE", size=10, bold=True, color=GREEN, spc=100)
    tb(s, LM + 0.3, y + 0.45, 5.0, 0.8, "Count E = empty wake-ups, P = picks caught, M = misses found later",
       size=12.5, color="C9CBE0", line=1.12)
    tb(s, LM + 0.3, y + 1.1, 5.0, 0.4, "m = M ÷ (P + M)", size=17, bold=True, color=WHITE)
    rules = [("UP", "one level more sensitive", "if m > 10% or M > 4", AMBER),
             ("DOWN", "one level less sensitive", "if E > 14 and m ≤ 5%", GREEN),
             ("HOLD", "stay", "otherwise", "C9CBE0")]
    ry = y + 1.7
    for t, what, cond, col in rules:
        tb(s, LM + 0.3, ry, 1.0, 0.3, t, size=14, bold=True, color=col)
        tb(s, LM + 1.3, ry, 4.0, 0.3, f"{what} · **{cond}**", size=12.5, color=WHITE, bold_color=WHITE)
        ry += 0.45
    tb(s, LM + 0.3, ry + 0.12, 5.0, 0.5, "Start on 7 · range 1–8 · level 3 blocked · thresholds = KPI 2.0 "
       "(> 7 a week, > 2 misses a week)", size=11, color="C9CBE0", line=1.1)
    xr = 6.55
    tb(s, xr, y - 0.12, 6, 0.3, "WORKED EXAMPLE · A BUSY 20 KG BIN", size=10.5, bold=True, color=PETROL, spc=120)
    table(s, xr, y + 0.25, SW - xr - LM, [1.0, 0.75, 0.6, 0.6, 0.6, 0.8, 2.6], [
        ["Weeks", "Level", "E", "P", "M", "m", "Decision"],
        ["1–2", "7", "22", "4", "0", "0%", "E > 14, m ≤ 5% → **down**"],
        ["3–4", "6", "15", "3", "0", "0%", "E > 14, m ≤ 5% → **down**"],
        ["5–6", "5", "6", "2", "1", "33%", "m > 10% → **up**"],
        ["7–8", "6", "12", "3", "0", "0%", "E ≤ 14 → **hold**"]], size=12, row_h=0.42, align="ccrrrrl", hl_rows=(4,))
    tb(s, xr, y + 2.5, SW - xr - LM, 1.3, f"The bin settles on level 6 with about two thirds fewer empty wake-ups "
       f"than on 8 (46 per two weeks). Counts are illustrative, scaled with the measured effects of the knob.",
       size=12, color=INK2, line=1.12)
    callout(s, LM, 5.75, CW, 0.95, f"**Why this rule:** it uses only counts the cloud already has, changes a scale about "
            f"once a month, and every decision can be audited. **Trade-off:** the typical scale ends at "
            f"{prange(min(miss20, miss6), max(miss20, miss6))} missed (limit 10%) because the rule keeps probing – the "
            "pilot tunes the 5% step-down threshold.", size=12.5)
    src(s, d, OWN + " · rule P7 simulated over 8 two-week cycles, 5 replications · Report §6.2–6.4")
    notes(s, f"""Time: 1.5 minutes. This is the method the client takes home – make it concrete.
Every two weeks the cloud counts three numbers per scale: empty wake-ups E, picks caught P, and misses M that a later weighing found. The miss share m is M over P plus M.
Then one of three things happens. Too many misses – more than 10 percent, or more than two a week – one step more sensitive. Many empty wake-ups – more than seven a week – while misses are comfortably low: one step less sensitive. Otherwise: stay.
The example: a busy bin starts on 7, steps down twice, misses one pick on level 5, steps back up to 6 and stays there – with about two thirds fewer empty wake-ups than today.
The honest trade-off: because the rule keeps testing the next level down, the typical scale ends at about 11 percent missed, just above the 10 percent limit. The pilot decides whether to make the step-down stricter.
Key line: "Every scale finds its own level – one step at a time." """)

    # ---- 14 WATCH -------------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "STEP 3 · WATCH", "KPI 2.0 and the existing 24-hour check keep every miss visible")
    card(s, LM, y - 0.05, 5.6, 4.4, "What happens when the sensor misses a pick?", [
        f"• **The next weighing finds it** – a later wake-up or the 24-hour idle check, which finds "
        f"{pc(N['idle_share_tf'])} of all detected misses.",
        f"• **Found after a median of {dl.loc['20kg', 'hours_median']:.0f} h (20 kg) / {dl.loc['6kg', 'hours_median']:.0f} h "
        f"(6 kg);** {pc(dl.loc['20kg', 'share_within_48h'])} / {pc(dl.loc['6kg', 'share_within_48h'])} within 48 h.",
        f"• **1% take {dl.loc['20kg', 'hours_p99'] / 24:.0f}–{dl.loc['6kg', 'hours_p99'] / 24:.0f} days** – already today. "
        f"Alarm when a scale has had no weighing for 48 h.",
        "• A one-day delay matters only if that pick crossed the reorder point and the rest lasts less than a day."],
         title_size=15, body_size=12.5, body_space=7)
    xr = 6.5
    wl = (W.loc["20kg", "share_of_too_few_from_top10pct"], W.loc["6kg", "share_of_too_few_from_top10pct"])
    ty = bignum(s, xr, y - 0.1, 6.2, f"10% → {pc(wl[0])} / {pc(wl[1])}", "the 10% of scales with the most misses "
                "produce this share of all misses (20 kg / 6 kg) → they go on a watch list: 12-hour check or one "
                "level higher", size=34, cap_size=12.5) + 0.3
    rect(s, xr, ty, SW - xr - LM, 1.55, fill=SAND)
    rect(s, xr, ty, 0.06, 1.55, fill=RED)
    tb(s, xr + 0.25, ty + 0.15, 5.6, 0.3, "Not recommended: a nightly check for every scale", size=14, bold=True,
       color=DEEP)
    tb(s, xr + 0.25, ty + 0.55, 5.7, 1.0, "It would add 7 weighings per scale and week – 33–40% of what the rule "
       "saves – to find misses that the 24-hour check already finds a day later.", size=12.5, color=INK2, line=1.12)
    tb(s, xr, ty + 1.8, SW - xr - LM, 0.6, "Nothing is removed: the existing 24-hour idle check stays as the safety "
       "net.", size=12.5, bold=True, color=PETROL_D)
    src(s, d, OWN + " · time from the previous weighing to a TOO_FEW · Report §3, §6.4")
    notes(s, f"""Time: 1 minute.
The natural worry: if the sensor sleeps more, does stock stay wrong for days?
The data answer it. A missed pick is found by the next weighing – any later wake-up, or the 24-hour idle check that Bossard already runs. That check finds {pc(N['idle_share_tf'])} of all detected misses. The median time to find a miss is about a day and a half; three quarters or more are found within two days.
One percent take up to a week – that is today already, probably scales that are offline. A cheap alarm for scales without any weighing for 48 hours closes that gap.
And misses are concentrated: ten percent of the scales produce half of them. Those go on a watch list.
What we do not recommend is a nightly check for every scale – it would eat a third of the savings.
Key line: "We keep the safety net – and point it at the few scales that need it." """)

    # ---- 15 result ------------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "RESULT", "Half the weighings – while misses stay low and visible")
    pic(s, ch_policies(plt, N, R, 7.3, 3.9), LM - 0.1, y - 0.05, w=7.3)
    xr = 7.75
    ty = y - 0.1
    ty = bignum(s, xr, ty, 4.9, f"{pct0(-save20 * 100)} / {pct0(-save6 * 100)}", "weighings with the "
                "rule (20 kg / 6 kg, scales without a loop)", size=34) + 0.22
    ty = bignum(s, xr, ty, 4.9, f"{pc(miss20)} / {pc(miss6)}", f"of the changes missed by the typical scale "
                f"(simulated today: {pc(miss0['20kg'])} / {pc(miss0['6kg'])})", size=34) + 0.22
    ty = bignum(s, xr, ty, 4.9, f"≈ {fleet / 1e6:.1f} M", f"weighings saved per week across "
                f"{N['installed']:,} installed scales – an illustration, not a forecast", size=34) + 0.22
    tb(s, xr, ty, 4.9, 0.8, f"Battery life +{p7['20kg'].battery_life_change_pct:.0f}% / +{p7['6kg'].battery_life_change_pct:.0f}%: "
       "sleep power dominates. The big gain is on radio, cloud and ARIMS – and on the looping scales.", size=11.5,
       color=INK2, line=1.12)
    tb(s, LM, y + 4.0, 7.0, 0.6, f"Same result when the knob effects are estimated without the loop scales: "
       f"{pct0(p7_nl['20kg'].meas_change_pct_without_loops)} / {pct0(p7_nl['6kg'].meas_change_pct_without_loops)}, "
       f"{pc(p7_nl['20kg'].median_miss_rate)} / {pc(p7_nl['6kg'].median_miss_rate)} missed.", size=11.5, color=INK2)
    src(s, d, OWN + " · policy simulation with each scale's own level-8 rates × measured rate ratios, 16 weeks · "
        "Report §6.3")
    notes(s, f"""Time: 1 minute.
We simulated nine policies on every scale, using each scale's own behaviour and the measured effect of the knob.
The chart shows the trade-off: further left means fewer weighings, higher means more misses. One level for all, tuned to Bossard's KPIs, saves the most – but misses 38 to 46 percent of all changes: it simply switches the sensor off. Steering by the fraction ends in the same place.
Our rule sits in the sweet spot: about {save20 * 100:.0f} and {save6 * 100:.0f} percent fewer weighings, with the typical scale missing about 11 to 12 percent – and those misses are found by the 24-hour check.
Across the installed base that would be roughly {fleet / 1e6:.1f} million weighings a week – an illustration, not a promise.
Key line: "Half the noise, and nothing goes unseen." """)

    # ---- 16 alternatives ------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "ALTERNATIVES", "What we tested and did not recommend")
    q = N["quiet"][N["quiet"].threshold == 0.01].set_index("scale_type")
    bm = N["burst"][(N["burst"].window_min == 10) & N["burst"].sequence.str.startswith("level 7")].set_index("scale_type")
    mw = N["minwait"][N["minwait"].minwait == 60].set_index("scale_type")
    p1 = {st: pol.loc[(st, P["P1"])] for st in SCALES}
    p8 = {st: pol.loc[(st, P["P8"])] for st in SCALES}
    rows = [["Option", "What happened in the data or the simulation (20 kg / 6 kg)", "Verdict"],
            ["One level for all, tuned to Bossard's KPIs", f"{pct0(p1['20kg'].meas_change_pct_without_loops)} / {pct0(p1['6kg'].meas_change_pct_without_loops)} weighings, but "
             f"{pc(p1['20kg'].median_miss_rate)} / {pc(p1['6kg'].median_miss_rate)} of the changes missed", "reject"],
            ["Steer by the TOO_MANY fraction", f"{pc(p8['20kg'].median_miss_rate)} / {pc(p8['6kg'].median_miss_rate)} "
             "missed: the fraction pushes scales towards level 1", "reject"],
            ["A table by pick size or item weight", "Same level as one-for-all for 95–99% of the scales", "no gain"],
            ["Quiet hours per site", f"Avoids {pc(q.loc['20kg', 'share_empty_avoided'])} / {pc(q.loc['6kg', 'share_empty_avoided'])} "
             f"of the empty wake-ups; {pc(q.loc['20kg', 'share_picks_in_quiet_hours'])} / {pc(q.loc['6kg', 'share_picks_in_quiet_hours'])} "
             "of the picks fall into them", "reject"],
            ["Nightly check for every scale", "Costs 33–40% of the savings", "reject"],
            ["Longer minimum wait time (60 min)", f"{pc(mw.loc['20kg', 'share_scales_top2_sites'])} / {pc(mw.loc['6kg', 'share_scales_top2_sites'])} "
             "of these scales sit at two sites; no scale ran both wait times", "not measurable"],
            ["Merge bursts into one weighing (10 min)", f"{pct0(-bm.loc['20kg', 'saved'] * 100)} / {pct0(-bm.loc['6kg', 'saved'] * 100)} on top of the rule; "
             f"{pc(bm.loc['20kg', 'picks_reported_later'])} of the picks reported about 5 min later", "optional firmware"]]
    table(s, LM, y, CW, [3.6, 6.4, 1.6], rows, size=12, row_h=0.5, align="lll",
          col_colors=[None, INK2, PETROL_D])
    callout(s, LM, 5.95, CW, 0.75, "**The pattern:** every shortcut either switches the sensor off or ignores what makes "
            "each bin different. A rule that learns per scale avoids both.", size=13)
    src(s, d, OWN + " · quiet hours learned on March–April and tested on May–June · Report §6.3, §6.5")
    notes(s, """Time: 1 minute – or skip and keep for questions.
We tested the obvious shortcuts. One level for everyone looks great on Bossard's KPIs and misses almost half of the changes. Steering by the fraction does the same.
Quiet hours sound attractive, but empty wake-ups follow the working day exactly like real picks – so you would block real picks too.
A nightly check for every scale eats a third of the savings. A longer wait time could not be measured, because it was only used at two sites.
Merging bursts is not wrong – it is an optional firmware item that adds a few percent.
Key line: "Every shortcut either switches the sensor off or ignores what makes each bin different." """)

    # ---- 17 roadmap and limits ------------------------------------------------------------------
    s = d.slide()
    y = head(s, "NEXT STEPS", "An eight-week pilot turns the simulation into a decision")
    phases = [("NOW", ["Loop detector runs daily", "Level 3 off the setting menu", "Factory default 8 → 7",
                       "KPI 2.0 in the dashboard"]),
              ("PILOT · WEEKS 1–8", ["Rule at ~10 sites vs. matched control sites", "Weekly KPI 2.0 per scale",
                                     "Test the step-down threshold: 5% vs. 3% missed"]),
              ("IN PARALLEL", ["Accelerometer / damping test at the 10 most coupled sites", "Firmware fix for loops",
                               "Confirm the idle-check rule and the meaning of the levels"])]
    lw_ = 2.75
    pw = (CW - lw_ - 3 * 0.2) / 3
    for i, (t, bullets) in enumerate(phases):
        x = LM + i * (pw + 0.2)
        rect(s, x, y, pw, 3.1, fill=SAND)
        rect(s, x, y, pw, 0.5, fill=PETROL if i < 2 else DEEP)
        tb(s, x + 0.2, y + 0.13, pw - 0.4, 0.3, t, size=12, bold=True, color=WHITE, spc=100)
        tb(s, x + 0.2, y + 0.7, pw - 0.4, 2.3, ["• " + bb for bb in bullets], size=12.5, color=DEEP, space_after=5)
    xr = LM + 3 * (pw + 0.2)
    wr = lw_
    rect(s, xr, y, wr, 3.1, fill=WHITE, line=LINE_C)
    tb(s, xr + 0.15, y + 0.13, wr - 0.3, 0.3, "LIMITS", size=11, bold=True, color=RED, spc=100)
    tb(s, xr + 0.15, y + 0.5, wr - 0.3, 2.6, ["• " + t_ for t_ in ["One season (Mar–Jun 2022)", "Observational: busier scales were "
                                               "switched first", "TOO_FEW is a lower bound of misses",
                                               "No rack positions in the data"]], size=11.5, color=INK2,
       line=1.08, space_after=6)
    callout(s, LM, y + 3.4, CW, 0.85, "**Decision for Bossard today:** start the loop detector and the new default 7 now "
            "– low risk, measured effect. Let the pilot decide the rule's thresholds.", size=13.5)
    src(s, d, OWN + " · Report §7–8")
    notes(s, """Time: 1 minute.
What should Bossard do on Monday? Two things are low-risk and can start now: run the loop detector, and change the factory default from 8 to 7 – we measured that step on hundreds of scales.
The rule itself goes into an eight-week pilot at about ten sites, each compared with a matched control site. The pilot also decides how strict the step-down threshold should be.
In parallel: a vibration test at the sites with the most coupling, a firmware fix for the loops, and three facts we inferred from the data and want Bossard to confirm.
Limits, briefly: one season of data, the switch was not randomised, and TOO_FEW undercounts misses. That is exactly why we recommend a pilot, not a big-bang roll-out.
Key line: "Start the safe part now, let the pilot decide the rest." """)

    # ---- 18 thank you ---------------------------------------------------------------------------
    s = d.slide(dark=True)
    rect(s, 0, 0, 0.22, SH, fill=PETROL)
    tb(s, 0.9, 1.0, 11, 0.3, "THANK YOU", size=11, bold=True, color=GREEN, spc=150)
    tb(s, 0.9, 1.4, 11, 1.0, "Questions?", size=44, bold=True, color=WHITE)
    recap = [("FLAG", "a two-number rule finds broken scales every day"),
             ("TUNE", "every scale learns its level, one step every two weeks"),
             ("WATCH", "KPI 2.0, the 24-hour check and a watch list keep misses visible")]
    yy = 3.0
    for t, body in recap:
        tb(s, 0.9, yy, 1.6, 0.4, t, size=17, bold=True, color=GREEN, spc=120)
        tb(s, 2.6, yy + 0.02, 9.5, 0.4, body, size=17, color=WHITE)
        yy += 0.62
    tb(s, 0.9, 5.6, 11.5, 0.8, "Backup: data cleaning · robustness without loop scales · methods · pick size · "
       "code · dashboard C · references", size=12, color="C9CBE0")
    tb(s, 0.9, 6.35, 11.5, 0.4, "Erenalp Çakıroğlu · code, report and pipeline in a private repository (case data "
       "under NDA)", size=11, color="9A9CB8")
    notes(s, "Leave this slide up during questions. Backup slides answer the likely technical questions: data "
             "cleaning, robustness without the loop scales, methods, pick size, the code and the KPI 2.0 dashboard.")

    # ================================= backup =================================
    # ---- B1 cleaning ----------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "BACKUP · DATA CLEANING", "Delete only what is broken – flag the rest and keep the chain intact")
    pic(s, ch_cleaning(plt, N, R, 7.2, 4.4), LM - 0.1, y - 0.1, w=7.2)
    tb(s, 7.6, y, 5.1, 4.5, [
        f"• **Duplicates:** {N['dupes']:,} copies with the same scale and millisecond. In 1,300 of 5,499 groups the "
        "copies carry different labels – one copy stays for the weight chain, its outcome counts as unknown.",
        f"• **Impossible weights:** {N['bad_weight']} rows (144 kg to 50 t) – exactly those D ONE marked invalid.",
        "• **Why flag instead of delete:** a weighing with an unknown level is useless for the KPI, but the next "
        "weighing needs it to compute its weight change, and it can be the pick that woke a neighbour.",
        "• **Checks:** D ONE's 'relevant period' rule re-computed on all 58,700 periods (100% match); empty wait-time "
        "values behave like the standard.",
        f"• **Result:** {N['rows_main'] / 1e6:.2f} M rows ({pc(N['share_main'])}) enter the KPI analysis; every excluded "
        "row is exported with its reason."], size=12, color=DEEP, space_after=6)
    src(s, d, OWN + " · Report §3, Appendix A")
    notes(s, "Likely question: 'Did you check for duplicates?' Yes – and they were not all identical, which is why "
             "conflicting copies keep the weight chain but lose their label.")

    # ---- B2 robustness --------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "BACKUP · ROBUSTNESS", "The findings hold when the 274 loop scales are left out")
    bs = R["base"]
    ka, kn = R["knob_all"], R["knob"]
    pa = {st: pol.loc[(st, P["P7"])] for st in SCALES}
    rows = [["Result", "All scales", "Without loop scales"],
            ["Typical scale on level 8: empty wake-ups per week (20 kg / 6 kg)",
             f"{bs.loc[('all scales', '20kg'), 'tm_per_week']:.1f} / {bs.loc[('all scales', '6kg'), 'tm_per_week']:.1f}",
             f"{bs.loc[('without loop scales', '20kg'), 'tm_per_week']:.1f} / {bs.loc[('without loop scales', '6kg'), 'tm_per_week']:.1f}"],
            ["Typical TOO_MANY fraction on level 8",
             f"{bs.loc[('all scales', '20kg'), 'frac_too_many']:.2f} / {bs.loc[('all scales', '6kg'), 'frac_too_many']:.2f}",
             f"{bs.loc[('without loop scales', '20kg'), 'frac_too_many']:.2f} / {bs.loc[('without loop scales', '6kg'), 'frac_too_many']:.2f}"],
            ["Empty wake-ups coupled to picking nearby", pc(c_both_all), f"**{pc(c_both)}**"],
            ["8 → 7: empty wake-ups", f"{sp(ka.loc[('20kg', 7.0), 'empty_wakeups_change'])} / {sp(ka.loc[('6kg', 7.0), 'empty_wakeups_change'])}",
             f"{sp(kn.loc[('20kg', 7.0), 'empty_wakeups_change'])} / {sp(kn.loc[('6kg', 7.0), 'empty_wakeups_change'])}"],
            ["8 → 6: empty wake-ups", f"{sp(ka.loc[('20kg', 6.0), 'empty_wakeups_change'])} / {sp(ka.loc[('6kg', 6.0), 'empty_wakeups_change'])}",
             f"{sp(kn.loc[('20kg', 6.0), 'empty_wakeups_change'])} / {sp(kn.loc[('6kg', 6.0), 'empty_wakeups_change'])}"],
            ["Recommended rule: weighings vs. today",
             f"{pct0(pa['20kg'].meas_change_pct_without_loops)} / {pct0(pa['6kg'].meas_change_pct_without_loops)}",
             f"{pct0(p7_nl['20kg'].meas_change_pct_without_loops)} / {pct0(p7_nl['6kg'].meas_change_pct_without_loops)}"],
            ["Recommended rule: changes missed (typical scale)", f"{pc(pa['20kg'].median_miss_rate)} / {pc(pa['6kg'].median_miss_rate)}",
             f"{pc(p7_nl['20kg'].median_miss_rate)} / {pc(p7_nl['6kg'].median_miss_rate)}"]]
    table(s, LM, y, CW, [6.2, 2.7, 2.7], rows, size=12.5, row_h=0.46, align="lrr", hl_rows=(3,))
    tb(s, LM, y + 3.9, CW, 0.9, "Why the coupling rises: a looping scale wakes up around the clock, also when nobody "
       "picks – its wake-ups dilute the link. Why the knob estimate moves: a few loop scales that were switched "
       "carried very large counts. The recommendation does not change.", size=12, color=INK2, line=1.12)
    src(s, d, OWN + " · report/robustness_loops.py · Report Appendix C")
    notes(s, "Question: 'Should the broken scales not be removed first?' Yes – this slide shows every key number with "
             "and without them. Medians and the recommendation are unchanged; the coupling gets stronger.")

    # ---- B3 methods -----------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "BACKUP · METHODS", "Four methods, each answering one question")
    meth = [("Does the knob cause the change?", "Difference-in-differences on 7,846 switched scales: each scale "
             "after vs. before, minus the trend of scales that stayed on 8 over the same weeks. Counts with a "
             "fixed-effects Poisson model, so each scale's own baseline cancels. 95% bootstrap intervals."),
            ("Are empty wake-ups linked to picks?", "Event coincidence analysis: a = (p_near − p_chance) ÷ "
             "(1 − p_chance); near = ±14 s (width of the peak), chance = 60–88 s later; placebo with a paired site."),
            ("How small a pick can a scale see?", "Weight change behind every label; threshold = 5th percentile of "
             "the missed changes (≈ 23 g / 5 g); detection rate per pick size and level."),
            ("Which policy is best?", "Simulator: each scale's own level-8 rates, shrunk towards its type, × the "
             "measured rate ratios; 9 policies, 16 weeks, 5 replications; back-test on the real switchers.")]
    cw2 = (CW - 0.25) / 2
    for i, (t, body) in enumerate(meth):
        x, yy = LM + (i % 2) * (cw2 + 0.25), y + (i // 2) * 2.0
        card(s, x, yy, cw2, 1.8, t, body, title_size=14, body_size=12.5)
    src(s, d, "Angrist & Pischke (2009) · Donges et al. (2016) · Green & Swets (1966) · Report §5.1, §4.5, §4.6, §6.3, Appendix A")
    notes(s, "Use only if asked about a method. The simulator is somewhat optimistic in the back-test and predicts "
             "single scales only roughly – it compares policies, it is not a forecast.")

    # ---- B4 pick size ---------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "BACKUP · PICK SIZE", "Below about 25 g, a 20 kg scale is practically blind")
    pic(s, ch_blind_zone(plt, N, R, 8.0, 3.9), LM - 0.1, y - 0.05, w=8.0)
    tb(s, 8.4, y, 4.3, 4.0, [
        f"• **Threshold:** about {N['floor_g']['20kg']:.0f} g (20 kg) and {N['floor_g']['6kg']:.0f} g (6 kg) – picks below "
        "it are invisible.",
        "• **Levels 1–5** never reach 95% detection for any pick size.",
        "• **Placement rule:** parts picked in amounts below about 25 g belong on 6 kg scales."],
       size=12.5, color=DEEP, space_after=7)
    src(s, d, OWN + " · threshold = 5th percentile of the weight change of missed changes · Report §4.6")
    notes(s, "Question: 'Does the pick size matter?' Yes for placement, not for a table: the efficient level is 6–8 "
             "for almost every pick size, which is why the rule starts on 7.")

    # ---- B5 code --------------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "BACKUP · CODE", "One pipeline from raw file to deck – rerun in about a minute")
    mods = [("io · cleaning", "load 7.6 M rows into DuckDB; stage A deletes, stage B flags with a reason"),
            ("features · tables", "event chain (previous weight, gap, change); one record per scale and level"),
            ("analysis M1–M7", "baseline, knob (DiD), parameters, sites, pick size, coupling, policy simulator"),
            ("M0 validation", "synthetic fleet with known physics: the pipeline must recover it first"),
            ("powerbi · report", "aggregated tables for Power BI; figures, report and both decks"),
            ("robustness_loops", "every key result again without the loop scales, placebo test, examples")]
    rows = [["Module", "What it does"]] + [[a, bb] for a, bb in mods]
    table(s, LM, y, 7.6, [2.2, 5.4], rows, size=12, row_h=0.5, align="ll")
    card(s, 8.5, y - 0.05, 4.2, 3.6, "Run it", ["bash run_bossard.sh", "python report/build_figures.py",
                                                "python report/robustness_loops.py", "quarto render report/report.qmd",
                                                "python presentation/build_cto_deck.py"], title_size=14, body_size=11.5)
    tb(s, LM, y + 3.75, CW, 0.6, "NDA: raw and row-level data and Power BI files never enter the repository; only "
       "aggregated tables do.", size=12, color=INK2)
    src(s, d, "Source: project repository (private)")
    notes(s, "For the data scientists in the room: everything is reproducible from the three raw files.")

    # ---- B6 Power BI C --------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "BACKUP · DASHBOARD", "KPI 2.0 monitoring: which scales need attention this week")
    placeholder(s, LM, y, CW, 6.75 - y, "C", "Page 3 (optional) – KPI 2.0 per scale: empty wake-ups per week, share of "
                "changes missed, watch list of the scales with the most misses",
                "Insert a screenshot of Power BI page 3 here · data: fact_device_sensitivity.csv")
    src(s, d, "Source: Power BI report on the aggregated pipeline output (fact_device_sensitivity.csv)")
    notes(s, "Optional: shows how the WATCH step would look in operation.")

    # ---- B7 references --------------------------------------------------------------------------
    s = d.slide()
    y = head(s, "REFERENCES", "Sources")
    tb(s, LM, y - 0.1, CW, 4.9, REFS, size=11, color=DEEP, line=1.08, space_after=5)
    footer(s, "", number=d.n)
    notes(s, "All sources cited on the slides. Case data are confidential and used under the module NDA.")

    d.save(DECK)
    return DECK, N, R


if __name__ == "__main__":
    path, _, _ = build()
    print(f"deck written: {path}")
