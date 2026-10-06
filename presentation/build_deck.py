"""
Client deck for the SmartBin case: 16:9 PowerPoint with speaker notes.

    python presentation/build_deck.py

Reads only the aggregated tables in report/data/ (written by report/build_figures.py), draws the slide
charts into presentation/charts/ and writes the deck. Every number on a slide is computed from those
tables. The Power BI dashboards are inserted by hand on the placeholder slides (POWER BI DASHBOARD A/B/C).
The look follows the group's inventory deck (Inter + Source Serif 4, navy / sand / orange).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "report" / "data"
OUT = ROOT / "presentation"
CHARTS = OUT / "charts"
DECK = OUT / "SmartBin – the story for the client.pptx"

# ---------------------------------------------------------------------------------------------
# design tokens
# ---------------------------------------------------------------------------------------------
NAVY, INK2, MUTED = "1E2A38", "4A5360", "8A8F98"
BG, CARD, CARD_LINE, CALLOUT, HEAD = "F7F5F0", "FFFDF9", "E2DDD2", "ECE8DF", "E5E3DE"
RULE, RULE2, DARK_LINE = "B7B5B1", "C5C4BF", "141413"
ORANGE, ORANGE_D, ORANGE_L = "D46A1F", "B5541A", "E8A06A"
BLUE, BLUE_L = "2F5D8C", "8DB3DD"
LIGHT, LIGHT2 = "F4F1EA", "DDE1E7"
GOOD, WARN, CRIT = "4E8F4E", "E0A030", "C0504D"
C = {"20kg": ORANGE, "6kg": BLUE}
SERIF, SANS = "Source Serif 4", "Inter"
SCALES = ["20kg", "6kg"]
LABEL = {"20kg": "20 kg", "6kg": "6 kg"}


# ---------------------------------------------------------------------------------------------
# facts: every number used on the slides
# ---------------------------------------------------------------------------------------------
def rd(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def pc(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f}%"


def n0(x: float) -> str:
    return f"{x:,.0f}"


def mi(text: str) -> str:
    """typographic minus for numbers shown on slides"""
    return text.replace("-", "\u2212")


def sp(x: float, d: int = 0) -> str:
    """signed percentage with a real minus sign, e.g. -0.61 -> '\u221261%'"""
    v = round(x * 100, d)
    return mi(f"{v:+.{d}f}%") if v != 0 else "0%"


def facts() -> dict:
    nums = json.loads((DATA / "numbers.json").read_text())
    F = nums["facts"]
    N: dict = {"rows": nums["rows_raw"], "scales": nums["scales_raw"], "rows_main": nums["rows_main"],
               "period": nums["period"], "installed": F["m7_policy"]["fleet_extrapolation_scales"]}
    N["share_main"] = N["rows_main"] / N["rows"]

    fu = rd("cleaning_funnel")
    N["funnel"] = fu
    N["deleted"] = int(fu.loc[fu.action == "delete row", "rows_removed"].sum())
    N["dupes"] = int(fu.loc[fu.rule == "A2_duplicate", "rows_removed"].iloc[0])
    N["bad_weight"] = int(fu.loc[fu.rule == "A5_weight_invalid", "rows_removed"].iloc[0])
    N["level3_rows"] = int(fu.loc[fu.rule == "B7_sensitivity_anomaly", "rows_removed"].iloc[0])

    lab = rd("label_counts")
    g = lambda i, c: int(lab[(lab.initiatortype == i) & (lab.cls == c)].n.sum())
    N["lab"] = dict(auto_ok=g("AUTOMATIC", "OK"), tm=g("AUTOMATIC", "TOO_MANY"), man_ok=g("MANUAL", "OK"),
                    tf=g("MANUAL", "TOO_FEW"), missing=int(lab[lab.cls == "missing"].n.sum()))
    N["auto_all"] = int(lab[lab.initiatortype == "AUTOMATIC"].n.sum())
    N["share_empty_auto"] = N["lab"]["tm"] / (N["lab"]["tm"] + N["lab"]["auto_ok"])

    N["base"] = rd("baseline_kpis").set_index("scale_type")
    N["kpi"] = rd("kpi_by_sensitivity")
    N["uq"] = rd("usage_quintiles")
    N["usage"] = {st: F["m1_baseline"][st]["tm_fraction_least_vs_most_used"] for st in SCALES}
    N["spearman"] = {st: F["m1_baseline"][st]["activity_spearman_tm_fraction"] for st in SCALES}

    N["root"] = rd("coupling_root_causes").set_index("scale_type")
    cor = rd("coupling_correlogram")
    N["cor"] = cor
    tm = cor[(cor.cls == "TOO_MANY") & (cor.lag_s == 0)].set_index("scale_type")
    N["peak_per_min"], N["base_per_min"] = tm.per_min, tm.baseline
    N["top_sites"] = rd("coupling_top_sites").set_index("scale_type")
    N["coupling_loc_r2"] = {st: F["m6_crosstalk"][st]["location_r2_explained"] for st in SCALES}
    hp = rd("hourly_profile")
    N["hourly"] = hp
    w = hp.groupby(["scale_type", "cls", "weekend"]).n.sum().unstack()
    N["weekend"] = (w[True] / w.sum(axis=1))           # share of weighings at weekends, by type and label

    N["floor_g"] = {st: F["m5_pick_size"][st]["significance_floor_g"] for st in SCALES}
    N["green_from"] = {st: F["m5_pick_size"][st]["green_from_g"] for st in SCALES}
    N["zones"] = rd("zones")
    N["pick"] = rd("pick_profile_summary").set_index("scale_type")

    N["loc"] = {st: F["m4_location"][st] for st in SCALES}
    N["spread"] = rd("parameter_spread")
    wc = pd.DataFrame(F["m3_importance"]["weight_check"])
    N["fill_same_scale"] = wc[wc.subset.str.startswith("all")].set_index("scale_type").median_delta_frac_tm_heavier

    ke = rd("knob_effects")
    ke["empty_change"] = np.exp(ke.beta_tm) - 1
    ke["miss_pp"] = ke.did_miss_rate * 100
    N["knob"] = ke
    N["switchers"] = F["m2_sensitivity"]["n_switchers_total"]

    lp = rd("loops_summary")
    N["loop_scales"] = int(lp.loop_scales_any_level.iloc[0])
    N["loop_auto"] = int(lp[lp.loop].n_auto.sum())
    N["loop_share_auto"] = N["loop_auto"] / N["auto_all"]
    N["loop_share_scales"] = N["loop_scales"] / N["scales"]
    L = lp.set_index(["group", "loop"])
    N["loop_l3"], N["loop_other"] = int(L.loc[("level 3", True), "scales"]), int(L.loc[("other levels", True), "scales"])
    N["loop_l3_auto"] = int(L.loc[("level 3", True), "n_auto"])
    N["l3_judged"] = int(L.loc[("level 3", True), "scales"] + L.loc[("level 3", False), "scales"])
    N["loop_rate_l3"] = float(L.loc[("level 3", True), "median_auto_per_day"])
    N["loop_rate_other"] = float(L.loc[("other levels", True), "median_auto_per_day"])
    N["loop_retrigger"] = float(L.loc[("level 3", True), "median_share_retrigger"])
    sig = rd("sens3_signature")
    a = sig[sig.grp == "anomaly devices on anomaly level"].set_index("scale_type")
    N["l3_scales"] = int(a.devices.sum())
    N["l3_rate"] = a.auto_per_device_day
    # battery model of the case: a weighing ~2 s at ~3,000x sleep power (config battery.*)
    e = 2.0 * 3000 / 3600                                      # hours of sleep energy per weighing
    typ = float(N["base"].loc["20kg", "meas_per_day_median"])
    N["loop_battery_days"] = 5 * 365 * (24 + typ * e) / (24 + N["loop_rate_l3"] * e)

    pol = rd("policies").set_index(["scale_type", "policy"])
    N["pol"] = pol
    P = {k: v for k, v in [("P0", "P0 status quo (8)"), ("P1", "P1 one setting"), ("P2", "P2 segment table"), ("P3", "P3 adaptive rule"),
                           ("P4", "P4 oracle"), ("P5", "P5 one setting (KPI 2.0)"), ("P6", "P6 adaptive rule (KPI 2.0)"),
                           ("P7", "P7 start 7 + rule (KPI 2.0)"), ("P8", "P8 fraction rule (start 6)")]}
    N["P"] = P
    N["achievable"] = {d["scale_type"]: d for d in F["m7_policy"]["achievable_target"]}
    N["one_setting"] = {st: F["m7_policy"][st]["one_setting"] for st in SCALES}
    N["one_setting_kpi2"] = {st: F["m7_policy"][st]["one_setting_kpi2"] for st in SCALES}
    sv = {st: pol.loc[(st, P["P7"]), "meas_saved_per_scale_week_without_loops"] for st in SCALES}
    n_dev = {st: pol.loc[(st, P["P0"]), "devices"] for st in SCALES}
    N["saved_per_scale_week"] = sum(sv[s] * n_dev[s] for s in SCALES) / sum(n_dev.values())
    N["fleet_saved_week"] = N["saved_per_scale_week"] * N["installed"]
    N["final"] = rd("final_levels")
    N["backtest"] = rd("policy_backtest")
    N["watch"] = rd("watch_numbers").set_index("scale_type")
    N["idle_share_manual"] = nums["share_manual_after_24h_idle"]
    N["idle_share_tf"] = nums["share_too_few_after_24h_idle"]
    N["idle_gap"] = nums["idle_check_gap_hours_q05_q25_q50_q75_q95"]
    N["burst"] = rd("burst_merge")
    N["quiet"] = rd("quiet_hours_test")
    N["minwait"] = rd("minwait_check")
    N["levers"] = nums.get("levers", {})
    rc = N["root"]
    N["coupled_pooled"] = (rc.neighbour_induced * rc.n_too_many).sum() / rc.n_too_many.sum()
    N["loop_other_auto"] = N["loop_auto"] - N["loop_l3_auto"]
    hp_ = hp.groupby(["scale_type", "cls", "hour"]).n.sum().unstack(1)
    N["hourly_r"] = {st: float(np.corrcoef(hp_.loc[st, "OK"], hp_.loc[st, "TOO_MANY"])[0, 1]) for st in SCALES}
    N["shapley"] = rd("shapley")
    N["check_hours"] = rd("manual_check_hours")
    N["event"] = rd("event_study")
    return N


# ---------------------------------------------------------------------------------------------
# charts (drawn at their size on the slide, so 1 pt in the chart = 1 pt on the slide)
# ---------------------------------------------------------------------------------------------
def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": SANS, "font.size": 12, "axes.labelsize": 12, "axes.labelcolor": "#" + INK2,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "xtick.color": "#" + INK2, "ytick.color": "#" + INK2,
        "axes.edgecolor": "#" + RULE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#" + HEAD, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 11.5,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2.4, "lines.solid_capstyle": "round",
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": "#" + NAVY,
        "text.color": "#" + NAVY, "axes.unicode_minus": True,
    })
    return plt


def hexc(c: str) -> str:
    return "#" + c


def save(fig, name: str):
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / f"{name}.png", dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return CHARTS / f"{name}.png"


def _legend_patches(colors, labels, fig_or_ax, **kw):
    from matplotlib.patches import Patch
    return fig_or_ax.legend([Patch(color=hexc(c)) for c in colors], labels, **kw)


def ch_cleaning(plt, N, w, h):
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
    fig.subplots_adjust(left=0.37, right=0.95, top=0.98, bottom=0.25)
    col = [CRIT if a == "delete row" else (ORANGE if r.startswith("B7") else RULE) for a, r in zip(x.action, x.rule)]
    ax.barh(x.name, x.rows_removed, color=[hexc(c) for c in col], height=0.62)
    for yi, v in enumerate(x.rows_removed):
        ax.text(v * 1.3, yi, n0(v), va="center", fontsize=11, color=hexc(INK2))
    ax.set_xscale("log")
    ax.set_xlim(2, 3e7)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=hexc(HEAD))
    ax.tick_params(axis="y", labelsize=11.5, colors=hexc(NAVY))
    ax.set_xticks([10, 1e3, 1e5, 1e7], ["10", "1,000", "100,000", "10 M"])
    ax.set_xlabel("rows (log scale)", fontsize=11)
    _legend_patches([CRIT, RULE, ORANGE], ["deleted (stage A)", "flagged, kept (stage B)", "flagged: level 3"], fig,
                    loc="lower center", ncol=3, fontsize=10.5, bbox_to_anchor=(0.6, 0.0), handlelength=1.0)
    return save(fig, "c01_cleaning")


def ch_one_sided(plt, N, w, h):
    b = N["base"]
    rows = [("Meets the TOO_MANY target\n(fraction ≤ 0.5)", "share_meets_tm"),
            ("Meets the TOO_FEW target\n(≤ 2 missed changes a week)", "share_meets_tf")]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.37, right=0.97, top=0.86, bottom=0.1)
    ys = np.arange(len(rows))[::-1] * 1.0
    for i, st in enumerate(SCALES):
        vals = [b.loc[st, k] * 100 for _, k in rows]
        yy = ys + (0.19 if i == 0 else -0.19)
        ax.barh(yy, vals, height=0.34, color=hexc(C[st]), label=f"{LABEL[st]} scales")
        for y, v in zip(yy, vals):
            ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=12.5, fontweight="bold", color=hexc(C[st]))
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=12, colors=hexc(NAVY))
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=hexc(HEAD))
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.16), ncol=2, fontsize=11.5)
    return save(fig, "c02_one_sided")


def ch_usage(plt, N, w, h):
    uq = N["uq"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.87, bottom=0.2)
    xs = np.arange(5)
    for st in SCALES:
        y = uq[uq.scale_type == st].frac_too_many_median.values
        ax.plot(xs, y, color=hexc(C[st]), marker="o", markersize=7, label=f"{LABEL[st]} scales")
        ax.text(xs[0] - 0.14, y[0], f"{y[0]:.2f}", ha="right", va="center", fontsize=11.5, color=hexc(C[st]),
                fontweight="bold")
        ax.text(xs[-1] + 0.14, y[-1] + (0.02 if st == "20kg" else -0.02), f"{y[-1]:.2f}", ha="left",
                va="center", fontsize=11.5, color=hexc(C[st]), fontweight="bold")
    ax.axhline(0.5, color=hexc(GOOD), lw=1.2, ls=(0, (4, 3)))
    ax.text(-0.55, 0.47, "Bossard target ≤ 0.5", ha="left", va="top", fontsize=10.5, color=hexc(GOOD))
    ax.set_xticks(xs, ["least\nused", "2", "3", "4", "most\nused"])
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(0.3, 1.08)
    ax.set_ylabel("median TOO_MANY fraction")
    ax.set_xlabel("fifths of the bins by real picks per week (level 8)", fontsize=11)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=11)
    return save(fig, "c03_usage")


def ch_coupling(plt, N, w, h):
    cor = N["cor"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.2)
    for st, col in [("20kg", ORANGE_L), ("6kg", BLUE_L)]:
        x = cor[(cor.scale_type == st) & (cor.cls == "TOO_MANY") & (cor.lag_s.abs() <= 116)].sort_values("lag_s")
        ax.plot(x.lag_s, x.per_min, color=hexc(col), lw=2.2, label=f"{LABEL[st]} scales")
        ax.axhline(x.baseline.iloc[0], color=hexc(col), lw=1, ls=(0, (3, 3)), alpha=0.8)
    ax.axvspan(-14, 14, color="#ffffff", alpha=0.07, lw=0)
    ax.text(17, 9.4, "±14 s", color=hexc(LIGHT2), fontsize=11)
    ax.text(114, 1.7, "chance level", color=hexc(LIGHT2), fontsize=10.5, ha="right")
    for sp_ in ["left", "bottom"]:
        ax.spines[sp_].set_color("#5d6878")
    ax.tick_params(colors=hexc(LIGHT2))
    ax.grid(color="#334255")
    ax.set_xlim(-116, 116)
    ax.set_ylim(0, 10.5)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xlabel("seconds from an empty wake-up", color=hexc(LIGHT2), fontsize=11)
    ax.set_ylabel("real picks per minute at\nother scales of the site", color=hexc(LIGHT2), fontsize=11)
    ax.legend(loc="upper left", fontsize=10.5, labelcolor=hexc(LIGHT))
    return save(fig, "c04_coupling")


def ch_blind_zone(plt, N, w, h):
    z = N["zones"]
    order = z.drop_duplicates("pick_bin").sort_values("bin_idx").pick_bin.tolist()
    lev = [(8, NAVY, 2.6), (6, "6E7F95", 2.0), (4, "A9B4C2", 1.8), (1, "CBD3DD", 1.8)]
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.25, wspace=0.08)
    for ax, st in zip(axes, SCALES):
        x = z[z.scale_type == st]
        floor_idx = x[~x.measurable].bin_idx.max() if (~x.measurable).any() else -1
        ax.axvspan(-0.5, floor_idx + 0.5, color=hexc(HEAD), lw=0, zorder=0)
        ax.text((floor_idx - 0.5) / 2 + 0.5 * (floor_idx < 2), 6, f"blind\nzone\n< {N['floor_g'][st]:.0f} g",
                ha="center", fontsize=10.5, color=hexc(INK2))
        ax.axhline(95, color=hexc(GOOD), lw=1.1, ls=(0, (4, 3)))
        ax.text(-0.35, 89, "95%", fontsize=10.5, color=hexc(GOOD))
        for lv, col, lw in lev:
            y = x[(x.sensitivity == lv) & x.measurable].sort_values("bin_idx")
            ax.plot(y.bin_idx, y.detection * 100, color=hexc(col), lw=lw, marker="o", markersize=4.5,
                    label=f"level {lv}")
        ax.set_title(f"{LABEL[st]} scales", color=hexc(C[st]))
        ax.set_xticks(range(len(order)), [o.replace(">500", "500+") for o in order], rotation=45, ha="right",
                      fontsize=10.5)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(0, 104)
        ax.set_xlabel("pick size (g)", fontsize=11)
    axes[0].set_ylabel("picks caught at once (%)")
    axes[1].legend(loc="lower right", fontsize=10.5, ncol=2)
    return save(fig, "c05_blind_zone")


def ch_parameters(plt, N, w, h):
    s = N["spread"]
    order = ["Site (location)", "Sensitivity level", "Fill level (weight bin)", "Box type", "Item weight (quartile)"]
    names = ["Site", "Sensitivity level", "Fill level of the box", "Box type", "Item weight"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.14)
    ys = np.arange(len(order))[::-1].astype(float)
    for i, st in enumerate(SCALES):
        v = s[s.scale_type == st].set_index("parameter").reindex(order).spread.values
        yy = ys + (0.18 if i == 0 else -0.18)
        ax.barh(yy, v, height=0.32, color=hexc(C[st]), label=f"{LABEL[st]} scales")
        for y, val in zip(yy, v):
            ax.text(val + 0.006, y, f"{val:.2f}", va="center", fontsize=11, color=hexc(C[st]), fontweight="bold")
    ax.set_yticks(ys, names)
    ax.tick_params(axis="y", labelsize=12, colors=hexc(NAVY))
    ax.set_xlim(0, 0.47)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=hexc(HEAD))
    ax.set_xlabel("difference in TOO_MANY fraction: best vs. worst group", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.15), ncol=2, fontsize=11)
    return save(fig, "c06_parameters")


def ch_knob(plt, N, w, h):
    k = N["knob"]
    lv = [7, 6, 5, 4, 2, 1]
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.78, bottom=0.12, wspace=0.28)
    xs = np.arange(len(lv))
    for i, st in enumerate(SCALES):
        x = k[k.scale_type == st].set_index("sensitivity").reindex([float(l) for l in lv])
        off = -0.19 if i == 0 else 0.19
        axes[0].bar(xs + off, x.empty_change * 100, width=0.36, color=hexc(C[st]), label=f"{LABEL[st]} scales")
        axes[1].bar(xs + off, x.miss_pp, width=0.36, color=hexc(C[st]))
        for xx, v in zip(xs + off, x.miss_pp):
            axes[1].text(xx, v + 1.2, f"+{v:.0f}", ha="center", fontsize=10, color=hexc(C[st]))
    axes[0].set_title("Empty wake-ups vs. level 8")
    axes[0].set_ylim(-100, 0)
    axes[0].set_yticks([0, -25, -50, -75, -100], ["0%", "−25%", "−50%", "−75%", "−100%"])
    axes[1].set_title("Extra changes missed (points)")
    axes[1].set_ylim(0, 72)
    for ax in axes:
        ax.set_xticks(xs, [f"8→{l}" for l in lv])
        ax.tick_params(axis="x", labelsize=11.5)
    axes[1].axvspan(-0.5, 1.5, color=hexc(GOOD), alpha=0.09, lw=0)
    axes[1].text(0.5, 64, "cheap", ha="center", fontsize=10.5, color=hexc(GOOD))
    _legend_patches([C["20kg"], C["6kg"]], ["20 kg scales", "6 kg scales"], fig, loc="upper left", ncol=2,
                    fontsize=11, bbox_to_anchor=(0.06, 1.0), handlelength=1.0)
    return save(fig, "c07_knob")


def ch_loops(plt, N, w, h):
    sc = N["loop_share_scales"] * 100
    l3 = N["loop_l3_auto"] / N["auto_all"] * 100
    lo = (N["loop_auto"] - N["loop_l3_auto"]) / N["auto_all"] * 100
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.36)
    rows = [("scales", [(sc, CRIT, None), (100 - sc, RULE, f"{100 - sc:.1f}%")]),
            ("automatic\nweighings", [(l3, CRIT, f"{l3:.1f}%"), (lo, "E3A19F", f"{lo:.1f}%"),
                                      (100 - l3 - lo, RULE, f"{100 - l3 - lo:.1f}%")])]
    for y, (_, parts) in zip([1, 0], rows):
        left = 0
        for v, c, lab in parts:
            ax.barh(y, v, left=left, color=hexc(c), height=0.58, edgecolor=hexc(BG), linewidth=1.5)
            if lab:
                ax.text(left + v / 2, y, lab, ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if c != RULE else hexc(NAVY))
            left += v
    ax.text(sc + 1.2, 1, f"{sc:.1f}% ({N['loop_scales']} scales)", va="center", fontsize=12, color=hexc(CRIT),
            fontweight="bold")
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=12, colors=hexc(NAVY))
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=hexc(HEAD))
    _legend_patches([CRIT, "E3A19F", RULE], ["trigger loop on level 3", "trigger loop on another level", "no loop"],
                    fig, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.58, 0.0))
    return save(fig, "c08_loops")


def ch_policies(plt, N, w, h):
    pol, P = N["pol"], N["P"]
    lab = {"P0": "today", "P1": "one level for all\n(Bossard KPIs)", "P5": "one level for all (KPI 2.0)",
           "P8": "steer by the fraction", "P7": "recommended rule"}
    # label offsets in points: (dx, dy, ha)
    place = {"20kg": {"P0": (-9, 0, "right"), "P1": (9, 0, "left"), "P5": (9, -1, "left"),
                      "P8": (9, 2, "left"), "P7": (9, 12, "left")},
             "6kg": {"P0": (-9, 0, "right"), "P1": (0, -19, "center"), "P5": (9, 0, "left"),
                     "P8": (9, 7, "left"), "P7": (10, 3, "left")}}
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.17, wspace=0.07)
    for ax, st in zip(axes, SCALES):
        ax.axhspan(0, 10, color=hexc(GOOD), alpha=0.1, lw=0)
        ax.text(-98, 1.6, "≤ 10% missed", fontsize=10, color=hexc(GOOD))
        for key in ["P0", "P1", "P5", "P8", "P7"]:
            r = pol.loc[(st, P[key])]
            x, y = r.meas_change_pct_without_loops, r.median_miss_rate * 100
            rec = key == "P7"
            ax.scatter(x, y, s=160 if rec else 62, color=hexc(C[st]) if rec else hexc(BG), edgecolor=hexc(C[st]),
                       linewidth=2, zorder=4)
            dx, dy, ha = place[st][key]
            ax.annotate(lab[key], (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center",
                        fontsize=10.5, color=hexc(NAVY) if rec else hexc(INK2), fontweight="bold" if rec else "normal",
                        linespacing=1.0)
        ax.set_title(f"{LABEL[st]} scales", color=hexc(C[st]))
        ax.set_xlim(-100, 10)
        ax.set_ylim(0, 56)
        ax.set_xticks([-100, -75, -50, -25, 0], ["−100%", "−75%", "−50%", "−25%", "0%"])
        ax.grid(axis="x", color=hexc(HEAD))
        ax.set_xlabel("weighings vs. today", fontsize=11)
    axes[0].set_ylabel("changes missed, median scale (%)", fontsize=11)
    return save(fig, "c09_policies")


def ch_final_levels(plt, N, w, h):
    f = N["final"]
    pol = N["P"]["P7"]
    lv = [1, 2, 4, 5, 6, 7, 8]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.9, bottom=0.2)
    xs = np.arange(len(lv))
    for i, st in enumerate(SCALES):
        v = (f[(f.policy == pol) & (f.scale_type == st)].set_index("sensitivity").share
             .reindex([float(l) for l in lv]).fillna(0).values * 100)
        off = -0.19 if i == 0 else 0.19
        ax.bar(xs + off, v, width=0.36, color=hexc(C[st]), label=f"{LABEL[st]} scales")
        for xx, vv in zip(xs + off, v):
            if vv >= 3:
                ax.text(xx, vv + 1, f"{vv:.0f}", ha="center", fontsize=10.5, color=hexc(C[st]))
    ax.set_xticks(xs, [str(l) for l in lv])
    ax.set_xlabel("sensitivity level after 16 weeks with the rule (start: 7)", fontsize=11)
    ax.set_ylabel("share of scales (%)")
    ax.set_ylim(0, 55)
    ax.legend(loc="upper left", fontsize=11)
    return save(fig, "c10_final_levels")


def ch_hourly(plt, N, w, h):
    hp = N["hourly"]
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.3, wspace=0.08)
    for ax, st in zip(axes, SCALES):
        x = hp[hp.scale_type == st].groupby(["cls", "hour"]).n.sum().unstack(0)
        x = x / x.sum() * 100
        ax.plot(x.index, x["OK"], color=hexc(NAVY), lw=2.2)
        ax.plot(x.index, x["TOO_MANY"], color=hexc(C[st]), lw=2.2, ls=(0, (4, 2)))
        ax.set_title(f"{LABEL[st]} scales", color=hexc(C[st]))
        ax.set_xticks([0, 6, 12, 18], ["0 h", "6 h", "12 h", "18 h"])
        ax.set_xlim(0, 23)
    axes[0].set_ylabel("share of the day's weighings (%)", fontsize=11)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=hexc(NAVY), lw=2.2), Line2D([], [], color=hexc(INK2), lw=2.2, ls=(0, (4, 2)))],
               ["real picks caught", "empty wake-ups (dashed, in the colour of the scale type)"], loc="lower center",
               ncol=2, fontsize=10.5, bbox_to_anchor=(0.54, 0.0))
    fig.text(0.54, 0.12, "hour of the day (time stamps as delivered)", ha="center", fontsize=10.5, color=hexc(INK2))
    return save(fig, "c11_hourly")


def ch_event_study(plt, N, w, h):
    e = N["event"].copy()
    e["tm_day"] = e.n_tm / e.active_days
    bands = [("to 1-2 (low)", "CBD3DD", "new level 1–2"), ("to 4-5 (mid)", "7F90A8", "new level 4–5"),
             ("to 6-7 (high)", NAVY, "new level 6–7")]
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.32, wspace=0.08)
    for ax, st in zip(axes, SCALES):
        for b, col, _ in bands:
            x = e[(e.scale_type == st) & (e.band == b)].sort_values("rel_week")
            ax.plot(x.rel_week, x.tm_day, color=hexc(col), lw=2.2, marker="o", markersize=3.5)
        ax.axvline(-0.5, color=hexc(INK2), lw=1, ls=(0, (3, 3)))
        ax.set_title(f"{LABEL[st]} scales", color=hexc(C[st]))
        ax.set_xticks([-4, -2, 0, 2, 4, 6, 8])
        ax.set_ylim(0, 16)
    axes[0].set_ylabel("empty wake-ups per scale-day", fontsize=11)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=hexc(c), lw=2.2) for _, c, _ in bands], [l for *_, l in bands],
               loc="lower center", ncol=3, fontsize=10.5, bbox_to_anchor=(0.54, 0.0))
    fig.text(0.54, 0.13, "weeks relative to the switch away from level 8", ha="center", fontsize=10.5,
             color=hexc(INK2))
    return save(fig, "c12_event_study")


def ch_check_timing(plt, N, w, h):
    t = N["check_hours"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.2)
    for idle, col, lab in [(True, NAVY, "after ≥ 24 h without any weighing (idle check)"),
                           (False, RULE, "within 24 h of the previous weighing (a person)")]:
        x = t[t.after_24h_idle == idle].groupby("hour").n.sum().reindex(range(24), fill_value=0)
        ax.plot(x.index, x / x.sum() * 100, color=hexc(col), lw=2.4, marker="o", markersize=3.5, label=lab)
    ax.set_xticks([0, 4, 8, 12, 16, 20], ["0 h", "4 h", "8 h", "12 h", "16 h", "20 h"])
    ax.set_xlim(0, 23)
    ax.set_ylabel("share of manual weighings (%)", fontsize=11)
    ax.set_xlabel("hour of the day (time stamps as delivered)", fontsize=11)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=10.5)
    return save(fig, "c13_check_timing")


def ch_shapley(plt, N, w, h):
    s = N["shapley"]
    s = s[(s.kpi == "frac_too_many") & s.scope.isin(SCALES)]
    order = ["Location (factory)", "Weight in box (fill level)", "Item weight (single part)", "Sensitivity setting",
             "Box type"]
    names = ["Site", "Fill level of the box", "Item weight", "Sensitivity level", "Box type"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.25, right=0.97, top=0.88, bottom=0.16)
    ys = np.arange(len(order))[::-1].astype(float)
    for i, st in enumerate(SCALES):
        v = s[s.scope == st].set_index("parameter").reindex(order).shapley_r2.fillna(0).values * 100
        yy = ys + (0.18 if i == 0 else -0.18)
        ax.barh(yy, v, height=0.32, color=hexc(C[st]), label=f"{LABEL[st]} scales")
        for y, val in zip(yy, v):
            ax.text(val + 0.3, y, f"{val:.1f}", va="center", fontsize=10.5, color=hexc(C[st]), fontweight="bold")
    ax.set_yticks(ys, names)
    ax.tick_params(axis="y", labelsize=12, colors=hexc(NAVY))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=hexc(HEAD))
    ax.set_xlim(0, 30)
    ax.set_xlabel("share of the variance between periods explained (Shapley R², %)", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.15), ncol=2, fontsize=11)
    return save(fig, "c14_shapley")


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

SW, SH = 13.333, 7.5
LM, CW = 0.89, 11.56                     # left margin and content width (inches), as in the group's deck
_FONTS: dict = {}


def _font(font, bold, size):
    """PIL font used to measure text, so that boxes can be stacked without overlaps."""
    key = (font, bold, size)
    if key not in _FONTS:
        try:
            from matplotlib import font_manager as fm
            from PIL import ImageFont
            path = fm.findfont(fm.FontProperties(family=font, weight="bold" if bold else "normal"),
                               fallback_to_default=False)
            _FONTS[key] = ImageFont.truetype(path, size=int(round(size * 10)))
        except Exception:                                           # font not installed: rough estimate
            _FONTS[key] = None
    return _FONTS[key]


def n_lines(text, width_in, size, font=SANS, bold=False):
    f = _font(font, bold, size)
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


def text_h(paras, width_in, size, line=1.12, font=SANS, bold=False, space_after=0, bullet_indent=0.2):
    if isinstance(paras, str):
        paras = paras.split("\n")
    h = 0.0
    for p in paras:
        w = width_in - (bullet_indent if p.startswith("• ") else 0)
        h += n_lines(p[2:] if p.startswith("• ") else p, w, size, font, bold) * size * 1.21 * line / 72
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


def tb(s, x, y, w, h, paras, size=14, color=NAVY, font=SANS, bold=False, italic=False, align="l", anchor="t",
       line=1.12, space_after=0, spc=None, bullet_indent=0.2, bold_color=None):
    """Text box. paras: str (\\n = new paragraph) or list of str. '• ' starts a bullet. **x** = bold.
    Returns the measured bottom (inches) of the text."""
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
            etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
            etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
        for t, b in _runs(ptxt) or [("", False)]:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name, f.size, f.bold, f.italic = font, Pt(size), (bold or b), italic
            f.color.rgb = rgb(bold_color if (b and bold_color) else color)
            if spc:
                r._r.get_or_add_rPr().set("spc", str(spc))
    return y + text_h(paras, w, size, line, font, bold, space_after, bullet_indent)


def rect(s, x, y, w, h, fill=None, line=None, lw=0.5, shape=MSO_SHAPE.RECTANGLE, radius=None, dash=False):
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


def panel(s, x, y, w, h, fill=CARD, line=CARD_LINE):
    """rounded card background with a constant corner radius of about 0.08 in"""
    return rect(s, x, y, w, h, fill=fill, line=line, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=min(0.5, 0.08 / min(w, h)))


def card(s, x, y, w, h, title=None, body=None, title_size=16, body_size=13, pad=0.22, title_color=NAVY,
         body_color=INK2, gap=0.08, fill=CARD, line=CARD_LINE, body_space=3, title_h=None):
    panel(s, x, y, w, h, fill, line)
    cy = y + pad
    if title:
        ty = tb(s, x + pad, cy, w - 2 * pad, 0.3, title, size=title_size, bold=True, color=title_color, line=1.0)
        cy = (cy + title_h if title_h else ty) + gap
    if body:
        cy = tb(s, x + pad, cy, w - 2 * pad, y + h - cy, body, size=body_size, color=body_color, line=1.15,
                space_after=body_space)
    return cy


def callout(s, x, y, w, h, text, size=15, fill=CALLOUT, color=NAVY):
    panel(s, x, y, w, h, fill=fill, line=None)
    tb(s, x + 0.25, y, w - 0.45, h, text, size=size, color=color, anchor="m", line=1.12)


def bignum(s, x, y, w, value, caption, color=ORANGE, size=54, cap_size=14, cap_color=NAVY):
    """big number + caption; returns the bottom of the caption"""
    nh = size / 72 * 1.08
    tb(s, x, y, w, nh, value, size=size, font=SERIF, bold=True, color=color, line=0.95)
    return tb(s, x, y + nh + 0.04, w, 0.3, caption, size=cap_size, color=cap_color, line=1.12)


def head(s, eyebrow, title, dark=False, size=32, lines=None):
    """Eyebrow + headline as in the group's deck. Returns the y (inches) where the content can start."""
    tb(s, LM, 0.78, CW + 0.3, 0.26, eyebrow, size=12, bold=True, color=ORANGE_L if dark else ORANGE_D, spc=150)
    lines = lines or n_lines(title, CW + 0.3, size, SERIF, True)
    h = 0.55 * lines * size / 32
    tb(s, LM, 1.08, CW + 0.3, h, title, size=size, font=SERIF, bold=True, color=LIGHT if dark else NAVY, line=1.0)
    return 1.08 + h + 0.3


def source(s, text, dark=False):
    tb(s, LM, 6.98, 11.2, 0.3, text, size=9.5, color="8C97A6" if dark else MUTED)


def pic(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def arrow(s, x1, y1, x2, y2, color=INK2, lw=1.25):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(lw)
    te = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"))
    te.set("type", "triangle")
    te.set("w", "med")
    te.set("len", "med")
    return c


def line(s, x1, y1, x2, y2, color=DARK_LINE, lw=1.0):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(lw)
    return c


def table(s, x, y, w, col_w, rows, size=12.5, row_h=0.36, align=None, bold_rows=(), header=True):
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
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = {"l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT, "c": PP_ALIGN.CENTER}[(align or "l" * ncol)[j]]
            for t, b in _runs(str(rows[i][j])) or [("", False)]:
                r = p.add_run()
                r.text = t
                r.font.name, r.font.size = SANS, Pt(size)
                r.font.bold = b or (header and i == 0) or (i in bold_rows)
                r.font.color.rgb = rgb(NAVY)
            tcPr = cell._tc.get_or_add_tcPr()
            for side, show, col in [("a:lnL", False, None), ("a:lnR", j < ncol - 1, RULE2),
                                    ("a:lnT", False, None), ("a:lnB", True, RULE)]:
                ln = etree.SubElement(tcPr, qn(side))
                if show:
                    ln.set("w", "12700")
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    etree.SubElement(sf, qn("a:srgbClr")).set("val", col)
                else:
                    ln.set("w", "0")
                    etree.SubElement(ln, qn("a:noFill"))
            if header and i == 0:
                sf = etree.SubElement(tcPr, qn("a:solidFill"))
                etree.SubElement(sf, qn("a:srgbClr")).set("val", HEAD)
            else:
                etree.SubElement(tcPr, qn("a:noFill"))
    line(s, x - 0.11, y - 0.11, x + w, y - 0.11)            # the group's table accent: a line on top ...
    line(s, x - 0.1, y - 0.11, x - 0.1, y + row_h * nrow)   # ... and on the left
    return gt


def placeholder(s, x, y, w, h, letter, what, how):
    rect(s, x, y, w, h, fill=CARD, line=ORANGE_D, lw=1.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=min(0.5, 0.08 / min(w, h)), dash=True)
    tb(s, x, y + h * 0.3, w, 0.5, f"POWER BI DASHBOARD {letter}", size=24, bold=True, color=ORANGE_D, align="c",
       spc=100)
    tb(s, x + 0.5, y + h * 0.3 + 0.62, w - 1.0, 0.9, what, size=14, color=NAVY, align="c", line=1.15)
    tb(s, x + 0.5, y + h - 0.75, w - 1.0, 0.5, how, size=11, color=MUTED, align="c", line=1.1)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = " ".join(t.strip() for t in text.strip().split("\n"))


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(SW), Inches(SH)
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, dark=False, number=True):
        s = self.prs.slides.add_slide(self.blank)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(NAVY if dark else BG)
        self.n += 1
        if number:
            tb(s, 12.35, 6.98, 0.5, 0.3, str(self.n), size=9.5, color="8C97A6" if dark else MUTED, align="r")
        return s

    def save(self, path):
        cp = self.prs.core_properties
        cp.author = cp.last_modified_by = "Erenalp Çakıroğlu"
        cp.title = "SmartBin Cloud – tuning the vibration sensor"
        cp.subject = "HSLU · Data-driven Supply Chain Management and Logistics · Smart Logistics Service Optimisation"
        cp.keywords = "SmartBin; IoT; sensitivity; Bossard; case study"
        cp.comments = ""
        import datetime as _dt
        cp.created = cp.modified = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0, tzinfo=None)
        cp.revision = 1
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(path))


# ---------------------------------------------------------------------------------------------
# the story
# ---------------------------------------------------------------------------------------------
REFS = [
    "Anastasi, G., Conti, M., Di Francesco, M., & Passarella, A. (2009). Energy conservation in wireless sensor "
    "networks: A survey. Ad Hoc Networks, 7(3), 537–568. https://doi.org/10.1016/j.adhoc.2008.06.003",
    "Angrist, J. D., & Pischke, J.-S. (2009). Mostly harmless econometrics: An empiricist's companion. "
    "Princeton University Press.",
    "Bossard AG. (n.d.). SmartBin Cloud. Retrieved 6 October 2026, from "
    "https://www.bossard.com/ch-en/smart-factory-logistics/systems/smartbin-cloud/",
    "Bossard AG & D ONE Solutions AG. (2026). Smart Factory Logistics – SmartBin Cloud [Lecture slides, case "
    "description and case data]. HSLU, module Data-driven Supply Chain Management and Logistics. Confidential (NDA).",
    "Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). "
    "CRISP-DM 1.0: Step-by-step data mining guide. SPSS Inc.",
    "Donges, J. F., Schleussner, C.-F., Siegmund, J. F., & Donner, R. V. (2016). Event coincidence analysis for "
    "quantifying statistical interrelationships between event time series. The European Physical Journal Special "
    "Topics, 225(3), 471–487. https://doi.org/10.1140/epjst/e2015-50233-y",
    "Green, D. M., & Swets, J. A. (1966). Signal detection theory and psychophysics. Wiley.",
    "Grömping, U. (2006). Relative importance for linear regression in R: The package relaimpo. Journal of "
    "Statistical Software, 17(1), 1–27. https://doi.org/10.18637/jss.v017.i01",
    "Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. "
    "Psychological Bulletin, 86(2), 420–428. https://doi.org/10.1037/0033-2909.86.2.420",
    "Strathern, M. (1997). 'Improving ratings': Audit in the British university system. European Review, 5(3), "
    "305–321.",
]
OWN = "Source: own analysis of the Bossard SmartBin data, 1 Mar – 20 Jun 2022"


def build():
    N = facts()
    plt = mpl()
    b, k = N["base"], N["knob"].set_index(["scale_type", "sensitivity"])
    pol, P = N["pol"], N["P"]
    kp = lambda st, lv, col: k.loc[(st, float(lv)), col]
    pv = lambda st, key, col: pol.loc[(st, P[key]), col]
    kpi = N["kpi"].set_index(["scale_type", "sensitivity"])
    w_ = N["watch"]
    bm = N["burst"].set_index(["window_min", "scale_type", "sequence"])
    qh = N["quiet"].set_index(["threshold", "scale_type"])
    mw = N["minwait"].set_index(["scale_type", "minwait", "delay"])
    root, loc = N["root"], N["loc"]
    spr = N["spread"].set_index(["scale_type", "parameter"]).spread
    fl = N["final"].set_index(["policy", "scale_type", "sensitivity"]).share

    c = dict(cleaning=ch_cleaning(plt, N, 6.8, 3.85), one_sided=ch_one_sided(plt, N, 7.2, 3.3),
             usage=ch_usage(plt, N, 6.7, 3.85), coupling=ch_coupling(plt, N, 6.0, 3.05),
             blind=ch_blind_zone(plt, N, 7.7, 3.75), params=ch_parameters(plt, N, 6.9, 3.75),
             knob=ch_knob(plt, N, 8.1, 3.3), loops=ch_loops(plt, N, 8.2, 2.05),
             policies=ch_policies(plt, N, 7.5, 3.75), final=ch_final_levels(plt, N, 6.4, 3.35),
             hourly=ch_hourly(plt, N, 7.6, 3.6), event=ch_event_study(plt, N, 7.6, 3.75),
             check=ch_check_timing(plt, N, 9.0, 3.0), shapley=ch_shapley(plt, N, 7.4, 3.7))

    d = Deck()
    M = lambda v: f"{v / 1e6:.1f} million"
    real = lambda st: kpi.loc[(st, 8), "real_picks_per_week"]
    e20, e6 = kp("20kg", 7, "empty_change"), kp("6kg", 7, "empty_change")
    m20, m6 = kp("20kg", 7, "miss_pp"), kp("6kg", 7, "miss_pp")
    sv20, sv6 = pv("20kg", "P7", "meas_change_pct_without_loops"), pv("6kg", "P7", "meas_change_pct_without_loops")
    mr20, mr6 = pv("20kg", "P7", "median_miss_rate"), pv("6kg", "P7", "median_miss_rate")
    m0_20, m0_6 = pv("20kg", "P0", "median_miss_rate"), pv("6kg", "P0", "median_miss_rate")
    fleet = N["fleet_saved_week"]
    loops_pct = pc(N["loop_share_auto"])
    weeks = N["loop_battery_days"] / 7
    e_ = 2.0 * 3000 / 3600
    typ_ = float(b.loc["20kg", "meas_per_day_median"])
    weeks_other = 5 * 365 * (24 + typ_ * e_) / (24 + N["loop_rate_other"] * e_) / 7
    wk_tm = max(N["weekend"].loc[("20kg", "TOO_MANY")], N["weekend"].loc[("6kg", "TOO_MANY")])
    aok1_20 = np.exp(kp("20kg", 1, "beta_aok")) - 1

    # 1 ---------------------------------------------------------------------------------------- title
    s = d.slide(dark=True, number=False)
    tb(s, LM, 2.0, 11.9, 0.3, "CASE STUDY · SMART LOGISTICS SERVICE OPTIMISATION · HSLU", size=12, bold=True,
       color=ORANGE_L, spc=150)
    tb(s, LM, 2.45, 11.2, 2.2, "How to stop smart bins from weighing themselves for nothing – without missing a pick",
       size=44, font=SERIF, bold=True, color=LIGHT, line=1.0)
    tb(s, LM, 4.95, 11.0, 0.4, "Bossard SmartBin Cloud · interim findings · the project explained step by step",
       size=16, color=LIGHT2)
    tb(s, LM, 5.45, 11.0, 0.4, "Erenalp Çakıroğlu · MSc Applied Information and Data Science · 23 October 2026",
       size=13, color="8C97A6")
    notes(s, f"""
Welcome. This deck tells the SmartBin story from start to finish, for someone who has not seen the details.
Bossard's SmartBin Cloud puts a battery-powered scale under every bin of small parts at a customer's site.
A vibration sensor wakes the scale up when the bin is touched; the scale weighs and reports the stock level,
and Bossard's ordering system ARIMS reorders automatically. The client's question: how should the sensitivity
of that vibration sensor be set, so that the scales do not weigh for nothing, but also do not miss real picks?
Our headline answer: there is no single right number. The right setting depends on where and how a bin is used,
and that is not in the master data. So every scale should learn its own level from its own weekly results,
with a simple rule that is reviewed every two weeks. In our simulation this cuts the weighings of normal scales
by {abs(sv6):.0f} to {abs(sv20):.0f} percent, while the sensor still catches almost nine out of ten
changes at once and the 24-hour check finds most of the rest within one to two days. A separate finding: in the
data, {N['loop_scales']} scales caught in a trigger loop – {pc(N['loop_share_scales'], 1)} of the scales – produced
{loops_pct} of all automatic weighings, most of them during a four-week trial of level 3 in April 2022. That is a
defect to repair, not a setting to tune. Every slide answers the question raised by the slide before it.""")

    # 2 ---------------------------------------------------------------------------------------- client
    s = d.slide()
    y = head(s, "THE CLIENT", "Bossard SmartBin Cloud: a scale under every bin")
    cards = [("≈ 110,000 scales", "installed at customer sites by March 2025; each reports the stock level of its "
              "bin. Our data: 18,455 of them."),
             ("Automatic reordering", "the stock level flows into Bossard's ARIMS system, which orders new parts "
              "before a bin runs empty."),
             ("A battery for ≥ 5 years", "a weighing draws about 3,000 times the power of sleep mode and loads the "
              "radio network, the IoT cloud and ARIMS."),
             ("A sensor decides", "the scale sleeps until its vibration sensor reports a touch. The sensitivity "
              "(1–10, default 8) sets how strong it must be.")]
    th = max(text_h(t, 2.76 - 0.44, 16, line=1.0, bold=True) for t, _ in cards)
    for i, (t, body) in enumerate(cards):
        card(s, LM + i * 2.93, y, 2.76, 2.25, t, body, body_size=13, title_h=th)
    sy = y + 2.55
    tb(s, LM, sy, CW, 0.3, "HOW A PICK BECOMES AN ORDER", size=11, bold=True, color=ORANGE_D, spc=120)
    chain = ["Someone takes parts", "Sensor wakes the scale", "Scale weighs", "Radio network", "IoT cloud",
             "ARIMS reorders"]
    for i, t in enumerate(chain):
        x = LM + i * 1.95
        rect(s, x, sy + 0.38, 1.7, 0.52, fill=HEAD if i < 5 else NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
        tb(s, x + 0.06, sy + 0.38, 1.58, 0.52, t, size=11.5, align="c", anchor="m", color=NAVY if i < 5 else LIGHT,
           bold=i == 5, line=1.0)
        if i < 5:
            arrow(s, x + 1.72, sy + 0.64, x + 1.93, sy + 0.64, color=RULE)
    callout(s, LM, sy + 1.25, CW, 0.78, "**So:** every unnecessary wake-up costs battery, radio and cloud capacity – "
            "and every change the sensor sleeps through leaves a wrong stock level in ARIMS.", size=14.5)
    source(s, "Source: Bossard AG & D ONE (2026), case material; Bossard AG (n.d.); Anastasi et al. (2009)")
    notes(s, """
Who is the client? Bossard supplies fasteners and other small C-parts to industrial customers. SmartBin Cloud is
its weight-based e-Kanban: every bin stands on a scale, the scale reports the stock level, and when it falls below
the reorder level, ARIMS – Bossard's replenishment system – orders automatically. About 110,000 scales were
installed by March 2025; our data cover 18,455 of them. The scales run on batteries that should last at least five
years. A weighing consumes about 3,000 times the power of sleep mode, and every weighing also travels over the radio
network and loads the IoT cloud and ARIMS. That is why the scale sleeps and only wakes up when its vibration sensor
reports a touch. The sensitivity setting, from 1 to 10 with a factory default of 8, decides how strong that vibration
must be. Two more settings exist – the minimum wait time and the measurement delay – but the case declared them out
of scope. In the language of sensor networks this is event-driven duty cycling: a node saves energy by sleeping and
waking only on events (Anastasi et al., 2009). The price of sleeping is that an event can be missed – and that is
exactly the trade-off of this case.""")

    # 3 ---------------------------------------------------------------------------------------- problem
    s = d.slide()
    y = head(s, "THE PROBLEM", "Every weighing ends in one of four outcomes")
    lab = N["lab"]

    def box(x, yy, ww, hh, t, f=CARD):
        panel(s, x, yy, ww, hh, fill=f)
        tb(s, x + 0.12, yy, ww - 0.24, hh, t, size=12.5, anchor="m", align="c", line=1.08)

    box(LM, 3.0, 1.75, 0.9, "Someone touches the bin")
    box(2.95, 2.85, 2.3, 1.2, "Is the vibration above the **sensitivity** threshold?", f=HEAD)
    box(5.6, 2.15, 2.35, 0.9, "**yes:** the scale weighs at once (automatic)")
    box(5.6, 3.8, 2.35, 1.0, "**no:** it sleeps on; a 24 h check or a person weighs later (manual)")
    arrow(s, LM + 1.75, 3.45, 2.95, 3.45)
    arrow(s, 5.25, 3.2, 5.6, 2.6)
    arrow(s, 5.25, 3.7, 5.6, 4.3)
    leaves = [(1.95, "**OK** · weight changed: real pick caught", lab["auto_ok"], GOOD, 2.6),
              (2.65, "**TOO_MANY** · no change: empty wake-up", lab["tm"], ORANGE, 2.6),
              (3.65, "**TOO_FEW** · change found later: missed", lab["tf"], CRIT, 4.3),
              (4.35, "**OK** · no change: nothing missed", lab["man_ok"], RULE, 4.3)]
    for yy, t, cnt, col, src_y in leaves:
        panel(s, 8.3, yy, 4.15, 0.6)
        rect(s, 8.3, yy + 0.06, 0.08, 0.48, fill=col)
        tb(s, 8.52, yy, 2.95, 0.6, t, size=12.5, anchor="m")
        tb(s, 11.3, yy, 1.05, 0.6, M(cnt).replace(" million", " M"), size=12.5, anchor="m", align="r", color=INK2)
        arrow(s, 7.95, src_y, 8.3, yy + 0.3)
    card(s, LM, 5.25, 5.6, 1.45, "Bossard's KPIs per scale",
         "**TOO_MANY fraction** = TOO_MANY / all automatic weighings · target ≤ 0.5 · alarm > 0.8\n"
         "**TOO_FEW per week** · target ≤ 2", title_size=14, body_size=12.5, pad=0.18)
    callout(s, 6.75, 5.25, 5.7, 1.45, "**So:** the setting moves one threshold. Too low wastes energy on empty "
            "wake-ups, too high misses changes – the classic false-alarm vs. miss trade-off of every detector.",
            size=14)
    source(s, "Source: Bossard AG & D ONE (2026); Green & Swets (1966). Counts after removing duplicates and impossible weights (M = million).")
    notes(s, f"""
Every weighing in the data ends in one of four outcomes, and the whole project is built on them. If the sensor
wakes the scale up, we have an automatic weighing: either the weight has changed – a real pick was caught, labelled
OK – or it has not – an empty wake-up, labelled TOO_MANY. If the sensor sleeps through a change, the change is only
found later by a manual weighing: either a person triggers it, or – in {pc(N['idle_share_manual'])} of the cases –
a check that runs when a scale has been idle for about 24 hours. If that weighing finds a change, it is labelled
TOO_FEW: the sensor missed it. In the data, after removing duplicates and impossible weights: {M(lab['auto_ok'])} caught picks, {M(lab['tm'])} empty
wake-ups – {pc(N['share_empty_auto'])} of all automatic weighings – {M(lab['man_ok'])} manual weighings without a
change and {M(lab['tf'])} missed changes. Bossard measures two KPIs per scale: the share of empty wake-ups among all
automatic weighings (target at most 0.5, alarm above 0.8) and the number of missed changes per week (target at
most 2). This is the classic trade-off of any detector: a lower threshold produces more false alarms, a higher one
more misses (Green & Swets, 1966). The sensitivity setting moves that threshold. Keep one thing in mind for later:
TOO_FEW can only come from a manual weighing, so it is a lower bound of the misses.""")

    # 4 ---------------------------------------------------------------------------------------- approach
    s = d.slide()
    y = head(s, "OUR APPROACH", "The client's three questions")
    qs = [("1", "Which parameters hardly matter?",
           "**What we did:** compared groups of scales on the same setting (level 8).",
           "**Short answer:** box type and item weight – clearly on 20 kg, less so on 6 kg scales."),
          ("2", "Which parameters matter?",
           f"**What we did:** measured sites, usage and timing – and the setting itself on the {N['switchers']:,} "
           "scales Bossard switched in April.",
           "**Short answer:** site and usage – and the setting, the only lever Bossard controls remotely."),
          ("3", "How should the sensitivity be set?",
           "**What we did:** simulated nine policies on the real history of every scale.",
           "**Short answer:** not one number – each scale learns its level, after defects are repaired.")]
    for i, (num, q, did, ans) in enumerate(qs):
        x = LM + i * 3.93
        panel(s, x, y, 3.7, 3.25)
        tb(s, x + 0.26, y + 0.2, 3.2, 0.5, num, size=26, font=SERIF, bold=True, color=ORANGE)
        cy = tb(s, x + 0.26, y + 0.75, 3.2, 0.6, q, size=16, bold=True, line=1.05)
        cy = tb(s, x + 0.26, cy + 0.12, 3.2, 0.8, did, size=12.5, color=INK2, line=1.12)
        tb(s, x + 0.26, cy + 0.14, 3.2, 0.75, ans, size=12.5, color=NAVY, line=1.12)
    steps = ["Data: 3 files", "Clean: rule book", "Observe: Power BI", "Analyse: 6 / 20 kg apart",
             "Simulate: 9 policies", "Recommend"]
    sy = y + 3.6
    for i, t in enumerate(steps):
        x = LM + i * 1.95
        rect(s, x, sy, 1.72, 0.56, fill=HEAD if i < 5 else ORANGE_D, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
        tb(s, x + 0.06, sy, 1.6, 0.56, t, size=11.5, align="c", anchor="m", color=NAVY if i < 5 else LIGHT,
           bold=i == 5, line=1.0)
        if i < 5:
            arrow(s, x + 1.74, sy + 0.28, x + 1.93, sy + 0.28, color=RULE)
    source(s, "Process model: CRISP-DM (Chapman et al., 2000)")
    notes(s, f"""
We turned the client's task into three questions, and the deck follows them. Questions one and two: which
parameters hardly influence the outcome and which do? We compared groups of scales on the same setting, measured how
strongly sites differ, analysed the timing of every empty wake-up – and used a natural experiment in the data: in
April Bossard moved {N['switchers']:,} scales away from level 8, so we can measure what the setting itself does to
the same scale. Question three: how should the sensitivity be set? We simulated nine policies on the real history of
every scale, from one fixed level for everybody to rules that adapt each scale. The process follows CRISP-DM
(Chapman et al., 2000): understand the business and the data, prepare the data with an explicit rule book, observe
it in Power BI, analyse it – always separately for 6 kg and 20 kg scales – simulate, and recommend. The short
answers: box type and item weight hardly matter on 20 kg scales and matter less than site and usage on 6 kg scales;
site, usage and the setting matter; and the setting should not be
one number, but learned by every scale, after a small group of defective scales has been repaired.""")

    # 5 ---------------------------------------------------------------------------------------- data + cleaning
    s = d.slide()
    y = head(s, "STEP 1 · DATA", "What we worked with – and how we cleaned it")
    stats = [(f"{N['rows'] / 1e6:.1f} M", "weighings in measurements.csv"),
             (f"{N['scales']:,}", "scales – type, box, site and item weight for most of them"),
             ("16 weeks", "1 March – 20 June 2022"),
             (pc(N["share_main"]), "of the rows enter the KPI analysis")]
    for i, (v, t) in enumerate(stats):
        x, yy = LM + (i % 2) * 2.42, y + (i // 2) * 1.62
        panel(s, x, yy, 2.25, 1.45)
        tb(s, x + 0.2, yy + 0.16, 1.95, 0.55, v, size=28, font=SERIF, bold=True, color=ORANGE if i == 3 else BLUE)
        tb(s, x + 0.2, yy + 0.78, 1.9, 0.6, t, size=12, color=INK2, line=1.1)
    pic(s, c["cleaning"], 5.75, y - 0.12, w=6.8)
    callout(s, LM, 5.85, CW, 0.8, f"**So:** delete only what is broken ({N['deleted']:,} rows). Flag the rest "
            "and keep it: a flagged row still counts for the weight change of the next weighing and as a "
            "neighbouring pick – it just does not enter the KPIs.", size=14)
    source(s, OWN + " (measurements, scale metadata, D ONE preprocessing flags)")
    notes(s, f"""
Three files: the measurements – {N['rows']:,} weighings of {N['scales']:,} scales from 1 March to 20 June 2022 –,
the scale metadata (scale type, box type, site, item weight) and D ONE's preprocessing flags. We cleaned with an
explicit rule book in two stages. Stage A deletes only what cannot be used anywhere: {N['dupes']:,} duplicate copies
(same scale, same millisecond) and {N['bad_weight']} physically impossible weights – exactly the rows D ONE had marked
as invalid. Stage B flags rows that are unsuitable for the KPIs but keeps them, because they are still real
weighings: we need them to compute the weight change of the next weighing and as neighbouring picks in the timing
analysis. The largest flag is level 3 with {N['level3_rows']:,} rows – we come back to it on slide 14. Four checks
shaped the rules: 1,300 duplicate groups carry different labels, so their outcome is unknown; D ONE's 'relevant
section' rule – at least 20 weighings over at least 5 days – holds for 100% of the sections; rows with an empty
wait-time setting behave like standard rows and are kept; and 985 scales have no type or site. In the end
{N['rows_main'] / 1e6:.2f} million rows ({pc(N['share_main'])}) enter the KPI analysis, and every excluded row is
exported with its reason. The unit of analysis is one scale on one sensitivity level. We summarise with the median
scale, the 'typical scale', so that every scale counts once – pooling all rows would let the noisiest scales
dominate.""")

    # 6 ---------------------------------------------------------------------------------------- Power BI A
    s = d.slide()
    y = head(s, "STEP 2 · OBSERVE · POWER BI", "First look: what each sensitivity level does")
    placeholder(s, LM, y, 7.65, 4.6, "A", "KPIs by sensitivity level (page 1): median TOO_MANY fraction, TOO_FEW per "
                "week and share of scales per fraction class, for each level – slicer 6 kg / 20 kg",
                "Insert a screenshot of the Power BI page here and delete this box.")
    t20 = (kpi.loc[("20kg", 8), "tm_fraction"], kpi.loc[("20kg", 1), "tm_fraction"])
    t6 = (kpi.loc[("6kg", 8), "tm_fraction"], kpi.loc[("6kg", 1), "tm_fraction"])
    tfmax = N["kpi"].tf_per_week.max()
    tb(s, 8.85, y, 3.65, 0.3, "WHAT WE SAW", size=12, bold=True, color=ORANGE_D, spc=150)
    tb(s, 8.85, y + 0.42, 3.65, 4.1, [
        f"• **Lower levels help – less than expected.** Median fraction of 20 kg scales: {t20[0]:.2f} on level 8, "
        f"{t20[1]:.2f} on level 1 (6 kg: {t6[0]:.2f} → {t6[1]:.2f}).",
        f"• **TOO_FEW per week stays low** on every level (at most {tfmax:.1f} for the typical scale) – even on "
        "level 1, where half of the changes are missed (slide 13).",
        "• **Huge spread within every level** – something other than the setting drives the result.",
        "• **Careful:** different scales on different levels – a description, not yet a cause."],
       size=13.5, space_after=9, line=1.12)
    source(s, OWN + "; Power BI report on the pipeline's export (one row = one scale on one level)")
    notes(s, f"""
[Presenter: show the 20 kg view first, then switch the slicer to 6 kg.] This is our observation dashboard in Power BI.
It is built on an export of our pipeline in which one row is one scale on one sensitivity level, so every ratio is
computed per scale before it is summarised – otherwise the busiest scales would dominate. What it shows: lowering the
setting improves the median TOO_MANY fraction, but much less than one would expect – on 20 kg scales from
{t20[0]:.2f} on level 8 to {t20[1]:.2f} on level 1, on 6 kg scales from {t6[0]:.2f} to {t6[1]:.2f}. TOO_FEW per
week stays far below Bossard's limit of 2 on every level – even on level 1, where the same scales miss half of their
changes (slide 13): a weekly count of misses hides how many changes are missed. And within every level the scales spread over the whole
range from 0 to 1. One caution: this view compares different scales on different levels and periods. It describes;
it does not prove cause. The causal effect of the setting follows on slide 13.""")

    # 7 ---------------------------------------------------------------------------------------- one-sided
    s = d.slide()
    y = head(s, "STEP 3 · ANALYSE · FINDING 1", "The problem is one-sided")
    pic(s, c["one_sided"], LM - 0.1, y + 0.05, w=7.2)
    yc = bignum(s, 8.6, y - 0.05, 3.85, f"{b.loc['20kg', 'too_many_per_week_median']:.1f} vs {real('20kg'):.1f}",
                "empty wake-ups vs. real picks caught per week – typical 20 kg scale on level 8", size=40)
    bignum(s, 8.6, yc + 0.3, 3.85, f"{b.loc['6kg', 'too_many_per_week_median']:.1f} vs {real('6kg'):.1f}",
           "the same for the typical 6 kg scale", size=40, color=BLUE)
    callout(s, LM, 5.75, CW, 0.82, "**So:** the question is not how to balance two targets – it is how far the "
            "sensitivity can go down before real changes are missed.")
    source(s, OWN + "; level 8 (factory default), typical scale = median")
    notes(s, f"""
At the factory default, level 8, {pc(b.loc['20kg', 'share_meets_tf'])} of the 20 kg and
{pc(b.loc['6kg', 'share_meets_tf'])} of the 6 kg scales meet the TOO_FEW target – but only
{pc(b.loc['20kg', 'share_meets_tm'])} and {pc(b.loc['6kg', 'share_meets_tm'])} meet the TOO_MANY target, and
{pc(b.loc['20kg', 'share_tm_alarm'])} of the 20 kg scales are even above the alarm level of 0.8. The typical 20 kg
scale wakes up empty {b.loc['20kg', 'too_many_per_week_median']:.1f} times a week and catches {real('20kg'):.1f} real
picks; the typical 6 kg scale {b.loc['6kg', 'too_many_per_week_median']:.1f} and {real('6kg'):.1f}. So this is not a
case of two equally violated targets. The question is how far the sensitivity can be lowered before real changes are
missed. One warning that will matter later: the TOO_FEW target is so loose that it is met even by scales that miss
half of their changes.""")

    # 8 ---------------------------------------------------------------------------------------- wrong compass
    s = d.slide()
    y = head(s, "FINDING 2", "The TOO_MANY fraction is the wrong compass")
    pic(s, c["usage"], LM - 0.15, y - 0.05, w=6.7)
    u20 = N["usage"]["20kg"]
    yc = bignum(s, 7.85, y - 0.05, 4.6, f"{u20[0]:.2f} vs {u20[1]:.2f}",
                "median fraction of the least- vs. the most-used fifth of the 20 kg bins – same setting", size=40)
    bignum(s, 7.85, yc + 0.25, 4.6, f"{sp(kp('20kg', 1, 'empty_change'))}  /  {sp(aok1_20)}",
           f"empty wake-ups / caught picks when the same 20 kg scales go from level 8 to 1 – the fraction falls by "
           f"only {abs(kp('20kg', 1, 'did_frac_too_many')):.2f}", size=40, color=BLUE)
    callout(s, LM, 5.75, CW, 0.82, "**So:** judge a scale by its empty wake-ups per week and the share of changes it "
            "misses – **KPI 2.0:** ≤ 7 empty wake-ups a week · ≤ 10% of changes missed · ≤ 2 TOO_FEW a week.",
            size=14)
    source(s, OWN + "; 'when a measure becomes a target, it ceases to be a good measure' (Strathern, 1997)")
    notes(s, f"""
The TOO_MANY fraction divides two numbers: empty wake-ups by all automatic weighings. A bin that is rarely used has
few real weighings in that denominator, so every empty wake-up counts heavily. On level 8, the least-used fifth of the
20 kg bins has a median fraction of {u20[0]:.2f}, the most-used fifth {u20[1]:.2f}; the rank correlation between real
picks per week and the fraction is {mi(f"{N['spearman']['20kg']:.2f}")} on 20 kg and
{mi(f"{N['spearman']['6kg']:.2f}")} on 6 kg scales. The typical 20 kg bin is picked only half as often as a 6 kg bin.
If a typical 6 kg bin were picked as rarely, its fraction would be about 0.85 instead of 0.74 – roughly 60% of the
gap between the two scale types. Second, the setting moves both
numbers. When the same 20 kg scales go from level 8 to level 1, their empty wake-ups fall by
{abs(kp('20kg', 1, 'empty_change')) * 100:.0f}%, but their caught picks fall by {abs(aok1_20) * 100:.0f}% as well, and
the fraction improves by only {abs(kp('20kg', 1, 'did_frac_too_many')):.2f}. A scale on level 1 has a 'good' fraction
mainly because its sensor is practically switched off. This is Goodhart's law in action – when a measure becomes a
target, it ceases to be a good measure (Strathern, 1997). We therefore propose KPI 2.0: empty wake-ups per week, at
most 7, because battery, radio and cloud pay per weighing, not per ratio; the share of real changes missed, at most
10%, because that is the service the customer receives; and Bossard's own floor of at most 2 TOO_FEW per week.""")

    # 9 ---------------------------------------------------------------------------------------- key insight: coupling
    s = d.slide(dark=True)
    head(s, "THE KEY INSIGHT", "Almost half of the empty wake-ups coincide with a pick at the same site", dark=True,
         size=34, lines=2)
    a20, a6 = root.loc["20kg", "neighbour_induced"], root.loc["6kg", "neighbour_induced"]
    yb = 2.6
    bignum(s, LM, yb, 2.5, pc(a20), "20 kg scales", color=ORANGE_L, size=54, cap_color=LIGHT2)
    yc = bignum(s, LM + 2.75, yb, 2.5, pc(a6), "6 kg scales", color=BLUE_L, size=54, cap_color=LIGHT2)
    tb(s, LM, yc + 0.22, 5.4, 1.6, "of the empty wake-ups come within ±14 seconds of a real pick at another scale of "
       "the same site – beyond what chance explains. At that moment, picks nearby jump from about 1 to about 9 per "
       "minute.", size=15, color=LIGHT, line=1.15)
    pic(s, c["coupling"], 6.75, yb - 0.1, w=6.0)
    tb(s, LM, 5.75, CW, 0.9, f"Empty wake-ups also follow the working day like real picks (hourly profiles: r = "
       f"{min(N['hourly_r'].values()):.2f}) – only about {wk_tm * 100:.1f}% happen at weekends. They go with activity on "
       "the site, not with idle hours.", size=15, color=LIGHT, line=1.15)
    source(s, "Method: event coincidence analysis (Donges et al., 2016), control window 60–88 s after the wake-up. "
              + OWN, dark=True)
    notes(s, f"""
This is the most important observation of the project. For every empty wake-up we looked for a real pick at another
scale of the same site within a short time window – an event coincidence analysis (Donges et al., 2016). If both were
unrelated, the picks would be spread evenly over time. Instead they pile up sharply at the moment of the empty
wake-up: from about {N['base_per_min']['20kg']:.1f} to about {N['peak_per_min']['20kg']:.1f} per minute on 20 kg
scales. The peak is about ±14 seconds wide. On level 8, {pc(root.loc['20kg', 'p_near'])} of the empty wake-ups of
20 kg scales have a pick at another scale of the site within ±14 seconds; in a control window one minute later only
{pc(root.loc['20kg', 'p_chance'])} do – that is the chance level of a busy site. Beyond chance, {pc(a20)} (20 kg)
and {pc(a6)} (6 kg) of the empty wake-ups are linked to picking at the same site – {pc(N['coupled_pooled'])} across
both scale types; the scale's own echo after its own pick explains only 1–3%. A second test points the same way:
empty wake-ups follow the working day like real picks – the hourly profiles correlate at r =
{min(N['hourly_r'].values()):.2f} – and only about {wk_tm * 100:.1f}% of them happen at weekends, so switching the
sensor off at night or at weekends would not help (appendix). What the data cannot show: the same peak appears around
real picks, because people pick several bins in a row, and machines working in shifts would show the same daily
rhythm. So we cannot separate vibration travelling through the rack from a worker touching the bin or from a pick too
small to count. A short on-site test with an accelerometer or a damping pad would settle it. The other half of the
empty wake-ups cannot be attributed to picking at the same site and remains unexplained.""")

    # 10 --------------------------------------------------------------------------------------- blind zone
    s = d.slide()
    y = head(s, "FINDING 3", "Every scale has a blind zone")
    pic(s, c["blind"], LM - 0.15, y - 0.05, w=7.7)
    pk = N["pick"]
    tb(s, 8.85, y + 0.05, 3.6, 3.6, [
        f"• **Below ≈ {N['floor_g']['20kg']:.0f} g (20 kg) and ≈ {N['floor_g']['6kg']:.0f} g (6 kg)** a pick is "
        "mostly invisible to the scale.",
        "• **Level 8 catches ≥ 95%** of the picks from 20 g up. Levels 1–5 never reach 95% for any pick size.",
        f"• **{pc(pk.loc['6kg', 'share_median_pick_below_25g'])} of the 6 kg scales** have a typical pick below "
        f"25 g – only {pc(pk.loc['20kg', 'share_median_pick_below_25g'])} of the 20 kg scales do."],
       size=13.5, space_after=10, line=1.12)
    callout(s, LM, 5.75, CW, 0.82, "**So:** small parts belong on 6 kg scales – and a 6 kg scale cannot go as low "
            "as a 20 kg scale.")
    source(s, OWN + "; pick size = weight drop since the previous weighing")
    notes(s, f"""
The cloud labels a weighing as 'changed' only if the weight changed by more than a significance threshold. The
threshold is not documented, so we estimated it from the data as the lower end of the changes reported as missed (the
5th percentile of the TOO_FEW pick sizes): about {N['floor_g']['20kg']:.0f} g on 20 kg scales and
{N['floor_g']['6kg']:.0f} g on 6 kg scales. Below it a pick is largely invisible: it is labelled TOO_MANY when the
sensor wakes up and is never reported when it does not. Above it, detection depends on the pick size and the setting.
The chart shows the share of picks the sensor caught at once, by pick size. On level 8, picks from 20 g upwards are
caught in at least 95% of cases on both scale types; levels 1 to 5 do not reach 95% for any pick size. This matters
for the recommendation: {pc(pk.loc['6kg', 'share_median_pick_below_25g'])} of the 6 kg scales have a typical pick
below 25 g – their median pick is {pk.loc['6kg', 'median_of_median_pick_g']:.0f} g, against
{pk.loc['20kg', 'median_of_median_pick_g']:.0f} g on 20 kg scales. They work right at the edge of what a scale can
see, so their sensitivity cannot go as low as that of a 20 kg scale. And parts that are picked in amounts below about
25 g belong on 6 kg scales.""")

    # 11 --------------------------------------------------------------------------------------- Power BI B / sites
    s = d.slide()
    y = head(s, "FINDING 4 · POWER BI", "Where a bin stands matters more than what is in it")
    placeholder(s, LM, y, 7.65, 4.6, "B", "Sites × sensitivity (page 2): median TOO_MANY fraction per site on "
                "level 8 and how the sites react on the other levels – one small chart per level, no dual axis",
                "Insert a screenshot of the Power BI page here and delete this box.")
    l20, l6 = loc["20kg"], loc["6kg"]
    ts = N["top_sites"]
    yc = bignum(s, 8.85, y - 0.05, 3.6, f"{l20['spread_p10_p90'][0]:.2f}–{l20['spread_p10_p90'][1]:.2f}",
                f"median fraction of the middle 80% of the 20 kg sites (6 kg: {l6['spread_p10_p90'][0]:.2f}–"
                f"{l6['spread_p10_p90'][1]:.2f})", size=36, cap_size=13)
    yc = bignum(s, 8.85, yc + 0.2, 3.6, f"{pc(l20['icc_frac_too_many'])} / {pc(l6['icc_frac_too_many'])}",
                f"of the differences between scales lie between sites – stable in two random halves "
                f"(r = {l20['split_half_r_frac_too_many']:.2f} / {l6['split_half_r_frac_too_many']:.2f})",
                size=36, color=BLUE, cap_size=13)
    bignum(s, 8.85, yc + 0.2, 3.6, "10 sites", f"hold {pc(ts.loc['20kg', 'share_coupled_top10_sites'])} / "
           f"{pc(ts.loc['6kg', 'share_coupled_top10_sites'])} of all coupled empty wake-ups", size=36, cap_size=13)
    source(s, OWN + "; intraclass correlation (Shrout & Fleiss, 1979), permutation test p ≤ 0.001")
    notes(s, f"""
[Presenter: show the Power BI page 2 – median TOO_MANY fraction per site, one small chart per level.] The site shows
the largest differences in the data. Among 20 kg scales on level 8, the middle 80% of the sites range from a median
fraction of {l20['spread_p10_p90'][0]:.2f} to {l20['spread_p10_p90'][1]:.2f} (6 kg: {l6['spread_p10_p90'][0]:.2f}
to {l6['spread_p10_p90'][1]:.2f}). The differences are real and stable: {pc(l20['icc_frac_too_many'])} (20 kg) and
{pc(l6['icc_frac_too_many'])} (6 kg) of the variation between scales lies between sites – the intraclass correlation
(Shrout & Fleiss, 1979); a permutation test with 999 shuffles of the site labels gives p ≤ 0.001; two random halves
of the scales of each site agree closely (split-half r = {l20['split_half_r_frac_too_many']:.2f} and
{l6['split_half_r_frac_too_many']:.2f}); and the effect remains when we account for how busy the bins are. The
coupling of slide 9 is concentrated as well: ten sites hold {pc(ts.loc['20kg', 'share_coupled_top10_sites'])} (20 kg)
and {pc(ts.loc['6kg', 'share_coupled_top10_sites'])} (6 kg) of all coupled empty wake-ups. Possible reasons are the
construction of the racks, machines nearby and the way people work. Bossard knows the site of every scale, but how a site behaves only shows in
its data – and the data contain the site, not the rack.""")

    # 12 --------------------------------------------------------------------------------------- parameters
    s = d.slide()
    y = head(s, "ANSWER TO QUESTIONS 1 AND 2", "What matters – and what hardly does")
    pic(s, c["params"], LM - 0.15, y - 0.05, w=6.9)
    fs = N["fill_same_scale"]
    fs20, fs6 = mi(f"{fs['20kg']:.3f}"), mi(f"{fs['6kg']:.3f}")
    uq = N["uq"].groupby("scale_type").frac_too_many_median.agg(lambda v: v.iloc[0] - v.iloc[-1])
    card(s, 7.95, y, 4.5, 1.8, "Hardly matters",
         f"**20 kg:** box type ({spr.loc[('20kg', 'Box type')]:.2f}) and item weight "
         f"({spr.loc[('20kg', 'Item weight (quartile)')]:.2f}). **Both types:** a fuller box on the same scale "
         f"({fs20} / {fs6}).", title_color=GOOD)
    card(s, 7.95, y + 2.0, 4.5, 1.8, "Matters",
         f"Site and usage ({min(spr.loc[('20kg', 'Site (location)')], uq.min()):.2f}–"
         f"{max(spr.loc[('6kg', 'Site (location)')], uq.max()):.2f}) and the setting – the only lever Bossard controls "
         f"remotely. On 6 kg scales box type ({spr.loc[('6kg', 'Box type')]:.2f}) and item weight "
         f"({spr.loc[('6kg', 'Item weight (quartile)')]:.2f}) differ too.", title_color=CRIT)
    tb(s, LM, 5.9, CW, 0.7, "How to read it: median TOO_MANY fraction of the best vs. the worst group of each "
       "parameter, all on level 8 except the setting (scales with ≥ 5 automatic weighings); sites with ≥ 20 scales: "
       "10th vs. 90th percentile; usage: least- vs. most-used fifth (slide 8).", size=12, color=INK2, line=1.12)
    source(s, OWN + "; Shapley decomposition in the appendix (Grömping, 2006)")
    notes(s, f"""
The answer to questions one and two. We compared the parameters by a simple measure: how far apart are the best and
the worst group? To keep the setting out of the comparison, all parameters except the sensitivity are compared on
level 8 only; for sites we use the 10th and 90th percentile of the site medians, so that a few tiny sites do not
dominate. On 20 kg scales the site moves the fraction most ({spr.loc[('20kg', 'Site (location)')]:.2f}), followed by
the sensitivity level ({spr.loc[('20kg', 'Sensitivity level')]:.2f}) and the fill level of the box
({spr.loc[('20kg', 'Fill level (weight bin)')]:.2f}); box type ({spr.loc[('20kg', 'Box type')]:.2f}) and item weight
({spr.loc[('20kg', 'Item weight (quartile)')]:.2f}) come last. On 6 kg scales the site also leads
({spr.loc[('6kg', 'Site (location)')]:.2f}), followed by the fill level ({spr.loc[('6kg', 'Fill level (weight bin)')]:.2f}),
box type ({spr.loc[('6kg', 'Box type')]:.2f}) and the sensitivity ({spr.loc[('6kg', 'Sensitivity level')]:.2f}). Usage
– the least- vs. the most-used fifth of the bins on slide 8 – spreads about as much as the site. Two numbers need
care. The fill-level difference almost disappears when the same scale changes its fill level: {mi(f"{fs['20kg']:.3f}")}
on 20 kg and {mi(f"{fs['6kg']:.3f}")} on 6 kg scales, over all scales that changed their weight bin – nearly empty
boxes are mostly rarely used boxes. (D ONE's subset 'relevant for weight analysis' shows larger values, but it was
selected on scales where something changes, that is, on the outcome.) And on 6 kg scales only two box types are
common, and they sit at different sites; item weight also differs more there. Verdict: on 20 kg scales box type and
item weight hardly matter; on 6 kg scales they matter less than site and usage. The site, the usage and the setting
matter most. Of these, the setting is the only lever Bossard can change remotely – how strong it is, we measure on the
next slide. A Shapley decomposition of the explained variance in the appendix ranks the site first as well; it gives
the setting a small share, partly because almost all data come from level 8.""")

    # 13 --------------------------------------------------------------------------------------- the knob
    s = d.slide()
    y = head(s, "THE SETTING · BEFORE AND AFTER", "Same scale, before and after: the first steps are cheap")
    tb(s, LM, y - 0.12, 7.9, 0.62, f"In April Bossard moved **{N['switchers']:,} scales** away from level 8 – a "
       "natural experiment. Each scale is compared with itself before the switch and with scales that stayed on 8.",
       size=13.5, color=INK2, line=1.12)
    pic(s, c["knob"], LM - 0.12, y + 0.6, w=8.1)
    yc = bignum(s, 9.35, y - 0.1, 3.1, f"{sp(e20)} / {sp(e6)}", "empty wake-ups for one step, 8 → 7 "
                "(20 kg / 6 kg) …", size=36, cap_size=13)
    yc = bignum(s, 9.35, yc + 0.2, 3.1, f"+{m20:.0f} / +{m6:.0f} pts", "… for this many more changes missed",
                size=36, color=BLUE, cap_size=13)
    tb(s, 9.35, yc + 0.25, 3.1, 1.2, "Below level 6 the misses climb steeply. 6 kg scales react more strongly – "
       "they behave like a different instrument.", size=13, color=NAVY, line=1.12)
    callout(s, LM, 5.95, CW, 0.7, "**So:** the setting works and the first step is cheap – how far a scale can go "
            "depends on the scale.", size=14.5)
    source(s, "Method: difference-in-differences (Angrist & Pischke, 2009), fixed-effects Poisson, bootstrap "
              "intervals. " + OWN)
    notes(s, f"""
Comparing different scales on different levels cannot show what the setting itself does, because the levels were not
assigned at random. But in April Bossard moved {N['switchers']:,} scales away from the default. We compared each of
them with itself before the switch, and subtracted the change of a stable group of scales that stayed on level 8
over the same weeks – a difference-in-differences design (Angrist & Pischke, 2009). Counts per week come from a
fixed-effects Poisson model; the intervals are 95% bootstrap intervals over scales. The result: the setting is
powerful. One step, from 8 to 7, removes about {abs(e20) * 100:.0f}% (20 kg) and {abs(e6) * 100:.0f}% (6 kg) of the
empty wake-ups and costs only {m20:.0f} and {m6:.0f} percentage points of missed changes. Below level 6 the price
rises steeply: from 8 to 5 the scales miss {kp('20kg', 5, 'miss_pp'):.0f} (20 kg) and {kp('6kg', 5, 'miss_pp'):.0f}
(6 kg) points more; on level 1 half or more of all changes are missed. The 20 kg estimates are less precise – for
8 → 6 the interval runs from about −24% to −67%. And Bossard seems to have switched some busy scales first: their empty
wake-ups rose in the weeks before the switch (appendix), so part of the drop may be a return to normal. A careful
reading of the first steps is therefore 'roughly half to two thirds fewer empty wake-ups'. The 6 kg scales react more strongly in both directions: the same step from 8 to 6 removes
{abs(kp('20kg', 6, 'empty_change')) * 100:.0f}% of the empty wake-ups on 20 kg scales but
{abs(kp('6kg', 6, 'empty_change')) * 100:.0f}% on 6 kg scales. Whether a level means the same vibration threshold on
both hardware versions is not documented – we treat them as two different instruments and set them separately.""")

    # 14 --------------------------------------------------------------------------------------- loops
    s = d.slide()
    y = head(s, "A DEFECT, NOT A SETTING", f"{pc(N['loop_share_scales'], 1)} of the scales caused {loops_pct} of the "
             "automatic weighings", lines=1)
    pic(s, c["loops"], LM - 0.12, y - 0.08, w=8.2)
    tb(s, 9.35, y - 0.05, 3.1, 2.1, f"**{N['loop_scales']} scales** were caught in a trigger loop: {N['loop_l3']} on "
       f"level 3 – a quarter of the scales in its April trial – and {N['loop_other']} on other levels "
       f"({N['loop_l3'] + N['loop_other'] - N['loop_scales']} on both).", size=13.5, line=1.15)
    yc = y + 2.15
    cards = [("The signature", f"A wake-up every ~90 s – the wait time (60 s) plus the delay (30 s) – day and night: "
              f"{N['loop_rate_l3']:.0f} a day for the typical looping level-3 scale."),
             ("The detector", "More than half of the wake-ups within 2 min of the previous one and more than 20 a "
              "day, judged over ≥ 5 days. Runs every day."),
             ("The cost", f"With the case's battery assumption, a level-3 loop empties a five-year battery in about "
              f"{weeks:.0f} weeks (about {weeks_other:.0f} weeks for the slower loops on other levels).")]
    for i, (t, body) in enumerate(cards):
        card(s, LM + i * 3.93, yc, 3.7, 1.72, t, body, body_size=12.5, title_size=15)
    callout(s, LM, 6.05, CW, 0.66, "**So:** repair it, don't tune it – keep level 3 off the menu and send looping "
            "scales to service.", size=14.5)
    source(s, OWN + "; battery model of the case: a weighing ≈ 2 s at ≈ 3,000 × sleep power")
    notes(s, f"""
The case description advises ignoring level 3. The data show why – and that the problem is not only level 3.
Level 3 was used only during a four-week trial in April 2022: {N['l3_scales']} scales ran on it, and together they woke
up about {N['l3_rate']['20kg']:.0f} (20 kg) and {N['l3_rate']['6kg']:.0f} (6 kg) times per scale and day – the same
scales on level 8 woke up about 11 to 12 times a day. Most of these wake-ups come in a loop: the scale wakes up,
weighs and wakes up again about 90 seconds later – exactly the minimum wait time of 60 seconds plus the measurement
delay of 30 seconds – day and night. We wrote a simple detector: a scale counts as looping on a level if more than
half of its wake-ups follow the previous one within 2 minutes and it wakes up more than 20 times a day, judged over at
least 5 days. It finds {N['loop_scales']} scales: {N['loop_l3']} on level 3 – about a quarter of the
{N['l3_judged']} level-3 scales it could judge – and {N['loop_other']} on other levels, mostly on level 8 (a few loop on
both). Together they produced {loops_pct} of all automatic weighings in the data, although they are only
{pc(N['loop_share_scales'], 1)} of the scales; the loops outside level 3 alone produced
{pc(N['loop_other_auto'] / N['auto_all'])}. The typical looping level-3 scale wakes up {N['loop_rate_l3']:.0f}
times a day. With the case's assumption that a weighing costs about 3,000 times the sleep power for about two
seconds, such a scale would empty a battery designed for five years in about {weeks:.0f} weeks; the slower loops on other
levels in about {weeks_other:.0f} weeks. A different sensitivity is not the fix for a loop: it is a defect. Level 3
should stay off the setting menu until the firmware is fixed, and the detector should run every day to send looping
scales to service. The looping scales are left out of the policy savings that follow.""")

    # 15 --------------------------------------------------------------------------------------- answer Q3 (dark)
    s = d.slide(dark=True)
    head(s, "ANSWER TO QUESTION 3", "The right setting is not in the master data – each scale has to learn it",
         dark=True, size=34, lines=2)
    pts = [("1", "What Bossard knows in advance – box type, item weight – matters little next to site and usage."),
           ("2", "What matters – how busy a bin is, its neighbours, its pick size, how its site behaves – only shows "
                 "once the scale runs."),
           ("3", "But every scale reports what we need, every week: how often it woke up for nothing, and how many "
                 "of its changes were missed.")]
    for i, (num, t) in enumerate(pts):
        x = LM + i * 3.93
        tb(s, x, 2.75, 3.6, 0.8, num, size=48, font=SERIF, bold=True, color=ORANGE_L)
        tb(s, x, 3.65, 3.55, 1.6, t, size=16, color=LIGHT, line=1.15)
    line(s, LM, 5.45, LM + CW, 5.45, color="3A4A5E", lw=1)
    tb(s, LM, 5.65, CW, 0.8, "So each scale finds its own level from its own results – with a simple rule that "
       "anyone can audit.", size=20, font=SERIF, bold=True, color=LIGHT, line=1.1)
    notes(s, """
This is the answer to question three, and it follows from everything before. First: what Bossard knows about a bin
before installing it – the box type, the item weight – hardly matters on 20 kg scales and matters less than site and
usage on 6 kg scales. Second: what does matter – how often the bin is used, what happens on the rack next to it, how
small the typical pick is, and how its site behaves – only shows once the scale is running. Bossard knows the site,
but not how it will behave. That is why a lookup table is the wrong tool: in our simulation, a table by pick-size class
chose the same setting as one level for everybody for 95 to 99% of the scales. Third, and this is the good news: every scale already reports exactly what is needed, every week – how
often it woke up for nothing, and how many of its changes were only found by a later check. So instead of guessing
the setting from the master data, we let each scale find its level from its own results, with a simple rule that
anybody can audit. We deliberately chose a transparent rule over a machine-learning model: it is easy to explain, to
check and to trust, and it needs no inputs that Bossard does not already have.""")

    # 16 --------------------------------------------------------------------------------------- recommendation
    s = d.slide()
    y = head(s, "RECOMMENDATION", "Flag, tune, watch")
    cols = [("FLAG", "Repair what no setting can fix",
             ["Run the loop detector every day; send looping scales to service.",
              "Take level 3 off the menu until the firmware is fixed."],
             f"loops caused {loops_pct} of all automatic weighings – no trade-off involved"),
            ("TUNE", "Let every scale learn its level",
             ["Start every scale on **7** instead of 8.",
              "Every 2 weeks: **one step less sensitive** if > 7 empty wake-ups a week and ≤ 5% missed; **one step "
              "more sensitive** if > 10% missed or > 2 TOO_FEW a week.",
              "Same rule, 6 kg and 20 kg monitored apart."],
             f"{sp(sv20 / 100)} (20 kg) / {sp(sv6 / 100)} (6 kg) weighings"),
            ("WATCH", "Keep the service level visible",
             ["Report KPI 2.0 per scale (Power BI).",
              f"Keep the 24 h check – it finds {pc(N['idle_share_tf'])} of the detected misses.",
              f"Watch list: 10% of the scales produce {pc(w_.loc['20kg', 'share_of_too_few_from_top10pct'])} / "
              f"{pc(w_.loc['6kg', 'share_of_too_few_from_top10pct'])} of all misses → 12 h check or a higher level."],
             "misses found within 1–2 days; the miss limit sets the trade-off")]
    for i, (tag, sub, items, eff) in enumerate(cols):
        x = LM + i * 3.93
        rect(s, x, y, 3.7, 0.5, fill=ORANGE_D if i == 1 else NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.25)
        tb(s, x, y, 3.7, 0.5, tag, size=15, bold=True, color=LIGHT, align="c", anchor="m", spc=200)
        if i < 2:
            arrow(s, x + 3.72, y + 0.25, x + 3.91, y + 0.25, color=RULE)
        panel(s, x, y + 0.62, 3.7, 3.3)
        tb(s, x + 0.22, y + 0.78, 3.3, 0.3, sub, size=14.5, bold=True)
        tb(s, x + 0.22, y + 1.2, 3.3, 2.05, ["• " + it for it in items], size=12, color=INK2, line=1.1,
           space_after=4, bullet_indent=0.16)
        rect(s, x + 0.22, y + 3.12, 3.26, 0.012, fill=CARD_LINE)
        tb(s, x + 0.22, y + 3.2, 3.3, 0.66, eff, size=12, bold=True, color=ORANGE_D if i == 1 else NAVY, line=1.1)
    callout(s, LM, 6.05, CW, 0.62, "**At the customer site:** parts picked in amounts below ≈ 25 g go on 6 kg scales · "
            "damping test at the 10 sites with the most coupled wake-ups", size=13)
    source(s, OWN + "; simulation of every scale's own history, scales in a trigger loop left out")
    notes(s, f"""
Our recommendation has three parts. FLAG – repair what no setting can fix. The loop detector runs every day; looping
scales go to service, and level 3 stays off the setting menu until the firmware is fixed. In the data the loops caused
{loops_pct} of all automatic weighings – most of them during the level-3 trial in April, but {N['loop_other']} scales
also looped on other levels. Repairing a loop involves no trade-off with missed changes. TUNE – let every
scale learn its level. Every scale starts on level 7 instead of 8: on the same scales, this one step removed
{abs(e20) * 100:.0f} to {abs(e6) * 100:.0f}% of the empty wake-ups for {m20:.0f} to {m6:.0f} points more misses.
Then each scale is checked every two weeks: if it woke up empty more than 7 times a week and missed at most 5% of its
changes, it goes one step less sensitive; if it missed more than 10% of its changes or had more than 2 TOO_FEW a week,
it goes one step more sensitive. 6 kg and 20 kg scales run the same rule but are monitored separately, because they
are different instruments. WATCH – keep the service level visible. KPI 2.0 is reported per scale in a dashboard; the
existing 24-hour idle check stays as the safety net – it finds {pc(N['idle_share_tf'])} of all detected misses
(TOO_FEW), usually one to two days after the last weighing; and a watch list covers the few scales where misses
concentrate: 10% of the scales produce {pc(w_.loc['20kg', 'share_of_too_few_from_top10pct'])} (20 kg) and
{pc(w_.loc['6kg', 'share_of_too_few_from_top10pct'])} (6 kg) of all TOO_FEW. They get a 12-hour check or a higher
level. We do not recommend a nightly check for every scale: it would cost up to 7 weighings per scale and week – about
a third of the savings. The miss limit of 10% is a management decision: a lower limit for critical parts keeps more
scales sensitive and saves less. At the customer site, two actions help: parts that are picked in amounts below about
25 grams go on 6 kg scales, and damping pads are tested at the ten sites that hold almost half of the coupled empty
wake-ups.""")

    # 17 --------------------------------------------------------------------------------------- result
    s = d.slide()
    y = head(s, "RESULT", "What it adds up to")
    yc = bignum(s, LM, y - 0.05, 4.0, f"{sp(sv20 / 100)} / {sp(sv6 / 100)}",
                "weighings of the 20 kg / 6 kg scales without a loop", size=44)
    yc = bignum(s, LM, yc + 0.22, 4.0, f"≈ {fleet / 1e6:.1f} million", "fewer weighings per week if the "
                f"{N['installed']:,} installed scales behaved like ours (≈ {N['saved_per_scale_week']:.0f} per scale; "
                "illustrative)", size=36, color=BLUE)
    bignum(s, LM, yc + 0.22, 4.0, f"{mr20 * 100:.0f}–{mr6 * 100:.0f}%", f"of the changes found late by the 24 h check "
           f"– just above the 10% limit (simulated today: {m0_20 * 100:.0f}–{m0_6 * 100:.0f}%)", size=36, color=NAVY)
    pic(s, c["policies"], 5.0, y - 0.12, w=7.5)
    ach = N["achievable"]
    tb(s, 5.05, y + 3.75, 7.4, 1.1, [
        f"No policy brings the typical scale to Bossard's 0.5 fraction (best: "
        f"{ach['20kg']['best_achievable_median_frac']:.2f} / {ach['6kg']['best_achievable_median_frac']:.2f}). Scales "
        f"meeting KPI 2.0: {pc(pv('20kg', 'P0', 'share_in_target_kpi2'))} → {pc(pv('20kg', 'P7', 'share_in_target_kpi2'))}"
        f" (20 kg), {pc(pv('6kg', 'P0', 'share_in_target_kpi2'))} → {pc(pv('6kg', 'P7', 'share_in_target_kpi2'))} "
        "(6 kg). A simulation: compare the policies, don't read the numbers as forecasts."], size=12, color=INK2,
       line=1.12)
    source(s, OWN + "; each scale's level-8 history × causal rate ratios, 5 replications, 16 weeks")
    p8_l1 = fl.loc[(P['P8'], '20kg', 1.0)]
    notes(s, f"""
What does it add up to? We simulated each policy on every scale's own history: the scale keeps its behaviour on level
8, and the causal effects of slide 13 translate it to the other levels; every rule was run five times. For scales
without a loop, the recommended rule cuts the weighings by {abs(sv20):.0f}% (20 kg) and {abs(sv6):.0f}% (6 kg) –
about {N['saved_per_scale_week']:.0f} weighings per scale and week; if the {N['installed']:,} installed scales behaved
like the ones in our data, that would be roughly {fleet / 1e6:.1f} million per week – an illustration, not a forecast.
The price: the typical scale misses {mr20 * 100:.0f} to {mr6 * 100:.0f}% of its changes at first, instead of
{m0_20 * 100:.0f} to {m0_6 * 100:.0f}% in the simulated status quo – just above the 10% limit of KPI 2.0, because the
rule keeps testing the next lower level. A stricter step-down threshold would miss less and save less; the pilot should
tune it. Missed changes are not lost – the 24-hour check finds most of them, usually within one to two days. The share of scales that meet KPI 2.0 rises
from {pc(pv('20kg', 'P0', 'share_in_target_kpi2'))} to {pc(pv('20kg', 'P7', 'share_in_target_kpi2'))} (20 kg) and
from {pc(pv('6kg', 'P0', 'share_in_target_kpi2'))} to {pc(pv('6kg', 'P7', 'share_in_target_kpi2'))} (6 kg), and a
scale changes its setting about once a month. The battery life of a typical scale grows by only
{pv('6kg', 'P7', 'battery_life_change_pct'):.0f} to {pv('20kg', 'P7', 'battery_life_change_pct'):.0f}%, because
sleep dominates its energy budget – the big gains are on the radio network, the cloud and ARIMS, and on the looping
scales. Why not the alternatives? The best single level for Bossard's KPIs still leaves most scales outside
Bossard's targets and misses {pv('20kg', 'P1', 'median_miss_rate') * 100:.0f} to
{pv('6kg', 'P1', 'median_miss_rate') * 100:.0f}% of the changes. One level tuned for KPI 2.0 saves less on 20 kg scales and misses
{pv('6kg', 'P5', 'median_miss_rate') * 100:.0f}% on 6 kg scales. A rule that steers by the fraction pushes
{pc(p8_l1)} of the 20 kg scales to level 1 and misses {pv('6kg', 'P8', 'median_miss_rate') * 100:.0f} to
{pv('20kg', 'P8', 'median_miss_rate') * 100:.0f}% of the changes. And no policy brings the typical scale to Bossard's
fraction target of 0.5. Caution: the simulation is a model. It is stricter than the observed status quo and somewhat
optimistic in a back-test on the real switchers, so its numbers compare policies; they are not forecasts – hence the
pilot on the next slides.""")

    # 18 --------------------------------------------------------------------------------------- honesty
    s = d.slide()
    y = head(s, "HONESTY", "What we cannot claim yet")
    bt = N["backtest"]
    lim = [("Observational data", "The before/after design removes what is constant per scale and calendar effects – "
            "not events that hit only the switched scales, e.g. a busy spell just before their switch."),
           ("Misses are a lower bound", "TOO_FEW comes only from a manual weighing. A missed pick followed by an "
            "automatic weighing counts as OK – true miss rates are higher."),
           ("Coupling: half explained", "Rack vibration, touching and picks below the threshold cannot be told apart "
            "in the data – and half of the empty wake-ups remains unexplained."),
           ("Site, not rack", "The data know only the site. Neighbouring scales were inferred from timing and must "
            "be confirmed on site."),
           ("The simulation is a model", f"Stricter than the observed status quo and somewhat optimistic in a back-test "
            f"(r = {bt['corr'].min():.2f}–{bt['corr'].max():.2f} per scale): it ranks policies, it does not forecast."),
           ("Assumptions to confirm", "The battery model (2 s at 3,000 × sleep power), the 24 h check rule and the "
            "significance threshold were inferred; the data cover one season.")]
    for i, (t, body) in enumerate(lim):
        x, yy = LM + (i % 3) * 3.93, y + (i // 3) * 2.2
        card(s, x, yy, 3.7, 2.02, t, body, title_size=15, body_size=12.5)
    source(s, OWN)
    notes(s, f"""
Every analysis has limits, and naming them makes the work more credible, not less. One: the data are observational.
The before/after design removes everything that is constant per scale and the calendar effects, but it cannot rule out
an event that happened only to the switched scales around their switch – and some of them were busier than usual just
before it. Two: TOO_FEW is a lower bound of the
misses, because only a manual weighing can produce it; a missed pick that is followed by an automatic weighing is
labelled OK. True miss rates are therefore somewhat higher – probably for every policy, though not necessarily to
the same degree. Three: the
coupling analysis links about half of the empty wake-ups to picking nearby, but it cannot separate vibration through
the rack from touching the bin or from picks below the threshold, and the other half remains unexplained. Four: the
data know only the site; groups of neighbouring scales were inferred from timing and must be confirmed on site. Five:
the policy simulation is stricter than the observed status quo and somewhat optimistic in a back-test on the real
switchers – the correlation between predicted and observed fraction per scale is only {bt['corr'].min():.2f} to
{bt['corr'].max():.2f}; it ranks policies, it does not forecast. Six: the battery model, the rule of the 24-hour check
and the significance threshold were inferred from the case material and the data and should be confirmed by Bossard;
and the data cover one season, March to June 2022.""")

    # 19 --------------------------------------------------------------------------------------- next steps
    s = d.slide()
    y = head(s, "NEXT STEPS", "From analysis to pilot")
    cols = [("For Bossard", ["Repair the looping scales; take level 3 off the menu.",
                             "Pilot the rule at 5–10 sites for 8 weeks, against matched control sites.",
                             "Damping test at one of the top coupling sites.",
                             "Confirm with the firmware team: what a level means per scale type, the threshold, "
                             "the 24 h check."]),
            ("For our project (until 27 Nov)", ["Work in the feedback from 23 October.",
                                                "Power BI: sites page and a KPI 2.0 watch-list page.",
                                                "Literature: IoT replenishment (e-Kanban), detector trade-offs.",
                                                "Final deck with full speaker notes; code and report in the repository."]),
            ("Expected results", ["A rule Bossard can switch on centrally – no new hardware.",
                                  f"About {abs(sv6):.0f}–{abs(sv20):.0f}% fewer weighings on normal scales, with "
                                  "about 1 in 9 changes found late – to be confirmed in the pilot.",
                                  f"A daily loop detector: in the data, {N['loop_other']} scales looped outside "
                                  f"level 3 ({pc(N['loop_other_auto'] / N['auto_all'])} of all automatic weighings)."])]
    for i, (t, items) in enumerate(cols):
        x = LM + i * 3.93
        panel(s, x, y, 3.7, 3.9)
        tb(s, x + 0.24, y + 0.2, 3.25, 0.35, t, size=15.5, bold=True, color=ORANGE_D if i == 0 else NAVY)
        tb(s, x + 0.24, y + 0.68, 3.25, 3.1, ["• " + it for it in items], size=12.5, color=INK2, line=1.12,
           space_after=6, bullet_indent=0.16)
    tb(s, LM, y + 4.1, CW, 0.62, "**Team:** Erenalp Çakıroğlu – SmartBin case (data pipeline, analysis, Power BI, "
       "deck) · Nikolaos Georgantopoulos – inventory case · both decks reviewed jointly before submission.",
       size=12.5, color=NAVY, line=1.12)
    source(s, OWN)
    notes(s, f"""
What happens next? For Bossard, four steps. First, repair the looping scales and take level 3 off the menu – this is
quick, cheap and safe. Second, pilot the rule: five to ten sites for eight weeks, each matched with a similar control
site that stays on the current setting, so that the effect can be measured cleanly – the same before/after logic we
used on the data. Third, an on-site damping test at one of the sites with the most coupled empty wake-ups, to find
out whether the coupling travels through the racks. Fourth, three facts to confirm with the firmware team: what a
sensitivity level physically means on each scale type, the exact significance threshold, and the rule of the 24-hour
check. For our project, until the final submission on 27 November: work in the feedback from today; add the sites
page and a KPI 2.0 watch-list page to the Power BI report; deepen the literature on IoT-based replenishment and on
detector trade-offs; and complete the speaker notes, the code and the written report in the repository. The work is
split as follows: Erenalp leads the SmartBin case – data pipeline, analysis, Power BI and this deck – and Nikolaos
leads the inventory case; both decks are reviewed jointly before submission. Expected results: a rule Bossard can
switch on centrally, without new hardware; about {abs(sv6):.0f} to {abs(sv20):.0f}% fewer weighings on normal scales
with roughly one in nine changes found late, to be confirmed in the pilot; and a daily loop detector – in the data,
{N['loop_other']} scales looped outside level 3 and produced {pc(N['loop_other_auto'] / N['auto_all'])} of all
automatic weighings.""")

    # ======================================================================================= APPENDIX
    s = d.slide()
    y = head(s, "APPENDIX A1 · POWER BI (OPTIONAL)", "Data quality at a glance")
    placeholder(s, LM, y, CW, 4.6, "C", "Cleaning funnel and data quality: rows per rule (deleted vs. flagged), rows "
                "and scales per stage, share entering the KPI analysis", "Optional – insert a screenshot of the "
                "Power BI cleaning page here, or delete this slide.")
    source(s, OWN)
    notes(s, "Optional backup: the cleaning funnel as a Power BI page (cleaning_funnel.csv in the Power BI pack).")

    s = d.slide()
    y = head(s, "APPENDIX A2", "Alternatives we tested – and why we rejected them")
    q1 = lambda st: qh.loc[(0.01, st)]
    pw = lambda st, key: sp(pv(st, key, "meas_change_pct_without_loops") / 100)
    rows = [["Alternative", "What we measured", "Result (20 kg / 6 kg)", "Verdict"],
            ["One level for all (Bossard KPIs)", f"best single level: {N['one_setting']['20kg']:.0f} (20 kg), "
             f"{N['one_setting']['6kg']:.0f} (6 kg)", f"misses {pv('20kg', 'P1', 'median_miss_rate') * 100:.0f}% / "
             f"{pv('6kg', 'P1', 'median_miss_rate') * 100:.0f}%", "reject"],
            ["One level for all (KPI 2.0)", f"level {N['one_setting_kpi2']['20kg']:.0f} / "
             f"{N['one_setting_kpi2']['6kg']:.0f}", f"weighings {pw('20kg', 'P5')} / {pw('6kg', 'P5')}; misses "
             f"{pv('20kg', 'P5', 'median_miss_rate') * 100:.0f}% / {pv('6kg', 'P5', 'median_miss_rate') * 100:.0f}%",
             "weaker than the rule"],
            ["Table by pick size", "best level per pick-size class", "same level as one level for all for 95–99% of the "
             "scales", "reject"],
            ["Steer by the fraction (daily)", "fraction rule, start 6",
             f"misses {pv('20kg', 'P8', 'median_miss_rate') * 100:.0f}% / {pv('6kg', 'P8', 'median_miss_rate') * 100:.0f}%",
             "reject (Goodhart)"],
            ["Nightly check for every scale", "+1 weighing per scale and night", f"+7 per scale and week = "
             f"{7 / pv('20kg', 'P7', 'meas_saved_per_scale_week_without_loops') * 100:.0f}–"
             f"{7 / pv('6kg', 'P7', 'meas_saved_per_scale_week_without_loops') * 100:.0f}% of the savings",
             "watch list instead"],
            ["Quiet hours learned per site", "trained Mar–Apr, tested May–Jun",
             f"avoids {pc(q1('20kg').share_empty_avoided)} / {pc(q1('6kg').share_empty_avoided)} of empty wake-ups; "
             f"puts {pc(q1('20kg').share_picks_in_quiet_hours)} / {pc(q1('6kg').share_picks_in_quiet_hours)} of picks "
             "at risk", "reject"],
            ["Longer minimum wait time", "60-min vs. 1-min scales",
             f"{pc(mw.loc[('20kg', 60.0, 30.0), 'share_scales_top2_sites'])} / "
             f"{pc(mw.loc[('6kg', 60.0, 30.0), 'share_scales_top2_sites'])} at two sites, "
             f"{pc(mw.loc[('20kg', 60.0, 30.0), 'share_on_level_1'])} / {pc(mw.loc[('6kg', 60.0, 30.0), 'share_on_level_1'])} "
             "on level 1: not measurable", "out of scope"],
            ["Merge bursts (firmware)", "one weighing per 10-min window",
             f"{sp(-bm.loc[(10.0, '20kg', 'today'), 'saved'])} / {sp(-bm.loc[(10.0, '6kg', 'today'), 'saved'])} alone, "
             f"{sp(-bm.loc[(10.0, '20kg', 'level 7 (simulated)'), 'saved'])} / "
             f"{sp(-bm.loc[(10.0, '6kg', 'level 7 (simulated)'), 'saved'])} on top of level 7; no misses",
             "optional backlog"]]
    table(s, LM + 0.1, y + 0.05, CW - 0.1, [2.55, 2.45, 4.45, 1.55], rows, size=11.5, row_h=0.48)
    source(s, OWN + "; weighings without the scales in a trigger loop")
    notes(s, f"""
Backup for questions. We tested several alternatives to the recommended rule. One level for everybody either misses
too many changes (Bossard's KPIs) or saves less (KPI 2.0). A table by pick-size class ended up with the same level as
one level for all for 95 to 99% of the scales. Steering by the fraction drives scales to level 1. A nightly check for every scale costs a third to
40% of the savings. Quiet hours: for every site we learned the hours of the week that held at most 1% of its real
picks in March and April – a median of about {q1('20kg').quiet_hours_per_week:.0f} hours a week – and tested them on
May and June: they would have avoided only {pc(q1('20kg').share_empty_avoided)} (20 kg) and
{pc(q1('6kg').share_empty_avoided)} (6 kg) of the empty wake-ups while {pc(q1('20kg').share_picks_in_quiet_hours)} and
{pc(q1('6kg').share_picks_in_quiet_hours)} of the real picks would have fallen into them. Empty wake-ups simply follow
the working day. A longer minimum wait time cannot be evaluated: the scales with 60 minutes sit mostly at two sites
and often on level 1, and no scale was observed with both wait times on the same level. Merging bursts – weighing once
at the end of a ten-minute window instead of after every wake-up – would save
{bm.loc[(10.0, '20kg', 'today'), 'saved'] * 100:.0f} to {bm.loc[(10.0, '6kg', 'today'), 'saved'] * 100:.0f}% on its
own and about {bm.loc[(10.0, '6kg', 'level 7 (simulated)'), 'saved'] * 100:.0f} to
{bm.loc[(10.0, '20kg', 'level 7 (simulated)'), 'saved'] * 100:.0f}% on top of level 7, without missing a change; about
{bm.loc[(10.0, '20kg', 'today'), 'picks_reported_later'] * 100:.0f}% of the picks would be reported about five minutes
later. It needs a firmware change, so it is an optional
backlog item.""")

    s = d.slide()
    y = head(s, "APPENDIX A3", "The rule in detail – and where the scales end up")
    card(s, LM, y, 4.6, 4.35, "Every two weeks, per scale",
         "**Measure** over the last 14 days: empty wake-ups per week (TM/wk), share of changes missed "
         "(TOO_FEW / (caught picks + TOO_FEW)) and TOO_FEW per week.\n"
         "**If** missed > 10% **or** TOO_FEW/wk > 2 → one level **up** (more sensitive).\n"
         "**Else if** TM/wk > 7 **and** missed ≤ 5% → one level **down**.\n"
         "**Else** → keep the level.\n"
         "**Start:** level 7. **Range:** levels 1–8 without 3. 6 kg and 20 kg are reported apart.",
         title_size=15, body_size=12.5, body_space=5)
    pic(s, c["final"], 5.75, y - 0.05, w=6.4)
    hi20 = fl.loc[(P['P7'], '20kg', 7.0)] + fl.loc[(P['P7'], '20kg', 8.0)]
    hi6 = fl.loc[(P['P7'], '6kg', 7.0)] + fl.loc[(P['P7'], '6kg', 8.0)]
    tb(s, 5.85, y + 3.45, 6.5, 0.9, f"After 16 weeks: {pc(hi20)} of the 20 kg and {pc(hi6)} of the 6 kg scales are on "
       f"level 7 or 8; almost no scale goes below 4. About one change per scale and month.", size=12, color=INK2,
       line=1.12)
    source(s, OWN + "; simulation, 5 replications")
    notes(s, "Backup: the exact rule and the distribution of levels after 16 simulated weeks. Many 6 kg scales return "
             "to level 8 – they handle small picks and cannot afford a lower setting.")

    s = d.slide()
    y = head(s, "APPENDIX A4", "All policies compared")
    show = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    names = {"P0": "P0 today (level 8, simulated)", "P1": "P1 one level (Bossard KPIs)", "P2": "P2 table by pick size", "P3": "P3 rule for Bossard's KPIs",
             "P4": "P4 best level per scale in hindsight", "P5": "P5 one level (KPI 2.0)",
             "P6": "P6 rule, start 8", "P7": "P7 rule, start 7 (recommended)", "P8": "P8 steer by the fraction"}
    rows = [["Policy", "Weighings 20 kg", "Weighings 6 kg", "Missed 20 kg", "Missed 6 kg", "KPI 2.0 met 20 kg",
             "KPI 2.0 met 6 kg"]]
    for key in show:
        rows.append([names[key], pw("20kg", key), pw("6kg", key),
                     f"{pv('20kg', key, 'median_miss_rate') * 100:.0f}%", f"{pv('6kg', key, 'median_miss_rate') * 100:.0f}%",
                     pc(pv('20kg', key, 'share_in_target_kpi2')), pc(pv('6kg', key, 'share_in_target_kpi2'))])
    table(s, LM + 0.1, y + 0.05, CW - 0.1, [3.4, 1.4, 1.4, 1.3, 1.3, 1.5, 1.5], rows, size=12, row_h=0.4,
          align="lrrrrrr", bold_rows=(show.index("P7") + 1,))
    tb(s, LM, y + 4.2, CW, 0.5, "Weighings: change vs. the simulated status quo, scales without a trigger loop. Missed: "
       "median share of changes missed. KPI 2.0 met: ≤ 7 empty wake-ups a week, ≤ 10% missed, ≤ 2 TOO_FEW a week.",
       size=11.5, color=INK2)
    source(s, OWN)
    notes(s, "Backup table with all simulated policies. P4 uses hindsight and is not implementable; it shows what "
             "even a perfect fixed level per scale would achieve under Bossard's KPIs.")

    s = d.slide()
    y = head(s, "APPENDIX A5", "When missed changes are found")
    gq = N["idle_gap"]
    for i, (v, t) in enumerate([(pc(N["idle_share_manual"]), "of the manual weighings come after ≥ 24 h without "
                                 "any weighing – a scheduled idle check"),
                                (pc(N["idle_share_tf"]), "of all TOO_FEW are found by these idle checks"),
                                (f"{gq[0]:.0f}–{gq[-1]:.0f} h", "after the previous weighing (5th–95th percentile): "
                                 "a missed change can stay invisible for up to two days")]):
        bignum(s, LM + i * 3.93, y - 0.05, 3.6, v, t, size=40, color=[ORANGE, BLUE, NAVY][i], cap_size=13)
    pic(s, c["check"], LM + 1.2, y + 1.85, w=9.0)
    source(s, OWN)
    notes(s, "Backup: the 24-hour idle check is the safety net that makes a lower sensitivity acceptable. A TOO_FEW is "
             "therefore a delay, not a loss – and it is a lower bound of the misses.")

    s = d.slide()
    y = head(s, "APPENDIX A6", "Methods in one page")
    meth = [("Cleaning", "Stage A deletes duplicates (same scale and ms) and impossible weights. Stage B flags "
             "conflicting duplicates, missing label or level, non-standard wait/delay, missing metadata, short "
             "periods, level 3 and rare levels."),
            ("KPIs", "Per scale and level; typical scale = median. Missed = TOO_FEW / (caught picks + TOO_FEW)."),
            ("Causal effect", "Difference-in-differences: switched scales vs. a balanced group that stayed on 8; "
             "fixed-effects Poisson rate ratios; 300 bootstrap samples over scales."),
            ("Coupling", "Real picks at other scales of the site within ±14 s of an empty wake-up vs. a control "
             "window 60–88 s later; beyond chance: a = (p_near − p_chance) / (1 − p_chance)."),
            ("Sites", "Intraclass correlation, permutation F-test (999 shuffles), split-half reliability (200 "
             "random halves); usage-adjusted variant."),
            ("Simulation", "Each scale's level-8 rates (shrunk to the type mean) × causal rate ratios; 16 weeks, "
             "5 replications; back-test on the real switchers.")]
    for i, (t, body) in enumerate(meth):
        x, yy = LM + (i % 2) * 5.85, y + (i // 2) * 1.55
        card(s, x, yy, 5.7, 1.42, t, body, title_size=13.5, body_size=11.5, pad=0.18, gap=0.04)
    source(s, "Angrist & Pischke (2009); Donges et al. (2016); Shrout & Fleiss (1979); Grömping (2006)")
    notes(s, "Backup: the methods behind each slide. All code is in the project repository; every number on the "
             "slides is computed from aggregated tables only.")

    s = d.slide()
    y = head(s, "APPENDIX A7", "Same scales, week by week around their switch")
    pic(s, c["event"], LM - 0.1, y - 0.05, w=7.6)
    tb(s, 8.85, y + 0.1, 3.6, 3.6, ["Empty wake-ups per scale and day collapse in the week of the switch and stay "
       "down – a far larger change than anything before it.", "The weeks before the switch show some extra activity "
       "(Bossard may have switched busy scales first); the difference-in-differences on slide 13 compares with "
       "scales that stayed on 8 over the same weeks."], size=13, color=INK2, line=1.15, space_after=8)
    source(s, OWN + "; event study, switched scales grouped by their new level")
    notes(s, "Backup: event study supporting the difference-in-differences estimates. The bump before the switch is "
             "small compared with the drop, so it cannot explain most of it; a careful reading of the first step is "
             "'roughly half fewer empty wake-ups'.")

    s = d.slide()
    y = head(s, "APPENDIX A8", "Shapley decomposition: the site explains most")
    pic(s, c["shapley"], LM - 0.1, y - 0.05, w=7.4)
    tb(s, 8.85, y + 0.1, 3.6, 3.4, "Share of the variance of the TOO_MANY fraction between periods explained by "
       "each parameter. The setting gets a small share partly because almost all data come from level 8 – its "
       "strength is measured on the same scales before and after, on slide 13.", size=13, color=INK2, line=1.15)
    source(s, "Method: LMG / Shapley R² (Grömping, 2006). " + OWN)
    notes(s, "Backup: variance explained is not the strength of a lever. That is why the main slides use the spread "
             "between groups and the causal effect instead.")

    s = d.slide()
    y = head(s, "APPENDIX A9", "Empty wake-ups follow the working day")
    pic(s, c["hourly"], LM - 0.1, y - 0.05, w=7.6)
    tb(s, 8.85, y + 0.1, 3.6, 3.4, [
        "Empty wake-ups (dashed) and real picks (solid) share the same daily rhythm.",
        f"Weekends: {pc(N['weekend'].loc[('20kg', 'TOO_MANY')], 1)} / {pc(N['weekend'].loc[('6kg', 'TOO_MANY')], 1)} "
        "of the empty wake-ups (20 kg / 6 kg).",
        f"So a night or weekend mode would not help: empty wake-ups go with activity on the site (r = "
        f"{min(N['hourly_r'].values()):.2f})."],
       size=13, color=INK2, line=1.15, space_after=8)
    source(s, OWN + "; scales in a trigger loop left out")
    notes(s, "Backup for the question 'did you check nights and weekends?'.")

    s = d.slide()
    y = head(s, "APPENDIX A10 · REFLECTION", "What we learned")
    refl = [("Surprising", f"The most surprising finding was not about the setting: {N['loop_scales']} looping scales "
             "caused more than a quarter of all automatic weighings in the data."),
            ("Expected", "The setting trades empty wake-ups against misses – but the first step costs almost nothing."),
            ("Hard", "Labels are not ground truth: TOO_FEW is a lower bound and tiny picks are invisible. We had to "
             "estimate the threshold from the data."),
            ("Next time", "Ask for rack positions and the firmware rules first – and design the pilot with control "
             "sites from day one.")]
    for i, (t, body) in enumerate(refl):
        x, yy = LM + (i % 2) * 5.85, y + (i // 2) * 2.05
        card(s, x, yy, 5.7, 1.85, t, body, title_size=15, body_size=13)
    source(s, OWN)
    notes(s, "Reflection for the final documentation: lessons learned, surprises and what we would do differently.")

    s = d.slide()
    y = head(s, "REFERENCES", "Sources")
    tb(s, LM, y - 0.1, CW, 4.9, REFS, size=11, color=NAVY, line=1.08, space_after=5)
    notes(s, "All sources cited on the slides. Case data are confidential and used under the module NDA.")

    d.save(DECK)
    return DECK, N


if __name__ == "__main__":
    path, _ = build()
    print(f"deck written: {path}")
