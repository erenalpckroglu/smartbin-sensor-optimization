"""
Numbers and sources for the CTO deck.

facts() reads the aggregated tables in report/data/ (written by report/build_figures.py) and returns every number
the deck needs; REFS is the reference list shown on the last slide.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "report" / "data"

SCALES = ["20kg", "6kg"]


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



REFS = [
    "Anastasi, G., Conti, M., Di Francesco, M., & Passarella, A. (2009). Energy conservation in wireless sensor "
    "networks: A survey. Ad Hoc Networks, 7(3), 537–568. https://doi.org/10.1016/j.adhoc.2008.06.003",
    "Angrist, J. D., & Pischke, J.-S. (2009). Mostly harmless econometrics: An empiricist's companion. "
    "Princeton University Press.",
    "Bossard AG. (n.d.). SmartBin Cloud. Retrieved 6 October 2026, from "
    "https://www.bossard.com/ch-en/smart-factory-logistics/systems/smartbin-cloud/",
    "Bossard AG & D ONE Solutions AG. (2026). Smart Factory Logistics: SmartBin Cloud [Lecture slides, case "
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
