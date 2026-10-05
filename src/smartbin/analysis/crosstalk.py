"""
M6 - Crosstalk: does a pick at one bin trigger the sensors of its NEIGHBOURS?

Hypothesis: a TOO_MANY at scale A is often caused by a real pick at scale B on the same rack -
the vibration travels through the shelf. If true, real picks at other scales of the same location
cluster in time exactly around A's empty measurement.

Method (event-coincidence analysis, as used for spike trains in neuroscience):
1. Sources = real picks (AUTOMATIC + OK) at OTHER scales of the same location.
   Targets = every automatic measurement (TOO_MANY and, as comparison, OK).
2. Cross-correlogram: count sources at every time lag (-120 s ... +120 s) around each target.
   Flat = no link. A sharp peak at lag ~ 0 = neighbour picks and empty measurements happen together.
   (Both sensors wait the same 30 s delay before weighing, so the peak sits at 0.)
3. Chance level from a control band of equal width 60+ s away (same time of day, same activity):
       attributable share = (P(neighbour pick within +-w) - P(chance)) / (1 - P(chance))
4. Root causes of TOO_MANY: neighbour-induced | own echo (bin touched again <= 5 min after own pick) | other.
5. Virtual racks: scale pairs that co-trigger far above chance are physically close. Community detection
   on that graph recovers rack-like groups - information that does not exist in the metadata.

Needs complete event chains -> skipped in sample mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .. import viz
from . import ModuleResult


def run(con, cfg, sample: bool, location_table: pd.DataFrame | None = None) -> ModuleResult:
    r = ModuleResult("m6_crosstalk", "Crosstalk - do neighbours trigger each other?",
                     "How many empty measurements are caused by picks at neighbouring bins?")
    if sample:
        r.status = "skipped"
        r.notes.append("Sample mode: random rows have no complete timelines; crosstalk needs the full file.")
        return r
    c = cfg.crosstalk
    W, bin_s = int(c.window_s), int(c.bin_s)
    std = ", ".join(str(float(x)) for x in cfg.cleaning.standard_delay)
    d0 = float(cfg.cleaning.standard_delay[0])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE xt_src AS
        SELECT location_id, deviceid AS src_dev, ts AS src_ts, CAST(floor(ts / {W}) AS BIGINT) AS b
        FROM events
        WHERE is_auto AND cls = 'OK' AND location_id IS NOT NULL AND coalesce(delay, {d0}) IN ({std})
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE xt_src3 AS
        SELECT s.location_id, s.src_dev, s.src_ts, s.b + o.o AS b
        FROM xt_src s CROSS JOIN (SELECT unnest([-1, 0, 1]) AS o) o
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE xt_tgt AS
        SELECT rid, location_id, location_label, deviceid, ts, cls, sensitivity, scale_type,
               (prev_initiator = 'AUTOMATIC' AND prev_cls = 'OK' AND gap_s <= {c.echo_window_s}) AS echo,
               gap_s, prev_cls, CAST(floor(ts / {W}) AS BIGINT) AS b
        FROM events
        WHERE is_auto AND cls IN ('OK', 'TOO_MANY') AND exclusion_reason IS NULL
          AND location_id IS NOT NULL AND coalesce(delay, {d0}) IN ({std})
    """)
    join = f"""FROM xt_tgt t JOIN xt_src3 s ON t.location_id = s.location_id AND t.b = s.b
               WHERE s.src_dev <> t.deviceid AND abs(t.ts - s.src_ts) <= {W}"""

    # 1) cross-correlogram -------------------------------------------------------------------
    hist = con.execute(f"""
        SELECT t.scale_type, t.cls, CAST(floor((s.src_ts - t.ts) / {bin_s}) AS INTEGER) * {bin_s} AS lag_s,
               count(*) AS n {join} GROUP BY ALL
    """).df()
    ntg = con.execute("SELECT scale_type, cls, count(*) AS n_targets FROM xt_tgt GROUP BY ALL").df()
    if hist.empty:
        r.status = "skipped"
        r.notes.append("No neighbour events found (locations missing?).")
        return r
    hist = hist.merge(ntg, on=["scale_type", "cls"])
    hist["per_min"] = hist.n / hist.n_targets / bin_s * 60      # neighbour picks per target per minute of lag
    base = (hist[hist.lag_s.abs() >= W / 2].groupby(["scale_type", "cls"]).per_min.mean().rename("baseline"))
    hist = hist.merge(base.reset_index(), on=["scale_type", "cls"])
    hist["excess"] = hist.per_min - hist.baseline
    r.tables["correlogram"] = hist.sort_values(["scale_type", "cls", "lag_s"])

    near, peak_found = _near_window(hist, c)
    r.facts["near_window_s"] = near
    r.facts["peak_found"] = peak_found
    c0 = float(c.control_offset_s)

    # 2) per-target flags ---------------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE xt_flags AS
        WITH f AS (
            SELECT t.rid,
                   bool_or(abs(t.ts - s.src_ts) <= {near}) AS near,
                   bool_or((s.src_ts - t.ts) BETWEEN {c0} AND {c0 + 2 * near}) AS ctrl
            {join} GROUP BY t.rid)
        SELECT t.*, coalesce(f.near, false) AS near, coalesce(f.ctrl, false) AS ctrl
        FROM xt_tgt t LEFT JOIN f USING (rid)
    """)
    agg = con.execute("""
        SELECT scale_type, cls, sensitivity, count(*) AS n,
               sum(near::INT) AS n_near, sum(ctrl::INT) AS n_ctrl, sum(echo::INT) AS n_echo,
               sum((echo AND NOT near)::INT) AS n_echo_only
        FROM xt_flags GROUP BY ALL
    """).df()
    agg = _attributable(agg)
    r.tables["attributable_by_level"] = agg.sort_values(["scale_type", "cls", "sensitivity"])

    b = float(cfg.analysis.baseline_sensitivity)
    tm8 = agg[(agg.cls == "TOO_MANY") & (agg.sensitivity == b)]
    ok8 = agg[(agg.cls == "OK") & (agg.sensitivity == b)]
    decomp = []
    for _, row in tm8.iterrows():
        f = max(0.0, row.attributable)
        echo = row.n_echo_only / row.n
        decomp.append(dict(scale_type=row.scale_type, neighbour_induced=f, own_echo=echo,
                           other=max(0.0, 1 - f - echo), n_too_many=row.n, p_near=row.p_near, p_chance=row.p_ctrl))
    decomp = pd.DataFrame(decomp)
    r.tables["root_causes_baseline"] = decomp

    # 3) per location: does crosstalk explain the factory differences? -------------------------
    loc = con.execute(f"""
        SELECT scale_type, location_label, count(*) AS n, sum(near::INT) AS n_near, sum(ctrl::INT) AS n_ctrl,
               count(DISTINCT deviceid) AS devices
        FROM xt_flags WHERE cls = 'TOO_MANY' AND sensitivity = {b} GROUP BY ALL
    """).df()
    loc = _attributable(loc.assign(cls="TOO_MANY"))
    expl = []
    if location_table is not None and len(location_table):
        loc = loc.merge(location_table[["scale_type", "location_label", "frac_too_many_median"]],
                        on=["scale_type", "location_label"], how="inner")
        for scale, g in loc.groupby("scale_type"):
            g = g[g.devices >= cfg.analysis.min_devices_location]
            if len(g) >= 5:
                rho = st.spearmanr(g.attributable, g.frac_too_many_median).statistic
                lin = st.linregress(g.attributable, g.frac_too_many_median)
                expl.append(dict(scale_type=scale, locations=len(g), spearman=rho, r2=lin.rvalue ** 2,
                                 slope=lin.slope))
                r.figures[f"location_link_{scale}"] = _loc_chart(g, scale, rho, cfg, r)
    r.tables["by_location"] = loc
    r.tables["location_explained"] = pd.DataFrame(expl)

    # 4) echo: time since own previous measurement -----------------------------------------------
    gaps = con.execute(f"""
        SELECT scale_type, cls, CAST(floor(gap_s / 10) AS INTEGER) * 10 AS gap_bin, count(*) AS n
        FROM xt_tgt WHERE gap_s <= 600 AND sensitivity = {b} GROUP BY ALL
    """).df()
    r.tables["gap_histogram"] = gaps

    # 5) virtual racks ------------------------------------------------------------------------------
    if c.infer_racks:
        try:
            racks, edges, fig = _infer_racks(con, join, near, c0, c, cfg, r)
            r.tables["virtual_racks"] = racks
            r.tables["rack_edges"] = edges
            if fig:
                r.figures["virtual_racks"] = fig
            if len(racks):
                r.facts["virtual_racks"] = {
                    "devices_clustered": int(racks.deviceid.nunique()),
                    "clusters": int(racks.cluster.nunique()),
                    "median_cluster_size": float(racks.groupby("cluster").size().median())}
        except Exception as e:  # optional module, never kill the run
            r.notes.append(f"virtual rack inference skipped: {e}")

    # Is the coupling SPECIFIC to empty measurements (mechanical crosstalk) or does it also surround real picks
    # (people working several bins in a row = activity coupling)?
    spec = {}
    for scale in tm8.scale_type.unique():
        tm_a = tm8[tm8.scale_type == scale].attributable.iloc[0]
        ok_a = ok8[ok8.scale_type == scale].attributable.iloc[0] if len(ok8[ok8.scale_type == scale]) else np.nan
        spec[scale] = tm_a / ok_a if ok_a and ok_a > 0 else np.nan
        r.facts.setdefault(scale, {})["tm_vs_ok_coupling_ratio"] = round(float(spec[scale]), 2)
        r.facts[scale]["mechanism"] = ("crosstalk-specific" if spec[scale] >= 1.5 else "activity-coupled")
    for scale in sorted(hist.scale_type.dropna().unique()):
        r.figures[f"correlogram_{scale}"] = _correlogram_chart(hist[hist.scale_type == scale], near, scale, cfg, r,
                                                               spec.get(scale, np.nan))
    if len(agg):
        r.figures["attributable_by_level"] = _level_chart(agg, cfg, r)
    if len(decomp):
        r.figures["root_causes"] = _decomp_chart(decomp, cfg, r)
    for _, row in decomp.iterrows():
        r.facts.setdefault(row.scale_type, {}).update({"neighbour_induced_share": round(row.neighbour_induced, 3),
                                                       "own_echo_share": round(row.own_echo, 3),
                                                       "other_share": round(row.other, 3)})
    for e in expl:
        r.facts.setdefault(e["scale_type"], {})["location_r2_explained"] = round(e["r2"], 3)
    ok = ok8.set_index("scale_type").attributable.to_dict() if len(ok8) else {}
    for scale, v in ok.items():
        r.facts.setdefault(scale, {})["co_picking_share_ok"] = round(v, 3)
    r.notes.append(f"Near window +-{near:.0f} s ({'detected from the correlogram' if peak_found else 'default, no clear peak'}); "
                   f"chance level from a {2 * near:.0f} s band starting {c0:.0f} s away.")
    return r


def _attributable(agg):
    agg = agg.copy()
    agg["p_near"] = agg.n_near / agg.n
    agg["p_ctrl"] = agg.n_ctrl / agg.n
    agg["attributable"] = ((agg.p_near - agg.p_ctrl) / (1 - agg.p_ctrl)).clip(lower=-1, upper=1)
    return agg


def _near_window(hist, c):
    if c.near_s:
        return float(c.near_s), True
    h = hist[hist.cls == "TOO_MANY"].groupby("lag_s").apply(
        lambda g: np.average(g.excess, weights=g.n_targets), include_groups=False)
    bin_s = float(c.bin_s)

    def cum(L):
        return float((h[(h.index >= -L) & (h.index < L)] * bin_s).sum())
    ref = cum(30)
    if ref <= 0 or h.max() <= 0:
        return 6.0, False
    for L in [2, 4, 6, 8, 10, 14, 20, 30]:
        if cum(L) >= 0.8 * ref:
            return float(L), True
    return 30.0, True


def _infer_racks(con, join, near, c0, c, cfg, r):
    import networkx as nx
    pairs = con.execute(f"""
        SELECT t.location_label, t.deviceid AS a, s.src_dev AS b,
               count(*) FILTER (WHERE abs(t.ts - s.src_ts) <= {near}) AS n_near,
               count(*) FILTER (WHERE (s.src_ts - t.ts) BETWEEN {c0} AND {c0 + 2 * near}) AS n_ctrl
        {join} AND t.cls = 'TOO_MANY' GROUP BY ALL HAVING n_near >= {int(c.rack_min_near)}
    """).df()
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    pairs["lift"] = (pairs.n_near + 1) / (pairs.n_ctrl + 1)
    pairs = pairs[pairs.lift >= c.rack_min_lift]
    key = np.where(pairs.a < pairs.b, pairs.a + "|" + pairs.b, pairs.b + "|" + pairs.a)
    pairs = pairs.assign(key=key, w=pairs.n_near - pairs.n_ctrl)
    und = pairs.groupby(["location_label", "key"]).w.sum().reset_index()
    und[["a", "b"]] = und.key.str.split("|", expand=True)
    rows = []
    for loc, g in und.groupby("location_label"):
        G = nx.Graph()
        G.add_weighted_edges_from(g[["a", "b", "w"]].itertuples(index=False, name=None))
        comms = nx.community.louvain_communities(G, weight="weight", seed=cfg.analysis.random_seed)
        for i, cm in enumerate(comms):
            if len(cm) >= 2:
                rows.extend(dict(location_label=loc, cluster=f"{loc}-R{i + 1:02d}", deviceid=d) for d in cm)
    racks = pd.DataFrame(rows)
    fig = _rack_chart(und, racks, cfg, r) if len(racks) else None
    return racks, und.drop(columns="key"), fig


# ------------------------------------------------------------------------------------- figures
def _correlogram_chart(h, near, scale, cfg, r, spec=np.nan):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.axvspan(-near, near, color=viz.GRID, alpha=0.6, lw=0)
    for cls, color, lab in [("TOO_MANY", viz.CAT[1], "around an EMPTY measurement (TOO_MANY)"),
                            ("OK", viz.CAT[0], "around a REAL pick (OK) - co-picking reference")]:
        g = h[h.cls == cls].sort_values("lag_s")
        ax.plot(g.lag_s + cfg.crosstalk.bin_s / 2, g.per_min, color=color, lw=1.6, label=lab)
    ax.set_xlabel("Time of the neighbour's real pick relative to the measurement (s)")
    ax.set_ylabel("Neighbour picks per measurement\n(per minute of lag)")
    if np.isfinite(spec) and spec < 1.5:
        ax.set_title(f"{scale}: empty measurements happen when people work at the rack - same pattern as real picks")
    else:
        ax.set_title(f"{scale}: neighbour picks pile up at the moment of an empty measurement")
    viz.subtitle(ax, f"gray band = +-{near:.0f} s 'near' window · flat far ends = chance level")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    return viz.save(fig, r.fig_path(cfg.paths.figures, "correlogram", scale))


def _level_chart(agg, cfg, r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    d = agg[agg.cls == "TOO_MANY"]
    for scale, g in d.groupby("scale_type"):
        g = g.sort_values("sensitivity")
        ax.plot(g.sensitivity, g.attributable * 100, "-o", color=viz.SCALE_COLOR.get(scale, viz.CAT[2]),
                label=scale, mec=viz.SURFACE, mew=1.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("% of TOO_MANY caused by neighbours")
    ax.set_title("The more sensitive the scale, the more it reacts to its neighbours")
    ax.legend()
    return viz.save(fig, r.fig_path(cfg.paths.figures, "attributable_by_level"))


def _decomp_chart(decomp, cfg, r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 2.2 + 0.5 * len(decomp)))
    parts = [("neighbour_induced", "real pick at another scale within the near window (beyond chance)", viz.CAT[1]),
             ("own_echo", "own echo (touched again <= 5 min)", viz.CAT[3]),
             ("other", "other / ambient / unexplained", viz.GRAY_BAR)]
    y = np.arange(len(decomp))
    left = np.zeros(len(decomp))
    for col, lab, color in parts:
        v = decomp[col].values * 100
        ax.barh(y, v, left=left, color=color, height=0.5, label=lab, edgecolor=viz.SURFACE, linewidth=2)
        for yi, (l, vi) in enumerate(zip(left, v)):
            if vi >= 7:
                ax.text(l + vi / 2, yi, f"{vi:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if color != viz.GRAY_BAR else viz.INK)
        left += v
    ax.set_yticks(y, decomp.scale_type)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of TOO_MANY measurements at sensitivity {cfg.analysis.baseline_sensitivity}")
    ax.set_title("Where do the empty measurements come from?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=8)
    ax.grid(axis="y", visible=False)
    return viz.save(fig, r.fig_path(cfg.paths.figures, "root_causes"))


def _loc_chart(g, scale, rho, cfg, r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(g.attributable * 100, g.frac_too_many_median, s=np.clip(g.devices, 10, 200),
               color=viz.SCALE_COLOR.get(scale, viz.CAT[0]), alpha=0.7, edgecolor=viz.SURFACE, linewidth=1)
    ax.set_xlabel("% of TOO_MANY caused by neighbours (per location)")
    ax.set_ylabel("Location median fraction TOO_MANY")
    ax.set_title(f"{scale}: crosstalk explains why factories differ")
    viz.subtitle(ax, f"Spearman rho = {rho:.2f} · dot size = scales")
    return viz.save(fig, r.fig_path(cfg.paths.figures, "location_link", scale))


def _rack_chart(und, racks, cfg, r):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    sizes = racks.groupby("location_label").deviceid.nunique()
    sizes = sizes[sizes <= 160]
    if sizes.empty:
        return None
    loc = sizes.idxmax()
    rk = racks[racks.location_label == loc].sort_values(["cluster", "deviceid"])
    order = rk.deviceid.tolist()
    idx = {d: i for i, d in enumerate(order)}
    M = np.zeros((len(order), len(order)))
    for row in und[und.location_label == loc].itertuples():
        if row.a in idx and row.b in idx:
            M[idx[row.a], idx[row.b]] = M[idx[row.b], idx[row.a]] = max(row.w, 0)
    cmap = LinearSegmentedColormap.from_list("seq", [viz.SURFACE] + viz.BLUE_RAMP[2:])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.imshow(np.log1p(M), cmap=cmap, interpolation="nearest")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(viz.AXIS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"Location {loc}: hidden rack structure from timing alone")
    viz.subtitle(ax, f"{len(order)} scales · {rk.cluster.nunique()} inferred groups · darker = co-trigger more")
    return viz.save(fig, r.fig_path(cfg.paths.figures, "virtual_racks"))
