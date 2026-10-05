"""Statistics toolbox: bootstrap, Shapley R2 decomposition, ICC, permutation & split-half tests."""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from scipy import stats as st


def boot_ci(x, func=np.median, reps=300, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (reps, len(x)))
    vals = func(x[idx], axis=1)
    return tuple(np.quantile(vals, [alpha / 2, 1 - alpha / 2]))


def summarize(df, by, cols, reps=300, seed=0, min_n=1):
    """Median + bootstrap CI per group for several columns."""
    rows = []
    for key, g in df.groupby(by, observed=True):
        if len(g) < min_n:
            continue
        row = dict(zip(by if isinstance(by, list) else [by], key if isinstance(key, tuple) else (key,)))
        row["n"] = len(g)
        for c in cols:
            row[f"{c}_median"] = g[c].median()
            row[f"{c}_lo"], row[f"{c}_hi"] = boot_ci(g[c], reps=reps, seed=seed)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- Shapley R2
def _design(df, factors, numeric=()):
    """One-hot blocks per factor (drop first level). Returns X (with intercept) and column index per factor."""
    blocks, index, start = [np.ones((len(df), 1))], {}, 1
    for f in factors:
        if f in numeric:
            v = df[f].astype(float).values
            b = ((v - v.mean()) / (v.std() or 1)).reshape(-1, 1)
        else:
            b = pd.get_dummies(df[f].astype(str), drop_first=True, dtype=float).values
        blocks.append(b)
        index[f] = list(range(start, start + b.shape[1]))
        start += b.shape[1]
    return np.hstack(blocks), index


def _gram(d, y, factors, weights=None, numeric=(), chunk=40_000):
    """X'WX and X'Wy accumulated in row chunks -> memory stays small even with 180 location dummies."""
    codes, idx, start = {}, {}, 1
    for f in factors:
        if f in numeric:
            v = d[f].astype(float).values
            codes[f] = ((v - v.mean()) / (v.std() or 1))
            idx[f] = [start]
            start += 1
        else:
            c = pd.Categorical(d[f].astype(str)).codes
            nlev = int(c.max()) + 1 if len(c) else 1
            codes[f] = c
            idx[f] = list(range(start, start + nlev - 1))
            start += nlev - 1
    k = start
    yv = d[y].astype(float).values
    w = np.ones(len(d)) if weights is None else d[weights].astype(float).values
    G, b = np.zeros((k, k)), np.zeros(k)
    for lo in range(0, len(d), chunk):
        hi = min(len(d), lo + chunk)
        X = np.zeros((hi - lo, k))
        X[:, 0] = 1.0
        for f in factors:
            if f in numeric:
                X[:, idx[f][0]] = codes[f][lo:hi]
            elif idx[f]:
                c = codes[f][lo:hi]
                m = c > 0
                X[np.nonzero(m)[0], idx[f][0] + c[m] - 1] = 1.0
        Xw = X * w[lo:hi, None]
        G += X.T @ Xw
        b += Xw.T @ yv[lo:hi]
    return G, b, idx


def shapley_r2(df, y, factors, weights=None, numeric=()):
    """
    Fair split of explained variance (R2) among correlated predictors (LMG / Shapley).
    For p factors all 2^p sub-models are fitted via the normal equations on a precomputed Gram matrix.
    Returns DataFrame: factor, shapley_r2 (sums to full-model R2), share (sums to 1).
    """
    d = df.dropna(subset=[y] + list(factors))
    G, b, idx = _gram(d, y, factors, weights, numeric)
    yv = d[y].astype(float).values
    w = np.ones(len(d)) if weights is None else d[weights].astype(float).values
    ybar = np.average(yv, weights=w)
    sst = np.sum(w * (yv - ybar) ** 2)
    yy = np.sum(w * yv ** 2)
    p = len(factors)
    cache = {}

    def r2(subset):
        key = tuple(sorted(subset))
        if key in cache:
            return cache[key]
        cols = [0] + [c for f in key for c in idx[factors[f]]]
        beta = np.linalg.lstsq(G[np.ix_(cols, cols)], b[cols], rcond=None)[0]
        sse = yy - beta @ b[cols]
        cache[key] = max(0.0, 1 - sse / sst) if sst > 0 else 0.0
        return cache[key]

    shap = np.zeros(p)
    for j in range(p):
        others = [i for i in range(p) if i != j]
        for k in range(p):
            wk = factorial(k) * factorial(p - k - 1) / factorial(p)
            for S in combinations(others, k):
                shap[j] += wk * (r2(S + (j,)) - r2(S))
    full = r2(tuple(range(p)))
    out = pd.DataFrame({"factor": factors, "shapley_r2": shap})
    out["share"] = out.shapley_r2 / full if full > 0 else np.nan
    out.attrs["r2_full"] = full
    out.attrs["n"] = len(d)
    return out.sort_values("shapley_r2", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------- group effects
def icc_oneway(values, groups):
    """ICC(1): share of variance that sits BETWEEN groups (0 = groups identical, 1 = all variance is group)."""
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    k = d.groupby("g").v.size()
    d = d[d.g.isin(k[k >= 2].index)]
    k = d.groupby("g").v.size()
    a, n = len(k), len(d)
    if a < 2:
        return np.nan
    gm = d.groupby("g").v.mean()
    msb = (k * (gm - d.v.mean()) ** 2).sum() / (a - 1)
    msw = ((d.v - d.g.map(gm)) ** 2).sum() / (n - a)
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def permutation_test_groups(values, groups, n_perm=999, seed=0):
    """F-statistic of a one-way layout; p-value by shuffling group labels across units."""
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    codes, _ = pd.factorize(d.g)
    v = d.v.values

    def f_stat(c):
        cnt = np.bincount(c)
        mean = np.bincount(c, weights=v) / cnt
        ssb = np.sum(cnt * (mean - v.mean()) ** 2)
        ssw = np.sum((v - mean[c]) ** 2)
        a, n = len(cnt), len(v)
        return (ssb / (a - 1)) / (ssw / (n - a))

    obs = f_stat(codes)
    rng = np.random.default_rng(seed)
    perm = np.array([f_stat(rng.permutation(codes)) for _ in range(n_perm)])
    return float(obs), float((1 + np.sum(perm >= obs)) / (n_perm + 1))


def split_half_reliability(values, groups, reps=200, seed=0, min_per_half=2):
    """
    Split the devices of every location randomly in two halves and rank locations in each half.
    If the location effect is real, the two rankings agree (Spearman r high). Returns mean r,
    Spearman-Brown corrected reliability, and one example split for plotting.
    """
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    rng = np.random.default_rng(seed)
    rs, example = [], None
    for i in range(reps):
        half = rng.random(len(d)) < 0.5
        a = d[half].groupby("g").v.agg(["median", "size"])
        b = d[~half].groupby("g").v.agg(["median", "size"])
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j.size_a >= min_per_half) & (j.size_b >= min_per_half)]
        if len(j) < 5:
            continue
        rs.append(st.spearmanr(j.median_a, j.median_b).statistic)
        if example is None:
            example = j[["median_a", "median_b"]].reset_index()
    if not rs:
        return np.nan, np.nan, None
    r = float(np.nanmean(rs))
    return r, (2 * r / (1 + r)) if r > -1 else np.nan, example


def logistic_fit(x, y, iters=50):
    """2-parameter logistic regression by Newton-Raphson: P(y=1) = 1 / (1 + exp(-(a + b*x)))."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - p)
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion (stable for small n and rates near 0 or 1)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        d = 1 + z ** 2 / n
        c = (p + z ** 2 / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h
