import numpy as np
import pandas as pd

from smartbin.stats import icc_oneway, logistic_fit, permutation_test_groups, shapley_r2


def test_shapley_sums_to_full_r2_and_finds_the_real_driver():
    rng = np.random.default_rng(0)
    n = 4000
    d = pd.DataFrame({"a": rng.integers(0, 5, n), "b": rng.integers(0, 3, n), "noise": rng.integers(0, 4, n)})
    d["y"] = 0.5 * d.a + 0.1 * d.b + rng.normal(0, 1, n)
    res = shapley_r2(d, "y", ["a", "b", "noise"])
    assert abs(res.shapley_r2.sum() - res.attrs["r2_full"]) < 1e-9
    assert res.iloc[0].factor == "a"
    assert res.set_index("factor").shapley_r2["noise"] < 0.005


def test_shapley_splits_collinear_factors_fairly():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.integers(0, 4, n)
    d = pd.DataFrame({"x1": x, "x2": x, "y": x + rng.normal(0, 1, n)})   # x2 is an exact copy
    res = shapley_r2(d, "y", ["x1", "x2"]).set_index("factor")
    assert abs(res.shapley_r2["x1"] - res.shapley_r2["x2"]) < 1e-6


def test_group_tests_detect_real_effect_and_ignore_noise():
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(30), 20)
    real = rng.normal(0, 1, 30)[g] + rng.normal(0, 1, len(g))
    noise = rng.normal(0, 1, len(g))
    assert icc_oneway(real, g) > 0.3
    assert abs(icc_oneway(noise, g)) < 0.1
    assert permutation_test_groups(real, g, n_perm=199)[1] < 0.01
    assert permutation_test_groups(noise, g, n_perm=199)[1] > 0.01


def test_logistic_fit_recovers_parameters():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 2, 20000)
    p = 1 / (1 + np.exp(-(-1.0 + 1.5 * x)))
    y = rng.random(len(x)) < p
    a, b = logistic_fit(x, y)
    assert abs(a + 1.0) < 0.1 and abs(b - 1.5) < 0.1
