import math

import numpy as np

from nanocomposite_hardness.uncertainty.conformal import (
    conformal_quantile,
    coverage_and_width,
)


def test_conformal_quantile_finite_sample_level():
    # For n=9, alpha=0.1: level = ceil(10*0.9)/9 = 9/9 = 1.0 -> the max residual.
    r = np.arange(1, 10, dtype=float)  # 1..9
    q = conformal_quantile(r, alpha=0.1)
    assert q == 9.0

    # For n=19, alpha=0.1: level = ceil(20*0.9)/19 = 18/19 -> 'higher' picks the 18th of 19.
    r = np.arange(1, 20, dtype=float)  # 1..19
    q = conformal_quantile(r, alpha=0.1)
    assert q == 18.0


def test_conformal_quantile_unbounded_when_too_few_points():
    # n=5, alpha=0.1: level = ceil(6*0.9)/5 = 6/5 > 1 -> unbounded.
    q = conformal_quantile(np.arange(1, 6, dtype=float), alpha=0.1)
    assert math.isinf(q)


def test_split_conformal_achieves_nominal_coverage_on_exchangeable_data():
    # Exchangeable residuals: calibrate on one draw, test on a fresh draw, average over trials.
    rng = np.random.default_rng(0)
    alpha = 0.1
    n_cal, n_test, trials = 500, 2000, 40
    coverages = []
    for _ in range(trials):
        cal_res = np.abs(rng.normal(size=n_cal))
        q = conformal_quantile(cal_res, alpha)
        # fresh test points: prediction error drawn from the same distribution
        err = rng.normal(size=n_test)
        cov = coverage_and_width(err, -q * np.ones(n_test), q * np.ones(n_test))
        coverages.append(cov["coverage"])
    mean_cov = float(np.mean(coverages))
    # Conformal guarantees >= nominal in expectation; allow a small band around 0.90.
    assert 0.88 <= mean_cov <= 0.95


def test_coverage_and_width_basic():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lower = np.array([-1.0, 0.0, 5.0, 2.5])
    upper = np.array([1.0, 2.0, 6.0, 3.5])
    out = coverage_and_width(y, lower, upper)
    assert out["coverage"] == 0.75  # third point (2.0) not in [5,6]
    # widths are [2, 2, 1, 1] -> mean 1.5
    assert np.isclose(out["mean_width"], 1.5)
