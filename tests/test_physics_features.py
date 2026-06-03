import math

import numpy as np

from nanocomposite_hardness.features.physics import (
    cte_mismatch_dislocation_proxy,
    hall_petch_feature,
    orowan_stress_mpa,
)


def test_hall_petch_decreases_with_grain_size():
    a = hall_petch_feature(100.0)
    b = hall_petch_feature(400.0)
    assert a > b > 0


def test_orowan_increases_with_volume_fraction():
    s1 = orowan_stress_mpa(50.0, 0.02, 26.0)
    s2 = orowan_stress_mpa(50.0, 0.08, 26.0)
    assert s2 > s1 > 0


def test_orowan_decreases_with_particle_size():
    s1 = orowan_stress_mpa(30.0, 0.05, 26.0)
    s2 = orowan_stress_mpa(90.0, 0.05, 26.0)
    assert s1 > s2 > 0


def test_cte_proxy_scales():
    p = cte_mismatch_dislocation_proxy(5e-6, 300.0, 0.05, 40.0)
    assert math.isfinite(p) and p > 0
    p2 = cte_mismatch_dislocation_proxy(5e-6, 300.0, 0.10, 40.0)
    assert p2 > p


def test_missing_grain_returns_zero():
    assert hall_petch_feature(None) == 0.0
    assert orowan_stress_mpa(None, 0.05, 26.0) == 0.0
