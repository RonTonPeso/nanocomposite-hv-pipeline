import numpy as np
import pandas as pd

from nanocomposite_hardness.io.units import hardness_to_hv, normalize_hardness_column


def test_hardness_to_hv_identity():
    assert hardness_to_hv(150.0, "hv") == 150.0


def test_gpa_conversion():
    v = hardness_to_hv(1.0, "gpa", gpa_to_hv=100.0)
    assert abs(v - 100.0) < 1e-9


def test_normalize_column():
    df = pd.DataFrame(
        {
            "hardness_value": [100.0, 1.0],
            "hardness_unit": ["hv", "gpa"],
        }
    )
    out = normalize_hardness_column(df, gpa_to_hv=100.0)
    assert np.isclose(out["hv"].iloc[0], 100.0)
    assert np.isclose(out["hv"].iloc[1], 100.0)
