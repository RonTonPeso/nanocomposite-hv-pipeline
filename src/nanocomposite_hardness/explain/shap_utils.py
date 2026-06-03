"""SHAP utilities for tree and ensemble models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap


def shap_summary_bar(
    model,
    X: np.ndarray,
    feature_names: list[str],
    out_path: str | Path,
    *,
    max_display: int = 25,
) -> None:
    """Write a SHAP summary bar plot (mean |SHAP|)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv,
        X,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
