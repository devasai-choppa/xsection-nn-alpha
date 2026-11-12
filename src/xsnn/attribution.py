"""Feature attribution that does not enter model selection.

Permutation importance is an out-of-sample explanation: each feature is shuffled
inside its month and the drop in Spearman IC is recorded. Integrated gradients
for the dashboard are computed separately and are not used to pick hyperparameters.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from xsnn.metrics import mean_ic
from xsnn.models.mlp import pd_period_codes, pd_unique


def permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    features: np.ndarray,
    realized: np.ndarray,
    dates: np.ndarray,
    feature_names: list[str],
    *,
    seed: int = 0,
) -> pd.Series:
    """Mean rank-IC drop from shuffling one feature at a time within each month.

    The shuffle preserves each date's marginal distribution and destroys only
    the cross-sectional alignment of that feature with the names.
    """
    baseline = mean_ic(predict_fn(features), realized, dates, method="spearman")
    rng = np.random.default_rng(seed)
    codes = pd_period_codes(dates)
    drops: list[float] = []
    for column in range(features.shape[1]):
        shuffled = features.copy()
        for code in pd_unique(codes):
            idx = np.flatnonzero(codes == code)
            if len(idx) < 2:
                continue
            shuffled[idx, column] = features[rng.permutation(idx), column]
        perturbed = mean_ic(predict_fn(shuffled), realized, dates, method="spearman")
        if np.isfinite(baseline) and np.isfinite(perturbed):
            drops.append(float(baseline - perturbed))
        else:
            drops.append(float("nan"))
    return pd.Series(drops, index=feature_names, name="ic_drop")


def integrated_gradients_for_ticker(
    model,
    features: np.ndarray,
    feature_names: list[str],
    *,
    device: str | None = None,
    n_steps: int = 32,
) -> pd.Series:
    """Captum integrated gradients of the scalar score with respect to inputs.

    The baseline is the zero vector, which is the cross-sectional median after
    rank normalization. Import of captum is local so the core package stays
    usable without the visualization extra.
    """
    import torch
    from captum.attr import IntegratedGradients

    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32)
    if device is not None:
        model.to(device)
        tensor = tensor.to(device)
    def _forward(inputs: torch.Tensor) -> torch.Tensor:
        scores = model(inputs)
        if isinstance(scores, tuple):
            scores = scores[0]
        return scores.unsqueeze(-1)

    explainer = IntegratedGradients(_forward)
    attributions = explainer.attribute(tensor, baselines=torch.zeros_like(tensor), n_steps=n_steps)
    values = attributions.detach().cpu().numpy()
    if values.ndim == 2:
        values = values[0]
    return pd.Series(values, index=feature_names, name="integrated_gradient")
