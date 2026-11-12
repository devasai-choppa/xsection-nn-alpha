"""Small checks for the score baseline, the ranking loss, and checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from xsnn.features import FEATURE_COLUMNS
from xsnn.models.baselines import MomentumReversalScore
from xsnn.models.mlp import MLP, load_mlp, pairwise_rank_loss, train_one_mlp


def test_momentum_score_is_equal_weight_minus_reversal() -> None:
    features = np.zeros((2, len(FEATURE_COLUMNS)))
    mom_cols = [FEATURE_COLUMNS.index(name) for name in ("mom_12_1", "mom_6_1", "mom_3_1")]
    rev = FEATURE_COLUMNS.index("rev_1m")
    features[0, mom_cols] = [1.0, 0.5, 0.0]
    features[0, rev] = 0.25
    features[1, mom_cols] = [-1.0, -1.0, -1.0]
    features[1, rev] = -0.5
    score = MomentumReversalScore().predict(features)
    assert score[0] == np.float64((1.0 + 0.5 + 0.0) / 3.0 - 0.25)
    assert score[1] == np.float64((-1.0 - 1.0 - 1.0) / 3.0 - (-0.5))


def test_pairwise_loss_prefers_the_correct_order() -> None:
    target = torch.tensor([-1.0, -0.2, 0.4, 1.0])
    good = torch.tensor([-0.8, -0.1, 0.2, 0.9])
    bad = torch.tensor([0.9, 0.2, -0.1, -0.8])
    assert float(pairwise_rank_loss(good, target)) < float(pairwise_rank_loss(bad, target))


def test_mlp_activation_widths_and_checkpoint(tmp_path: Path) -> None:
    model = MLP()
    batch = torch.randn(16, 12)
    output, activations = model(batch, return_activations=True)
    assert output.shape == (16,)
    assert [tensor.shape[1] for tensor in activations] == [64, 32, 16]
    dates = np.repeat(np.array(["2020-01-31", "2020-02-29"], dtype="datetime64[ns]"), 16)
    features = np.random.default_rng(0).normal(size=(32, 12))
    target = np.random.default_rng(1).normal(size=32)
    result = train_one_mlp(
        features,
        target,
        dates,
        hidden=(64, 32, 16),
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        rank_lambda=0.2,
        epochs=1,
        seed=0,
        feature_names=FEATURE_COLUMNS,
        window_id="test",
        primary_checkpoint_dir=tmp_path,
    )
    loaded, payload = load_mlp(tmp_path / "epoch_1.pt")
    assert payload["epoch"] == 1
    with torch.no_grad():
        original = result.model(torch.as_tensor(features, dtype=torch.float32))
        restored = loaded(torch.as_tensor(features, dtype=torch.float32))
    assert torch.allclose(original, restored, atol=1e-6)
