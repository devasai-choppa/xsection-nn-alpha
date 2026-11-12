"""MLP ranker: 12 → 64 → 32 → 16 → 1, SiLU, batch norm, dropout.

The training target is the cross-sectional rank of next-month return mapped to
[-1, 1]. The loss is mean squared error plus an optional pairwise logistic
ranking penalty computed inside each month.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch. CUDA is seeded when it is present."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Fully connected ranker with a linear head.

    Each hidden block is Linear → BatchNorm → SiLU → Dropout. Dropout is idle
    in ``eval`` mode, which is how activations are dumped for the animations.
    """

    def __init__(self, in_dim: int = 12, hidden: tuple[int, ...] = (64, 32, 16), dropout: float = 0.1) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError("in_dim must be positive")
        dims = [in_dim, *hidden]
        blocks: list[nn.Sequential] = []
        for src, dst in zip(dims[:-1], dims[1:], strict=True):
            linear = nn.Linear(src, dst)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            blocks.append(
                nn.Sequential(
                    linear,
                    nn.BatchNorm1d(dst),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )
        self.hidden = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden[-1] if hidden else in_dim, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.in_dim = in_dim
        self.hidden_dims = tuple(hidden)
        self.dropout_p = float(dropout)

    def forward(
        self,
        features: torch.Tensor,
        return_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        activations: list[torch.Tensor] = []
        hidden = features
        for block in self.hidden:
            hidden = block(hidden)
            if return_activations:
                activations.append(hidden)
        output = self.head(hidden).squeeze(-1)
        if return_activations:
            return output, activations
        return output


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean logistic loss over upper-triangle pairs with a nonzero label gap.

    ``softplus(-(p_i - p_j) * sign(y_i - y_j))`` is zero when the predicted
    order matches the realized order and grows when it does not.
    """
    if pred.ndim != 1 or target.ndim != 1:
        raise ValueError("pairwise_rank_loss expects 1-D tensors")
    diff_pred = pred[:, None] - pred[None, :]
    diff_target = target[:, None] - target[None, :]
    sign = torch.sign(diff_target)
    mask = torch.triu(torch.ones_like(sign, dtype=torch.bool), diagonal=1) & (sign != 0)
    if int(mask.sum()) == 0:
        return pred.sum() * 0.0
    return torch.nn.functional.softplus(-diff_pred[mask] * sign[mask]).mean()


@dataclass
class MLPTrainResult:
    model: MLP
    history: list[dict[str, float]]


def _month_batches(dates: np.ndarray, seed: int, epoch: int, min_names: int = 8) -> list[np.ndarray]:
    codes = pd_period_codes(dates)
    groups: list[np.ndarray] = []
    for code in pd_unique(codes):
        idx = np.flatnonzero(codes == code)
        if len(idx) >= min_names:
            groups.append(idx)
    rng = np.random.default_rng(seed + epoch * 10_007)
    order = np.arange(len(groups))
    rng.shuffle(order)
    return [groups[int(i)] for i in order]


def pd_period_codes(dates: np.ndarray) -> np.ndarray:
    return pd.to_datetime(pd.Series(dates)).dt.to_period("M").astype(str).to_numpy()


def pd_unique(codes: np.ndarray) -> np.ndarray:
    return pd.unique(codes)


def _batch_loss(
    model: MLP,
    features: torch.Tensor,
    target: torch.Tensor,
    rank_lambda: float,
) -> torch.Tensor:
    pred = model(features)
    loss = torch.nn.functional.mse_loss(pred, target)
    if rank_lambda > 0.0:
        loss = loss + float(rank_lambda) * pairwise_rank_loss(pred, target)
    return loss


def _as_float_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    """Copy into a writable float32 tensor. Pandas/NumPy views are often read-only."""
    array = np.array(values, dtype=np.float32, copy=True)
    return torch.tensor(array, device=device)


def predict_mlp(model: MLP, features: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """Eval-mode scores. Batch-norm uses running statistics."""
    device = device or next(model.parameters()).device
    model.eval()
    tensor = _as_float_tensor(features, device)
    with torch.no_grad():
        output = model(tensor)
    if isinstance(output, tuple):
        output = output[0]
    return output.detach().cpu().numpy().astype(float)


def predict_with_activations(
    model: MLP,
    features: np.ndarray,
    device: torch.device | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return scores and the post-SiLU activation of every hidden block."""
    device = device or next(model.parameters()).device
    model.eval()
    tensor = _as_float_tensor(features, device)
    with torch.no_grad():
        output, activations = model(tensor, return_activations=True)
    scores = output.detach().cpu().numpy().astype(np.float32)
    layers = [act.detach().cpu().numpy().astype(np.float32) for act in activations]
    return scores, layers


def save_checkpoint(
    path: Path,
    model: MLP,
    *,
    epoch: int,
    train_loss: float,
    seed: int,
    feature_names: list[str],
    rank_lambda: float,
    lr: float,
    weight_decay: float,
    window_id: str,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "train_loss": float(train_loss),
        "seed": int(seed),
        "in_dim": int(model.in_dim),
        "hidden": list(model.hidden_dims),
        "dropout": float(model.dropout_p),
        "feature_names": list(feature_names),
        "rank_lambda": float(rank_lambda),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "window_id": window_id,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_mlp(path: Path, device: torch.device | None = None) -> tuple[MLP, dict[str, Any]]:
    """Rebuild an MLP from a checkpoint saved by :func:`save_checkpoint`."""
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = MLP(
        in_dim=int(payload["in_dim"]),
        hidden=tuple(int(size) for size in payload["hidden"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload


def train_one_mlp(
    features: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    hidden: tuple[int, ...] = (64, 32, 16),
    dropout: float,
    lr: float,
    weight_decay: float,
    rank_lambda: float,
    epochs: int,
    seed: int,
    feature_names: list[str],
    window_id: str,
    checkpoint_dir: Path | None = None,
    primary_checkpoint_dir: Path | None = None,
    device: torch.device | None = None,
    grad_clip: float = 1.0,
    val_features: np.ndarray | None = None,
    val_target: np.ndarray | None = None,
    val_dates: np.ndarray | None = None,
) -> MLPTrainResult:
    """Train one seed. Optionally write ``epoch_{n}.pt`` after every epoch.

    When ``val_*`` arrays are provided, each epoch records validation rank IC.
    Weights are left at the last epoch; epoch selection happens in the caller
    so the checkpoint trajectory matches the weights that were trained.
    """
    from xsnn.metrics import mean_ic

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    set_seed(seed)
    device = device or get_device()
    model = MLP(in_dim=features.shape[1], hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_tensor = _as_float_tensor(features, device)
    y_tensor = _as_float_tensor(target, device)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        batches = _month_batches(dates, seed=seed, epoch=epoch)
        if not batches:
            raise RuntimeError("No month has enough names to train the MLP")
        losses: list[float] = []
        for idx in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, x_tensor[idx], y_tensor[idx], rank_lambda)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses))
        row: dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss, "seed": float(seed)}
        if val_features is not None and val_target is not None and val_dates is not None:
            val_pred = predict_mlp(model, val_features, device)
            row["val_rank_ic"] = mean_ic(val_pred, val_target, val_dates, method="spearman")
        history.append(row)
        logger.info(
            "window=%s seed=%s epoch=%s train_loss=%.5f val_rank_ic=%s",
            window_id,
            seed,
            epoch,
            train_loss,
            f"{row['val_rank_ic']:.4f}" if "val_rank_ic" in row else "n/a",
        )
        if checkpoint_dir is not None:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch}.pt",
                model,
                epoch=epoch,
                train_loss=train_loss,
                seed=seed,
                feature_names=feature_names,
                rank_lambda=rank_lambda,
                lr=lr,
                weight_decay=weight_decay,
                window_id=window_id,
                extra={"val_rank_ic": row.get("val_rank_ic", float("nan"))},
            )
        if primary_checkpoint_dir is not None:
            save_checkpoint(
                primary_checkpoint_dir / f"epoch_{epoch}.pt",
                model,
                epoch=epoch,
                train_loss=train_loss,
                seed=seed,
                feature_names=feature_names,
                rank_lambda=rank_lambda,
                lr=lr,
                weight_decay=weight_decay,
                window_id=window_id,
                extra={"val_rank_ic": row.get("val_rank_ic", float("nan"))},
            )
    model.eval()
    return MLPTrainResult(model=model, history=history)


class EnsembleMLP:
    """Arithmetic mean of several trained MLPs."""

    def __init__(self, models: list[MLP], device: torch.device | None = None) -> None:
        if not models:
            raise ValueError("EnsembleMLP requires at least one model")
        self.models = models
        self.device = device or get_device()
        for model in self.models:
            model.to(self.device)
            model.eval()

    def predict(self, features: np.ndarray) -> np.ndarray:
        stacked = [predict_mlp(model, features, self.device) for model in self.models]
        return np.mean(np.vstack(stacked), axis=0)
