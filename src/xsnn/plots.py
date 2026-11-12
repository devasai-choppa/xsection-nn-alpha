"""Static figures for the walk-forward report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = ("momentum", "ridge", "lgbm", "mlp")
MODEL_LABELS = {
    "momentum": "Momentum + reversal",
    "ridge": "Ridge",
    "lgbm": "LightGBM",
    "mlp": "MLP ensemble",
}
MODEL_COLORS = {
    "momentum": "#4C4C4C",
    "ridge": "#3B6D8C",
    "lgbm": "#D17A22",
    "mlp": "#B23A48",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.titlesize": 13,
            "legend.frameon": False,
        }
    )


def _ordered(models: list[str] | pd.Index) -> list[str]:
    present = set(models)
    return [name for name in MODEL_ORDER if name in present]


def plot_cumulative(monthly: pd.DataFrame, path: Path, column: str, title: str) -> None:
    """Compound a long/short monthly column for every model."""
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    books = monthly.loc[monthly["book"] == "long_short"]
    for model in _ordered(books["model"].unique()):
        group = books.loc[books["model"] == model].sort_values("signal_date")
        wealth = (1.0 + group[column].astype(float)).cumprod()
        ax.plot(
            group["signal_date"],
            wealth,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "black"),
            linewidth=2.0 if model == "mlp" else 1.4,
        )
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Signal month")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_drawdown(monthly: pd.DataFrame, path: Path) -> None:
    """Peak-to-trough drawdown of net long/short wealth."""
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    books = monthly.loc[monthly["book"] == "long_short"]
    for model in _ordered(books["model"].unique()):
        group = books.loc[books["model"] == model].sort_values("signal_date")
        wealth = (1.0 + group["net"].astype(float)).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        ax.plot(
            group["signal_date"],
            drawdown,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "black"),
            linewidth=2.0 if model == "mlp" else 1.4,
        )
    ax.set_title("Long/short drawdown, net of costs")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Signal month")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_rolling_ic(ic: pd.DataFrame, path: Path, window: int = 12) -> None:
    """Rolling mean of monthly Spearman IC."""
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for model in _ordered(ic["model"].unique()):
        group = ic.loc[ic["model"] == model].sort_values("signal_date")
        rolled = group["rank_ic"].rolling(window, min_periods=max(6, window // 2)).mean()
        ax.plot(
            group["signal_date"],
            rolled,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "black"),
            linewidth=2.0 if model == "mlp" else 1.4,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"{window}-month rolling rank IC")
    ax.set_ylabel("Spearman IC")
    ax.set_xlabel("Signal month")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_decile_bars(deciles: pd.DataFrame, path: Path) -> None:
    """Mean gross forward return by score decile."""
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    models = _ordered(deciles["model"].unique())
    decile_ids = sorted(int(value) for value in deciles["decile"].unique())
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(decile_ids))
    for offset, model in enumerate(models):
        group = deciles.loc[deciles["model"] == model]
        means = group.groupby("decile")["ret"].mean().reindex(decile_ids)
        ax.bar(
            x + offset * width - 0.4 + width / 2,
            means.to_numpy(dtype=float),
            width=width,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "black"),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(decile) for decile in decile_ids])
    ax.set_xlabel("Score decile (1 = lowest predicted rank)")
    ax.set_ylabel("Mean next-month return")
    ax.set_title("Decile monotonicity, gross of costs")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_turnover(monthly: pd.DataFrame, path: Path) -> None:
    """One-way turnover of the long/short book."""
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    books = monthly.loc[monthly["book"] == "long_short"]
    for model in _ordered(books["model"].unique()):
        group = books.loc[books["model"] == model].sort_values("signal_date")
        ax.plot(
            group["signal_date"],
            group["turnover"],
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "black"),
            linewidth=1.6 if model == "mlp" else 1.2,
        )
    ax.set_title("Long/short one-way turnover")
    ax.set_ylabel("0.5 · Σ|Δw|")
    ax.set_xlabel("Signal month")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    """Average out-of-sample rank-IC drop from within-month permutation."""
    _style()
    models = _ordered(importance["model"].unique())
    features = list(dict.fromkeys(importance["feature"].tolist()))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
    for ax in axes.ravel()[len(models) :]:
        ax.axis("off")
    for ax, model in zip(axes.ravel(), models, strict=False):
        group = (
            importance.loc[importance["model"] == model]
            .groupby("feature")["ic_drop"]
            .mean()
            .reindex(features)
            .sort_values()
        )
        colors = ["#3B6D8C" if value >= 0 else "#B23A48" for value in group.to_numpy()]
        ax.barh(group.index, group.to_numpy(), color=colors)
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Rank-IC drop when shuffled")
    fig.suptitle("Permutation importance on walk-forward test months", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_all_figures(
    monthly: pd.DataFrame,
    ic: pd.DataFrame,
    deciles: pd.DataFrame,
    importance: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    """Write the report figures and return their paths."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cumulative_ls_net.png": lambda path: plot_cumulative(
            monthly, path, "net", "Long/short growth of $1, net of costs"
        ),
        "cumulative_ls_gross.png": lambda path: plot_cumulative(
            monthly, path, "gross", "Long/short growth of $1, gross of costs"
        ),
        "rolling_rank_ic.png": lambda path: plot_rolling_ic(ic, path),
        "decile_bars.png": lambda path: plot_decile_bars(deciles, path),
        "turnover.png": lambda path: plot_turnover(monthly, path),
        "drawdown.png": lambda path: plot_drawdown(monthly, path),
        "feature_importance.png": lambda path: plot_feature_importance(importance, path),
    }
    paths: list[Path] = []
    for name, writer in outputs.items():
        path = figure_dir / name
        writer(path)
        paths.append(path)
    return paths
