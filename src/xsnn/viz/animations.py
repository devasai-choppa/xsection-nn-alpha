"""Lightweight GIF loops for the README. Each file is kept under 8 MB."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from xsnn.plots import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER
from xsnn.viz.loaders import (
    artifacts,
    decile_returns,
    explainer_cross_section,
    ic_series,
    inference_checkpoint,
    monthly_returns,
)

logger = logging.getLogger(__name__)

MAX_BYTES = 8 * 1024 * 1024


def build_all_gifs(cfg: dict | None = None) -> list[Path]:
    """Write the four README animations and return their paths."""
    cfg = cfg or artifacts()
    out = Path(cfg["outputs"]["reports"]) / "gifs"
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        rolling_ic_gif(cfg, out / "rolling_rank_ic.gif"),
        decile_gif(cfg, out / "decile_months.gif"),
        pnl_race_gif(cfg, out / "pnl_race.gif"),
        forward_pass_gif(cfg, out / "forward_pass.gif"),
    ]
    for path in paths:
        _shrink(path)
        logger.info("GIF %s is %.2f MB", path.name, path.stat().st_size / 1e6)
        if path.stat().st_size >= MAX_BYTES:
            raise RuntimeError(f"{path} is still over 8 MB after optimization")
    return paths


def _frame(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(buffer[..., :3], mode="RGB")
    plt.close(fig)
    return image.resize((960, 540), Image.Resampling.LANCZOS)


def _save(frames: list[Image.Image], path: Path, duration_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=64) for frame in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _shrink(path: Path) -> None:
    gifsicle = shutil.which("gifsicle")
    if gifsicle is None:
        logger.warning("gifsicle is not installed; leaving %s unoptimized", path.name)
        return
    temporary = path.with_suffix(".tmp.gif")
    subprocess.run(
        [gifsicle, "-O3", "--colors", "64", "--lossy=120", "-o", str(temporary), str(path)],
        check=True,
    )
    temporary.replace(path)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("#101418")
    ax.tick_params(colors="#d0d0d0")
    ax.xaxis.label.set_color("#d0d0d0")
    ax.yaxis.label.set_color("#d0d0d0")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.grid(True, alpha=0.25, color="#666666")


def rolling_ic_gif(cfg: dict, path: Path) -> Path:
    """Rolling 12-month rank IC drawing through the walk-forward sample."""
    frame = ic_series(cfg)
    dates = sorted(frame["signal_date"].unique())
    stride = 2
    images = []
    for stop in range(12, len(dates), stride):
        fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=100, facecolor="#101418")
        _style(ax)
        cutoff = dates[stop]
        for model in MODEL_ORDER:
            group = frame.loc[frame["model"] == model].sort_values("signal_date")
            group = group.loc[group["signal_date"] <= cutoff]
            rolled = group["rank_ic"].rolling(12, min_periods=6).mean()
            ax.plot(group["signal_date"], rolled, color=MODEL_COLORS[model], label=MODEL_LABELS[model], lw=2)
        ax.axhline(0.0, color="white", lw=0.6)
        ax.set_title(f"12-month rolling rank IC through {pd.Timestamp(cutoff).date()}")
        ax.set_ylabel("Spearman IC")
        ax.legend(facecolor="#101418", labelcolor="white", frameon=False, loc="upper left")
        ax.set_xlim(dates[0], dates[-1])
        images.append(_frame(fig))
    _save(images, path, duration_ms=90)
    return path


def decile_gif(cfg: dict, path: Path) -> Path:
    """MLP decile returns, one test month per frame."""
    deciles = decile_returns(cfg)
    deciles = deciles.loc[deciles["model"] == "mlp"].copy()
    dates = sorted(deciles["signal_date"].unique())
    limit = float(np.nanpercentile(np.abs(deciles["ret"]), 98))
    images = []
    stride = 2
    for date in dates[::stride]:
        group = deciles.loc[deciles["signal_date"] == date].sort_values("decile")
        fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=100, facecolor="#101418")
        _style(ax)
        colors = ["#E85D4C" if decile <= 2 else "#3DDC97" if decile >= 9 else "#8aa0b4" for decile in group["decile"]]
        ax.bar(group["decile"].astype(int), group["ret"], color=colors)
        ax.axhline(0.0, color="white", lw=0.6)
        ax.set_ylim(-limit, limit)
        ax.set_xticks(range(1, 11))
        ax.set_title(f"MLP decile returns  ·  signal {pd.Timestamp(date).date()}")
        ax.set_ylabel("Next-month return")
        ax.set_xlabel("Decile (1 = lowest score)")
        images.append(_frame(fig))
    _save(images, path, duration_ms=120)
    return path


def pnl_race_gif(cfg: dict, path: Path) -> Path:
    """Cumulative net long/short wealth, models racing forward together."""
    books = monthly_returns(cfg)
    books = books.loc[books["book"] == "long_short"].copy()
    dates = sorted(books["signal_date"].unique())
    series = {}
    for model in MODEL_ORDER:
        group = books.loc[books["model"] == model].sort_values("signal_date")
        series[model] = (1.0 + group["net"].to_numpy(dtype=float)).cumprod()
    hi = max(float(values.max()) for values in series.values())
    images = []
    for stop in range(3, len(dates), 2):
        fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=100, facecolor="#101418")
        _style(ax)
        for model in MODEL_ORDER:
            ax.plot(
                dates[: stop + 1],
                series[model][: stop + 1],
                color=MODEL_COLORS[model],
                lw=2.4 if model == "mlp" else 1.6,
                label=MODEL_LABELS[model],
            )
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylim(0.6, hi * 1.05)
        ax.set_title(f"Long/short growth of $1, net  ·  {pd.Timestamp(dates[stop]).date()}")
        ax.set_ylabel("Wealth")
        ax.legend(facecolor="#101418", labelcolor="white", frameon=False, loc="upper left")
        images.append(_frame(fig))
    _save(images, path, duration_ms=80)
    return path


def forward_pass_gif(cfg: dict, path: Path) -> Path:
    """Three real names — top, middle, and bottom decile — looped through the MLP."""
    from xsnn.models.mlp import load_mlp

    bundle = explainer_cross_section(cfg)
    model, payload = load_mlp(inference_checkpoint(cfg, bundle["window_id"]))
    state = model.state_dict()
    weights = [
        state["hidden.0.0.weight"].detach().cpu().numpy(),
        state["hidden.1.0.weight"].detach().cpu().numpy(),
        state["hidden.2.0.weight"].detach().cpu().numpy(),
        state["head.weight"].detach().cpu().numpy(),
    ]
    shown = [
        list(range(weights[0].shape[1])),
        _take(weights[0].shape[0], 8),
        _take(weights[1].shape[0], 6),
        _take(weights[2].shape[0], 6),
        [0],
    ]
    activation = bundle["activation"]
    tickers = np.asarray(activation["tickers"]).astype(str)
    lookup = {ticker: index for index, ticker in enumerate(tickers)}
    images = []
    for pick in bundle["picks"]:
        row = lookup[str(pick["ticker"])]
        inputs = np.asarray(activation["inputs"][row], dtype=float)
        layers = [np.asarray(activation[f"layer_{i}"][row], dtype=float) for i in range(3)]
        output = float(activation["outputs"][row])
        values = [inputs, *layers, np.array([output])]
        for lit in range(1, 5):
            fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=100, facecolor="#101418")
            ax.set_xlim(-0.4, 4.6)
            ax.set_ylim(-1.3, 1.5)
            ax.axis("off")
            ax.set_facecolor("#101418")
            xs = [0, 1, 2, 3, 4]
            coords: list[list[tuple[float, float]]] = []
            for layer, indices in enumerate(shown):
                ys = np.linspace(1.15, -1.05, len(indices))
                coords.append([(xs[layer], float(y)) for y in ys])
                scale = max(float(np.percentile(np.abs(values[layer]), 90)), 0.25)
                active = layer < lit
                for (x, y), index in zip(coords[-1], indices, strict=True):
                    value = float(values[layer][index])
                    color = "#3DDC97" if value >= 0 else "#E85D4C"
                    if not active:
                        color = "#2a3340"
                    size = 40 + (160 if active else 40) * min(1.0, abs(value) / scale)
                    ax.scatter([x], [y], s=size, c=color, zorder=3)
            for layer, weight in enumerate(weights):
                scale = max(float(np.percentile(np.abs(weight), 90)), 1e-4)
                active = layer < lit - 1
                for out_i, out_idx in enumerate(shown[layer + 1]):
                    for in_i, in_idx in enumerate(shown[layer]):
                        value = float(weight[out_idx, in_idx])
                        ax.plot(
                            [coords[layer][in_i][0], coords[layer + 1][out_i][0]],
                            [coords[layer][in_i][1], coords[layer + 1][out_i][1]],
                            color=("#4C78A8" if value >= 0 else "#E45756") if active else "#243040",
                            lw=0.3 + 1.8 * min(1.0, abs(value) / scale),
                            alpha=0.55 if active else 0.25,
                            zorder=1,
                        )
            ax.set_title(
                f"{pick['ticker']}   decile {int(pick['decile'])}   score {output:+.3f}",
                color="white",
                fontsize=14,
            )
            images.append(_frame(fig))
    _save(images, path, duration_ms=450)
    return path


def _take(count: int, keep: int) -> list[int]:
    if count <= keep:
        return list(range(count))
    return [int(round(value)) for value in np.linspace(0, count - 1, keep)]
