"""Disk-backed feature build, walk-forward training, and the results report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from xsnn.data import load_factors, load_market_data, load_pit_constituents
from xsnn.features import build_feature_panel, model_frame
from xsnn.reporting import evaluate_predictions, write_report
from xsnn.serialize import json_ready
from xsnn.train import TrainOutputs, fit_walk_forward

logger = logging.getLogger(__name__)


def _processed(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["outputs"]["processed"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    if "ticker" in frame.columns:
        return frame.set_index(["date", "ticker"]).sort_index()
    return frame


def build_feature_store(cfg: dict[str, Any]) -> pd.DataFrame:
    """Compute the point-in-time panel from the raw cache and write it to disk."""
    market = load_market_data(cfg)
    membership = load_pit_constituents(cfg)
    features = cfg["features"]
    panel = build_feature_panel(
        market.prices,
        market.volume,
        market.spy,
        membership,
        beta_window=int(features["beta_window"]),
        vol_short=int(features["vol_short"]),
        vol_long=int(features["vol_long"]),
        max_missing_frac=float(features["max_missing_frac"]),
    )
    out = _processed(cfg)
    panel.reset_index().to_parquet(out / "panel.parquet", index=False)
    market.factors.to_parquet(out / "factors.parquet")
    usable = model_frame(panel)
    logger.info(
        "Feature panel rows=%s usable=%s dates=%s..%s",
        len(panel),
        len(usable),
        panel.index.get_level_values("date").min().date() if len(panel) else "n/a",
        panel.index.get_level_values("date").max().date() if len(panel) else "n/a",
    )
    return panel


def load_feature_panel(cfg: dict[str, Any]) -> pd.DataFrame:
    path = _processed(cfg) / "panel.parquet"
    if not path.exists():
        logger.info("Processed panel missing; building it from the raw cache")
        return build_feature_store(cfg)
    return _read_indexed(path)


def load_processed_factors(cfg: dict[str, Any]) -> pd.DataFrame:
    path = _processed(cfg) / "factors.parquet"
    if path.exists():
        factors = pd.read_parquet(path)
    else:
        factors = load_factors(cfg)
    factors.index = pd.DatetimeIndex(pd.to_datetime(factors.index))
    return factors.sort_index()


def run_train(cfg: dict[str, Any]) -> TrainOutputs:
    """Fit the walk-forward models and persist predictions."""
    panel = load_feature_panel(cfg)
    usable = model_frame(panel)
    if usable.empty:
        raise RuntimeError("No supervised rows are available. Check the price cache and feature windows.")
    outputs = fit_walk_forward(usable, cfg)
    out = _processed(cfg)
    reports = Path(cfg["outputs"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    outputs.predictions.to_parquet(out / "predictions.parquet", index=False)
    outputs.scored.to_parquet(out / "scored.parquet", index=False)
    outputs.importance.to_parquet(out / "importance.parquet", index=False)
    payload = json_ready(outputs.hyperparams)
    (reports / "hyperparams.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "hyperparams.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved predictions to %s", out / "predictions.parquet")
    return outputs


def run_backtest(cfg: dict[str, Any]) -> Path:
    """Evaluate saved predictions and write the report."""
    out = _processed(cfg)
    pred_path = out / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError("Predictions are missing. Run `make train` first.")
    predictions = pd.read_parquet(pred_path)
    predictions["date"] = pd.to_datetime(predictions["date"])
    importance_path = out / "importance.parquet"
    if importance_path.exists():
        importance = pd.read_parquet(importance_path)
    else:
        importance = pd.DataFrame(columns=["window", "model", "feature", "ic_drop"])
    hyper_path = Path(cfg["outputs"]["reports"]) / "hyperparams.json"
    hyperparams = json.loads(hyper_path.read_text(encoding="utf-8")) if hyper_path.exists() else []
    factors = load_processed_factors(cfg)
    bundle = evaluate_predictions(predictions, factors, importance, cfg)
    return write_report(bundle, cfg, hyperparams, predictions)
