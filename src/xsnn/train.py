"""Walk-forward fitting. Hyperparameters are chosen inside the training window."""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xsnn.attribution import permutation_importance
from xsnn.features import FEATURE_COLUMNS, add_months, month_end
from xsnn.metrics import mean_ic
from xsnn.models.baselines import LightGBMRanker, MomentumReversalScore, RidgeRanker
from xsnn.models.mlp import (
    EnsembleMLP,
    get_device,
    predict_with_activations,
    train_one_mlp,
)
from xsnn.serialize import json_ready
from xsnn.walkforward import WalkWindow, generate_windows, inner_time_split

logger = logging.getLogger(__name__)

MODEL_NAMES = ("momentum", "ridge", "lgbm", "mlp")


@dataclass
class TrainOutputs:
    """Artifacts of one walk-forward run, before the portfolio backtest."""

    predictions: pd.DataFrame
    scored: pd.DataFrame
    importance: pd.DataFrame
    hyperparams: list[dict[str, Any]] = field(default_factory=list)


def _slice_months(panel: pd.DataFrame, months: tuple[pd.Timestamp, ...] | list[pd.Timestamp]) -> pd.DataFrame:
    allowed = pd.DatetimeIndex([month_end(month) for month in months])
    dates = pd.DatetimeIndex(pd.to_datetime(panel.index.get_level_values("date"))).map(month_end)
    mask = pd.DatetimeIndex(dates).isin(allowed)
    return panel.loc[np.asarray(mask)]


def _matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    ranked = frame["fwd_ret_rank"].to_numpy(dtype=np.float64)
    raw = frame["fwd_ret"].to_numpy(dtype=np.float64)
    dates = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values("date"))).map(month_end).to_numpy()
    tickers = frame.index.get_level_values("ticker").astype(str).to_numpy()
    return features, ranked, raw, dates, tickers


def _mask(dates: np.ndarray, months: tuple[pd.Timestamp, ...]) -> np.ndarray:
    allowed = pd.DatetimeIndex([month_end(month) for month in months])
    converted = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates))).map(month_end)
    return pd.DatetimeIndex(converted).isin(allowed)


def _best_ic(records: list[dict[str, Any]], key: str = "inner_rank_ic") -> dict[str, Any]:
    finite = [row for row in records if np.isfinite(row.get(key, np.nan))]
    if not finite:
        logger.warning("Hyperparameter search produced no finite validation IC; using the first candidate")
        return records[0]
    return max(finite, key=lambda row: float(row[key]))


def _tune_ridge(
    features: np.ndarray,
    target: np.ndarray,
    raw_return: np.ndarray,
    dates: np.ndarray,
    train_months: tuple[pd.Timestamp, ...],
    cfg: dict[str, Any],
) -> tuple[RidgeRanker, dict[str, Any]]:
    validation = cfg["validation"]
    inner_train, inner_valid = inner_time_split(
        train_months,
        inner_valid_months=int(validation["inner_valid_months"]),
        embargo_months=int(validation["embargo_months"]),
        horizon_months=int(validation["horizon_months"]),
        min_inner_train_months=int(validation.get("min_inner_train_months", 24)),
    )
    train_mask = _mask(dates, inner_train)
    valid_mask = _mask(dates, inner_valid)
    records: list[dict[str, Any]] = []
    for alpha in cfg["models"]["ridge"]["alphas"]:
        model = RidgeRanker(alpha=float(alpha))
        model.fit(features[train_mask], target[train_mask], dates[train_mask])
        pred = model.predict(features[valid_mask])
        score = mean_ic(pred, raw_return[valid_mask], dates[valid_mask], method="spearman")
        records.append({"alpha": float(alpha), "inner_rank_ic": score})
        logger.info("Ridge alpha=%s inner rank IC=%.4f", alpha, score)
    chosen = _best_ic(records)
    final = RidgeRanker(alpha=float(chosen["alpha"]))
    final.fit(features, target, dates)
    return final, {"model": "ridge", "chosen": chosen, "grid": records}


def _tune_lgbm(
    features: np.ndarray,
    target: np.ndarray,
    raw_return: np.ndarray,
    dates: np.ndarray,
    train_months: tuple[pd.Timestamp, ...],
    cfg: dict[str, Any],
    seed: int,
) -> tuple[LightGBMRanker, dict[str, Any]]:
    validation = cfg["validation"]
    inner_train, inner_valid = inner_time_split(
        train_months,
        inner_valid_months=int(validation["inner_valid_months"]),
        embargo_months=int(validation["embargo_months"]),
        horizon_months=int(validation["horizon_months"]),
        min_inner_train_months=int(validation.get("min_inner_train_months", 24)),
    )
    train_mask = _mask(dates, inner_train)
    valid_mask = _mask(dates, inner_valid)
    grid = cfg["models"]["lgbm"]
    records: list[dict[str, Any]] = []
    combinations = itertools.product(
        grid["n_estimators"],
        grid["learning_rate"],
        grid["num_leaves"],
        grid["min_child_samples"],
    )
    for n_estimators, learning_rate, num_leaves, min_child in combinations:
        model = LightGBMRanker(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            min_child_samples=int(min_child),
            subsample=float(grid["subsample"]),
            colsample_bytree=float(grid["colsample_bytree"]),
            seed=seed,
        )
        model.fit(features[train_mask], target[train_mask], dates[train_mask])
        pred = model.predict(features[valid_mask])
        score = mean_ic(pred, raw_return[valid_mask], dates[valid_mask], method="spearman")
        row = {
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "min_child_samples": int(min_child),
            "inner_rank_ic": score,
        }
        records.append(row)
        logger.info("LightGBM %s inner rank IC=%.4f", row, score)
    chosen = _best_ic(records)
    final = LightGBMRanker(
        n_estimators=int(chosen["n_estimators"]),
        learning_rate=float(chosen["learning_rate"]),
        num_leaves=int(chosen["num_leaves"]),
        min_child_samples=int(chosen["min_child_samples"]),
        subsample=float(grid["subsample"]),
        colsample_bytree=float(grid["colsample_bytree"]),
        seed=seed,
    )
    final.fit(features, target, dates)
    return final, {"model": "lgbm", "chosen": chosen, "grid": records}


def _select_mlp_hyperparams(
    features: np.ndarray,
    target: np.ndarray,
    raw_return: np.ndarray,
    dates: np.ndarray,
    train_months: tuple[pd.Timestamp, ...],
    cfg: dict[str, Any],
    window_id: str,
) -> dict[str, Any]:
    """Pick dropout, rank-loss weight, and epoch count on the inner split.

    Only seed 0 is trained during the search. The epoch that maximizes
    validation rank IC, among epochs at or after ``min_epochs``, becomes the
    training budget of the final ensemble.
    """
    validation = cfg["validation"]
    mlp_cfg = cfg["models"]["mlp"]
    inner_train, inner_valid = inner_time_split(
        train_months,
        inner_valid_months=int(validation["inner_valid_months"]),
        embargo_months=int(validation["embargo_months"]),
        horizon_months=int(validation["horizon_months"]),
        min_inner_train_months=int(validation.get("min_inner_train_months", 24)),
    )
    train_mask = _mask(dates, inner_train)
    valid_mask = _mask(dates, inner_valid)
    hidden = tuple(int(size) for size in mlp_cfg["hidden"])
    min_epochs = int(mlp_cfg["min_epochs"])
    tune_epochs = int(mlp_cfg["tune_epochs"])
    records: list[dict[str, Any]] = []
    device = get_device()
    for dropout, rank_lambda in itertools.product(mlp_cfg["dropout_grid"], mlp_cfg["rank_loss_lambda_grid"]):
        result = train_one_mlp(
            features[train_mask],
            target[train_mask],
            dates[train_mask],
            hidden=hidden,
            dropout=float(dropout),
            lr=float(mlp_cfg["lr"]),
            weight_decay=float(mlp_cfg["weight_decay"]),
            rank_lambda=float(rank_lambda),
            epochs=tune_epochs,
            seed=int(cfg["models"]["seeds"][0]),
            feature_names=FEATURE_COLUMNS,
            window_id=f"{window_id}-tune",
            device=device,
            grad_clip=float(mlp_cfg["grad_clip"]),
            val_features=features[valid_mask],
            val_target=raw_return[valid_mask],
            val_dates=dates[valid_mask],
        )
        eligible = [row for row in result.history if int(row["epoch"]) >= min_epochs]
        pool = eligible or result.history
        best_row = max(
            pool,
            key=lambda row: row["val_rank_ic"] if np.isfinite(row.get("val_rank_ic", np.nan)) else -np.inf,
        )
        record = {
            "dropout": float(dropout),
            "rank_lambda": float(rank_lambda),
            "epochs": int(best_row["epoch"]),
            "inner_rank_ic": float(best_row.get("val_rank_ic", np.nan)),
        }
        records.append(record)
        logger.info("MLP search %s", record)
    chosen = _best_ic(records)
    epochs = int(min(int(mlp_cfg["max_epochs"]), max(min_epochs, int(chosen["epochs"]))))
    chosen = {**chosen, "epochs": epochs}
    return {"model": "mlp", "chosen": chosen, "grid": records}


def _train_mlp_ensemble(
    features: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    cfg: dict[str, Any],
    selection: dict[str, Any],
    window: WalkWindow,
) -> EnsembleMLP:
    mlp_cfg = cfg["models"]["mlp"]
    chosen = selection["chosen"]
    hidden = tuple(int(size) for size in mlp_cfg["hidden"])
    seeds = [int(seed) for seed in cfg["models"]["seeds"]]
    device = get_device()
    window_dir = Path(cfg["outputs"]["checkpoints"]) / window.window_id
    models = []
    histories: list[list[dict[str, float]]] = []
    for seed in seeds:
        result = train_one_mlp(
            features,
            target,
            dates,
            hidden=hidden,
            dropout=float(chosen["dropout"]),
            lr=float(mlp_cfg["lr"]),
            weight_decay=float(mlp_cfg["weight_decay"]),
            rank_lambda=float(chosen["rank_lambda"]),
            epochs=int(chosen["epochs"]),
            seed=seed,
            feature_names=FEATURE_COLUMNS,
            window_id=window.window_id,
            checkpoint_dir=window_dir / f"seed_{seed}",
            primary_checkpoint_dir=window_dir if seed == seeds[0] else None,
            device=device,
            grad_clip=float(mlp_cfg["grad_clip"]),
        )
        models.append(result.model)
        histories.append(result.history)
    meta = {
        "window_id": window.window_id,
        "train_start": str(pd.Timestamp(window.train_months[0]).date()),
        "train_end": str(pd.Timestamp(window.train_end).date()),
        "test_start": str(pd.Timestamp(window.test_start).date()),
        "test_end_exclusive": str(pd.Timestamp(window.test_end_exclusive).date()),
        "feature_names": FEATURE_COLUMNS,
        "hidden": list(hidden),
        "seeds": seeds,
        "selection": selection,
        "history": histories,
    }
    window_dir.mkdir(parents=True, exist_ok=True)
    (window_dir / "meta.json").write_text(json.dumps(json_ready(meta), indent=2), encoding="utf-8")
    return EnsembleMLP(models, device=device)


def _prediction_frame(
    model_name: str,
    scores: np.ndarray,
    raw_return: np.ndarray,
    ranked: np.ndarray,
    dates: np.ndarray,
    tickers: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    frame.insert(0, "ticker", tickers)
    frame.insert(0, "date", pd.to_datetime(dates))
    frame["fwd_ret"] = raw_return
    frame["fwd_ret_rank"] = ranked
    frame["model"] = model_name
    frame["score"] = scores
    return frame


def _dump_activations(
    ensemble: EnsembleMLP,
    frame: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    """Write one compressed dump per test month using the primary seed."""
    out_dir = Path(cfg["outputs"]["activations"])
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = ensemble.models[0]
    device = ensemble.device
    for date, group in frame.groupby(level="date", sort=True):
        features, _, _, _, tickers = _matrices(group)
        seed_scores, layers = predict_with_activations(primary, features, device)
        ensemble_scores = ensemble.predict(features)
        payload: dict[str, np.ndarray] = {
            "inputs": features.astype(np.float32),
            "outputs": seed_scores.astype(np.float32),
            "ensemble_outputs": ensemble_scores.astype(np.float32),
            "tickers": tickers.astype(str),
            "feature_names": np.array(FEATURE_COLUMNS),
        }
        for index, layer in enumerate(layers):
            payload[f"layer_{index}"] = layer
        stamp = pd.Timestamp(date).strftime("%Y-%m-%d")
        np.savez_compressed(out_dir / f"{stamp}.npz", **payload)


def fit_window(
    panel: pd.DataFrame,
    window: WalkWindow,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    """Fit every model on one window and score its test months."""
    logger.info(
        "Window %s train %s → %s (%s months), test %s months",
        window.window_id,
        pd.Timestamp(window.train_months[0]).date(),
        pd.Timestamp(window.train_end).date(),
        len(window.train_months),
        len(window.test_months),
    )
    train = _slice_months(panel, window.train_months)
    test = _slice_months(panel, window.test_months)
    x_train, y_train, raw_train, dates_train, _ = _matrices(train)
    x_test, y_test, raw_test, dates_test, tickers_test = _matrices(test)
    seeds = [int(seed) for seed in cfg["models"]["seeds"]]
    primary_seed = seeds[0]

    momentum = MomentumReversalScore(FEATURE_COLUMNS)
    momentum.fit(x_train, y_train, dates_train)
    ridge, ridge_info = _tune_ridge(x_train, y_train, raw_train, dates_train, window.train_months, cfg)
    lgbm, lgbm_info = _tune_lgbm(
        x_train, y_train, raw_train, dates_train, window.train_months, cfg, primary_seed
    )
    mlp_info = _select_mlp_hyperparams(
        x_train, y_train, raw_train, dates_train, window.train_months, cfg, window.window_id
    )
    ensemble = _train_mlp_ensemble(x_train, y_train, dates_train, cfg, mlp_info, window)

    fitted = {
        "momentum": momentum,
        "ridge": ridge,
        "lgbm": lgbm,
        "mlp": ensemble,
    }
    frames = []
    wide = test.copy()
    for name, model in fitted.items():
        scores = np.asarray(model.predict(x_test), dtype=float)
        frames.append(_prediction_frame(name, scores, raw_test, y_test, dates_test, tickers_test, x_test))
        wide[f"score_{name}"] = scores
    _dump_activations(ensemble, test, cfg)

    importance_rows = []
    for name, model in fitted.items():
        series = permutation_importance(
            model.predict,
            x_test,
            raw_test,
            dates_test,
            FEATURE_COLUMNS,
            seed=primary_seed,
        )
        for feature, drop in series.items():
            importance_rows.append(
                {"window": window.window_id, "model": name, "feature": feature, "ic_drop": float(drop)}
            )
    for info in (ridge_info, lgbm_info, mlp_info):
        info["window"] = window.window_id
    hyper = [
        {"model": "momentum", "window": window.window_id, "chosen": {"rule": "equal-weight mom minus reversal"}},
        ridge_info,
        lgbm_info,
        mlp_info,
    ]
    return pd.concat(frames, ignore_index=True), wide, hyper, pd.DataFrame(importance_rows)


def fit_walk_forward(panel: pd.DataFrame, cfg: dict[str, Any]) -> TrainOutputs:
    """Expanding-window refits. The panel must already be the model frame."""
    months = pd.DatetimeIndex(pd.to_datetime(panel.index.get_level_values("date"))).map(month_end)
    windows = generate_windows(pd.DatetimeIndex(months.unique()), cfg)
    if not windows:
        raise RuntimeError(
            "No walk-forward windows were produced. Extend the sample or lower min_train_months."
        )
    predictions: list[pd.DataFrame] = []
    scored_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    hyperparams: list[dict[str, Any]] = []
    for window in windows:
        preds, wide, hyper, importance = fit_window(panel, window, cfg)
        predictions.append(preds)
        scored_frames.append(wide.reset_index())
        importance_frames.append(importance)
        hyperparams.extend(hyper)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    scored = pd.concat(scored_frames, ignore_index=True)
    importance = pd.concat(importance_frames, ignore_index=True)
    # The last label month of each test block is realized one month later.
    # Keep the column explicit for readers of the parquet.
    prediction_frame["realized_date"] = prediction_frame["date"].map(lambda value: add_months(value, 1))
    logger.info(
        "Walk-forward scored %s rows across %s models",
        len(prediction_frame),
        prediction_frame["model"].nunique(),
    )
    return TrainOutputs(
        predictions=prediction_frame,
        scored=scored,
        importance=importance,
        hyperparams=hyperparams,
    )
