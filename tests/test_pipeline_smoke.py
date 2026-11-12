"""One walk-forward window on synthetic prices, including checkpoints and the report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xsnn.features import build_feature_panel, model_frame, month_end
from xsnn.models.mlp import load_mlp
from xsnn.pipeline import run_backtest
from xsnn.reporting import evaluate_predictions
from xsnn.train import fit_walk_forward


def _market() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2013-01-02", periods=1400)
    n_stocks = 24
    spy_ret = rng.normal(0.0003, 0.01, len(index))
    betas = rng.uniform(0.5, 1.4, n_stocks)
    rets = betas * spy_ret[:, None] + rng.normal(0.0, 0.012, size=(len(index), n_stocks))
    prices = pd.DataFrame(
        40.0 * np.exp(np.cumsum(rets, axis=0)),
        index=index,
        columns=[f"S{i:02d}" for i in range(n_stocks)],
    )
    volume = pd.DataFrame(rng.integers(80_000, 400_000, size=prices.shape), index=index, columns=prices.columns)
    spy = pd.Series(80.0 * np.exp(np.cumsum(spy_ret)), index=index, name="SPY")
    return prices, volume, spy


def _cfg(tmp_path: Path, first_test: pd.Timestamp) -> dict:
    return {
        "_config_path": "configs/smoke.yaml",
        "_root": str(tmp_path),
        "data": {},
        "features": {},
        "models": {
            "seeds": [0],
            "ridge": {"alphas": [1.0]},
            "lgbm": {
                "n_estimators": [20],
                "learning_rate": [0.1],
                "num_leaves": [7],
                "min_child_samples": [5],
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "mlp": {
                "hidden": [64, 32, 16],
                "dropout_grid": [0.1],
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "rank_loss_lambda_grid": [0.0],
                "tune_epochs": 1,
                "min_epochs": 1,
                "max_epochs": 1,
                "grad_clip": 1.0,
            },
        },
        "validation": {
            "first_test_start": str(first_test.date()),
            "retrain_every_months": 36,
            "embargo_months": 1,
            "horizon_months": 1,
            "inner_valid_months": 6,
            "min_train_months": 18,
            "min_inner_train_months": 8,
        },
        "backtest": {"cost_bps": 10, "n_deciles": 5, "min_names": 10, "hac_lags": 1},
        "outputs": {
            "checkpoints": str(tmp_path / "checkpoints"),
            "activations": str(tmp_path / "activations"),
            "reports": str(tmp_path / "reports"),
            "figures": str(tmp_path / "reports" / "figures"),
            "processed": str(tmp_path / "processed"),
            "logs": str(tmp_path / "logs"),
        },
    }


def test_smoke_walk_forward_writes_checkpoints_activations_and_report(tmp_path: Path) -> None:
    prices, volume, spy = _market()
    panel = model_frame(build_feature_panel(prices, volume, spy))
    months = pd.DatetimeIndex(sorted(panel.index.get_level_values("date").unique()))
    assert len(months) > 30
    cfg = _cfg(tmp_path, months[20])
    outputs = fit_walk_forward(panel, cfg)
    assert set(outputs.predictions["model"].unique()) == {"momentum", "ridge", "lgbm", "mlp"}
    window_id = outputs.hyperparams[0]["window"]
    checkpoint = tmp_path / "checkpoints" / window_id / "epoch_1.pt"
    assert checkpoint.exists()
    loaded, payload = load_mlp(checkpoint)
    assert payload["hidden"] == [64, 32, 16]
    assert loaded.in_dim == 12
    activation_files = list((tmp_path / "activations").glob("*.npz"))
    assert activation_files
    with np.load(activation_files[0], allow_pickle=False) as dump:
        assert dump["inputs"].shape[1] == 12
        assert dump["layer_0"].shape[1] == 64
        assert dump["layer_1"].shape[1] == 32
        assert dump["layer_2"].shape[1] == 16
        assert len(dump["tickers"]) == dump["outputs"].shape[0]
        assert dump["ensemble_outputs"].shape == dump["outputs"].shape
    index = pd.date_range(months[0], periods=len(months) + 2, freq="ME")
    rng = np.random.default_rng(1)
    factors = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.005, 0.04, len(index)),
            "SMB": rng.normal(0.0, 0.02, len(index)),
            "HML": rng.normal(0.0, 0.02, len(index)),
            "RMW": rng.normal(0.0, 0.015, len(index)),
            "CMA": rng.normal(0.0, 0.015, len(index)),
            "Mom": rng.normal(0.0, 0.03, len(index)),
            "RF": np.full(len(index), 0.001),
        },
        index=index.map(month_end),
    )
    processed = tmp_path / "processed"
    processed.mkdir()
    outputs.predictions.to_parquet(processed / "predictions.parquet", index=False)
    outputs.importance.to_parquet(processed / "importance.parquet", index=False)
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "hyperparams.json").write_text("[]", encoding="utf-8")
    factors.to_parquet(processed / "factors.parquet")
    bundle = evaluate_predictions(outputs.predictions, factors, outputs.importance, cfg)
    assert not bundle["metrics"].empty
    assert (bundle["metrics"]["ann_return"].notna()).any()
    path = run_backtest(cfg)
    text = path.read_text(encoding="utf-8")
    assert "Long/short, net of" in text
    for name in (
        "cumulative_ls_net.png",
        "cumulative_ls_gross.png",
        "rolling_rank_ic.png",
        "decile_bars.png",
        "turnover.png",
        "drawdown.png",
        "feature_importance.png",
    ):
        figure = tmp_path / "reports" / "figures" / name
        assert figure.exists()
        assert figure.read_bytes().startswith(b"\x89PNG")
