"""The shipped config stays on the stated validation design."""

from __future__ import annotations

from xsnn.config import load_config


def test_default_config_loads() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg["models"]["mlp"]["hidden"] == [64, 32, 16]
    assert cfg["validation"]["embargo_months"] == 1
    assert cfg["validation"]["retrain_every_months"] == 12
    assert cfg["backtest"]["cost_bps"] == 10
    assert cfg["data"]["cache_dir"].endswith("data/raw")
