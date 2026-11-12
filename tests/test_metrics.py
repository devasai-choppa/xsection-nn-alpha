"""Closed-form checks for the published performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xsnn.metrics import (
    annualized_return,
    calmar_ratio,
    factor_attribution,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize_ic,
)


def _monthly_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-31", periods=n, freq="ME")


def test_annualized_return_of_constant_monthly_gain() -> None:
    returns = pd.Series([0.01] * 24, index=_monthly_index(24))
    expected = (1.01**12) - 1.0
    assert annualized_return(returns) == pytest_approx(expected)


def test_sharpe_matches_hand_calculation() -> None:
    returns = pd.Series([0.02, 0.00] * 12, index=_monthly_index(24))
    # 12 observations at 0.02 and 12 at 0.00. Sample variance uses n - 1.
    std = np.sqrt(0.0024 / 23.0)
    expected = (0.01 / std) * np.sqrt(12.0)
    assert sharpe_ratio(returns) == pytest_approx(expected)


def test_sortino_uses_downside_second_moment() -> None:
    returns = pd.Series([0.03, -0.01, 0.02, -0.02], index=_monthly_index(4))
    downside = np.sqrt(np.mean(np.square([0.0, -0.01, 0.0, -0.02])))
    expected = returns.mean() / downside * np.sqrt(12.0)
    assert sortino_ratio(returns) == pytest_approx(expected)


def test_drawdown_and_calmar() -> None:
    returns = pd.Series([-0.10, 0.05, -0.20, 0.10], index=_monthly_index(4))
    # Wealth 0.90, 0.945, 0.756, 0.8316. Peak 0.945 falls to 0.756.
    assert max_drawdown(returns) == pytest_approx(-0.20)
    assert calmar_ratio(returns) == pytest_approx(annualized_return(returns) / 0.20)


def test_ic_summary_t_stat() -> None:
    series = pd.Series([0.1, 0.2, 0.0, -0.1])
    stats = summarize_ic(series)
    std = np.sqrt(0.05 / 3.0)
    assert stats["mean"] == pytest_approx(0.05)
    assert stats["std"] == pytest_approx(std)
    assert stats["icir"] == pytest_approx(0.05 / std)
    assert stats["t_stat"] == pytest_approx(0.05 / (std / np.sqrt(4.0)))


def test_factor_alpha_recovers_planted_intercept() -> None:
    rng = np.random.default_rng(0)
    index = _monthly_index(60)
    market = rng.normal(0.0, 0.04, len(index))
    factors = pd.DataFrame(
        {
            "Mkt-RF": market,
            "SMB": rng.normal(0.0, 0.02, len(index)),
            "HML": rng.normal(0.0, 0.02, len(index)),
            "RMW": rng.normal(0.0, 0.015, len(index)),
            "CMA": rng.normal(0.0, 0.015, len(index)),
            "Mom": rng.normal(0.0, 0.03, len(index)),
            "RF": np.full(len(index), 0.001),
        },
        index=index,
    )
    alpha = 0.002
    returns = pd.Series(alpha + 0.8 * market + rng.normal(0.0, 0.001, len(index)), index=index)
    result = factor_attribution(returns, factors, excess=False, hac_lags=1)
    assert result["alpha_monthly"] == pytest_approx(alpha, abs=0.001)
    assert result["alpha_t"] > 2.0


def pytest_approx(expected: float, abs: float | None = None):
    import pytest

    if abs is None:
        return pytest.approx(expected)
    return pytest.approx(expected, abs=abs)
