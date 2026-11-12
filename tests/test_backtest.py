"""Turnover cost accounting for the decile books."""

from __future__ import annotations

import pandas as pd

from xsnn.backtest import run_backtest


def test_cost_math_on_two_name_books() -> None:
    predictions = pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"],
            "ticker": ["A", "B", "A", "B"],
            "score": [1.0, -1.0, -1.0, 1.0],
            "fwd_ret": [0.10, 0.00, 0.00, 0.05],
        }
    )
    result = run_backtest(predictions, n_deciles=2, cost_bps=10.0, min_names=2)
    long_short = result.long_short.set_index("signal_date")
    # Entering +1/-1 from flat costs one unit of one-way turnover.
    assert long_short.loc[pd.Timestamp("2020-01-31"), "gross"] == pytest_approx(0.10)
    assert long_short.loc[pd.Timestamp("2020-01-31"), "turnover"] == pytest_approx(1.0)
    assert long_short.loc[pd.Timestamp("2020-01-31"), "cost"] == pytest_approx(0.001)
    assert long_short.loc[pd.Timestamp("2020-01-31"), "net"] == pytest_approx(0.099)
    # Flipping both legs replaces two units of gross exposure on each side.
    assert long_short.loc[pd.Timestamp("2020-02-29"), "gross"] == pytest_approx(0.05)
    assert long_short.loc[pd.Timestamp("2020-02-29"), "turnover"] == pytest_approx(2.0)
    assert long_short.loc[pd.Timestamp("2020-02-29"), "cost"] == pytest_approx(0.002)
    assert long_short.loc[pd.Timestamp("2020-02-29"), "net"] == pytest_approx(0.048)

    long_only = result.long_only.set_index("signal_date")
    assert long_only.loc[pd.Timestamp("2020-01-31"), "turnover"] == pytest_approx(0.5)
    assert long_only.loc[pd.Timestamp("2020-01-31"), "net"] == pytest_approx(0.10 - 0.0005)
    assert long_only.loc[pd.Timestamp("2020-02-29"), "turnover"] == pytest_approx(1.0)
    assert long_only.loc[pd.Timestamp("2020-02-29"), "net"] == pytest_approx(0.05 - 0.001)


def test_zero_cost_leaves_gross_equal_to_net() -> None:
    predictions = pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"],
            "ticker": ["A", "B", "A", "B"],
            "score": [1.0, -1.0, 1.0, -1.0],
            "fwd_ret": [0.02, -0.01, 0.03, 0.00],
        }
    )
    result = run_backtest(predictions, n_deciles=2, cost_bps=0.0, min_names=2)
    assert (result.long_short["gross"] - result.long_short["net"]).abs().max() == 0.0


def pytest_approx(expected: float):
    import pytest

    return pytest.approx(expected)
