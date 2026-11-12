"""Embargo and inner time-series split."""

from __future__ import annotations

import pandas as pd

from xsnn.features import add_months
from xsnn.walkforward import generate_windows, inner_time_split


def test_expanding_window_keeps_a_one_month_embargo() -> None:
    months = pd.date_range("2008-01-31", "2016-12-31", freq="ME")
    cfg = {
        "validation": {
            "first_test_start": "2012-01-31",
            "retrain_every_months": 12,
            "embargo_months": 1,
            "horizon_months": 1,
            "min_train_months": 36,
        }
    }
    windows = generate_windows(months, cfg)
    first = windows[0]
    assert first.window_id == "2012-01"
    assert first.train_end == pd.Timestamp("2011-11-30")
    assert max(first.train_months) == pd.Timestamp("2011-11-30")
    embargo_month = pd.Timestamp("2011-12-31")
    assert embargo_month not in first.train_months
    assert embargo_month not in first.test_months
    assert min(first.test_months) == pd.Timestamp("2012-01-31")
    assert max(first.test_months) == pd.Timestamp("2012-12-31")
    assert windows[1].test_start == pd.Timestamp("2013-01-31")
    assert set(first.train_months).isdisjoint(first.test_months)
    # The newest training label ends one month before the test block.
    assert add_months(max(first.train_months), 1) == embargo_month


def test_inner_split_repeats_the_embargo() -> None:
    months = pd.date_range("2006-01-31", "2011-11-30", freq="ME")
    inner_train, valid = inner_time_split(
        months,
        inner_valid_months=24,
        embargo_months=1,
        horizon_months=1,
        min_inner_train_months=24,
    )
    assert valid[0] == pd.Timestamp("2009-12-31")
    assert valid[-1] == pd.Timestamp("2011-11-30")
    assert max(inner_train) == pd.Timestamp("2009-10-31")
    assert pd.Timestamp("2009-11-30") not in inner_train
    assert pd.Timestamp("2009-11-30") not in valid
