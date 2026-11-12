"""Expanding-window walk-forward splits with an embargo before each test block."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from xsnn.features import add_months, month_end

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkWindow:
    """One refit and the test months it is allowed to score.

    ``train_months`` are feature dates whose labels are fully known before the
    test block, after removing ``embargo_months``. With a one-month horizon and
    a one-month embargo, the latest training feature date is two months before
    ``test_start``. The skipped month is the embargo: its label ends on the
    first test date, so it is withheld.
    """

    window_id: str
    train_months: tuple[pd.Timestamp, ...]
    test_months: tuple[pd.Timestamp, ...]
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end_exclusive: pd.Timestamp


def _as_months(months: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> tuple[pd.Timestamp, ...]:
    unique = {month_end(value) for value in months}
    return tuple(sorted(unique))


def generate_windows(available_months: Sequence[pd.Timestamp], cfg: dict[str, Any]) -> list[WalkWindow]:
    """Build non-overlapping test blocks on an expanding training history.

    A block is skipped when the expanding history is shorter than
    ``min_train_months``. Later blocks are still emitted once that bar is cleared.
    """
    months = _as_months(available_months)
    if not months:
        return []
    validation = cfg["validation"]
    first = month_end(validation["first_test_start"])
    step = int(validation["retrain_every_months"])
    gap = int(validation["embargo_months"]) + int(validation["horizon_months"])
    min_train = int(validation.get("min_train_months", 36))
    windows: list[WalkWindow] = []
    test_start = first
    last = months[-1]
    while test_start <= last:
        test_end = add_months(test_start, step)
        train_end = add_months(test_start, -gap)
        train = tuple(month for month in months if month <= train_end)
        test = tuple(month for month in months if test_start <= month < test_end)
        if test and len(train) >= min_train:
            windows.append(
                WalkWindow(
                    window_id=pd.Timestamp(test_start).strftime("%Y-%m"),
                    train_months=train,
                    test_months=test,
                    train_end=pd.Timestamp(train_end),
                    test_start=pd.Timestamp(test_start),
                    test_end_exclusive=pd.Timestamp(test_end),
                )
            )
        elif test:
            logger.info(
                "Skipping test block %s: %s training months available, %s required",
                pd.Timestamp(test_start).strftime("%Y-%m"),
                len(train),
                min_train,
            )
        test_start = test_end
    logger.info("Walk-forward produced %s refit windows", len(windows))
    return windows


def inner_time_split(
    train_months: Sequence[pd.Timestamp],
    *,
    inner_valid_months: int,
    embargo_months: int,
    horizon_months: int,
    min_inner_train_months: int,
) -> tuple[tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...]]:
    """Hold out the tail of a training window for hyperparameter selection.

    The same embargo used in the outer walk-forward separates the inner training
    block from the inner validation block. Months inside that gap are unused
    during selection and are included again only in the final refit.
    """
    months = _as_months(train_months)
    needed = inner_valid_months + embargo_months + horizon_months + min_inner_train_months
    if len(months) < needed:
        raise ValueError(
            f"Inner time-series split needs at least {needed} months; received {len(months)}"
        )
    valid = months[-inner_valid_months:]
    cutoff = add_months(valid[0], -(embargo_months + horizon_months))
    inner_train = tuple(month for month in months if month <= cutoff)
    if len(inner_train) < min_inner_train_months:
        raise ValueError("Inner training block is shorter than min_inner_train_months")
    return inner_train, valid
