"""Monthly decile portfolios with turnover costs charged per side."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xsnn.features import add_months, month_end


def assign_deciles(scores: pd.Series, n_deciles: int) -> pd.Series:
    """Equal-count deciles from 1 (lowest score) to ``n_deciles`` (highest).

    Ties are broken by ``rank(method="first")`` so buckets stay balanced.
    """
    if n_deciles < 2:
        raise ValueError("n_deciles must be at least 2")
    if len(scores) < n_deciles:
        raise ValueError(f"Need at least {n_deciles} names to form deciles, got {len(scores)}")
    buckets = pd.qcut(scores.rank(method="first"), q=n_deciles, labels=False, duplicates="drop")
    if int(buckets.max()) + 1 != n_deciles:
        raise ValueError("Decile bins collapsed; scores do not span enough distinct names")
    return pd.Series(buckets.to_numpy(dtype=int) + 1, index=scores.index, name="decile")


def _turnover_and_cost(weights: pd.Series, previous: pd.Series, cost_rate: float) -> tuple[float, float]:
    """One-way turnover ``0.5 * sum(|Δw|)`` and the return drag it causes.

    Names that enter or leave are treated as coming from a zero weight. A
    cost of ``cost_bps`` per side is ``turnover * cost_bps / 10_000`` because
    one-way turnover already counts each side once. A full replacement of a
    +1/-1 book has one-way turnover 2 and therefore costs ``2 * cost_bps``.
    """
    names = weights.index.union(previous.index)
    current = weights.reindex(names).fillna(0.0)
    prior = previous.reindex(names).fillna(0.0)
    turnover = 0.5 * float((current - prior).abs().sum())
    return turnover, turnover * cost_rate


@dataclass
class BacktestResult:
    """Monthly books, decile returns, and the score panel used to build them."""

    long_short: pd.DataFrame
    long_only: pd.DataFrame
    deciles: pd.DataFrame


def run_backtest(
    predictions: pd.DataFrame,
    *,
    score_col: str = "score",
    return_col: str = "fwd_ret",
    date_col: str = "date",
    ticker_col: str = "ticker",
    n_deciles: int = 10,
    cost_bps: float = 10.0,
    min_names: int = 30,
) -> BacktestResult:
    """Form equal-weight top-decile / bottom-decile books from a score.

    Parameters
    ----------
    predictions:
        Must contain the date, ticker, score, and realized next-month return.
        Returns are the simple return from this month-end to the next.
    cost_bps:
        Transaction cost in basis points charged on one-way turnover.
    """
    required = {date_col, ticker_col, score_col, return_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Predictions missing columns: {sorted(missing)}")
    frame = predictions.loc[:, [date_col, ticker_col, score_col, return_col]].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame[date_col] = pd.to_datetime(frame[date_col]).map(month_end)
    frame[ticker_col] = frame[ticker_col].astype(str)

    cost_rate = float(cost_bps) / 10_000.0
    previous_ls = pd.Series(dtype=float)
    previous_lo = pd.Series(dtype=float)
    long_short_rows: list[dict[str, float | pd.Timestamp]] = []
    long_only_rows: list[dict[str, float | pd.Timestamp]] = []
    decile_rows: list[dict[str, float | pd.Timestamp | int]] = []

    for signal_date, group in frame.groupby(date_col, sort=True):
        group = group.drop_duplicates(ticker_col).set_index(ticker_col)
        if len(group) < max(min_names, n_deciles):
            continue
        deciles = assign_deciles(group[score_col], n_deciles)
        labeled = group.assign(decile=deciles)
        for decile, bucket in labeled.groupby("decile", sort=True):
            decile_rows.append(
                {
                    "signal_date": pd.Timestamp(signal_date),
                    "realized_date": add_months(pd.Timestamp(signal_date), 1),
                    "decile": int(decile),
                    "ret": float(bucket[return_col].mean()),
                    "n": int(len(bucket)),
                }
            )
        long_names = labeled.index[labeled["decile"] == n_deciles]
        short_names = labeled.index[labeled["decile"] == 1]
        if len(long_names) == 0 or len(short_names) == 0:
            continue
        weights_ls = pd.Series(0.0, index=labeled.index)
        weights_lo = pd.Series(0.0, index=labeled.index)
        weights_ls.loc[long_names] = 1.0 / len(long_names)
        weights_ls.loc[short_names] = -1.0 / len(short_names)
        weights_lo.loc[long_names] = 1.0 / len(long_names)
        gross_ls = float((weights_ls * labeled[return_col]).sum())
        gross_lo = float((weights_lo * labeled[return_col]).sum())
        turn_ls, cost_ls = _turnover_and_cost(weights_ls, previous_ls, cost_rate)
        turn_lo, cost_lo = _turnover_and_cost(weights_lo, previous_lo, cost_rate)
        realized = add_months(pd.Timestamp(signal_date), 1)
        long_short_rows.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "realized_date": realized,
                "gross": gross_ls,
                "cost": cost_ls,
                "net": gross_ls - cost_ls,
                "turnover": turn_ls,
                "n_long": int(len(long_names)),
                "n_short": int(len(short_names)),
            }
        )
        long_only_rows.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "realized_date": realized,
                "gross": gross_lo,
                "cost": cost_lo,
                "net": gross_lo - cost_lo,
                "turnover": turn_lo,
                "n_long": int(len(long_names)),
                "n_short": 0,
            }
        )
        previous_ls = weights_ls
        previous_lo = weights_lo

    long_short = pd.DataFrame(long_short_rows)
    long_only = pd.DataFrame(long_only_rows)
    deciles_out = pd.DataFrame(decile_rows)
    return BacktestResult(long_short=long_short, long_only=long_only, deciles=deciles_out)
