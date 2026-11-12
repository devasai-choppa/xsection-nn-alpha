"""Point-in-time feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xsnn.features import (
    FEATURE_COLUMNS,
    apply_pit_membership,
    assemble_panel,
    build_feature_panel,
    drop_incomplete_trailing_month,
    rolling_beta_idio,
)


def _raw_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-31", "2020-01-31", "2020-01-31", "2020-01-31"])
    tickers = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
    raw = pd.DataFrame(index=index, columns=[*FEATURE_COLUMNS, "fwd_ret"], dtype=float)
    for column in FEATURE_COLUMNS:
        raw[column] = [1.0, 2.0, 3.0, 4.0]
    raw.loc[("2020-01-31", "C"), "mom_12_1"] = np.nan
    raw["fwd_ret"] = [0.01, 0.02, -0.01, 0.04]
    return raw


def test_rank_normalization_endpoints_and_median_fill() -> None:
    raw = _raw_panel()
    raw["mom_12_1"] = [1.0, 2.0, 3.0, 4.0]
    raw["amihud"] = [1.0, 2.0, np.nan, 4.0]
    panel = assemble_panel(raw)
    mom = panel["mom_12_1"].xs(pd.Timestamp("2020-01-31"), level="date")
    span = pd.Series({"A": -1.0, "B": -1.0 / 3.0, "C": 1.0 / 3.0, "D": 1.0})
    pd.testing.assert_series_equal(mom.reindex(span.index), span, check_names=False, atol=1e-12)
    # Median of 1, 2, 4 is 2, so the filled name ties B and both map to 0.
    amihud = panel["amihud"].xs(pd.Timestamp("2020-01-31"), level="date")
    expected = pd.Series({"A": -1.0, "B": 0.0, "C": 0.0, "D": 1.0})
    pd.testing.assert_series_equal(amihud.reindex(expected.index), expected, check_names=False)
    assert panel["mom_12_1"].between(-1.0, 1.0).all()
    target = panel["fwd_ret_rank"].xs(pd.Timestamp("2020-01-31"), level="date")
    assert target.loc["C"] == -1.0
    assert target.loc["D"] == 1.0
    dropped = _raw_panel()
    kept = assemble_panel(dropped)
    assert "C" not in kept.xs(pd.Timestamp("2020-01-31"), level="date").index


def test_pit_interval_filter() -> None:
    raw = _raw_panel()
    # Give February rows as well so the interval can cut them.
    feb = raw.copy()
    feb.index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-02-29"]), ["A", "B", "C", "D"]],
        names=["date", "ticker"],
    )
    both = pd.concat([raw, feb])
    membership = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "start_date": ["2020-01-01", "2020-02-01"],
            "end_date": ["2020-02-29", "2020-02-29"],
        }
    )
    kept = apply_pit_membership(both, membership)
    tickers_jan = set(kept.xs(pd.Timestamp("2020-01-31"), level="date").index)
    tickers_feb = set(kept.xs(pd.Timestamp("2020-02-29"), level="date").index)
    assert tickers_jan == {"A"}
    assert tickers_feb == {"A", "B"}


def test_long_lookback_features_are_not_median_filled() -> None:
    prices, volume, spy = _market(n_days=700, n_stocks=8, seed=3)
    panel = build_feature_panel(prices, volume, spy)
    first = panel.index.get_level_values("date").min()
    assert first >= prices.index.min() + pd.DateOffset(months=12)
    assert panel["mom_12_1_raw"].notna().all()
    assert panel["beta_252d_raw"].notna().all()


def test_partial_trailing_month_is_dropped() -> None:
    index = pd.bdate_range("2020-01-02", "2020-02-10")
    prices = pd.DataFrame({"A": np.arange(len(index), dtype=float) + 10.0}, index=index)
    trimmed = drop_incomplete_trailing_month(prices)
    assert trimmed.index.max() == pd.Timestamp("2020-01-31")
    assert trimmed.index.min() == index.min()


def test_rolling_beta_recovers_known_exposure() -> None:
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2010-01-04", periods=600)
    spy = pd.Series(rng.normal(0.0, 0.01, len(index)), index=index)
    noise = rng.normal(0.0, 0.002, len(index))
    stock = pd.DataFrame({"AAA": 1.5 * spy.to_numpy() + noise}, index=index)
    beta, idio = rolling_beta_idio(stock, spy, window=252)
    assert abs(float(beta["AAA"].dropna().iloc[-1]) - 1.5) < 0.15
    assert float(idio["AAA"].dropna().iloc[-1]) > 0.0


def _market(n_days: int = 900, n_stocks: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-02", periods=n_days)
    spy_ret = rng.normal(0.0002, 0.01, n_days)
    betas = rng.uniform(0.4, 1.6, n_stocks)
    idio = rng.normal(0.0, 0.012, size=(n_days, n_stocks))
    rets = betas * spy_ret[:, None] + idio
    prices = pd.DataFrame(
        50.0 * np.exp(np.cumsum(rets, axis=0)),
        index=index,
        columns=[f"S{i:02d}" for i in range(n_stocks)],
    )
    volume = pd.DataFrame(
        rng.integers(50_000, 500_000, size=prices.shape),
        index=index,
        columns=prices.columns,
        dtype=float,
    )
    spy = pd.Series(100.0 * np.exp(np.cumsum(spy_ret)), index=index, name="SPY")
    return prices, volume, spy


def test_features_unchanged_when_future_is_appended() -> None:
    prices, volume, spy = _market()
    cutoff = prices.loc[: "2017-06-30"].index.max()
    full = build_feature_panel(prices, volume, spy)
    truncated = build_feature_panel(prices.loc[:cutoff], volume.loc[:cutoff], spy.loc[:cutoff])
    feature_dates = truncated.index.get_level_values("date")
    last_month = feature_dates.max()
    comparable = feature_dates < last_month
    left = truncated.loc[comparable, FEATURE_COLUMNS].sort_index()
    right = full.reindex(left.index)[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-10, atol=1e-8)
    # The label is allowed to change once the next month exists; features are not.
    mutated = prices.copy()
    mutated.loc[mutated.index > cutoff] = mutated.loc[mutated.index > cutoff] * 3.0
    shocked = build_feature_panel(mutated, volume, spy)
    shocked_aligned = shocked.reindex(left.index)[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(left, shocked_aligned, check_exact=False, rtol=1e-10, atol=1e-8)
    label_date = last_month
    full_label = full.xs(label_date, level="date")["fwd_ret"]
    trunc_label = truncated.xs(label_date, level="date")["fwd_ret"]
    assert trunc_label.isna().all()
    assert full_label.notna().any()
