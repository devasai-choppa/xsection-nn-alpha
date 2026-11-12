"""Point-in-time cross-sectional features.

Every feature dated ``t`` is a function of prices and volume on or before the
month-end ``t``. ``fwd_ret`` is the simple return realized over the following
month and is a label, never a model input.

Momentum definitions at month-end ``t`` (``P`` is the month-end adjusted close):

- ``mom_12_1 = P_{t-1} / P_{t-12} - 1`` (skip the most recent month)
- ``mom_6_1 = P_{t-1} / P_{t-6} - 1``
- ``mom_3_1 = P_{t-1} / P_{t-3} - 1``
- ``rev_1m = P_t / P_{t-1} - 1`` (short-term reversal input)

Cross-sectional transform, applied independently on each date: values with more
than ``max_missing_frac`` of features missing are dropped; remaining missing
feature entries are filled with that date's median; average ranks are then
affinely mapped onto ``[-1, 1]`` so the worst name is -1 and the best is +1.
An all-tie column maps to 0. The ranked target uses the same map and is not
median-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "mom_12_1",
    "mom_6_1",
    "mom_3_1",
    "rev_1m",
    "vol_20d",
    "vol_60d",
    "downside_vol",
    "log_dollar_volume",
    "amihud",
    "beta_252d",
    "idio_vol",
    "max_daily_ret",
]

# These have to be observed. Median-filling a 12-month momentum or a 252-day beta
# that was never measured would invent a signal. Liquidity features can still be
# filled when a print is missing.
REQUIRED_RAW: list[str] = [
    "mom_12_1",
    "mom_6_1",
    "mom_3_1",
    "rev_1m",
    "vol_20d",
    "vol_60d",
    "downside_vol",
    "beta_252d",
    "idio_vol",
    "max_daily_ret",
]

RAW_SUFFIX = "_raw"


def month_end(ts: pd.Timestamp | str) -> pd.Timestamp:
    """Normalize a timestamp to a timezone-naive calendar month-end."""
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(stamp.to_period("M").to_timestamp("M"))


def add_months(ts: pd.Timestamp | str, months: int) -> pd.Timestamp:
    """Shift by whole calendar months and snap to month-end."""
    stamp = month_end(ts)
    return pd.Timestamp((stamp.to_period("M") + int(months)).to_timestamp("M"))


def drop_incomplete_trailing_month(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop the last calendar month when the sample does not reach its last session.

    A monthly bar labeled with that month-end would otherwise mix a partial month
    into the next-month return of the previous bar.
    """
    if prices.empty:
        return prices
    last = pd.Timestamp(prices.index.max())
    if last.tzinfo is not None:
        last = last.tz_convert("UTC").tz_localize(None)
    last = last.normalize()
    period = last.to_period("M")
    sessions = pd.bdate_range(period.to_timestamp(), period.to_timestamp(how="end").normalize())
    if len(sessions) == 0 or last >= sessions[-1]:
        return prices
    keep = pd.DatetimeIndex(prices.index).to_period("M") < period
    return prices.loc[keep]


def _as_naive_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _as_naive_series(series: pd.Series) -> pd.Series:
    out = series.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.astype(float)


def _sqrt_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.sqrt(frame.to_numpy(dtype=float)), index=frame.index, columns=frame.columns)


def _resample_month_end(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    try:
        return frame.resample("ME").last() if not isinstance(frame, pd.Series) else frame.resample("ME").last()
    except ValueError:
        return frame.resample("M").last()


def _month_max(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        return frame.resample("ME").max()
    except ValueError:
        return frame.resample("M").max()


def signed_rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Map each column within each row-group date from average rank onto [-1, 1].

    ``frame`` must be indexed by ``(date, ticker)`` or share a same-length
    ``date`` index level. NaNs stay NaN and do not consume a rank.
    """
    dates = frame.index.get_level_values("date")
    ranks = frame.groupby(dates, sort=False).rank(method="average")
    counts = frame.notna().groupby(dates, sort=False).transform("sum")
    denom = (counts - 1.0).where(counts > 1)
    return (ranks - 1.0) / denom * 2.0 - 1.0


def signed_rank_series(series: pd.Series) -> pd.Series:
    """Cross-sectionally map one MultiIndex series onto [-1, 1]."""
    dates = series.index.get_level_values("date")
    ranks = series.groupby(dates, sort=False).rank(method="average")
    counts = series.notna().groupby(dates, sort=False).transform("sum")
    denom = (counts - 1.0).where(counts > 1)
    scaled = (ranks - 1.0) / denom * 2.0 - 1.0
    scaled.name = series.name
    return scaled


def rolling_beta_idio(
    daily_ret: pd.DataFrame,
    spy_ret: pd.Series,
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling OLS beta and annualized residual volatility versus SPY.

    Sample moments (pandas ``ddof=1``) over ``window`` observations. Residual
    variance is ``var(stock) - beta^2 var(SPY)``, floored at zero.
    """
    var_m = spy_ret.rolling(window, min_periods=window).var()
    cov = daily_ret.rolling(window, min_periods=window).cov(spy_ret)
    beta = cov.div(var_m.replace(0.0, np.nan), axis=0)
    var_s = daily_ret.rolling(window, min_periods=window).var()
    idio_var = (var_s - beta.pow(2).mul(var_m, axis=0)).clip(lower=0.0)
    idio = _sqrt_frame(idio_var) * np.sqrt(252.0)
    return beta, idio


def compute_monthly_raw_features(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    spy: pd.Series,
    *,
    beta_window: int = 252,
    vol_short: int = 20,
    vol_long: int = 60,
) -> pd.DataFrame:
    """Long panel of raw features and the next-month return.

    The panel uses only rows of ``prices``, ``volume``, and ``spy``. Truncating
    those inputs at date ``T`` leaves features on month-ends ``<= T`` unchanged.
    """
    px = drop_incomplete_trailing_month(_as_naive_frame(prices).apply(pd.to_numeric, errors="coerce"))
    px = px.where(px > 0.0)
    vol = _as_naive_frame(volume).apply(pd.to_numeric, errors="coerce")
    vol = vol.reindex(index=px.index, columns=px.columns)
    market = _as_naive_series(spy).reindex(px.index)

    daily_ret = px.pct_change()
    spy_ret = market.pct_change()
    ann = float(np.sqrt(252.0))
    vol_20 = daily_ret.rolling(vol_short, min_periods=vol_short).std() * ann
    vol_60 = daily_ret.rolling(vol_long, min_periods=vol_long).std() * ann
    downside_var = daily_ret.clip(upper=0.0).pow(2).rolling(vol_long, min_periods=vol_long).mean()
    downside = _sqrt_frame(downside_var) * ann

    dollar_volume = px * vol
    adv = dollar_volume.rolling(vol_short, min_periods=vol_short).mean()
    log_values = np.log(adv.where(adv > 0.0).to_numpy(dtype=float))
    log_dv = pd.DataFrame(log_values, index=adv.index, columns=adv.columns)
    amihud = (
        daily_ret.abs()
        .div(dollar_volume.where(dollar_volume > 0.0))
        .rolling(vol_short, min_periods=vol_short)
        .mean()
    )
    beta, idio = rolling_beta_idio(daily_ret, spy_ret, beta_window)

    monthly_price = _resample_month_end(px)
    assert isinstance(monthly_price, pd.DataFrame)
    mom_12_1 = monthly_price.shift(1) / monthly_price.shift(12) - 1.0
    mom_6_1 = monthly_price.shift(1) / monthly_price.shift(6) - 1.0
    mom_3_1 = monthly_price.shift(1) / monthly_price.shift(3) - 1.0
    rev_1m = monthly_price / monthly_price.shift(1) - 1.0
    fwd_ret = monthly_price.shift(-1) / monthly_price - 1.0

    monthly = {
        "mom_12_1": mom_12_1,
        "mom_6_1": mom_6_1,
        "mom_3_1": mom_3_1,
        "rev_1m": rev_1m,
        "vol_20d": _resample_month_end(vol_20),
        "vol_60d": _resample_month_end(vol_60),
        "downside_vol": _resample_month_end(downside),
        "log_dollar_volume": _resample_month_end(log_dv),
        "amihud": _resample_month_end(amihud),
        "beta_252d": _resample_month_end(beta),
        "idio_vol": _resample_month_end(idio),
        "max_daily_ret": _month_max(daily_ret),
        "fwd_ret": fwd_ret,
    }
    pieces = [_stack_wide(frame, name) for name, frame in monthly.items()]
    panel = pd.concat(pieces, axis=1, join="outer")
    panel = panel.replace([np.inf, -np.inf], np.nan).sort_index()
    return panel


def _stack_wide(wide: pd.DataFrame | pd.Series, name: str) -> pd.Series:
    frame = wide.copy()
    if isinstance(frame, pd.Series):
        raise TypeError(f"{name} must be a wide DataFrame")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).to_period("M").to_timestamp("M")
    frame.index.name = "date"
    long = frame.reset_index().melt(id_vars="date", var_name="ticker", value_name=name)
    long["ticker"] = long["ticker"].astype(str)
    long["date"] = pd.to_datetime(long["date"])
    return long.set_index(["date", "ticker"])[name].sort_index()


def apply_pit_membership(panel: pd.DataFrame, membership: pd.DataFrame | None) -> pd.DataFrame:
    """Restrict rows to a point-in-time constituent list.

    Accepted shapes:

    - ``date``, ``ticker``: one row per constituent-month
    - ``ticker``, ``start_date``, ``end_date``: inclusive intervals

    ``None`` or an empty frame leaves the panel unchanged.
    """
    if membership is None or len(membership) == 0:
        return panel
    mem = membership.copy()
    mem.columns = [str(column).strip().lower() for column in mem.columns]
    if {"date", "ticker"}.issubset(mem.columns):
        dates = pd.to_datetime(mem["date"]).map(month_end)
        tickers = mem["ticker"].astype(str).str.replace(".", "-", regex=False)
        allowed = pd.MultiIndex.from_arrays([pd.DatetimeIndex(dates), tickers], names=["date", "ticker"])
        return panel[panel.index.isin(allowed)].copy()
    if {"ticker", "start_date", "end_date"}.issubset(mem.columns):
        intervals = mem.copy()
        intervals["ticker"] = intervals["ticker"].astype(str).str.replace(".", "-", regex=False)
        intervals["start_date"] = pd.to_datetime(intervals["start_date"]).map(month_end)
        intervals["end_date"] = pd.to_datetime(intervals["end_date"]).map(month_end)
        dates = pd.to_datetime(panel.index.get_level_values("date")).map(month_end)
        dates_np = pd.DatetimeIndex(dates).to_numpy()
        tickers = panel.index.get_level_values("ticker").astype(str).to_numpy()
        keep = np.zeros(len(panel), dtype=bool)
        for row in intervals.itertuples(index=False):
            start = np.datetime64(pd.Timestamp(row.start_date).to_datetime64())
            end = np.datetime64(pd.Timestamp(row.end_date).to_datetime64())
            keep |= (tickers == row.ticker) & (dates_np >= start) & (dates_np <= end)
        return panel.loc[keep].copy()
    raise ValueError(
        "Point-in-time constituents need columns date,ticker or ticker,start_date,end_date"
    )


def assemble_panel(
    raw: pd.DataFrame,
    membership: pd.DataFrame | None = None,
    *,
    max_missing_frac: float = 0.5,
) -> pd.DataFrame:
    """Filter, median-fill, rank-normalize, and attach the ranked label.

    Raw feature values are preserved with a ``_raw`` suffix. Canonical feature
    names are the normalized model inputs.
    """
    missing_cols = [column for column in FEATURE_COLUMNS + ["fwd_ret"] if column not in raw.columns]
    if missing_cols:
        raise KeyError(f"Raw panel missing columns: {missing_cols}")
    panel = apply_pit_membership(raw, membership)
    panel = panel.dropna(subset=REQUIRED_RAW)
    observed = panel[FEATURE_COLUMNS]
    keep = observed.isna().mean(axis=1) <= max_missing_frac
    panel = panel.loc[keep].copy()
    raw_block = panel[FEATURE_COLUMNS].rename(columns={col: f"{col}{RAW_SUFFIX}" for col in FEATURE_COLUMNS})
    dates = panel.index.get_level_values("date")
    medians = panel[FEATURE_COLUMNS].groupby(dates, sort=False).transform("median")
    filled = panel[FEATURE_COLUMNS].fillna(medians)
    normalized = signed_rank_frame(filled)
    out = pd.concat(
        [raw_block, normalized, panel[["fwd_ret"]]],
        axis=1,
    )
    out["fwd_ret_rank"] = signed_rank_series(out["fwd_ret"])
    out.index = out.index.set_names(["date", "ticker"])
    return out.sort_index()


def build_feature_panel(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    spy: pd.Series,
    membership: pd.DataFrame | None = None,
    *,
    beta_window: int = 252,
    vol_short: int = 20,
    vol_long: int = 60,
    max_missing_frac: float = 0.5,
) -> pd.DataFrame:
    """End-to-end point-in-time feature panel."""
    raw = compute_monthly_raw_features(
        prices,
        volume,
        spy,
        beta_window=beta_window,
        vol_short=vol_short,
        vol_long=vol_long,
    )
    return assemble_panel(raw, membership, max_missing_frac=max_missing_frac)


def model_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows with a complete feature vector and a finite ranked target."""
    columns = FEATURE_COLUMNS + ["fwd_ret", "fwd_ret_rank"]
    out = panel.dropna(subset=columns).copy()
    return out.sort_index()
