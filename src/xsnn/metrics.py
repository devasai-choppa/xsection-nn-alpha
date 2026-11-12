"""Performance, information-coefficient, and factor-attribution statistics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTOR_COLUMNS: list[str] = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def _clean(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def annualized_return(returns: pd.Series, periods_per_year: float = 12.0) -> float:
    """Compound growth of wealth, annualized by the number of observations."""
    values = _clean(returns)
    if values.empty:
        return float("nan")
    wealth = float((1.0 + values).prod())
    years = len(values) / periods_per_year
    if years <= 0.0 or wealth <= 0.0:
        return float("nan")
    return float(wealth ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: float = 12.0) -> float:
    values = _clean(returns)
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: float = 12.0) -> float:
    """Mean divided by sample standard deviation, annualized. No risk-free rate.

    Long/short spreads are already excess returns. Long-only Sharpes reported
    by this function are therefore not excess of T-bills; the factor regression
    is the excess-return attribution.
    """
    values = _clean(returns)
    if len(values) < 2:
        return float("nan")
    vol = float(values.std(ddof=1))
    if vol == 0.0 or np.isnan(vol):
        return float("nan")
    return float(values.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float = 12.0) -> float:
    """Mean over the square root of the second moment of non-positive returns."""
    values = _clean(returns)
    if len(values) < 2:
        return float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(values.to_numpy(dtype=float), 0.0) ** 2)))
    if downside == 0.0 or np.isnan(downside):
        return float("nan")
    return float(values.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough decline of compounded wealth. Always ``<= 0``."""
    values = _clean(returns)
    if values.empty:
        return float("nan")
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: float = 12.0) -> float:
    drawdown = max_drawdown(returns)
    if drawdown == 0.0 or np.isnan(drawdown):
        return float("nan")
    return float(annualized_return(returns, periods_per_year) / abs(drawdown))


def hit_rate(returns: pd.Series) -> float:
    values = _clean(returns)
    if values.empty:
        return float("nan")
    return float((values > 0.0).mean())


def summarize_returns(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    periods_per_year: float = 12.0,
) -> dict[str, float]:
    """Standard backtest summary for a monthly return stream."""
    stats = {
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_vol": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "hit_rate": hit_rate(returns),
        "n_months": float(len(_clean(returns))),
    }
    if turnover is not None:
        aligned = _clean(turnover.reindex(returns.index))
        stats["turnover"] = float(aligned.mean()) if len(aligned) else float("nan")
    else:
        stats["turnover"] = float("nan")
    return stats


def _pair(pred: pd.Series, realized: pd.Series) -> pd.DataFrame:
    frame = pd.concat(
        [pred.rename("pred").reset_index(drop=True), realized.rename("realized").reset_index(drop=True)],
        axis=1,
    )
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def _corr(pred: pd.Series, realized: pd.Series, method: str) -> float:
    pair = _pair(pred, realized)
    if len(pair) < 3:
        return float("nan")
    if pair["pred"].nunique(dropna=True) < 2 or pair["realized"].nunique(dropna=True) < 2:
        return float("nan")
    return float(pair["pred"].corr(pair["realized"], method=method))


def information_coefficient(pred: pd.Series, realized: pd.Series) -> float:
    """Pearson correlation of predictions and realized returns."""
    return _corr(pred, realized, "pearson")


def rank_information_coefficient(pred: pd.Series, realized: pd.Series) -> float:
    """Spearman correlation of predictions and realized returns."""
    return _corr(pred, realized, "spearman")


def ic_by_date(
    pred: Iterable[float],
    realized: Iterable[float],
    dates: Iterable[pd.Timestamp],
    method: str = "spearman",
) -> pd.Series:
    """One information coefficient per date."""
    frame = pd.DataFrame(
        {
            "pred": np.asarray(list(pred), dtype=float),
            "realized": np.asarray(list(realized), dtype=float),
            "date": pd.to_datetime(pd.Series(list(dates))),
        }
    )
    fn = information_coefficient if method == "pearson" else rank_information_coefficient
    rows: list[float] = []
    index: list[pd.Timestamp] = []
    for date, group in frame.groupby("date", sort=True):
        rows.append(fn(group["pred"], group["realized"]))
        index.append(pd.Timestamp(date))
    name = "ic" if method == "pearson" else "rank_ic"
    return pd.Series(rows, index=pd.DatetimeIndex(index), name=name)


def mean_ic(
    pred: Iterable[float],
    realized: Iterable[float],
    dates: Iterable[pd.Timestamp],
    method: str = "spearman",
) -> float:
    series = ic_by_date(pred, realized, dates, method=method)
    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return float("nan")
    return float(cleaned.mean())


def summarize_ic(series: pd.Series) -> dict[str, float]:
    """Mean, standard deviation, ICIR, and t-statistic of a monthly IC series.

    The t-statistic is ``mean / (std / sqrt(n))`` with a sample standard deviation.
    ICIR is ``mean / std`` and is not annualized.
    """
    values = _clean(series)
    n = int(len(values))
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "icir": float("nan"), "t_stat": float("nan"), "n": 0.0}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else float("nan")
    if std == 0.0 or np.isnan(std):
        icir = float("nan")
        t_stat = float("nan")
    else:
        icir = float(mean / std)
        t_stat = float(mean / (std / np.sqrt(n)))
    return {"mean": mean, "std": std, "icir": icir, "t_stat": t_stat, "n": float(n)}


def decile_monotonicity(mean_return_by_decile: pd.Series) -> float:
    """Spearman correlation between decile number and mean forward return."""
    values = mean_return_by_decile.dropna()
    if len(values) < 3:
        return float("nan")
    x = pd.Series(np.asarray(values.index, dtype=float))
    y = pd.Series(values.to_numpy(dtype=float))
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def factor_attribution(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    excess: bool = False,
    hac_lags: int = 3,
) -> dict[str, float]:
    """Newey-West alpha of a monthly book on FF5 + momentum.

    ``returns`` and ``factors`` are indexed by the month the return is realized.
    Factor columns are decimals. Set ``excess`` for long-only books so the
    regression uses return minus ``RF``. Long/short spreads skip that subtraction
    because the financing legs cancel.
    """
    factor_frame = factors.copy()
    factor_frame.index = pd.DatetimeIndex(pd.to_datetime(factor_frame.index)).to_period("M").to_timestamp("M")
    target = returns.copy()
    target.index = pd.DatetimeIndex(pd.to_datetime(target.index)).to_period("M").to_timestamp("M")
    joined = target.rename("y").to_frame().join(factor_frame, how="inner")
    if excess:
        if "RF" not in joined.columns:
            raise KeyError("RF is required to compute long-only excess returns")
        joined["y"] = joined["y"] - joined["RF"]
    needed = ["y", *FACTOR_COLUMNS]
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    empty = {
        "alpha_monthly": float("nan"),
        "alpha_annual": float("nan"),
        "alpha_t": float("nan"),
        "n": float(len(joined)),
        "r2": float("nan"),
    }
    if len(joined) < len(FACTOR_COLUMNS) + 5:
        return empty
    design = sm.add_constant(joined[FACTOR_COLUMNS], has_constant="add")
    fitted = sm.OLS(joined["y"], design).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
    alpha = float(fitted.params["const"])
    return {
        "alpha_monthly": alpha,
        "alpha_annual": float((1.0 + alpha) ** 12 - 1.0),
        "alpha_t": float(fitted.tvalues["const"]),
        "n": float(len(joined)),
        "r2": float(fitted.rsquared),
    }
