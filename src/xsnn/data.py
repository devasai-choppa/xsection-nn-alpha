"""Download and cache prices, S&P 500 membership, and Ken French factors.

Raw files live under ``data.cache_dir``. A file that already exists is reused
and is never downloaded again. Point-in-time constituent files are optional;
when none is configured the universe is today's S&P 500, which embeds
survivorship bias (documented in the README).
"""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xsnn.features import month_end

logger = logging.getLogger(__name__)

FRENCH_URLS = {
    "F-F_Research_Data_5_Factors_2x3": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    ),
    "F-F_Momentum_Factor": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
    ),
}

USER_AGENT = "xsection-nn-alpha/1.0 (research; educational; +https://github.com/TechieGoku2623)"


@dataclass
class MarketData:
    """Daily prices, the benchmark, monthly factors in decimal, and the universe."""

    prices: pd.DataFrame
    volume: pd.DataFrame
    spy: pd.Series
    factors: pd.DataFrame
    universe: pd.DataFrame


def _cache_dir(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["data"]["cache_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _request(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _retry(label: str, fn, attempts: int = 4):
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # network and parser errors are retried
            last_error = exc
            wait = 2**attempt
            logger.warning("%s failed (attempt %s/%s): %s", label, attempt + 1, attempts, exc)
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def fetch_sp500_wikipedia(url: str) -> pd.DataFrame:
    """Current S&P 500 constituents from the Wikipedia membership table."""
    html = _retry("Wikipedia S&P 500", lambda: _request(url))
    tables = pd.read_html(io.BytesIO(html))
    for table in tables:
        columns = {str(column).strip().lower(): column for column in table.columns}
        if "symbol" not in columns:
            continue
        frame = table.rename(columns={columns["symbol"]: "ticker"})
        frame["ticker"] = (
            frame["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
        )
        keep = ["ticker"]
        for source, target in (("security", "name"), ("gics sector", "sector"), ("gics sub-industry", "industry")):
            if source in columns:
                frame = frame.rename(columns={columns[source]: target})
                keep.append(target)
        out = frame.loc[:, keep].drop_duplicates("ticker").reset_index(drop=True)
        if out.empty:
            break
        logger.info("Parsed %s current S&P 500 tickers from Wikipedia", len(out))
        return out
    raise RuntimeError(f"Could not parse an S&P 500 symbol table from {url}")


def load_universe(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the cached constituent list, downloading it once if needed."""
    cache = _cache_dir(cfg)
    path = cache / "universe.csv"
    meta_path = cache / "universe_meta.json"
    if path.exists():
        universe = pd.read_csv(path)
        logger.info("Using cached universe at %s (%s names)", path, len(universe))
        return universe
    universe = fetch_sp500_wikipedia(str(cfg["data"]["wikipedia_url"]))
    max_tickers = cfg["data"].get("max_tickers")
    if max_tickers:
        universe = universe.sort_values("ticker").head(int(max_tickers)).reset_index(drop=True)
        logger.info("Restricted universe to %s tickers via max_tickers", len(universe))
    path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "source": cfg["data"]["wikipedia_url"],
                "n_tickers": int(len(universe)),
                "note": "Current constituents only. Historical membership is not point-in-time.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return universe


def load_pit_constituents(cfg: dict[str, Any]) -> pd.DataFrame | None:
    """Optional user-supplied point-in-time membership file."""
    raw_path = cfg["data"].get("pit_constituents_csv")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.exists():
        raise FileNotFoundError(f"Point-in-time constituent file not found: {path}")
    frame = pd.read_csv(path)
    logger.info("Loaded point-in-time constituents from %s (%s rows)", path, len(frame))
    return frame


def _extract_close_volume(downloaded: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Normalize the several shapes ``yfinance.download`` returns."""
    frames: dict[str, pd.DataFrame] = {}
    if downloaded is None or downloaded.empty:
        return frames
    columns = downloaded.columns
    if isinstance(columns, pd.MultiIndex):
        levels = [set(map(str, columns.get_level_values(i))) for i in range(columns.nlevels)]
        field_level = 0 if {"Close", "Adj Close", "Volume"} & levels[0] else 1
        ticker_level = 1 - field_level
        for ticker in tickers:
            if ticker not in columns.get_level_values(ticker_level):
                continue
            block = downloaded.xs(ticker, axis=1, level=ticker_level)
            close = _first_column(block, ("Close", "Adj Close"))
            volume = _first_column(block, ("Volume",))
            if close is None:
                continue
            frames[ticker] = _price_frame(close, volume)
        return frames
    close = _first_column(downloaded, ("Close", "Adj Close"))
    volume = _first_column(downloaded, ("Volume",))
    if close is not None and len(tickers) == 1:
        frames[tickers[0]] = _price_frame(close, volume)
    return frames


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            series = frame[name]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return series
    return None


def _price_frame(close: pd.Series, volume: pd.Series | None) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(close.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out = pd.DataFrame({"close": pd.to_numeric(close, errors="coerce").to_numpy()}, index=idx.normalize())
    if volume is None:
        out["volume"] = np.nan
    else:
        vol_idx = pd.DatetimeIndex(pd.to_datetime(volume.index))
        if vol_idx.tz is not None:
            vol_idx = vol_idx.tz_convert("UTC").tz_localize(None)
        vol = pd.Series(pd.to_numeric(volume, errors="coerce").to_numpy(), index=vol_idx.normalize())
        out["volume"] = vol.reindex(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.loc[out["close"] > 0]
    return out


def download_prices(cfg: dict[str, Any], tickers: list[str]) -> None:
    """Cache one parquet per ticker. Existing files are left untouched."""
    import yfinance as yf

    cache = _cache_dir(cfg) / "prices"
    cache.mkdir(parents=True, exist_ok=True)
    start = str(cfg["data"]["start"])
    end = cfg["data"].get("end")
    missing = [ticker for ticker in tickers if not (cache / f"{_safe_name(ticker)}.parquet").exists()]
    if not missing:
        logger.info("All %s price files are already cached", len(tickers))
        return
    logger.info("Downloading %s tickers (%s already cached)", len(missing), len(tickers) - len(missing))
    batch_size = 40
    for start_idx in range(0, len(missing), batch_size):
        batch = missing[start_idx : start_idx + batch_size]
        logger.info("Price batch %s-%s / %s", start_idx + 1, start_idx + len(batch), len(missing))
        downloaded = _retry(
            f"yfinance {batch[0]}..{batch[-1]}",
            lambda batch=batch: yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            ),
        )
        frames = _extract_close_volume(downloaded, batch)
        failed = [ticker for ticker in batch if ticker not in frames or frames[ticker].empty]
        for ticker in failed:
            single = _retry(
                f"yfinance {ticker}",
                lambda ticker=ticker: yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                ),
                attempts=3,
            )
            one = _extract_close_volume(single, [ticker])
            if ticker in one and not one[ticker].empty:
                frames[ticker] = one[ticker]
        for ticker, frame in frames.items():
            if frame.empty:
                continue
            frame.to_parquet(cache / f"{_safe_name(ticker)}.parquet")
        still_missing = [ticker for ticker in batch if not (cache / f"{_safe_name(ticker)}.parquet").exists()]
        if still_missing:
            logger.warning("No price history returned for %s", ", ".join(still_missing))
        time.sleep(1.0)


def _safe_name(ticker: str) -> str:
    return ticker.replace("/", "-")


def load_prices(cfg: dict[str, Any], tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Read cached per-ticker files into aligned close and volume matrices."""
    cache = _cache_dir(cfg) / "prices"
    closes: list[pd.Series] = []
    volumes: list[pd.Series] = []
    found: list[str] = []
    for ticker in tickers:
        path = cache / f"{_safe_name(ticker)}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty or "close" not in frame.columns:
            continue
        close = frame["close"].copy()
        close_idx = pd.DatetimeIndex(pd.to_datetime(close.index))
        if close_idx.tz is not None:
            close_idx = close_idx.tz_convert("UTC").tz_localize(None)
        close.index = close_idx.normalize()
        close.name = ticker
        if "volume" in frame.columns:
            volume = frame["volume"].copy()
        else:
            volume = pd.Series(np.nan, index=close.index)
        volume.index = close.index
        volume.name = ticker
        closes.append(close)
        volumes.append(volume)
        found.append(ticker)
    if not found:
        raise FileNotFoundError(f"No cached prices in {cache}. Run `make data` first.")
    prices = pd.concat(closes, axis=1).sort_index()
    volume = pd.concat(volumes, axis=1).sort_index().reindex(prices.index)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    volume = volume.reindex(columns=prices.columns)
    benchmark = str(cfg["data"].get("benchmark", "SPY"))
    if benchmark not in prices.columns:
        raise FileNotFoundError(f"Benchmark {benchmark} is not in the price cache")
    spy = prices[benchmark].copy()
    spy.name = benchmark
    keep = [column for column in prices.columns if column != benchmark]
    logger.info(
        "Loaded prices for %s names from %s to %s",
        len(keep),
        prices.index.min().date(),
        prices.index.max().date(),
    )
    return prices.loc[:, keep], volume.loc[:, keep], spy


def _parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French monthly CSV, stopping before the annual block."""
    lines = text.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if "Mkt-RF" in line or ",Mom" in line or line.startswith(",Mom"):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("French CSV header not found")
    header = [part.strip() for part in lines[header_idx].split(",")]
    rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts[0]) != 6 or not parts[0].isdigit():
            if rows:
                break
            continue
        rows.append(parts)
    columns = header[1:] if header and header[0] == "" else header
    # The first cell of the header row is blank, so data rows have an extra date.
    frame = pd.DataFrame(rows, columns=["date", *columns[: len(rows[0]) - 1]])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m")
    frame["date"] = frame["date"].map(lambda value: month_end(value))
    for column in frame.columns:
        if column == "date":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.set_index("date").sort_index()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _french_via_datareader(dataset: str, start: str) -> pd.DataFrame:
    import pandas_datareader.data as web

    raw = web.DataReader(dataset, "famafrench", start=start)
    monthly = raw[0]
    frame = monthly.copy()
    if hasattr(frame.index, "to_timestamp"):
        frame.index = frame.index.to_timestamp("M")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).map(month_end)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame.index.name = "date"
    return frame.sort_index().apply(pd.to_numeric, errors="coerce")


def _french_via_http(dataset: str) -> pd.DataFrame:
    url = FRENCH_URLS[dataset]
    payload = _retry(f"French {dataset}", lambda: _request(url))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = archive.namelist()[0]
        text = archive.read(name).decode("latin1")
    return _parse_french_csv(text)


def _load_one_french(dataset: str, start: str, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).map(month_end)
        logger.info("Using cached French dataset %s", dataset)
        return frame.sort_index()
    try:
        frame = _french_via_datareader(dataset, start)
        logger.info("Downloaded %s via pandas-datareader", dataset)
    except Exception as exc:
        logger.warning("pandas-datareader failed for %s (%s); using the French CSV mirror", dataset, exc)
        frame = _french_via_http(dataset)
    # Ken French files are in percent.
    frame = frame / 100.0
    frame.to_parquet(cache_path)
    return frame


def load_factors(cfg: dict[str, Any]) -> pd.DataFrame:
    """Monthly FF5 + momentum in decimal units, indexed by month-end."""
    cache = _cache_dir(cfg)
    start = str(cfg["data"]["start"])
    ff5 = _load_one_french(str(cfg["data"]["french_ff5"]), start, cache / "ff5.parquet")
    mom = _load_one_french(str(cfg["data"]["french_mom"]), start, cache / "mom.parquet")
    mom_col = "Mom" if "Mom" in mom.columns else str(mom.columns[0])
    factors = ff5.join(mom[[mom_col]].rename(columns={mom_col: "Mom"}), how="inner")
    rename = {column: column.strip() for column in factors.columns}
    factors = factors.rename(columns=rename)
    required = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]
    missing = [column for column in required if column not in factors.columns]
    if missing:
        raise KeyError(f"French factors missing {missing}; columns are {list(factors.columns)}")
    factors = factors.loc[:, required].sort_index()
    factors = factors.loc[factors.index >= month_end(start)]
    factors.to_parquet(cache / "factors.parquet")
    logger.info(
        "Factors cover %s to %s (%s months)",
        factors.index.min().date(),
        factors.index.max().date(),
        len(factors),
    )
    return factors


def download_all(cfg: dict[str, Any]) -> MarketData:
    """Populate the raw cache and return the aligned market data."""
    universe = load_universe(cfg)
    tickers = universe["ticker"].astype(str).tolist()
    benchmark = str(cfg["data"].get("benchmark", "SPY"))
    download_prices(cfg, sorted(set([*tickers, benchmark])))
    prices, volume, spy = load_prices(cfg, [*tickers, benchmark])
    factors = load_factors(cfg)
    return MarketData(prices=prices, volume=volume, spy=spy, factors=factors, universe=universe)


def load_market_data(cfg: dict[str, Any]) -> MarketData:
    """Load the cache built by :func:`download_all` without hitting the network."""
    universe = load_universe(cfg) if (Path(cfg["data"]["cache_dir"]) / "universe.csv").exists() else None
    if universe is None:
        raise FileNotFoundError("Universe cache is missing. Run `make data` first.")
    tickers = universe["ticker"].astype(str).tolist()
    benchmark = str(cfg["data"].get("benchmark", "SPY"))
    prices, volume, spy = load_prices(cfg, [*tickers, benchmark])
    factors_path = Path(cfg["data"]["cache_dir"]) / "factors.parquet"
    if factors_path.exists():
        factors = pd.read_parquet(factors_path)
        factors.index = pd.DatetimeIndex(pd.to_datetime(factors.index)).map(month_end)
        factors = factors.sort_index()
    else:
        factors = load_factors(cfg)
    return MarketData(prices=prices, volume=volume, spy=spy, factors=factors, universe=universe)
