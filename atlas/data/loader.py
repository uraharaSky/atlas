"""Market data loaders (CSV and Yahoo Finance) producing validated ``MarketData``."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.config.settings import (
    CLOSE,
    HIGH,
    INDEX_NAME,
    LOW,
    OPEN,
    REQUIRED_COLUMNS,
    VOLUME,
)
from atlas.data.models import MarketData
from atlas.data.validator import DataValidationError

_CANONICAL = {c.lower(): c for c in REQUIRED_COLUMNS}


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns case-insensitively and keep only the OHLCV columns.

    This changes labels and drops extra columns (e.g. 'Adj Close'); it never
    alters values. Missing columns are left missing so validation reports them.
    """
    renamed = df.rename(columns=lambda c: _CANONICAL.get(str(c).strip().lower(), c))
    present = [c for c in REQUIRED_COLUMNS if c in renamed.columns]
    out = renamed.loc[:, present]
    out.index.name = INDEX_NAME
    return out


def load_csv(path: str | Path, symbol: str | None = None, date_column: str = INDEX_NAME) -> MarketData:
    """Load OHLCV data from a CSV file and validate it.

    The file must contain a date column plus Open/High/Low/Close/Volume
    (column names are matched case-insensitively). Rows are kept exactly as in
    the file: unsorted or duplicated data raises rather than being repaired.
    """
    path = Path(path)
    raw = pd.read_csv(path)
    date_candidates = {c.lower(): c for c in raw.columns}
    if date_column.lower() not in date_candidates:
        raise DataValidationError(f"Date column '{date_column}' not found in {path.name}")
    date_col = date_candidates[date_column.lower()]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="raise")
    raw = raw.set_index(date_col)
    return MarketData(symbol=symbol or path.stem, frame=_standardize(raw), source=f"csv:{path.name}")


def load_yahoo(
    symbol: str,
    start: date | str | None = None,
    end: date | str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> MarketData:
    """Download OHLCV data from Yahoo Finance and validate it.

    ``auto_adjust=True`` returns split/dividend-adjusted prices (consistently
    across O/H/L/C). Requires network access and the ``yfinance`` package.
    Timezone information is stripped from the index (a format change only).
    """
    import yfinance as yf  # imported lazily so the rest of ATLAS works offline

    raw = yf.download(
        symbol, start=start, end=end, interval=interval, auto_adjust=auto_adjust, progress=False
    )
    if raw is None or raw.empty:
        raise DataValidationError(f"Yahoo Finance returned no data for '{symbol}'")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if isinstance(raw.index, pd.DatetimeIndex) and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    return MarketData(symbol=symbol, frame=_standardize(raw), source="yahoo")


def load_market_data(
    source: str,
    symbol: str,
    start: date | str | None = None,
    end: date | str | None = None,
    interval: str = "1d",
    csv_path: str | Path | None = None,
) -> MarketData:
    """Dispatch to a provider by name (``'csv'`` or ``'yahoo'``)."""
    if source == "csv":
        if csv_path is None:
            raise ValueError("csv_path is required when source='csv'")
        return load_csv(csv_path, symbol=symbol)
    if source == "yahoo":
        return load_yahoo(symbol, start=start, end=end, interval=interval)
    raise ValueError(f"Unknown data source '{source}' (expected 'csv' or 'yahoo')")


def generate_synthetic_ohlcv(
    n_bars: int = 750,
    seed: int = 42,
    start: str = "2020-01-01",
    initial_price: float = 100.0,
    drift: float = 0.0003,
    volatility: float = 0.01,
) -> MarketData:
    """Generate seeded random-walk OHLCV data.

    **For pipeline smoke tests only.** Results computed on this data say nothing
    about real markets. The output is deterministic for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    gaps = rng.normal(0.0, volatility / 3, n_bars)
    intraday = rng.normal(drift, volatility, n_bars)
    opens = np.empty(n_bars)
    closes = np.empty(n_bars)
    prev_close = initial_price
    for i in range(n_bars):
        opens[i] = prev_close * (1 + gaps[i])
        closes[i] = opens[i] * (1 + intraday[i])
        prev_close = closes[i]
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, volatility / 3, n_bars)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, volatility / 3, n_bars)))
    volume = rng.integers(1_000_000, 5_000_000, n_bars).astype(float)
    index = pd.bdate_range(start=start, periods=n_bars, name=INDEX_NAME)
    frame = pd.DataFrame(
        {OPEN: opens, HIGH: highs, LOW: lows, CLOSE: closes, VOLUME: volume}, index=index
    )
    return MarketData(symbol="SYNTHETIC", frame=frame, source=f"synthetic(seed={seed})")
