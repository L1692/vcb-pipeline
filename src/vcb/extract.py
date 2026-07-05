"""Extract layer: market data acquisition from heterogeneous sources.

Three supported sources, normalized to a single OHLC schema
(UTC DatetimeIndex + Open/High/Low/Close float columns):

1. Yahoo Finance (1h bars, resampled to the target timeframe);
2. TradingView "Export chart data" CSV (unix seconds/ms or ISO timestamps);
3. MetaTrader 5 bar export (tab-separated, ``<DATE>``/``<TIME>`` columns).

``yfinance`` is imported lazily so the CSV path works without it installed.
"""

from __future__ import annotations

import os

import pandas as pd

#: Yahoo Finance tickers for the supported instruments.
YAHOO_TICKERS = {
    "DAX": "^GDAXI",
    "NASDAQ": "^NDX",
    "NIKKEI": "^N225",
    "XAUUSD": "GC=F",
    "SP500": "^GSPC",
}

_RAW_CACHE: dict = {}

OHLC = ("Open", "High", "Low", "Close")


def load_yahoo_1h(ticker: str, start: str) -> pd.DataFrame:
    """Download 1h bars from Yahoo Finance (cached per process).

    Note: Yahoo serves at most ~730 days of hourly history.
    """
    key = (ticker, start)
    if key not in _RAW_CACHE:
        import yfinance as yf  # lazy: optional dependency

        df = yf.download(ticker, start=start, interval="1h",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            df.index = df.index.tz_convert("UTC")
        _RAW_CACHE[key] = df
    return _RAW_CACHE[key]


def resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample OHLC bars to a coarser timeframe (e.g. ``"2h"``, ``"4h"``)."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    return df.resample(tf).agg(agg).dropna()


def load_csv(path: str) -> pd.DataFrame:
    """Load a price CSV exported from TradingView or MetaTrader 5.

    Tolerant to separator (auto-detected), column naming, and time format:
    unix seconds, unix milliseconds, ISO strings, or MT5's split
    ``<DATE>`` + ``<TIME>`` columns (``2024.07.01`` / ``06:00:00``).

    Timestamps are treated as UTC. MT5 exports use broker server time, so
    day boundaries may be shifted by a few hours; this only affects the
    daily-stop reset and is an accepted approximation.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower().strip().strip("<>"): c for c in df.columns}

    if "date" in cols and "time" in cols and "datetime" not in cols:
        # MT5 format: separate date and time columns
        t = (df[cols["date"]].astype(str).str.replace(".", "-", regex=False)
             + " " + df[cols["time"]].astype(str))
        idx = pd.to_datetime(t, utc=True)
    else:
        tcol = next((cols[k] for k in ("time", "datetime", "date") if k in cols), None)
        if tcol is None:
            raise ValueError(f"{path}: no time column found ({list(df.columns)})")
        t = df[tcol]
        if pd.api.types.is_numeric_dtype(t):
            unit = "ms" if float(t.iloc[0]) > 1e11 else "s"
            idx = pd.to_datetime(t, unit=unit, utc=True)
        else:
            idx = pd.to_datetime(t, utc=True)

    out = pd.DataFrame(index=idx)
    for k in ("open", "high", "low", "close"):
        if k not in cols:
            raise ValueError(f"{path}: missing '{k}' column")
        out[k.capitalize()] = pd.to_numeric(df[cols[k]].values, errors="coerce")
    return out.dropna().sort_index()


def load_instrument(name: str, start: str, tf: str,
                    csv_dir: str | None = None) -> pd.DataFrame:
    """Return raw OHLC bars for one instrument.

    If ``csv_dir`` is given, ``{csv_dir}/{name}.csv`` is used verbatim
    (the exported chart's timeframe applies and ``tf`` is ignored);
    otherwise Yahoo 1h data is downloaded and resampled to ``tf``.
    """
    if csv_dir:
        path = os.path.join(csv_dir, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return load_csv(path)
    raw = load_yahoo_1h(YAHOO_TICKERS[name], start)
    if raw.empty:
        return raw
    return resample_ohlc(raw, tf)
