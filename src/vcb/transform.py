"""Transform layer: derive strategy features from raw OHLC bars.

Implements the VCB (Volatility Compression Breakout) feature set:

- Bollinger Bands vs Keltner Channels squeeze detection (TTM-style);
- LSMA momentum oscillator (least-squares endpoint, computed as
  ``3*WMA - 2*SMA`` which is algebraically identical and much faster
  than a rolling regression);
- EMA trend filter and Wilder-style ATR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEN_BB = 20      #: lookback for Bollinger/Keltner/momentum
MULT_BB = 2.0    #: Bollinger standard-deviation multiplier
ATR_LEN = 14     #: ATR smoothing length


def wma(s: pd.Series, n: int) -> pd.Series:
    """Linearly weighted moving average."""
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def add_indicators(df: pd.DataFrame, mult_kc: float, trend_len: int) -> pd.DataFrame:
    """Append feature columns to an OHLC frame and drop the warm-up rows.

    Added columns: ``sqz_on``, ``release``, ``val``, ``val_prev``,
    ``ema``, ``atr``.
    """
    df = df.copy()
    c, h, low = df["Close"], df["High"], df["Low"]

    basis = c.rolling(LEN_BB).mean()
    dev = MULT_BB * c.rolling(LEN_BB).std(ddof=0)
    tr = pd.concat([h - low, (h - c.shift()).abs(), (low - c.shift()).abs()],
                   axis=1).max(axis=1)
    kc = mult_kc * tr.rolling(LEN_BB).mean()

    df["sqz_on"] = ((basis - dev) > (basis - kc)) & ((basis + dev) < (basis + kc))
    df["release"] = df["sqz_on"].shift(1).fillna(False) & ~df["sqz_on"]

    mid = ((h.rolling(LEN_BB).max() + low.rolling(LEN_BB).min()) / 2
           + c.rolling(LEN_BB).mean()) / 2
    src = c - mid
    df["val"] = 3 * wma(src, LEN_BB) - 2 * src.rolling(LEN_BB).mean()
    df["val_prev"] = df["val"].shift(1)

    df["ema"] = c.ewm(span=trend_len, adjust=False).mean()
    df["atr"] = tr.ewm(alpha=1 / ATR_LEN, adjust=False).mean()
    return df.dropna()
