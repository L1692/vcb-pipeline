"""Strategy plug-ins: signal generators consumed by the engine.

Each ``prepare_*`` function takes raw OHLC bars and returns a frame with an
``atr`` column plus precomputed boolean ``go_long``/``go_short`` columns.
Frames without those columns fall back to the engine's built-in VCB logic.

Design note: TPB deliberately uses only *continuous* indicators (EMAs, RSI).
Cross-feed validation of the squeeze-based VCB showed that binary threshold
conditions are fragile to bar composition on CFD indices; smooth indicators
degrade gracefully when the feed changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .transform import ATR_LEN, add_indicators

__all__ = ["prepare_vcb", "prepare_tpb"]


def prepare_vcb(df: pd.DataFrame, mult_kc: float, trend_len: int) -> pd.DataFrame:
    """VCB features (engine computes signals inline from these columns)."""
    return add_indicators(df, mult_kc, trend_len)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def prepare_tpb(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50,
                ema_regime: int = 200, rsi_len: int = 14,
                arm_lo: float = 45.0) -> pd.DataFrame:
    """TPB — Trend Pullback: pullback-resumption entries within an EMA regime.

    Long setup: regime up (``EMA_slow > EMA_regime`` and close above the
    regime EMA); a pullback *arms* the setup when RSI dips below ``arm_lo``;
    entry triggers when price closes back above the fast EMA. Shorts mirror
    the logic with ``arm_hi = 100 - arm_lo``.
    """
    df = df.copy()
    c, h, low = df["Close"], df["High"], df["Low"]

    ef = c.ewm(span=ema_fast, adjust=False).mean()
    es = c.ewm(span=ema_slow, adjust=False).mean()
    er = c.ewm(span=ema_regime, adjust=False).mean()
    rsi = _rsi(c, rsi_len)

    tr = pd.concat([h - low, (h - c.shift()).abs(), (low - c.shift()).abs()],
                   axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_LEN, adjust=False).mean()

    cross_up = (c > ef) & (c.shift() <= ef.shift())
    cross_dn = (c < ef) & (c.shift() >= ef.shift())

    arm_hi = 100.0 - arm_lo
    armed_l = np.zeros(len(df), dtype=bool)
    armed_s = np.zeros(len(df), dtype=bool)
    al = as_ = False
    for i, (r_, cu, cd) in enumerate(zip(rsi.values, cross_up.values,
                                         cross_dn.values)):
        if r_ < arm_lo:
            al = True
        if r_ > arm_hi:
            as_ = True
        armed_l[i], armed_s[i] = al, as_
        if cu:            # trigger consumes the armed state
            al = False
        if cd:
            as_ = False

    df["go_long"] = armed_l & cross_up.values & (es > er).values & (c > er).values
    df["go_short"] = armed_s & cross_dn.values & (es < er).values & (c < er).values
    return df.dropna()
