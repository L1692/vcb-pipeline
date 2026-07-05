"""End-to-end tests on synthetic data (no network required)."""

import numpy as np
import pandas as pd
import pytest

from vcb.engine import COMMISSION, portfolio_backtest
from vcb.extract import load_csv
from vcb.report import summarize
from vcb.transform import add_indicators


def synthetic_ohlc(seed: int, n: int = 4000, freq: str = "2h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-07-01", periods=n, freq=freq, tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0.00005, 0.004, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    op = np.roll(close, 1)
    op[0] = close[0]
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close},
                        index=idx)


@pytest.fixture
def prepared():
    return {n: add_indicators(synthetic_ohlc(s), 1.5, 100)
            for n, s in (("A", 1), ("B", 2), ("C", 3))}


BASE = {"tf": "2h", "mult_kc": 1.5, "trend_len": 100,
        "risk": 0.012, "use_flip": True}


def _span(data):
    lo = min(d.index[0] for d in data.values())
    hi = max(d.index[-1] for d in data.values())
    return lo, hi


# -- extract ------------------------------------------------------------------

def test_load_csv_formats(tmp_path):
    df = synthetic_ohlc(9, n=50)
    base = {"open": df.Open, "high": df.High, "low": df.Low, "close": df.Close}

    tv_unix = tmp_path / "unix.csv"
    pd.DataFrame({"time": [int(t.timestamp()) for t in df.index], **base}
                 ).to_csv(tv_unix, index=False)
    tv_iso = tmp_path / "iso.csv"
    pd.DataFrame({"time": df.index.strftime("%Y-%m-%dT%H:%M:%S+00:00"), **base}
                 ).to_csv(tv_iso, index=False)
    mt5 = tmp_path / "mt5.csv"
    pd.DataFrame({"<DATE>": df.index.strftime("%Y.%m.%d"),
                  "<TIME>": df.index.strftime("%H:%M:%S"),
                  "<OPEN>": df.Open, "<HIGH>": df.High,
                  "<LOW>": df.Low, "<CLOSE>": df.Close,
                  "<TICKVOL>": 1}).to_csv(mt5, index=False, sep="\t")

    for path in (tv_unix, tv_iso, mt5):
        out = load_csv(str(path))
        assert list(out.columns) == ["Open", "High", "Low", "Close"]
        assert len(out) == 50
        assert str(out.index[0]).startswith("2024-07-01")


# -- transform ----------------------------------------------------------------

def test_indicators_columns():
    df = add_indicators(synthetic_ohlc(4), 1.5, 100)
    for col in ("sqz_on", "release", "val", "val_prev", "ema", "atr"):
        assert col in df.columns
    assert df["atr"].min() > 0
    assert not df.isna().any().any()


# -- engine -------------------------------------------------------------------

def test_backtest_produces_trades(prepared):
    lo, hi = _span(prepared)
    res = portfolio_backtest(prepared, BASE, lo, hi)
    assert len(res.trades) > 10
    assert len(res.eq_curve) > 0
    assert res.worst_dd <= 0


def test_stress_degrades_profit_factor(prepared):
    lo, hi = _span(prepared)
    plain = summarize(portfolio_backtest(prepared, BASE, lo, hi), BASE)
    stressed_p = dict(BASE, comm=COMMISSION * 2, slip_atr=0.1)
    stressed = summarize(portfolio_backtest(prepared, stressed_p, lo, hi), stressed_p)
    assert stressed["PF"] <= plain["PF"]


def test_challenge_mode_resets_and_continues(prepared):
    lo, hi = _span(prepared)
    plain = portfolio_backtest(prepared, BASE, lo, hi)
    chall = portfolio_backtest(prepared, dict(BASE, challenge=True), lo, hi)
    # the plain run truncates at the first halt; challenge mode keeps going
    assert len(chall.trades) >= len(plain.trades)
    if plain.halt != "no":
        assert len(chall.challenges) >= 1


def test_custom_limits_respected(prepared):
    lo, hi = _span(prepared)
    tight = dict(BASE, challenge=True, kill=0.02)     # 2% kill: fails fast
    loose = dict(BASE, challenge=True, kill=0.20)
    r_tight = portfolio_backtest(prepared, tight, lo, hi)
    r_loose = portfolio_backtest(prepared, loose, lo, hi)
    fails_tight = sum(1 for ok, _ in r_tight.challenges if not ok)
    fails_loose = sum(1 for ok, _ in r_loose.challenges if not ok)
    assert fails_tight >= fails_loose


# -- report -------------------------------------------------------------------

def test_summarize_challenge_fields(prepared):
    lo, hi = _span(prepared)
    p = dict(BASE, challenge=True)
    out = summarize(portfolio_backtest(prepared, p, lo, hi), p)
    assert {"ch_ok", "ch_ko", "med_months"} <= set(out)
    assert "P&L%" not in out          # truncated P&L is hidden in challenge mode
