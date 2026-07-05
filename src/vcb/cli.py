"""Command-line orchestrator: wires extract -> transform -> engine -> report.

Two modes:

- ``single``: one configuration, portfolio backtest (optionally in challenge
  mode and/or under stressed costs);
- ``grid``: walk-forward parameter search in challenge mode. Configurations
  are ranked by challenges passed, then failed, then median months, then
  profit factor — never by truncated P&L. The best in-sample combination is
  re-evaluated out-of-sample and over the full period.
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd

from .engine import COMMISSION, portfolio_backtest
from .extract import YAHOO_TICKERS, load_instrument
from .report import summarize
from .transform import LEN_BB, add_indicators

DEFAULT_SET = "DAX,NASDAQ,XAUUSD"

GRID = {
    "tf":        ["2h", "4h"],
    "mult_kc":   [1.2, 1.5],
    "trend_len": [100, 200],
    "risk":      [0.008, 0.010, 0.012],
    "use_flip":  [True, False],
}

_PREP: dict = {}


def get_data(names: list[str], start: str, p: dict,
             csv_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """Extract + transform each instrument, with per-config caching."""
    data = {}
    for n in names:
        key = (n, start, p["tf"], p["mult_kc"], p["trend_len"], csv_dir)
        if key not in _PREP:
            try:
                raw = load_instrument(n, start, p["tf"], csv_dir)
            except FileNotFoundError as e:
                print(f"[!] {e} not found, skipping {n}.")
                _PREP[key] = pd.DataFrame()
                raw = _PREP[key]
            if not raw.empty:
                _PREP[key] = add_indicators(raw, p["mult_kc"], p["trend_len"])
            else:
                _PREP[key] = raw
        df = _PREP[key]
        if df.empty or len(df) < max(p["trend_len"], LEN_BB) + LEN_BB:
            print(f"[!] Not enough data for {n}, skipping.")
            continue
        data[n] = df
    return data


def run_single(names, start, p, csv_dir=None):
    data = get_data(names, start, p, csv_dir)
    if not data:
        return
    src = f"CSV ({csv_dir})" if csv_dir else f"Yahoo from {start}"
    for n, df in data.items():
        print(f"    {n}: {len(df)} bars, {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    t0 = max(df.index[0] for df in data.values())
    t1 = max(df.index[-1] for df in data.values())
    out = summarize(portfolio_backtest(data, p, t0, t1), p)
    print(f"\n=== VCB portfolio [{', '.join(data)}] — data: {src} ===")
    print(pd.DataFrame([out]).to_string(index=False))


def run_grid(names, start, is_frac=0.7, stress=False, limits=None):
    extra = {"challenge": True,
             "comm": COMMISSION * 2 if stress else COMMISSION,
             "slip_atr": 0.1 if stress else 0.0, **(limits or {})}
    combos = [dict(zip(GRID, v), **extra)
              for v in itertools.product(*GRID.values())]
    is_rows, cache = [], {}
    for p in combos:
        data = get_data(names, start, p)
        if not data:
            return
        lo = max(df.index[0] for df in data.values())
        hi = max(df.index[-1] for df in data.values())
        split = lo + (hi - lo) * is_frac
        cache[tuple(sorted(p.items()))] = (data, lo, split, hi)
        is_rows.append(summarize(portfolio_backtest(data, p, lo, split), p))

    is_df = pd.DataFrame(is_rows)
    valid = is_df[(is_df["maxDD%"] > -8.0) & (is_df["trades"] >= 15)]
    ranked = (valid if not valid.empty else is_df).sort_values(
        ["ch_ok", "ch_ko", "med_months", "PF"],
        ascending=[False, True, True, False], na_position="last")
    print(f"\n=== GRID (challenge mode) — in-sample (first {int(is_frac*100)}%) — top 10 ===")
    print(ranked.head(10).to_string(index=False))

    best = ranked.iloc[0]
    p = {"tf": best["tf"], "mult_kc": float(best["multKC"]),
         "trend_len": int(best["trend"]), "risk": float(best["risk%"]) / 100,
         "use_flip": best["flip"] == "yes", **extra}
    data, lo, split, hi = cache[tuple(sorted(p.items()))]
    oos = summarize(portfolio_backtest(data, p, split, hi), p)
    print("\n=== Best in-sample combo -> out-of-sample check ===")
    print(pd.DataFrame([oos]).to_string(index=False))

    full = summarize(portfolio_backtest(data, p, lo, hi), p)
    print("\n=== Best combo — full period (consecutive challenges) ===")
    print(pd.DataFrame([full]).to_string(index=False))
    print("\nReading: ch_ok/ch_ko = challenges passed/failed; med_months = median"
          "\nmonths to pass; maxDD% = worst intra-challenge peak drawdown."
          "\nIf OOS confirms the in-sample ch_ok/ch_ko ratio, the combo is robust.")


def main():
    ap = argparse.ArgumentParser(
        description="VCB pipeline — prop-firm portfolio backtester")
    ap.add_argument("mode", choices=["single", "grid"])
    ap.add_argument("--start", default="2024-07-01")
    ap.add_argument("--tickers", default=DEFAULT_SET,
                    help=f"Comma list among {','.join(YAHOO_TICKERS)} "
                         f"(default: {DEFAULT_SET})")
    ap.add_argument("--tf", default="4h", choices=["2h", "4h"])
    ap.add_argument("--mult-kc", type=float, default=1.5)
    ap.add_argument("--trend", type=int, default=200,
                    help="Trend EMA length (0 disables the filter)")
    ap.add_argument("--risk", type=float, default=0.8, help="Risk per trade in %%")
    ap.add_argument("--flip", action="store_true",
                    help="Enable trigger B (momentum flip during the squeeze)")
    ap.add_argument("--challenge", action="store_true",
                    help="Challenge mode: reset to initial balance after target/kill")
    ap.add_argument("--stress", action="store_true",
                    help="Stress test: 2x commissions and 0.1 ATR slippage on stops")
    ap.add_argument("--daily-stop", type=float, default=2.5,
                    help="Software daily-loss guard in %% (default 2.5)")
    ap.add_argument("--kill", type=float, default=9.0,
                    help="Software kill switch in %% from initial balance "
                         "(e.g. 5.0 for a prop with a 6%% max-loss rule)")
    ap.add_argument("--target", type=float, default=10.0, help="Profit target in %%")
    ap.add_argument("--csv-dir", default=None,
                    help="Directory with {TICKER}.csv exports (TradingView or MT5); "
                         "replaces Yahoo data. single mode only; --tf is ignored")
    a = ap.parse_args()

    names = [n.strip().upper() for n in a.tickers.split(",") if n.strip()]
    bad = [n for n in names if n not in YAHOO_TICKERS]
    if bad:
        print(f"[!] Unknown tickers: {bad}")
        return

    limits = {"daily": a.daily_stop / 100, "kill": a.kill / 100,
              "target": a.target / 100}
    if a.mode == "single":
        p = {"tf": a.tf, "mult_kc": a.mult_kc, "trend_len": a.trend,
             "risk": a.risk / 100, "use_flip": a.flip, "challenge": a.challenge,
             "comm": COMMISSION * 2 if a.stress else COMMISSION,
             "slip_atr": 0.1 if a.stress else 0.0, **limits}
        run_single(names, a.start, p, csv_dir=a.csv_dir)
    else:
        if a.csv_dir:
            print("[!] --csv-dir is only supported in single mode.")
            return
        run_grid(names, a.start, stress=a.stress, limits=limits)


if __name__ == "__main__":
    main()
