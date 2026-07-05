# VCB Pipeline

![CI](https://github.com/<you>/vcb-pipeline/actions/workflows/ci.yml/badge.svg)

An ETL-style backtesting pipeline for validating trading strategies under
**prop-firm constraints** (daily loss limits, max drawdown, profit targets),
built around a Volatility Compression Breakout (VCB) strategy.

The interesting part is not the strategy — it is the **validation
methodology**: the same engine runs on three heterogeneous data sources
(Yahoo Finance, TradingView exports, MetaTrader 5 exports), which exposed how
much backtest results depend on the data feed, and led to evidence-based
decisions (dropping instruments whose "edge" did not survive a feed change).

```
┌─────────────── EXTRACT ───────────────┐   ┌── TRANSFORM ──┐   ┌────────── LOAD / ANALYZE ──────────┐
│ Yahoo Finance (1h → resampled)        │   │ BB/KC squeeze │   │ Shared-equity portfolio engine     │
│ TradingView CSV (unix/ISO timestamps) │ → │ LSMA momentum │ → │ Prop-firm guards (daily/max DD)    │
│ MetaTrader 5 CSV (<DATE>/<TIME>)      │   │ EMA trend/ATR │   │ Challenge simulation & stress test │
└───────────────────────────────────────┘   └───────────────┘   └────────────────────────────────────┘
```

## Why this exists

Prop-firm challenges are not "make money" problems, they are **constrained
optimization** problems: hit +10% without ever breaching a daily loss limit
or a max-drawdown limit measured on the whole account. Standard backtest
metrics (total P&L, Sharpe) answer the wrong question. This pipeline answers
the right one:

> Over N years of data, how many consecutive challenges does this
> configuration pass, how many does it fail, and how long does the median
> pass take — under realistic costs?

## Key design decisions

- **Portfolio-level risk guards.** The daily stop and kill switch are
  evaluated on the shared account mark-to-market across all instruments, not
  per instrument — because that is how prop firms actually monitor accounts.
- **Challenge mode.** On hitting the target (pass) or the kill switch (fail)
  the account resets and a new simulated challenge starts. Ranking
  configurations by truncated P&L is misleading (every passing run stops near
  the target); ranking by passes/failures/median-months is not.
- **Walk-forward grid search.** Parameter combinations are selected on the
  first 70% of the data and verified out-of-sample on the last 30%, then on
  the full period.
- **Conservative fill model.** When a bar touches both stop and target, the
  stop fills first. Stress mode doubles commissions and degrades stop/trail
  fills by 0.1 ATR.
- **Feed-agnostic extract layer.** One OHLC schema, three loaders, lazy
  optional dependency on `yfinance` (the CSV path has zero network needs).

## Install

```bash
git clone https://github.com/<you>/vcb-pipeline.git
cd vcb-pipeline
pip install -e ".[yahoo,dev]"
pytest            # 7 tests, synthetic data, no network required
```

## Usage

```bash
# Single configuration, Yahoo data, portfolio of three instruments
vcb single --tf 2h --trend 100 --risk 1.0 --flip --challenge

# Same, on broker data exported from MT5/TradingView into ./data
vcb single --csv-dir ./data --tickers XAUUSD --trend 100 --risk 0.75 \
    --flip --challenge --kill 5.0 --stress

# Walk-forward grid search (48 combinations, challenge mode)
vcb grid --start 2024-07-01
```

Prop limits are configurable: `--daily-stop`, `--kill`, `--target`
(e.g. `--kill 5.0` models a 6% max-loss rule with a 1% safety buffer).

Sample output:

```
tf  multKC  trend  risk% flip stress  trades  win%   PF  maxDD%  day_stops  ch_ok  ch_ko  med_months
4h     1.5    100   0.75  yes     no     160  45.6 1.45   -5.79          0      3      0         7.2
```

## What the cross-feed validation found

The same strategy, same parameters, three feeds:

| Instrument | Yahoo (resampled 1h) | TradingView (prop feed) | MT5 (broker demo) | Verdict |
|---|---|---|---|---|
| Gold (XAUUSD) | strong | consistent signal count | strong, survives stress | **kept** |
| DAX | strong | weak (PF < 1) | flat (PF ≈ 1.0) | dropped |
| NASDAQ | strong | weak recently | regime-dependent | dropped |

Signal counts on gold matched across feeds (28 vs 26 trades on the aligned
window); on indices they diverged by up to 7x. A squeeze-based edge that
appears and disappears with the data provider is not an edge — it is an
artifact of bar composition. The pipeline made that visible.

## Repository layout

```
src/vcb/
  extract.py     # data acquisition: Yahoo, TradingView CSV, MT5 CSV
  transform.py   # indicators: squeeze, LSMA momentum, EMA, ATR
  engine.py      # portfolio backtester with prop-firm guards
  report.py      # metrics aggregation (challenge-aware)
  cli.py         # single / grid orchestration
pine/            # TradingView Pine Script v6 implementation (live signals)
tests/           # pytest suite on synthetic data
data/            # your CSV exports (git-ignored)
```

## Roadmap

- GitHub Actions CI (pytest on push)
- Strategy plug-in interface (second strategy: trend-following on indices)
- Direct MT5 ingestion via the `MetaTrader5` package (Windows)
- Rolling multi-window walk-forward

## Disclaimer

This is a research and engineering project. Nothing here is financial
advice; past backtest performance does not guarantee future results.
Trading leveraged products involves substantial risk of loss.

## License

MIT
