"""VCB pipeline: ETL-style backtesting for prop-firm trading strategies."""

from .engine import Position, Result, portfolio_backtest
from .extract import load_csv, load_instrument
from .report import summarize
from .transform import add_indicators

__all__ = ["Position", "Result", "portfolio_backtest", "load_csv",
           "load_instrument", "summarize", "add_indicators"]
__version__ = "1.0.0"
