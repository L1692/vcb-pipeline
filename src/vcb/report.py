"""Load/report layer: aggregate engine results into comparable metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import CAPITAL, Result


def summarize(res: Result, p: dict) -> dict:
    """Flatten a :class:`~vcb.engine.Result` into a metrics row.

    In challenge mode, cumulative P&L and single-halt fields are replaced by
    the challenge tally (``ch_ok``/``ch_ko``) and the median months to pass,
    because the target halt truncates P&L and makes it misleading.
    """
    eq = pd.Series(res.eq_curve, dtype=float)
    tr = res.trades
    wins = [x for x in tr if x > 0]
    losses = [x for x in tr if x <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
    final = eq.iloc[-1] if len(eq) else CAPITAL
    out = {
        "strat": p.get("strategy", "VCB"),
        "tf": p.get("tf", "-"),
        "multKC": p.get("mult_kc", "-"),
        "trend": p.get("trend_len", "-"),
        "risk%": round(p["risk"] * 100, 2),
        "flip": "yes" if p.get("use_flip") else "no",
        "stress": "yes" if p.get("slip_atr", 0.0) > 0 else "no",
        "trades": len(tr),
        "win%": round(100 * len(wins) / len(tr), 1) if tr else 0.0,
        "PF": round(pf, 2) if tr else 0.0,
        "P&L%": round(100 * (final / CAPITAL - 1), 2),
        "maxDD%": round(100 * res.worst_dd, 2),
        "day_stops": res.daily_stops,
        "halt": res.halt,
        "months_to_target": round(res.months_to_target, 1) if res.months_to_target else "-",
    }
    if p.get("challenge"):
        passed = [m for ok, m in res.challenges if ok]
        failed = [m for ok, m in res.challenges if not ok]
        out["ch_ok"] = len(passed)
        out["ch_ko"] = len(failed)
        out["med_months"] = round(float(np.median(passed)), 1) if passed else np.nan
        for k in ("P&L%", "halt", "months_to_target"):
            out.pop(k)
    return out
