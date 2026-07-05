"""Portfolio backtest engine with prop-firm risk constraints.

Design highlights:

- **Shared account equity** across instruments: the daily-loss guard and
  the kill switch are evaluated on the portfolio mark-to-market, matching
  how prop firms actually monitor accounts.
- **Challenge mode**: on hitting the profit target (challenge passed) or
  the kill switch (challenge failed) the account resets to the initial
  balance and a new simulated challenge starts, yielding the metric that
  matters for prop economics: passes vs failures over the whole dataset.
- **Conservative fills**: when a bar touches both stop and target, the
  stop is assumed to fill first; optional stress mode worsens stop/trail
  fills by a configurable ATR fraction and multiplies commissions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

CAPITAL = 100_000.0   #: initial account balance
COMMISSION = 0.0002   #: commission per side (fraction of notional)
SL_MULT = 2.0         #: stop-loss distance in ATRs
TP_MULT = 3.5         #: take-profit distance in ATRs
TRAIL_MULT = 3.0      #: chandelier trailing distance in ATRs
DAILY_STOP = 0.025    #: software daily-loss guard (fraction of day-start equity)
KILL = 0.09           #: software kill switch (fraction of initial balance)
TARGET = 0.10         #: profit target (fraction of initial balance)
MAX_POS = 2           #: max simultaneous open positions


@dataclass
class Position:
    side: int          # +1 long, -1 short
    qty: float
    entry: float
    sl: float
    tp: float
    trail: float


@dataclass
class Result:
    trades: list = field(default_factory=list)
    eq_curve: list = field(default_factory=list)
    daily_stops: int = 0
    halt: str = "no"
    months_to_target: float | None = None
    challenges: list = field(default_factory=list)   # (passed: bool, months: float)
    worst_dd: float = 0.0                            # worst intra-challenge peak DD


def portfolio_backtest(data: dict[str, pd.DataFrame], p: dict, t0, t1) -> Result:
    """Run the VCB strategy over ``data`` (prepared frames) in ``[t0, t1]``.

    ``p`` keys: ``risk``, ``use_flip``, ``trend_len`` (0 disables the trend
    filter), plus optional ``challenge``, ``comm``, ``slip_atr``, ``daily``,
    ``kill``, ``target``.
    """
    rows = {n: {r.Index: r for r in df.itertuples()} for n, df in data.items()}
    timeline = sorted(set().union(*[set(d.keys()) for d in rows.values()]))
    timeline = [t for t in timeline if t0 <= t <= t1]
    res = Result()
    if not timeline:
        return res

    equity = CAPITAL                      # realized equity
    positions: dict[str, Position] = {}
    last_close: dict[str, float] = {}
    cur_day, day_start, blocked = None, CAPITAL, False
    start_ts = timeline[0]
    challenge = bool(p.get("challenge", False))
    comm = p.get("comm", COMMISSION)
    slip = p.get("slip_atr", 0.0)
    daily = p.get("daily", DAILY_STOP)
    kill = p.get("kill", KILL)
    target = p.get("target", TARGET)
    ch_start, peak = start_ts, CAPITAL

    def mtm() -> float:
        unreal = sum(pos.side * pos.qty * (last_close[n] - pos.entry)
                     for n, pos in positions.items())
        return equity + unreal

    def close_pos(name: str, px: float) -> None:
        nonlocal equity
        pos = positions.pop(name)
        pnl = pos.side * pos.qty * (px - pos.entry) \
            - comm * pos.qty * (pos.entry + px)
        equity += pnl
        res.trades.append(pnl)

    for t in timeline:
        if t.date() != cur_day:
            cur_day = t.date()
            day_start = mtm() if last_close else equity
            blocked = False

        # 1) exits for instruments with a bar at this timestamp
        for name in list(positions.keys()):
            r = rows[name].get(t)
            if r is None:
                continue
            pos, atr = positions[name], r.atr
            exit_px = None
            if pos.side > 0:
                pos.trail = max(pos.trail, r.High - atr * TRAIL_MULT)
                if r.Low <= pos.sl:
                    exit_px = pos.sl - slip * atr      # stop: degraded fill
                elif r.High >= pos.tp:
                    exit_px = pos.tp                   # limit: clean fill
                elif r.Close < pos.trail:
                    exit_px = r.Close - slip * atr     # trail: market order
            else:
                pos.trail = min(pos.trail, r.Low + atr * TRAIL_MULT)
                if r.High >= pos.sl:
                    exit_px = pos.sl + slip * atr
                elif r.Low <= pos.tp:
                    exit_px = pos.tp
                elif r.Close > pos.trail:
                    exit_px = r.Close + slip * atr
            if exit_px is not None:
                close_pos(name, exit_px)

        for name in rows:
            r = rows[name].get(t)
            if r is not None:
                last_close[name] = r.Close

        # 2) account-level guards
        acct = mtm()
        if day_start > 0 and (acct - day_start) / day_start <= -daily:
            for name in list(positions.keys()):
                close_pos(name, last_close[name])
            if not blocked:
                res.daily_stops += 1
            blocked = True
            acct = equity
        peak = max(peak, acct)
        res.worst_dd = min(res.worst_dd, (acct - peak) / peak)
        hit_kill = acct <= CAPITAL * (1 - kill)
        hit_target = acct >= CAPITAL * (1 + target)
        if hit_kill or hit_target:
            for name in list(positions.keys()):
                close_pos(name, last_close[name])
            months = (t - ch_start).days / 30.44
            if challenge:
                res.challenges.append((hit_target, months))
                equity, day_start, blocked = CAPITAL, CAPITAL, False
                ch_start, peak = t, CAPITAL
                res.eq_curve.append(CAPITAL)
                continue
            res.halt = "target" if hit_target else "kill switch"
            if hit_target:
                res.months_to_target = months
            res.eq_curve.append(acct)
            break
        res.eq_curve.append(acct)

        # 3) new entries
        if blocked or len(positions) >= MAX_POS:
            continue
        for name in rows:
            if name in positions or len(positions) >= MAX_POS:
                continue
            r = rows[name].get(t)
            if r is None or r.atr <= 0:
                continue
            flip_up = r.sqz_on and r.val > 0 >= r.val_prev
            flip_dn = r.sqz_on and r.val < 0 <= r.val_prev
            long_raw = (r.release and r.val > 0) or (p["use_flip"] and flip_up)
            short_raw = (r.release and r.val < 0) or (p["use_flip"] and flip_dn)
            go_long = long_raw and (p["trend_len"] == 0 or r.Close > r.ema)
            go_short = short_raw and (p["trend_len"] == 0 or r.Close < r.ema)
            if not (go_long or go_short):
                continue
            side = 1 if go_long else -1
            stop_dist = r.atr * SL_MULT
            qty = mtm() * p["risk"] / stop_dist
            positions[name] = Position(
                side=side, qty=qty, entry=r.Close,
                sl=r.Close - side * stop_dist,
                tp=r.Close + side * r.atr * TP_MULT,
                trail=(r.High - r.atr * TRAIL_MULT) if side > 0
                      else (r.Low + r.atr * TRAIL_MULT))

    return res
