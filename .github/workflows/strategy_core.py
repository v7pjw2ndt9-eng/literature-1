"""
strategy_core.py -- THE single source of truth for the 600989 strategy.

WHY THIS FILE EXISTS
--------------------
There used to be three implementations of "the strategy": the live web tool's own simulator inside
index.html, leverage_backtest_gated.py, and the parameterised strategy_variants.py. They drifted, and
every drift produced a wrong answer that took real work to find:
  - the web simulator never maintained the 90% baseline, so the breaker state shown in the live tool
    was computed from a different strategy than the one being researched (worth 47pp and a 3-month
    shift in one resume date);
  - both engines valued the soft-cap check against the book that SURVIVED the day's take-profits,
    which is lookahead;
  - both engines only rested tranche take-profits on bracket-mode days, contradicting the written
    rule and the live tool's own daily checklist.
Since there is no JS runtime available here, a cross-language conformance test is not possible. So the
resolution is one-directional: Python is the only engine, and index.html becomes a pure renderer that
consumes daily_state.json emitted by this file.

THE INTRADAY ORDERING, STATED ONCE
----------------------------------
Every rule below is expressed against this timeline. Anything that gates a day-t order may only use
information available at that point in day t.

  overnight   accrue margin interest on borrowed cash
  09:25       the open is known. SNAPSHOT the book. Decide today's mode from prob[t], which was
              fitted on data through t-1. Place all of today's orders:
                - take-profit limits at cost*(1+target) for EVERY open tranche, every day
                - standard-T day: market sell the slice at the open
                - bracket day:    buy limit at open*(1-B), sell limit at open*(1+X)
  intraday    resolve fills against low/high
  close       resolve anything that had to settle at the close, mark to market, then rebalance the
              baseline if its weight is outside the band

FIXES APPLIED RELATIVE TO THE OLD CANONICAL (each measured; see notes in-line)
  F1  soft-cap check uses the 09:25 book snapshot, not the post-take-profit residual   (+0.9pp)
  F2  tranche take-profits rest on standard-T days too, as the spec always said        (measured below)
  F3  margin financing cost is charged                                                 (-29.9pp @5%)
  F4  the breaker signal comes from a FROZEN REFERENCE STRATEGY, so execution-layer
      parameters can no longer reshape breaker history                                 (architectural)
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, asdict, field, replace
from datetime import date, timedelta

import numpy as np

# load_data belongs to the research tree; parse_ymd is the only thing the engine needs at runtime.
# Falling back keeps this file deployable on its own (e.g. into a CI job that only needs simulate()
# and the breaker), without dragging the whole backtest corpus and its data files along.
try:
    from leverage_backtest_cap3 import load_data, parse_ymd, THRESH
except ImportError:                                        # standalone deployment
    load_data = None
    THRESH = 0.779                      # fallback only; the canonical value lives in cap3

    def parse_ymd(s):
        return date(int(s[:4]), int(s[5:7]), int(s[8:10]))

SCRATCH = os.path.dirname(os.path.abspath(__file__))


# ======================================================================================
# Configuration
# ======================================================================================
@dataclass(frozen=True)
class Config:
    # --- account structure ---
    base_frac: float = 0.90          # target baseline weight
    slice_frac: float = 0.10         # daily operating slice, as a fraction of equity
    rebalance_band: float = 0.03     # rebalance baseline when |weight - target| exceeds this
    cost_leg: float = 0.001          # transaction cost per leg
    fin_rate: float = 0.05           # annualised margin financing rate on borrowed cash  (F3)

    # --- the intraday model switch ---
    #: Gate threshold, taken from the canonical module rather than written here. It was hardcoded at
    #: 0.75 while the engine moved to 0.779, which broke the identity check -- the same
    #: constant-in-two-places failure that has now happened four times in this project.
    thresh: float = THRESH
    std_t_action: str = "trade"      # "trade" = do the T; "none" = abstain that day (see notes)

    # --- bracket mode ---
    buy_depth: float = 0.009         # buy limit at open*(1-buy_depth)
    sell_pop: float = 0.03           # sell limit at open*(1+sell_pop)
    target_profit: float = 0.05      # persistent tranche take-profit at cost*(1+this)
    max_sells_per_day: int = 3       # cap on tranche take-profit fills per day, oldest first
    min_cost_spacing: float | None = None   # new tranche must be this much below the cheapest held

    # --- exposure limit ---
    lev_cap: float = 1.5             # soft cap; gates NEW exposure only, valued at the open

    # --- execution realism ---
    #: A-share limit orders can only be quoted to 0.01. Both engines used to test and fill against the
    #: unrounded open*(1-depth), which is a price you cannot actually enter -- the live checklist has
    #: always had to print a 2-decimal number. Rounding makes the backtest describe an executable
    #: order. It is not a free win either way: a buy limit rounded UP fills more easily, one rounded
    #: DOWN fills less, and the same cuts both ways on the sell side.
    tick: float = 0.01
    # Require the low to pierce the limit by this much before believing the fill. A resting limit
    # exactly at the intraday extreme is not reliably filled in reality, and the deeper the limit the
    # more of an extreme it is -- so this is the differential realism penalty that deeper entries have
    # to survive. Used both as a standing assumption and as the perturbation axis for robust
    # parameter selection.
    fill_margin: float = 0.0

    # --- optional inventory controls (off by default) ---
    max_hold_days: int | None = None
    max_open_tranches: int | None = None

    # --- bug-compatibility switches, for MIGRATION VERIFICATION ONLY ---
    # These reproduce the two defects the old engines had. They exist so the new engine can be proven
    # to reproduce the old numbers exactly before anything is attributed to a fix; a migration that
    # cannot reproduce the thing it replaces has not been verified, it has been asserted. Never turn
    # these on for research or for live use.
    legacy_levcheck_post_tp: bool = False       # soft cap valued after the day's take-profits (F1 off)
    legacy_tp_on_bracket_days_only: bool = False  # take-profits skipped on standard-T days (F2 off)
    legacy_cap_denom_prev: bool = False         # cap ratio divided by yesterday's close equity
    # What to do when the bracket's UPPER leg fills but the lower one does not. Both old engines
    # ignored this case entirely, yet the live tool tells the user to sell and logs it -- so it is a
    # real, unmodelled rule, not a parameter. The three readings are genuinely different strategies:
    #   "none"            reproduce the old engines: pretend it never happened
    #   "to_cash"         sell the slice out of the baseline, leave proceeds in cash, let the
    #                     +/-3% band redeploy them later
    #   "buy_back_close"  sell the slice at open*(1+pop), buy it back at the close -- the exact
    #                     mirror of standard-T, and the reading the "double-sided T" name implies
    #   "close_tranche"   sell an EXISTING tranche at open*(1+pop); if no tranche is open, place no
    #                     sell at all. This is the only reading where the sale is not immediately
    #                     undone: selling baseline shares fights the 90% target, so the +/-3% band
    #                     buys them straight back (measured: 210 rebalances instead of 86, leverage
    #                     unchanged at 1.380, 1.89% of equity burned in fees for nothing). Selling a
    #                     tranche instead retires borrowed exposure, which is real deleveraging and
    #                     needs no rebalance to reverse it.
    sell_only_mode: str = "none"
    #: which tranche the upper leg retires: "oldest" (same priority as the take-profit queue) or
    #: "highest_cost" (retires the positions the cost*1.05 rule can never reach)
    sell_only_pick: str = "oldest"
    #: whether the upper leg's tranche sale consumes the max_sells_per_day budget
    sell_only_counts_to_cap: bool = True
    #: on a day the regime gate refuses new inventory, does the upper leg still rest? The gate's
    #: purpose is "add nothing today", which says nothing about selling -- and selling is what
    #: reduces margin usage. Same class of mistake as blocking sells during a breaker stop.
    gate_blocks_sell: bool = True

    def json(self) -> dict:
        return asdict(self)


#: The FROZEN reference strategy that generates the breaker signal, and nothing else.  (F4)
#: Both external reviews independently converged on this: while the breaker's trigger is a function of
#: the strategy's own P&L, and the strategy's P&L is a function of every parameter, ANY parameter can
#: reshape breaker history. Measured consequence: moving the buy depth by 0.05pp (from -0.95% to
#: -1.00%) bought a whole extra 15-day stop worth about +16pp, which manufactured a false "deeper
#: entry is better" result; and a +0.9pp lookahead fix moved the headline by +24.2pp. Pinning the
#: signal to a frozen reference removes that feedback path: execution parameters may now be varied
#: without rewriting the breaker's history. It does NOT make the breaker's own parameters continuous
#: -- that is mathematically impossible for a latched threshold -- but those are now the only
#: parameters with that problem, and they are changed only on a declared version bump.
#:
#: v2 (2026-08-02): the reference is now the EXECUTED rule set with its parameters frozen, rather than
#: the older, more levered variant. The previous reference still did the standard-T round trip and
#: never retired a position through the upper leg, so it sat at the 1.5x cap 75.4% of the time and
#: swung much further from buy-and-hold -- a high-gain thermometer whose +20% line was calibrated to
#: it. Two references that finish within 0.2pp of each other (78.2% vs 78.4%) therefore fired 4 times
#: versus 1 at the same threshold, which made the line, not the signal, the real parameter.
#:
#: Pinning the reference to the executed rules and re-calibrating the line to 8% was adopted on
#: evidence that is, unusually for this project, robust:
#:   - frequency-matched RANDOM stop placement (same count, same lengths, shuffled dates) scored
#:     75-81% against the real signal's 96-184%; the real signal beat all 60 draws at 5/8/10/15%.
#:     Downtime alone is worth nothing -- the timing is the whole effect.
#:   - matching AVERAGE LEVERAGE by lowering the soft cap instead scored 57-78%, i.e. 18-122pp worse.
#:     "Hold less always" is not a substitute for "hold less at the right times".
#:   - the ranking 8 > 5 > 10 > 15 > 20 is identical on 2020-2026, 2020-2025 and 2020-2024, and 8% is
#:     the peak in all three. (Nested windows, so this is robustness to dropping the recent years, not
#:     three independent confirmations.)
#: What still argues for caution: roughly 20 (reference, threshold) pairs have now been examined, so
#: the maximum of that set is a multiple-comparison result. The plateau at 6/7/8% (176/176/184) and
#: the cross-window stability are what make it more than a lucky cell.
REFERENCE_CONFIG = Config(
    fin_rate=0.05,
    std_t_action="none",
    sell_only_mode="close_tranche",
    buy_depth=0.009,
    min_cost_spacing=None,
    max_hold_days=None,
    max_open_tranches=None,
)
REFERENCE_VERSION = "theta0-2026-08-03"

BREAKER_STOP_THRESH = 0.08      # 180-day rolling excess of the reference strategy
BREAKER_RESUME_THRESH = 0.0
BREAKER_CAL_DAYS = 180


# ======================================================================================
# The one simulation
# ======================================================================================
def simulate(bars, probs, warmup, gate, cfg: Config, event_log=None):
    """Run the strategy once.

    gate[t] is the breaker's permission for day t+1's NEW bracket exposure: a bool (True = stopped)
    or a float multiplier in [0,1]. Day t reads gate[t-1], mirroring the live tool, where each morning
    you check a status derived from data through the prior close.

    Returns equity / bench / leverage series plus diagnostics. Never mutates cfg.
    """
    n = len(bars)
    o_ = np.array([b["open"] for b in bars], float)
    h_ = np.array([b["high"] for b in bars], float)
    l_ = np.array([b["low"] for b in bars], float)
    c_ = np.array([b["close"] for b in bars], float)
    p_ = np.asarray(probs, float)
    dates = [b["date"] for b in bars]

    yrs = np.zeros(n)
    if cfg.fin_rate:
        dd = [parse_ymd(b["date"]) for b in bars]
        for t in range(1, n):
            yrs[t] = (dd[t] - dd[t - 1]).days / 365.0

    baseline_shares = cfg.base_frac / c_[0]
    cash = 1.0 - cfg.base_frac
    tranches: list[dict] = []
    equity, bench, leverage = [1.0], [1.0], [cfg.base_frac]

    diag = {k: 0 for k in ("std_t_days", "std_t_abstain_days", "tranche_opens", "same_day_roundtrips",
                            "tranche_closes", "tranche_closes_on_std_t_days", "forced_exits",
                            "blocked_breaker", "blocked_levcap", "blocked_spacing", "blocked_maxopen",
                            "rebalances", "sell_only_legs")}
    diag["financing_paid"] = 0.0
    diag["rebalance_fees"] = 0.0

    def log(**kw):
        if event_log is not None:
            event_log.append(kw)

    for t in range(1, n):
        d = dates[t]
        bench.append(bench[-1] * (c_[t] / c_[t - 1]))

        # ---- overnight: margin interest on borrowed cash -------------------------------- F3
        if cfg.fin_rate and cash < 0:
            interest = (-cash) * cfg.fin_rate * yrs[t]
            cash -= interest
            diag["financing_paid"] += interest

        eq_prev = equity[-1]
        baseline_val_prev = baseline_shares * c_[t - 1]

        if t < warmup:
            # No trading before the model is usable, but the baseline band still applies -- the
            # account is a 90% holding from day one, and it drifts whether or not we are trading.
            _mark(equity, leverage, baseline_shares * c_[t], tranches, c_[t], cash)
            baseline_shares, cash = _rebalance(equity, leverage, baseline_shares, cash,
                                                tranches, c_[t], cfg, diag, log, d)
            continue

        o, hi, lo, c = o_[t], h_[t], l_[t], c_[t]

        # ---- 09:25 --------------------------------------------------------------------------
        # Snapshot the book BEFORE any of today's fills. The soft-cap check must see the exposure
        # that actually exists when the order is placed, not what survives the day.            F1
        book_at_open = list(tranches)
        # Equity as it truly stands at 09:25, captured BEFORE the day's take-profit loop runs. It has
        # to be taken here: a few lines down `cash` already carries today's take-profit proceeds while
        # `book_at_open` still holds those same shares, so combining them there double-counts. This is
        # a fact about the account and does not depend on which subset the cap check chooses to count.
        eq_at_open = baseline_shares * o + sum(tr["shares"] * o for tr in book_at_open) + cash
        gate_today = _gate(gate[t - 1])
        stopped_today = gate_today <= 0.0
        p = p_[t] if t < len(p_) else float("nan")
        std_t_fires = p >= cfg.thresh          # NaN >= x is False, which is what we want

        # ---- tranche take-profits rest EVERY day, standard-T or not --------------------- F2
        # The old engines put this inside the bracket-mode branch, so on the 8.9% of days that fired
        # standard-T the cost*1.05 orders silently did not exist -- contradicting both the written
        # rule and the live tool's own daily checklist, and making exits slower than advertised.
        sold_today = 0
        still_open = []
        skip_tp = cfg.legacy_tp_on_bracket_days_only and std_t_fires
        for tr in ([] if skip_tp else tranches):
            age = t - tr["i0"]
            target_px = tr["cost"] * (1 + cfg.target_profit)
            if hi >= target_px and sold_today < cfg.max_sells_per_day:
                cash += tr["shares"] * target_px * (1 - cfg.cost_leg)
                sold_today += 1
                diag["tranche_closes"] += 1
                if std_t_fires:
                    diag["tranche_closes_on_std_t_days"] += 1
                log(date=d, type="老仓位止盈", shares=tr["shares"], cost=tr["cost"],
                    sell_price=target_px, opened_date=tr["opened_date"], held_days=age)
            elif cfg.max_hold_days is not None and age >= cfg.max_hold_days:
                cash += tr["shares"] * c * (1 - cfg.cost_leg)
                diag["forced_exits"] += 1
                log(date=d, type="超期强制平仓", shares=tr["shares"], cost=tr["cost"],
                    sell_price=c, opened_date=tr["opened_date"], held_days=age)
            else:
                still_open.append(tr)
        if not skip_tp:
            tranches = still_open

        # ---- today's mode -------------------------------------------------------------------
        if std_t_fires and cfg.std_t_action == "none" and not cfg.gate_blocks_sell:
            # gated: no new buying, but the upper leg still rests and can retire a tranche
            diag["std_t_abstain_days"] += 1
            sell_trigger = o * (1 + cfg.sell_pop)
            if hi >= sell_trigger and tranches and (
                    not cfg.sell_only_counts_to_cap or sold_today < cfg.max_sells_per_day):
                i = (0 if cfg.sell_only_pick == "oldest"
                     else max(range(len(tranches)), key=lambda k: tranches[k]["cost"]))
                tr = tranches.pop(i)
                cash += tr["shares"] * sell_trigger * (1 - cfg.cost_leg)
                diag["sell_only_closed_tranche"] = diag.get("sell_only_closed_tranche", 0) + 1
                log(date=d, type="闸门日卖出(平掉一笔加仓)", shares=tr["shares"], cost=tr["cost"],
                    sell_price=sell_trigger, opened_date=tr["opened_date"], held_days=t - tr["i0"])
            baseline_val_t = baseline_shares * c

        elif std_t_fires and cfg.std_t_action == "none":
            # Measured A/B/C/D result: doing the intraday round-trip was worth -2.2pp, while merely
            # declining to add overnight inventory that day was worth +67.4pp. So "abstain" is a
            # first-class action, not a degenerate case.
            diag["std_t_abstain_days"] += 1
            baseline_val_t = baseline_shares * c

        elif std_t_fires:
            diag["std_t_days"] += 1
            slice_val = cfg.slice_frac * eq_prev
            r = _std_t_factor(o, lo, c, c_[t - 1], cfg.cost_leg)
            remaining = baseline_val_prev - slice_val
            baseline_val_t = remaining * (c / c_[t - 1]) + slice_val * r
            baseline_shares = baseline_val_t / c
            log(date=d, type="标准T", sell_price=o,
                buy_price=(o * 0.99 if lo <= o * 0.99 else c),
                note=("低点触及-1%回补" if lo <= o * 0.99 else "全天未触及,收盘价回补"))

        else:
            buy_trigger = _tick(o * (1 - cfg.buy_depth), cfg.tick)
            sell_trigger = _tick(o * (1 + cfg.sell_pop), cfg.tick)
            buy_fires = lo <= buy_trigger * (1 - cfg.fill_margin)
            bought = False

            if buy_fires:
                reason = None
                if stopped_today:
                    reason = "breaker"
                elif cfg.max_open_tranches is not None and len(tranches) >= cfg.max_open_tranches:
                    reason = "maxopen"
                elif cfg.min_cost_spacing is not None and tranches:
                    # price-indexed entry: bounds the tranche count by construction, since inside a
                    # drawdown of depth D at most ~log(1-D)/log(1-spacing) tranches can coexist
                    if buy_trigger > min(tr["cost"] for tr in tranches) * (1 - cfg.min_cost_spacing):
                        reason = "spacing"

                if reason is None:
                    cap_book = tranches if cfg.legacy_levcheck_post_tp else book_at_open
                    open_val = sum(tr["shares"] * o for tr in cap_book)          # F1
                    cur_exposure = baseline_shares * o + open_val
                    # Denominator marked at the SAME moment as the numerator. It used to be eq_prev --
                    # yesterday's close equity -- so the ratio mixed two dates inside one fraction and
                    # overstated headroom on any morning the stock gapped up. At 09:25 the open is
                    # known, so there is no reason to fall back on yesterday.
                    new_shares = (cfg.slice_frac * gate_today * eq_prev) / buy_trigger
                    _den = eq_prev if cfg.legacy_cap_denom_prev else eq_at_open
                    if (cur_exposure + new_shares * buy_trigger) / max(_den, 1e-9) <= cfg.lev_cap:
                        bought = True
                        cash -= new_shares * buy_trigger * (1 + cfg.cost_leg)
                        if hi >= sell_trigger:
                            cash += new_shares * sell_trigger * (1 - cfg.cost_leg)
                            diag["same_day_roundtrips"] += 1
                            log(date=d, type="双边(即日了结)", shares=new_shares,
                                buy_price=buy_trigger, sell_price=sell_trigger)
                        else:
                            tranches.append({"shares": new_shares, "cost": buy_trigger,
                                              "opened_date": d, "i0": t})
                            diag["tranche_opens"] += 1
                            log(date=d, type="开新仓(未当日了结)", shares=new_shares,
                                buy_price=buy_trigger,
                                target_sell_price=buy_trigger * (1 + cfg.target_profit))
                    else:
                        diag["blocked_levcap"] += 1
                else:
                    diag["blocked_" + reason] += 1
            # The upper leg is evaluated whenever it was NOT consumed by a same-day round trip on
            # freshly bought shares. Critically this is independent of why no buy happened -- not
            # firing, the soft cap, the spacing rule, or the BREAKER. A stop must only block new
            # buying; writing this branch as `elif buy_fires` (as the first version did) silently
            # swallowed the sale on any stopped day whose low also reached the buy limit.
            if (not bought) and hi >= sell_trigger:
                # The bracket's upper leg filled while the lower one did not. Both old engines
                # ignored this case entirely (the sell was only ever evaluated inside the
                # buy-fired branch), so a real, recordable trade was missing from every backtest.
                # Faithful reading of the rule: the slice is sold out of the baseline, proceeds sit
                # in cash, and the +/-3% band later redeploys them.
                diag["sell_only_legs"] += 1
                if cfg.sell_only_mode == "close_tranche":
                    can = tranches and (not cfg.sell_only_counts_to_cap
                                        or sold_today < cfg.max_sells_per_day)
                    if can:
                        i = (0 if cfg.sell_only_pick == "oldest"
                             else max(range(len(tranches)), key=lambda k: tranches[k]["cost"]))
                        tr = tranches.pop(i)
                        cash += tr["shares"] * sell_trigger * (1 - cfg.cost_leg)
                        if cfg.sell_only_counts_to_cap:
                            sold_today += 1
                        diag["sell_only_closed_tranche"] = diag.get("sell_only_closed_tranche", 0) + 1
                        log(date=d, type="双边卖出(平掉一笔加仓)", shares=tr["shares"],
                            cost=tr["cost"], sell_price=sell_trigger,
                            opened_date=tr["opened_date"], held_days=t - tr["i0"])
                    else:
                        # nothing above the baseline to sell -> the order is simply not placed,
                        # rather than selling core holding that the band would buy right back
                        diag["sell_only_no_tranche"] = diag.get("sell_only_no_tranche", 0) + 1
                elif cfg.sell_only_mode != "none":
                    sold_shares = min(cfg.slice_frac * eq_prev, baseline_shares * sell_trigger) / sell_trigger
                    proceeds = sold_shares * sell_trigger * (1 - cfg.cost_leg)
                    baseline_shares -= sold_shares
                    if cfg.sell_only_mode == "to_cash":
                        cash += proceeds
                        log(date=d, type="双边卖出(留现金)", shares=sold_shares,
                            sell_price=sell_trigger)
                    elif cfg.sell_only_mode == "buy_back_close":
                        back = proceeds / (c * (1 + cfg.cost_leg))
                        baseline_shares += back
                        log(date=d, type="双边卖出(收盘买回)", shares=sold_shares,
                            sell_price=sell_trigger, buy_price=c)
                    else:
                        raise ValueError(f"unknown sell_only_mode {cfg.sell_only_mode}")

            baseline_val_t = baseline_shares * c

        _mark(equity, leverage, baseline_val_t, tranches, c, cash)
        baseline_shares, cash = _rebalance(equity, leverage, baseline_shares, cash,
                                            tranches, c, cfg, diag, log, d)

    return {"equity": equity, "bench": bench, "leverage": leverage, "diag": diag,
            "open_tranches": tranches, "dates": dates}


def _rebalance(equity, leverage, baseline_shares, cash, tranches, close, cfg, diag,
                log=None, date_str=None):
    """Pull the baseline back to its target weight if it has drifted outside the band.

    Charges cost_leg on the traded notional and reflects the fee in the recorded equity and leverage
    for that day, so the drag is never invisible. Runs every day including warmup, because the
    account holds the baseline from day one whether or not the model is trading yet.
    """
    eq = equity[-1]
    if cfg.rebalance_band is None or eq <= 0:
        return baseline_shares, cash
    cur_val = baseline_shares * close
    if abs(cur_val / eq - cfg.base_frac) <= cfg.rebalance_band:
        return baseline_shares, cash
    fee = abs(cfg.base_frac * eq - cur_val) * cfg.cost_leg
    eq_after = eq - fee
    target_val = cfg.base_frac * eq_after
    baseline_shares = target_val / close
    tv = sum(tr["shares"] * close for tr in tranches)
    cash = eq_after - target_val - tv
    equity[-1] = eq_after
    leverage[-1] = (target_val + tv) / eq_after if eq_after > 0 else 1.0
    diag["rebalances"] += 1
    diag["rebalance_fees"] += fee
    if log is not None and date_str is not None:
        log(date=date_str, type="底仓再平衡", direction=("买入" if target_val > cur_val else "卖出"),
            traded=abs(target_val - cur_val), fee=fee)
    return baseline_shares, cash


def _tick(px, tick):
    """Round a limit price to the exchange's quotable increment."""
    return round(px / tick) * tick if tick else px


def _mark(equity, leverage, baseline_val, tranches, close, cash):
    tv = sum(tr["shares"] * close for tr in tranches)
    total = baseline_val + tv
    eq = total + cash
    equity.append(eq)
    leverage.append(total / eq if eq > 0 else 1.0)


def _std_t_factor(o, lo, c, prev_c, cost_leg):
    """Multiplicative factor on the slice's value for one standard-T day.

    Sold at the open; bought back at open*0.99 if the low touched it, else at the close. Both legs
    pay cost_leg. Expressed relative to the previous close so it composes with the baseline's own
    return, which is how every prior version of this engine did it.
    """
    if lo <= o * 0.99:
        return (c / prev_c) * (1 / 0.99) * (1 - 2 * cost_leg)
    return (o / prev_c) * (1 - 2 * cost_leg)


def _gate(v):
    if isinstance(v, (bool, np.bool_)):
        return 0.0 if bool(v) else 1.0
    return float(v)


# ======================================================================================
# The breaker, driven only by the frozen reference strategy                             F4
# ======================================================================================
def breaker_signal(bars, ref_result):
    """180-calendar-day rolling excess return of the reference strategy vs 100% buy-and-hold."""
    n = len(bars)
    dd = [parse_ymd(b["date"]) for b in bars]
    eq, bh = ref_result["equity"], ref_result["bench"]
    sig = [None] * n
    for t in range(1, n):
        ref = None
        for tp in range(t, -1, -1):
            if (dd[t] - dd[tp]).days >= BREAKER_CAL_DAYS:
                ref = tp
                break
        if ref is None:
            continue
        sig[t] = (eq[t] / eq[ref] - 1) - (bh[t] / bh[ref] - 1)
    return sig


def breaker_states(sig, stop_thresh=None, resume_thresh=None):
    """Latched state machine. stopped[t] is the state as of day t's close, gating day t+1.

    The thresholds are explicit arguments because a caller that is reproducing a HISTORICAL calibration
    must pin them rather than inherit whatever the module currently holds. The legacy-reproduction
    check did inherit them, so when the live calibration moved from 20% to 8% the check silently
    followed and reported the identity as broken -- the same way verify_baseline() once went stale.
    """
    st = BREAKER_STOP_THRESH if stop_thresh is None else stop_thresh
    rt = BREAKER_RESUME_THRESH if resume_thresh is None else resume_thresh
    n = len(sig)
    stopped = [False] * n
    on = False
    for t in range(n):
        s = sig[t]
        if s is not None:
            if not on and s >= st:
                on = True
            elif on and s <= rt:
                on = False
        stopped[t] = on
    return stopped


def reference_breaker(bars, probs, warmup):
    """The ONLY way the breaker schedule is ever produced.

    The reference strategy is run with the breaker permanently OFF (it is the counterfactual "what if
    we never stopped"), under REFERENCE_CONFIG, which is frozen. Nothing about the strategy being
    executed or researched enters here.
    """
    n = len(bars)
    ref = simulate(bars, probs, warmup, [False] * n, REFERENCE_CONFIG)
    sig = breaker_signal(bars, ref)
    return breaker_states(sig), sig, ref


def transitions(stopped, dates):
    out = []
    for t in range(1, len(dates)):
        if stopped[t] and not stopped[t - 1]:
            out.append({"date": dates[t], "type": "STOP"})
        elif not stopped[t] and stopped[t - 1]:
            out.append({"date": dates[t], "type": "RESUME"})
    return out


# ======================================================================================
# Metrics
# ======================================================================================
def max_drawdown(eq):
    peak, worst = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1)
    return worst


def summarize(res, warmup, label=""):
    eq = np.array(res["equity"], float)
    bh = np.array(res["bench"], float)
    lv = np.array(res["leverage"], float)
    return {
        "label": label,
        "return_pct": (eq[-1] - 1) * 100,
        "bench_pct": (bh[-1] - 1) * 100,
        "excess_pp": ((eq[-1] - 1) - (bh[-1] - 1)) * 100,
        "ratio": float(eq[-1] / bh[-1]),
        "maxdd_pct": max_drawdown(list(eq)) * 100,
        "lev_mean": float(lv[warmup:].mean()),
        "lev_max": float(lv[warmup:].max()),
        "time_at_cap_pct": float(np.mean(lv[warmup:] >= 1.45) * 100),
        "diag": res["diag"],
    }


if __name__ == "__main__":
    bars, probs, warmup = load_data()
    dates = [b["date"] for b in bars]
    stopped, sig, ref = reference_breaker(bars, probs, warmup)
    print(f"参考策略 {REFERENCE_VERSION} 产生的熔断时间表:")
    for tr in transitions(stopped, dates):
        print(f"   {tr['date']}  {tr['type']}")
    live = simulate(bars, probs, warmup, stopped, Config())
    s = summarize(live, warmup, "strategy_core 默认口径")
    print(f"\n收益 {s['return_pct']:.2f}%  基准 {s['bench_pct']:.2f}%  超额 {s['excess_pp']:+.2f}pp"
          f"  回撤 {s['maxdd_pct']:.2f}%")
    print(f"杠杆 均值 {s['lev_mean']:.3f}  max {s['lev_max']:.3f}  顶格 {s['time_at_cap_pct']:.1f}%")
    print(f"诊断 {json.dumps(s['diag'], ensure_ascii=False)}")
