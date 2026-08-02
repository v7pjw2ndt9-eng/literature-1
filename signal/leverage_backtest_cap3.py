"""
leverage_backtest_cap3.py

Faithful Python port of JS_REFERENCE's runCounterfactualSimulation +
computeSignalAndBreaker, run twice:

  (a) NEW rule: MAX_TRANCHE_SELLS_PER_DAY = 3, oldest-tranche priority
  (b) OLD rule: no cap at all (same code, cap check removed)

plus (c) the 100%-buy-and-hold benchmark (identical for both variants,
produced once as cf.bench).

Data:
  raw_data_2y.json      -> stock bars (bfny), ascending date order
  touch_model_2y.json   -> true walk-forward oos_prob[t] (NaN before "start")

Critical adaptation vs. the live JS tool: warmup = start = 171 (not 21),
because we substitute the true walk-forward oos_prob (only valid from
index "start" onward) for JS_REFERENCE's probs[t]. For t < start we take
the JS's final "else" branch verbatim: pure 90% baseline, zero ops.
For t >= start we run the exact active-mode logic (standard-T vs
bracket/tranche), treating NaN oos_prob as "not >= THRESH" (a plain
Python `x >= THRESH` with NaN x is already False, so this falls through
to the bracket/tranche branch exactly like JS's null-check semantics --
no special-casing needed, but we double-check this explicitly below).

This script is idempotent: re-running it just overwrites its two output
artifacts (the result JSON and the PNG chart) with identical content.
"""

import json
import math
import os
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates

SCRATCH = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(SCRATCH, "raw_data_2y.json")
# The gate's probability series. Switched 2026-08-02 to the day-session-open features: the daily
# bar's "open" for a commodity is the PRIOR EVENING's 21:00 night open, so on a Monday it predates the
# whole weekend. Rebuilding the 8 commodity features on the 09:00 day-session open (recovered by
# chaining contract-month 120-min bars) lifted walk-forward AUC 0.588 -> 0.605, stable for C from 0.03
# to 10, paired bootstrap 96.6% positive, biggest gain on Mondays. Cost: the series needs 171 valid
# rows to start, and the new features only begin 2020-02-10, so the backtest window shortens from
# 2020-01-23 to 2020-09-16 (1579 -> 1422 days). Set to touch_model_2y.json to go back.
TOUCH_PATH = os.path.join(SCRATCH, "touch_model_dayopen.json")
TOUCH_PATH_LEGACY = os.path.join(SCRATCH, "touch_model_2y.json")
RESULT_PATH = os.path.join(SCRATCH, "leverage_backtest_cap3_result.json")
CHART_PATH = os.path.join(SCRATCH, "leverage_backtest_cap3_chart.png")

# ----------------------------------------------------------------------
# Fixed parameters, exactly as in JS_REFERENCE.
# ----------------------------------------------------------------------
COST_LEG = 0.001
# Recalibrated with the day-open model. The new features shift the whole probability distribution
# up, so the old 0.75 would gate 176 days where the old model gated 127 -- swapping the model would
# have silently also changed the trading frequency by 39%, which is exactly the confound that made
# three earlier comparisons meaningless. 0.779 reproduces the historical gate rate (~9% of days).
THRESH = 0.779
TARGET_PROFIT = 0.05
LEV_CAP = 1.5
SLICE_FRAC = 0.10
BASE_FRAC = 0.90
X = 0.03   # bracket sell-side offset: sell at open*(1+X). Live tool default (bracketUpperSlider=0.03).
B = 0.009  # bracket buy-side offset: buy at open*(1-B).  Live tool default (bracketLowerSlider=0.009).
MAX_TRANCHE_SELLS_PER_DAY = 3


def parse_ymd(s):
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def load_data(touch_path=None):
    with open(RAW_PATH) as f:
        raw = json.load(f)
    stock_bars_raw = raw["bfny"]

    with open(touch_path or TOUCH_PATH) as f:
        touch = json.load(f)

    # touch_model_2y.json's "dates" is authoritative (full_model_2y.py already aligned it to the
    # intersection of the stock and all 5 commodity series, dropping any sporadic single-day gaps
    # e.g. exchange-holiday-calendar mismatches). raw_data_2y.json's "bfny" may still contain those
    # dropped dates, so filter stock_bars down to exactly touch["dates"] rather than assuming 1:1.
    touch_dates = set(touch["dates"])
    stock_bars = [b for b in stock_bars_raw if b["date"] in touch_dates]
    assert [b["date"] for b in stock_bars] == touch["dates"], (
        "stock_bars dates must line up 1:1 with touch_model_2y.json dates after filtering"
    )
    start = touch["start"]
    oos_prob = touch["oos_prob"]  # list, NaN encoded as None or nan from json (json has no NaN literal support by default -> python json loads NaN as float('nan') if written by python's json.dump; test)
    # json module (python) does support NaN as an extension when written with allow_nan=True
    # (the default). Confirm oos_prob values are floats (possibly nan).
    oos_prob = [float(x) if x is not None else float("nan") for x in oos_prob]

    return stock_bars, oos_prob, start


def day_factor_standard(stock_bars, t):
    """dayFactorStandard(t, execute=True) from JS_REFERENCE."""
    base = stock_bars[t]["close"] / stock_bars[t - 1]["close"]
    hit = stock_bars[t]["low"] <= stock_bars[t]["open"] * 0.99
    if hit:
        return base * (1 / 0.99) * (1 - 2 * COST_LEG)
    return (stock_bars[t]["open"] / stock_bars[t - 1]["close"]) * (1 - 2 * COST_LEG)


def run_counterfactual_simulation(stock_bars, oos_prob, warmup, enforce_cap, max_sells_per_day=MAX_TRANCHE_SELLS_PER_DAY):
    """
    Port of runCounterfactualSimulation.

    enforce_cap=True  -> NEW rule (MAX_TRANCHE_SELLS_PER_DAY, oldest-first priority)
    enforce_cap=False -> OLD rule (no cap; every eligible tranche closes same day)

    Returns dict with equity, bench, leverage (lists), plus per-day
    diagnostics: n_eligible_today (how many open tranches had h >= target
    price on that day, BEFORE any capping is applied -- used to detect
    days where the cap actually binds).
    """
    n = len(stock_bars)

    baseline_shares = BASE_FRAC / stock_bars[0]["close"]
    cash = 1.0 - BASE_FRAC
    tranches = []  # list of dicts {shares, cost}, appended oldest-last

    equity = [1.0]
    bench = [1.0]
    leverage = [BASE_FRAC]
    n_eligible_today = [0]  # index 0 unused/placeholder to align with t index
    n_open_tranches = [0]   # open tranche count AFTER today's activity, aligned to t index
    buy_fires_count = 0     # total number of days the buy trigger (l <= o*(1-B)) fired

    for t in range(1, n):
        bench.append(bench[-1] * (stock_bars[t]["close"] / stock_bars[t - 1]["close"]))
        eq_prev = equity[-1]
        baseline_val_prev = baseline_shares * stock_bars[t - 1]["close"]

        p = oos_prob[t] if t < len(oos_prob) else float("nan")
        # NaN-safe: `p >= THRESH` is False whenever p is NaN, matching JS's
        # `probs[t] !== null && probs[t] >= THRESH` semantics (falls through
        # to the bracket/tranche branch). No special-casing required; we
        # assert this behavior holds.
        standard_t_condition = (t >= warmup) and (p >= THRESH)
        assert not (t >= warmup and math.isnan(p) and standard_t_condition), "NaN must not satisfy standard-T condition"

        eligible_today = 0

        if standard_t_condition:
            r = day_factor_standard(stock_bars, t)
            slice_val = SLICE_FRAC * eq_prev
            remaining = baseline_val_prev - slice_val
            baseline_val_t = remaining * (stock_bars[t]["close"] / stock_bars[t - 1]["close"]) + slice_val * r
            baseline_shares = baseline_val_t / stock_bars[t]["close"]

        elif t >= warmup:
            o = stock_bars[t]["open"]
            h = stock_bars[t]["high"]
            l = stock_bars[t]["low"]
            c = stock_bars[t]["close"]

            # Count how many currently-open tranches are eligible to close
            # today (h >= target), BEFORE applying any per-day cap. This is
            # the diagnostic used to determine when the NEW rule's cap
            # actually binds (i.e. eligible_today > MAX_TRANCHE_SELLS_PER_DAY).
            eligible_today = sum(1 for tr in tranches if h >= tr["cost"] * (1 + TARGET_PROFIT))

            sold_today = 0
            still_open = []
            for tr in tranches:
                target_hit = h >= tr["cost"] * (1 + TARGET_PROFIT)
                can_sell = target_hit and (not enforce_cap or sold_today < max_sells_per_day)
                if can_sell:
                    cash += tr["shares"] * tr["cost"] * (1 + TARGET_PROFIT) * (1 - COST_LEG)
                    sold_today += 1
                else:
                    still_open.append(tr)
            tranches = still_open

            buy_trigger = o * (1 - B)
            buy_fires = l <= buy_trigger
            if buy_fires:
                buy_fires_count += 1
            # LOOKAHEAD FIX: value existing exposure at the day's OPEN, not its close. The bracket
            # orders are placed at/just after the 9:25 open, so open[t] is known but close[t] is not.
            open_val = sum(tr["shares"] * o for tr in tranches)
            cur_exposure = baseline_shares * o + open_val
            if buy_fires:
                new_shares = (SLICE_FRAC * eq_prev) / buy_trigger
                if (cur_exposure + new_shares * buy_trigger) / eq_prev <= LEV_CAP:
                    cash -= new_shares * buy_trigger * (1 + COST_LEG)
                    if h >= o * (1 + X):
                        cash += new_shares * o * (1 + X) * (1 - COST_LEG)
                    else:
                        tranches.append({"shares": new_shares, "cost": buy_trigger})
                # else: leverage cap breached -> buy simply doesn't happen this day

            baseline_val_t = baseline_shares * c

        else:
            baseline_val_t = baseline_shares * stock_bars[t]["close"]

        tranche_val = sum(tr["shares"] * stock_bars[t]["close"] for tr in tranches)
        stock_total = baseline_val_t + tranche_val
        eq_t = stock_total + cash
        equity.append(eq_t)
        leverage.append(stock_total / eq_t if eq_t > 0 else 1.0)
        n_eligible_today.append(eligible_today)
        n_open_tranches.append(len(tranches))

    return {
        "equity": equity,
        "bench": bench,
        "leverage": leverage,
        "n_eligible_today": n_eligible_today,
        "n_open_tranches": n_open_tranches,
        "buy_fires_count": buy_fires_count,
        "max_open_tranches": max(n_open_tranches),
    }


def compute_signal_and_breaker(stock_bars, cf):
    """Port of computeSignalAndBreaker: 180-CALENDAR-day rolling excess
    return + STOP/RESUME hysteresis, using actual calendar dates."""
    n = len(stock_bars)
    dates = [parse_ymd(b["date"]) for b in stock_bars]
    STOP_THRESH = 0.20
    RESUME_THRESH = 0.0
    CAL_DAYS = 180

    stopped = False
    signal = []
    transitions = []

    for t in range(1, n):
        ref = None
        for tp in range(t, -1, -1):
            if (dates[t] - dates[tp]).days >= CAL_DAYS:
                ref = tp
                break
        if ref is None:
            continue
        sig = (cf["equity"][t] / cf["equity"][ref] - 1) - (cf["bench"][t] / cf["bench"][ref] - 1)
        signal.append({"date": stock_bars[t]["date"], "sig": sig * 100})
        if not stopped and sig >= STOP_THRESH:
            stopped = True
            transitions.append({"date": stock_bars[t]["date"], "type": "STOP"})
        elif stopped and sig <= RESUME_THRESH:
            stopped = False
            transitions.append({"date": stock_bars[t]["date"], "type": "RESUME"})

    return {"signal": signal, "transitions": transitions}


def main():
    stock_bars, oos_prob, start = load_data()
    warmup = start  # critical adaptation, see module docstring

    n = len(stock_bars)
    dates = [b["date"] for b in stock_bars]

    cf_new = run_counterfactual_simulation(stock_bars, oos_prob, warmup, enforce_cap=True)
    cf_old = run_counterfactual_simulation(stock_bars, oos_prob, warmup, enforce_cap=False)

    sb_new = compute_signal_and_breaker(stock_bars, cf_new)
    sb_old = compute_signal_and_breaker(stock_bars, cf_old)

    # ---- Diagnostics: days where the cap actually binds (old rule would
    # have closed MORE than MAX_TRANCHE_SELLS_PER_DAY tranches). Note the
    # eligibility count is identical between new/old runs up to the point
    # a divergence in tranches held could occur; we use the NEW-rule run's
    # eligibility count as ground truth since that's the scenario the cap
    # applies to, but also cross check against the OLD-rule run.
    binding_days_new = [
        {"date": dates[t], "n_eligible": cf_new["n_eligible_today"][t]}
        for t in range(1, n)
        if cf_new["n_eligible_today"][t] > MAX_TRANCHE_SELLS_PER_DAY
    ]
    binding_days_old = [
        {"date": dates[t], "n_eligible": cf_old["n_eligible_today"][t]}
        for t in range(1, n)
        if cf_old["n_eligible_today"][t] > MAX_TRANCHE_SELLS_PER_DAY
    ]

    # ---- Divergence detection: first index/date where new_rule and
    # old_rule equity paths actually differ (beyond floating-point noise).
    EPS = 1e-9
    first_divergence_idx = None
    for t in range(n):
        if abs(cf_new["equity"][t] - cf_old["equity"][t]) > EPS:
            first_divergence_idx = t
            break
    first_divergence = None
    if first_divergence_idx is not None:
        t = first_divergence_idx
        first_divergence = {
            "date": dates[t],
            "index": t,
            "equity_new": cf_new["equity"][t],
            "equity_old": cf_old["equity"][t],
            "leverage_new": cf_new["leverage"][t],
            "leverage_old": cf_old["leverage"][t],
            "n_eligible_that_day": cf_new["n_eligible_today"][t],
        }

    result = {
        "new_rule": {
            "equity": cf_new["equity"],
            "bench": cf_new["bench"],
            "leverage": cf_new["leverage"],
            "signal": sb_new["signal"],
            "transitions": sb_new["transitions"],
        },
        "old_rule": {
            "equity": cf_old["equity"],
            "bench": cf_old["bench"],
            "leverage": cf_old["leverage"],
            "signal": sb_old["signal"],
            "transitions": sb_old["transitions"],
        },
        "dates": dates,
        "meta": {
            "start": start,
            "warmup_used": warmup,
            "thresh": THRESH,
            "x": X,
            "b": B,
            "target_profit": TARGET_PROFIT,
            "lev_cap": LEV_CAP,
            "cost_leg": COST_LEG,
            "max_tranche_sells_per_day_new_rule": MAX_TRANCHE_SELLS_PER_DAY,
            "note_bench": "bench is identical between new_rule and old_rule (100% buy-and-hold); included in both for convenience.",
        },
        "diagnostics": {
            "binding_days_new_rule_run": binding_days_new,
            "binding_days_old_rule_run": binding_days_old,
            "buy_fires_count_new_rule_run": cf_new["buy_fires_count"],
            "buy_fires_count_old_rule_run": cf_old["buy_fires_count"],
            "max_open_tranches_new_rule_run": cf_new["max_open_tranches"],
            "max_open_tranches_old_rule_run": cf_old["max_open_tranches"],
            "first_divergence_new_vs_old": first_divergence,
        },
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, allow_nan=True)

    # ------------------------------------------------------------------
    # Chart
    # ------------------------------------------------------------------
    font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    dt = [parse_ymd(d) for d in dates]

    cum_new = [(v - 1.0) * 100 for v in cf_new["equity"]]
    cum_old = [(v - 1.0) * 100 for v in cf_old["equity"]]
    cum_bench = [(v - 1.0) * 100 for v in cf_new["bench"]]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # (i) cumulative return
    ax = axes[0]
    ax.plot(dt, cum_new, label="新规则 (每日封顶3笔)", color="#1f77b4", linewidth=1.4)
    ax.plot(dt, cum_old, label="旧规则 (无封顶)", color="#ff7f0e", linewidth=1.4)
    ax.plot(dt, cum_bench, label="100%满仓持有", color="#7f7f7f", linewidth=1.2, linestyle="--")
    ax.set_title("累计收益率对比：新规则 vs 旧规则 vs 满仓持有", fontsize=13)
    ax.set_ylabel("累计收益率 (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # (ii) 180-day rolling excess return signal
    ax = axes[1]
    sig_new_dates = [parse_ymd(p["date"]) for p in sb_new["signal"]]
    sig_new_vals = [p["sig"] for p in sb_new["signal"]]
    sig_old_dates = [parse_ymd(p["date"]) for p in sb_old["signal"]]
    sig_old_vals = [p["sig"] for p in sb_old["signal"]]
    ax.plot(sig_new_dates, sig_new_vals, label="新规则 180日超额收益", color="#1f77b4", linewidth=1.2)
    ax.plot(sig_old_dates, sig_old_vals, label="旧规则 180日超额收益", color="#ff7f0e", linewidth=1.2)
    ax.axhline(20, color="red", linestyle="--", linewidth=1, label="停摆20%")
    ax.axhline(0, color="green", linestyle="--", linewidth=1, label="解除0%")
    for tr in sb_new["transitions"]:
        d = parse_ymd(tr["date"])
        color = "red" if tr["type"] == "STOP" else "green"
        ax.axvline(d, color=color, linestyle=":", linewidth=1.3, alpha=0.85)
    ax.set_title("180日滚动超额收益与熔断信号（叠加新规则的停摆/解除节点）", fontsize=13)
    ax.set_ylabel("超额收益 (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # (iii) leverage ratio
    ax = axes[2]
    ax.plot(dt, cf_new["leverage"], label="新规则杠杆率", color="#1f77b4", linewidth=1.2)
    ax.plot(dt, cf_old["leverage"], label="旧规则杠杆率", color="#ff7f0e", linewidth=1.2)
    ax.axhline(1.5, color="red", linestyle="--", linewidth=1, label="软顶1.5")
    ax.set_title("持仓杠杆率对比（敞口/净值）", fontsize=13)
    ax.set_ylabel("杠杆率")
    ax.set_xlabel("日期")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)
    plt.close(fig)

    print("Wrote:", RESULT_PATH, os.path.getsize(RESULT_PATH), "bytes")
    print("Wrote:", CHART_PATH, os.path.getsize(CHART_PATH), "bytes")

    # ---- Console summary for quick sanity check ----
    print("\n=== Summary ===")
    print(f"X={X}  B={B}  start/warmup = {start}")
    print(f"Final cumulative return: new={cum_new[-1]:.2f}%  old={cum_old[-1]:.2f}%  bench={cum_bench[-1]:.2f}%")
    print(f"Leverage new: min={min(cf_new['leverage']):.4f} max={max(cf_new['leverage']):.4f}")
    print(f"Leverage old: min={min(cf_old['leverage']):.4f} max={max(cf_old['leverage']):.4f}")
    print(f"Buy trigger fires (new-rule run): {cf_new['buy_fires_count']}")
    print(f"Buy trigger fires (old-rule run): {cf_old['buy_fires_count']}")
    print(f"Max concurrently open tranches (new-rule run): {cf_new['max_open_tranches']}")
    print(f"Max concurrently open tranches (old-rule run): {cf_old['max_open_tranches']}")
    print(f"Binding days (new-rule run): {len(binding_days_new)}")
    for bd in binding_days_new:
        print("   ", bd)
    print(f"Binding days (old-rule run): {len(binding_days_old)}")
    for bd in binding_days_old:
        print("   ", bd)
    print(f"First divergence new vs old: {first_divergence}")
    print(f"Transitions new: {sb_new['transitions']}")
    print(f"Transitions old: {sb_old['transitions']}")


if __name__ == "__main__":
    main()
