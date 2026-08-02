"""
daily_signal.py -- build the 09:15 morning briefing.

TIMING
The two things this email carries become available at different moments:
  09:00  the commodity day session opens -> today's probability, hence the GATE, is computable
  09:25  the stock's call auction ends   -> the stock's open is fixed, hence the bracket PRICES
GitHub's scheduled runners drift by 5-30 minutes and can be skipped entirely, so the job is scheduled
EARLY and polls for the stock's open rather than trying to land on 09:25. A late run is not a wrong
run: the open, once set, does not change for the rest of the session, so the prices stay correct --
only the time left to act shrinks. The message always states the Beijing timestamp it was built at.

WHY 09:15 WORKS AT ALL
The model's features are commodity overnight gaps, and the commodity day session opens at 09:00 --
so by 09:15 today's probability is already computable. The stock's own open is not known until the
09:15-09:25 call auction concludes, so the briefing can state the GATE (trade or not) and every order
whose price does not depend on today's open, and gives the two bracket prices as formulas to fill in
at 09:25.

WHAT IT DELIBERATELY DOES NOT DO
It does not send anything. Delivery needs a mail credential, and handling one is not something I will
do -- this writes the message to stdout / a file and exits. Whatever schedules it owns the sending.

POSITIONS
Open positions come from the Gist the tool already syncs to (secret gists are readable from their raw
URL without a token, which is why no credential is needed here either). Without a Gist URL the
briefing still covers the gate, the breaker, and the baseline check -- just not the per-position
take-profit prices.
"""

import argparse
import json
import os
import re
import ssl
import sys
import urllib.request
from datetime import datetime, timezone, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy_core import (Config, simulate, breaker_signal, breaker_states, transitions,
                           REFERENCE_CONFIG, REFERENCE_VERSION,
                           BREAKER_STOP_THRESH, BREAKER_RESUME_THRESH)
# Gate threshold and headline stats are read from the engine / the shipped state file, never written
# here: the threshold moved 0.75 -> 0.779 with the day-open model, and a hardcoded copy in this file
# would be the fifth instance of the same constant-in-two-places failure.
from leverage_backtest_cap3 import THRESH

CN = timezone(timedelta(hours=8))
HISTORY_START = "2019-05-16"
FUT = [("MA0", "ma"), ("TA0", "ta"), ("SC0", "sc"), ("JM0", "jm"), ("PP0", "pp")]
#: copied verbatim from index.html:435 -- these are the MAIN-CONTRACT ids ("...M"), not the
#: continuous-series ids used for the daily K-lines. Guessing the latter returns data:null.
EM_SECIDS = {"u": "1.600989", "ma": "115.MAM", "ta": "115.TAM",
             "sc": "142.scm", "jm": "114.jmm", "pp": "114.ppm"}
_CTX = ssl.create_default_context()


def _get(url, timeout=25):
    # Both providers reject requests carrying the other one's Referer, so pick it per host.
    ref = "https://quote.eastmoney.com/" if "eastmoney" in url else "https://finance.sina.com.cn"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": ref})
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
        return r.read().decode("utf-8", "replace")


def _jsonp(text):
    """Sina wraps the payload as  var _=([{...}]);  and prefixes an anti-hotlink <script> comment,
    so slice from the first [ to the last ] rather than trying to match the assignment."""
    i, j = text.find("["), text.rfind("]")
    if i < 0 or j <= i:
        raise ValueError("unparseable JSONP: " + text[:140])
    return json.loads(re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', text[i:j + 1]))


def fetch_stock():
    t = _get("https://money.finance.sina.com.cn/quotes_service/api/jsonp.php/var%20_=/"
             "CN_MarketData.getKLineData?symbol=sh600989&scale=240&ma=no&datalen=2000")
    return [{"date": r["day"], "open": float(r["open"]), "high": float(r["high"]),
             "low": float(r["low"]), "close": float(r["close"]), "volume": float(r["volume"])}
            for r in _jsonp(t) if r["day"] >= HISTORY_START]


def fetch_futures(sym):
    t = _get(f"https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_=/"
             f"InnerFuturesNewService.getDailyKLine?symbol={sym}")
    return [{"date": r["d"], "open": float(r["o"]), "high": float(r["h"]),
             "low": float(r["l"]), "close": float(r["c"])}
            for r in _jsonp(t) if r["d"] >= HISTORY_START]


def fetch_realtime():
    out = {}
    for k, secid in EM_SECIDS.items():
        t = _get("https://push2delay.eastmoney.com/api/qt/stock/get?fltt=2&invt=2&"
                 f"fields=f43,f46,f60&secid={secid}")
        d = json.loads(t).get("data")
        if not d:
            raise RuntimeError(f"实时行情无数据: {k} ({secid}) —— 合约代码可能已变更")
        out[k] = {"open": d.get("f46"), "prev_close": d.get("f60"), "last": d.get("f43")}
    return out


def features(bars, fut, t, rt_open=None):
    """Same 14 features as the engine, all strictly from data through t-1 plus today's commodity opens."""
    c = np.array([b["close"] for b in bars]); o = np.array([b["open"] for b in bars])
    h = np.array([b["high"] for b in bars]); l = np.array([b["low"] for b in bars])
    v = np.array([b["volume"] for b in bars])
    row = []
    for _, k in FUT:                                   # |overnight gap| for all five
        fc = np.array([b["close"] for b in fut[k]]); fo = np.array([b["open"] for b in fut[k]])
        g = (rt_open[k] / fc[t-1] - 1.0) if (rt_open and t >= len(fo)) else (fo[t] / fc[t-1] - 1.0)
        row.append(abs(g))
    for _, k in FUT[:3]:                               # signed gap for ma/ta/sc
        fc = np.array([b["close"] for b in fut[k]]); fo = np.array([b["open"] for b in fut[k]])
        g = (rt_open[k] / fc[t-1] - 1.0) if (rt_open and t >= len(fo)) else (fo[t] / fc[t-1] - 1.0)
        row.append(g)
    row.append((h[t-1] - l[t-1]) / o[t-1])
    row.append(c[t-1] / c[t-2] - 1.0)
    a5 = v[t-6:t-1].mean()
    row.append(v[t-1] / a5 if a5 > 0 else np.nan)
    row.append(c[t-1] / c[t-5-1:t-1].mean() - 1.0)
    row.append(c[t-1] / c[t-20-1:t-1].mean() - 1.0)
    row.append(c[t-5-1:t-1].mean() / c[t-20-1:t-1].mean() - 1.0)
    return np.array(row, float)


def train_and_predict(bars, fut, rt_open, warmup=21):
    n = len(bars)
    X = np.array([features(bars, fut, t) for t in range(warmup, n)])
    y = np.array([1.0 if bars[t]["low"] <= bars[t]["open"] * 0.99 else 0.0 for t in range(warmup, n)])
    ok = np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(4000):                              # same trainer the page uses
        p = 1 / (1 + np.exp(-(Z @ w + b))); g = p - y
        w -= 0.5 * (Z.T @ g / len(y) + 1.0 * w / len(y)); b -= 0.5 * g.mean()
    xt = (features(bars, fut, n, rt_open) - mu) / sd
    return float(1 / (1 + np.exp(-(xt @ w + b))))


def load_positions(gist_raw_url):
    if not gist_raw_url:
        return None
    try:
        return json.loads(_get(gist_raw_url))
    except Exception as e:
        print(f"[warn] 读取持仓失败: {e}", file=sys.stderr)
        return None


def wait_for_stock_open(max_wait_s=900, poll_s=20):
    """Poll until the stock's call auction has produced an open (f46 > 0), or give up.

    Returns (realtime_dict, waited_seconds, got_open). Giving up is not fatal -- the briefing still
    carries the gate and every order whose price does not depend on today's open.
    """
    waited = 0
    while True:
        rt = fetch_realtime()
        op = rt["u"]["open"]
        if op and op > 0:
            return rt, waited, True
        if waited >= max_wait_s:
            return rt, waited, False
        print(f"[wait] 个股开盘价尚未产生,{poll_s}s 后重试(已等 {waited}s)", file=sys.stderr)
        time.sleep(poll_s); waited += poll_s


def build(gist_url=None, wait_open=False):
    bars = fetch_stock()
    fut = {k: fetch_futures(s) for s, k in FUT}
    dates = set(b["date"] for b in bars)
    for k in fut:
        dates &= set(b["date"] for b in fut[k])
    bars = [b for b in bars if b["date"] in dates]
    for k in fut:
        fut[k] = [b for b in fut[k] if b["date"] in dates]

    if wait_open:
        rt, waited, got_open = wait_for_stock_open()
    else:
        rt, waited = fetch_realtime(), 0
        got_open = bool(rt["u"]["open"] and rt["u"]["open"] > 0)
    rt_open = {k: rt[k]["open"] for _, k in FUT}
    prob = train_and_predict(bars, fut, rt_open)

    # Breaker: frozen reference, run over the same bars, never stopping. The HISTORICAL probabilities
    # must come from daily_state.json -- they are walk-forward values, each fitted only on data before
    # its own day, so they are data and never change. Feeding NaN here instead (as a first version did)
    # silently removes all 140 gated days from the reference, i.e. simulates a different strategy, and
    # the signal came out 0.4pp off. Only TODAY's probability is computed live.
    # Look next to this script first, then one level up. In the deployed repo daily_state.json lives
    # at the root because GitHub Pages serves it to the web tool; keeping a second copy in signal/
    # would be the same file in two places, which is the failure mode this whole project kept hitting.
    here = os.path.dirname(os.path.abspath(__file__))
    ds_path = next((p for p in (os.path.join(here, "daily_state.json"),
                                 os.path.join(here, os.pardir, "daily_state.json"))
                     if os.path.exists(p)), None)
    if ds_path is None:
        raise RuntimeError("找不到 daily_state.json(已找 signal/ 和仓库根目录)—— 请先运行 export_state.py")
    ds = json.load(open(ds_path, encoding="utf-8"))
    pmap = dict(zip(ds["model"]["dates"], ds["model"]["oos_prob"]))
    missing = [b["date"] for b in bars if b["date"] not in pmap]
    probs_hist = np.array([pmap.get(b["date"], np.nan) if pmap.get(b["date"]) is not None else np.nan
                            for b in bars], dtype=float)
    ref = simulate(bars, probs_hist, 171, [False] * len(bars), REFERENCE_CONFIG)
    sig = breaker_signal(bars, ref)
    se = breaker_states(sig)
    stopped = bool(se[-1])
    last_sig = sig[-1]
    trs = transitions(se, [b["date"] for b in bars])

    st = load_positions(gist_url) or {}
    tranches = st.get("tranches", []) if isinstance(st, dict) else []
    total_eq = st.get("totalEquity") if isinstance(st, dict) else None
    base_val = st.get("baselineValue") if isinstance(st, dict) else None

    return {
        "now": datetime.now(CN).strftime("%Y-%m-%d %H:%M"),
        "last_bar": bars[-1]["date"], "last_close": bars[-1]["close"],
        "stock_rt": rt["u"], "prob": prob, "gate_open": prob >= THRESH,
        "stopped": stopped, "signal": last_sig,
        "last_transition": trs[-1] if trs else None,
        "tranches": tranches, "total_equity": total_eq, "baseline_value": base_val,
        "ref_version": REFERENCE_VERSION,
        "backtest": ds.get("backtest_A", {}),
        "stock_open": rt["u"]["open"] if got_open else None,
        "waited_s": waited,
        "stale_days": len(missing),
        "state_through": ds["generated_from"]["last_date"],
    }


def render(d):
    L = []
    A = L.append
    gate, stop = d["gate_open"], d["stopped"]
    if stop:
        head = "熔断停摆中 — 今日不挂买单"
    elif gate:
        head = "闸门开启 — 今日不新开仓"
    else:
        head = "正常 — 9:25 挂双边单"
    A(f"600989 {d['now']}   【{head}】")
    A("")
    A(f"模型概率 {d['prob']*100:.1f}%  (阈值 {THRESH*100:.1f}%,{'≥ 闸门开' if gate else '< 闸门关'})")
    sg = "n/a" if d["signal"] is None else f"{d['signal']*100:+.2f}%"
    A(f"熔断信号 {sg}  (停摆线 +{BREAKER_STOP_THRESH*100:.0f}% / 解除线 {BREAKER_RESUME_THRESH*100:.0f}%)"
      f"  状态: {'停摆中' if stop else '正常'}")
    if d["last_transition"]:
        t = d["last_transition"]
        A(f"最近切换 {t['date']} {'停摆' if t['type']=='STOP' else '解除'}")
    A("")

    op = d.get("stock_open")
    if op:
        A(f"── 今日挂单价(开盘 {op:.2f},已确定)──")
        buy, sell = round(op * 0.991, 2), round(op * 1.03, 2)
        if stop or gate:
            A("  不挂买单。" + ("熔断只挡买入,卖单照挂。" if stop else "闸门开启,整套双边都不做。"))
            if stop:
                A(f"  上方卖单: {sell:.2f}   → 平掉最老的一笔加仓"
                  + ("" if d["tranches"] else "(当前无加仓,不挂)"))
        else:
            A(f"  买单: {buy:.2f}    (先确认买入后杠杆不超 1.5 倍,超了就只挂卖单)")
            A(f"  卖单: {sell:.2f}    → 买单也成交则当日了结;否则平掉最老的一笔加仓"
              + ("" if d["tranches"] else ";当前无加仓,若买单没成交则此单不挂"))
    else:
        A("── 9:25 开盘价出来后 ──")
        A(f"  (未取到开盘价,等了 {d.get('waited_s', 0)}s;下面给公式,请自行按开盘价换算)")
        if stop or gate:
            A("  不挂买单。" + ("熔断只挡买入,卖单照挂。" if stop else "闸门期间整套双边都不做。"))
            if stop:
                A("  上方卖单: 开盘价 × 1.03  → 平掉最老的一笔加仓(没有加仓就不挂)")
        else:
            A("  买单: 开盘价 × 0.991     (先确认买入后杠杆不超 1.5 倍,超了就只挂卖单)")
            A("  卖单: 开盘价 × 1.03      → 若买单也成交则为当日了结;否则平掉最老的一笔加仓")
    A("")

    trs = d["tranches"]
    if trs:
        A(f"── 老仓位止盈单(共 {len(trs)} 笔,每天最多成交 3 笔,最老优先)──")
        for t in sorted(trs, key=lambda x: x.get("date", "")):
            A(f"  {t.get('date','?')}  {t.get('shares','?')}股  成本 {t.get('cost',0):.2f}"
              f"  → 挂 {t.get('cost',0)*1.05:.2f}")
        A("  这些单子任何时候都要挂,熔断和闸门都不影响。")
    else:
        A("── 当前没有加仓仓位 ──")
        A("  上方卖单不挂(它的作用是平掉一笔加仓,不是卖底仓)。")
    A("")

    if d["total_equity"] and d["baseline_value"]:
        w = d["baseline_value"] / d["total_equity"]
        A(f"── 底仓 ──")
        A(f"  占净值 {w*100:.1f}% (目标 90%,±3% 内不用动)")
        if abs(w - 0.90) > 0.03:
            gap = 0.90 * d["total_equity"] - d["baseline_value"]
            A(f"  偏离超过带宽,建议{'买入' if gap>0 else '卖出'}约 {abs(gap):,.0f} 元调回")
        A("")

    if d.get("stale_days"):
        A(f"⚠ daily_state.json 只到 {d['state_through']},之后有 {d['stale_days']} 个交易日没有历史概率,")
        A(f"   熔断信号可能不准。请重新运行 export_state.py 并更新该文件。")
        A("")
    A(f"参考策略 {d['ref_version']}  |  最新收盘 {d['last_bar']} {d['last_close']:.2f}"
      f"  |  实时 今开 {d['stock_rt'].get('open')} 昨收 {d['stock_rt'].get('prev_close')}")
    bt = d.get("backtest") or {}
    if bt:
        A(f"回测 {bt.get('window','?')}: {bt.get('return_pct','?')}%,最大回撤 {bt.get('maxdd_pct','?')}%,"
          f"日均融资占用 {bt.get('borrow_mean_pct','?')}%。单标的单路径,尾部风险不在回测里,请自行判断。")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gist", default=os.environ.get("POSITIONS_GIST_RAW_URL"),
                    help="raw URL of the gist the web tool syncs positions to (optional)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    msg = render(build(a.gist))
    if a.out:
        open(a.out, "w", encoding="utf-8").write(msg)
    print(msg)
