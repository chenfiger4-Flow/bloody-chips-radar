#!/usr/bin/env python3
"""
带血筹码雷达
恐慌触发研究，量价确认触发交易。
状态机: S0 常态 / S1 压力观察 / S2 SC候选 / S3 测试中 / S4 确认 / S5 趋势 / S9 破位
"""
import os, sys, json, time, datetime as dt
import requests, yaml
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
STATE_FILE = os.path.join(DOCS, "state.json")
os.makedirs(DOCS, exist_ok=True)

MODE = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()   # us / crypto / all
NOW = dt.datetime.now(dt.timezone.utc)
BJT = NOW + dt.timedelta(hours=8)

with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

STATE_NAME = {
    "S0": "常态", "S1": "压力观察", "S2": "SC候选", "S3": "测试中",
    "S4": "确认·可试探", "S5": "趋势持有", "S9": "破位/失效", "NA": "数据缺失",
}

# ---------------- 数据获取 ----------------
def fetch_yf(ticker, period="1y"):
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
            if df is not None and len(df) > 30:
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df, "yfinance"
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return None, None

def fetch_coingecko(coin_id):
    """备用：CoinGecko 只有收盘和成交量，无 OHLC；上影线将标记 N/A"""
    try:
        r = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": 200, "interval": "daily"}, timeout=20)
        j = r.json()
        p = pd.DataFrame(j["prices"], columns=["ts", "Close"])
        v = pd.DataFrame(j["total_volumes"], columns=["ts", "Volume"])
        df = p.merge(v, on="ts")
        df.index = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
        df = df[~df.index.duplicated(keep="last")]
        df["Open"] = df["Close"].shift(1)
        df["High"] = df[["Open", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "Close"]].min(axis=1)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df, "coingecko(无OHLC)"
    except Exception:
        return None, None

CG_MAP = {"BTC-USD": "bitcoin", "HYPE32196-USD": "hyperliquid"}

def fetch_hyperliquid(coin, days=400):
    """Hyperliquid 官方 API 日线 K（永续合约，真实 OHLC + 成交量，无需密钥）。
    自动丢弃当天未走完的 K 线，避免运行时刻的半根 K 造成假缩量。"""
    try:
        end = int(time.time() * 1000); start = end - days * 86400 * 1000
        r = requests.post("https://api.hyperliquid.xyz/info",
                          json={"type": "candleSnapshot",
                                "req": {"coin": coin, "interval": "1d", "startTime": start, "endTime": end}},
                          timeout=20)
        j = r.json()
        if not isinstance(j, list) or len(j) < 30:
            print(f"[HL] {coin} 返回异常: {str(j)[:120]}"); return None, None
        df = pd.DataFrame(j)
        # 注意：必须先转成 DatetimeIndex 再 normalize（Series 没有 .normalize()，旧版在此抛错后静默回退到 yfinance）
        df.index = pd.DatetimeIndex(pd.to_datetime(df["t"].astype("int64"), unit="ms")).normalize()
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df = df[~df.index.duplicated(keep="last")].dropna()
        today_utc = pd.Timestamp(NOW.date())
        df = df[df.index < today_utc]          # 丢掉今天未收盘的 K
        print(f"[HL] {coin} 取得 {len(df)} 根日线，最新 {df.index[-1].date()}")
        return (df, "hyperliquid") if len(df) > 30 else (None, None)
    except Exception as e:
        print(f"[HL] {coin} 获取失败: {e}"); return None, None

HL_MAP = {"HYPE32196-USD": "HYPE"}   # 原生交易所优先
CRYPTO_TICKERS = {t["ticker"] for t in CFG["watchlist"] if t["type"].startswith("crypto")}

def drop_open_candle(df):
    """加密 7x24：运行时刻当天(UTC)那根 K 还没走完，成交量只有一部分，
    量比会虚低而误触发“缩量回踩”(S3/LPS)。统一丢掉 index >= 今日UTC 的行。
    美股不适用：22:30 UTC 运行时当日 K 已收盘。"""
    if df is None: return None
    today_utc = pd.Timestamp(NOW.date())
    out = df[df.index < today_utc]
    return out if len(out) > 30 else df

def fetch_fng():
    try:
        j = requests.get("https://api.alternative.me/fng/?limit=8", timeout=15).json()["data"]
        cur = int(j[0]["value"]); prev = int(j[7]["value"]) if len(j) > 7 else None
        return {"value": cur, "label": j[0]["value_classification"], "chg7d": (cur - prev) if prev is not None else None,
                "source": "alternative.me", "ts": dt.datetime.utcfromtimestamp(int(j[0]["timestamp"])).strftime("%Y-%m-%d")}
    except Exception:
        return None

def fetch_hl_funding():
    """Hyperliquid 资金费率（BTC / HYPE 永续）"""
    try:
        r = requests.post("https://api.hyperliquid.xyz/info", json={"type": "metaAndAssetCtxs"}, timeout=15).json()
        names = [u["name"] for u in r[0]["universe"]]
        out = {}
        for n, c in zip(names, r[1]):
            if n in ("BTC", "HYPE"):
                out[n] = {"funding_8h": float(c["funding"]) , "oi": float(c["openInterest"]), "mark": float(c["markPx"])}
        return out
    except Exception:
        return {}

# ---------------- 指标 ----------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def enrich(df):
    d = df.copy()
    d["ema5"], d["ema10"], d["ema20"] = ema(d.Close, 5), ema(d.Close, 10), ema(d.Close, 20)
    d["vol20"] = d.Volume.rolling(20).mean()
    d["vol_ratio"] = d.Volume / d.vol20
    tr = pd.concat([d.High - d.Low, (d.High - d.Close.shift()).abs(), (d.Low - d.Close.shift()).abs()], axis=1).max(axis=1)
    d["atr20"] = tr.rolling(20).mean()
    rng = (d.High - d.Low).replace(0, np.nan)
    d["upper_shadow"] = (d.High - d.Close) / rng
    d["close_pos"] = (d.Close - d.Low) / rng
    d["hi60"] = d.High.rolling(60, min_periods=20).max()   # 次新股不足60根时用已有K，避免 NaN
    d["lo60"] = d.Low.rolling(60, min_periods=20).min()
    d["dd60"] = d.Close / d.hi60 - 1
    return d

def rel_strength(df, bench_df, n):
    if bench_df is None: return None
    a = df.Close.iloc[-1] / df.Close.iloc[-1 - n] - 1
    b = bench_df.Close.iloc[-1] / bench_df.Close.iloc[-1 - n] - 1
    return round((a - b) * 100, 2)

# ---------------- 状态机 ----------------
def diagnose(d, item, has_ohlc=True):
    last = d.iloc[-1]
    res = {"notes": [], "flags": []}
    dd_trig, vol_sc = item["dd_trigger"], item["vol_sc"]

    # 派发/危险标记
    if has_ohlc and last.vol_ratio >= 1.5 and last.upper_shadow > 0.5:
        res["flags"].append("放量长上影>50% → 派发信号，不追")
    if last.vol_ratio >= 1.5 and last.Close < last.ema20 and d.Close.iloc[-2] >= d.ema20.iloc[-2]:
        res["flags"].append("放量跌破EMA20 → 趋势破坏")
    ext = last.Close / last.ema5 - 1
    if ext > 0.15:
        res["flags"].append(f"偏离EMA5 {ext*100:.1f}% → 超买不追")

    # 寻找最近30根内的 SC 候选
    look = d.iloc[-30:]
    sc_idx = None
    for i in range(len(look) - 1, -1, -1):
        r = look.iloc[i]
        cond = (r.vol_ratio >= vol_sc and (r.High - r.Low) >= 1.5 * r.atr20
                and r.Low <= r.lo60 * 1.03)
        if has_ohlc: cond = cond and r.close_pos >= 0.35
        if cond:
            sc_idx = look.index[i]; break

    state = "S0"
    if last.dd60 <= -dd_trig:
        state = "S1"
        res["notes"].append(f"距60日高点回撤 {last.dd60*100:.1f}% ≥ 触发线 {dd_trig*100:.0f}%")

    sc = None
    if sc_idx is not None:
        sc = d.loc[sc_idx]
        after = d.loc[sc_idx:].iloc[1:]
        state = "S2"
        res["notes"].append(f"SC（抛售高潮）候选 {sc_idx.date()}：量比 {sc.vol_ratio:.1f}x，收盘位于振幅 {sc.close_pos*100:.0f}% 处")
        # 破位检查
        broke = after[(after.Close < sc.Low) & (after.vol_ratio >= 1.5)]
        if len(broke):
            state = "S9"
            res["notes"].append(f"放量跌破 SC 低点 {sc.Low:.2f} → 供应未出清")
        else:
            # S3 测试：后续存在缩量回踩（近3根均量比<1）且未收破SC低点
            if len(after) >= 3 and after.vol_ratio.iloc[-3:].mean() < 1.0 and after.Close.min() >= sc.Low:
                state = "S3"
                res["notes"].append("ST（二次测试）：回踩缩量、未破SC低点 → 供应衰竭迹象")
            # Spring：曾跌破SC低点但当日收回且未放量破位
            spring = after[(after.Low < sc.Low) & (after.Close > sc.Low)]
            if len(spring):
                res["notes"].append(f"Spring（弹簧效应）{spring.index[-1].date()}：假跌破收回")
            # S4 确认：收复 SC 高点 + EMA20，放量且上影不长
            us_ok = (last.upper_shadow < 0.3) if has_ohlc else True
            if last.Close > sc.High and last.Close > last.ema20 and last.vol_ratio >= 1.5 and us_ok:
                state = "S4"
                res["notes"].append("SOS（强势信号）：放量收复 SC 高点与 EMA20")
            elif len(after) and (after.Close > after.ema20).any() and last.Close > last.ema20 and last.vol_ratio < 1.0 \
                    and last.Low >= min(last.ema20, sc.High) * 0.99 and state != "S9":
                state = "S4"
                res["notes"].append("LPS（最后支撑点）：突破后缩量回踩守住")
    # S5 趋势
    if last.ema5 > last.ema10 > last.ema20 and last.Close > last.ema5 and ext <= 0.15 and state in ("S0", "S4"):
        if state == "S4" or last.dd60 > -0.05:
            state = "S5" if state == "S4" else state
    if state == "S0" and last.ema5 > last.ema10 > last.ema20 and last.Close > last.ema5:
        res["notes"].append("EMA5/10/20 多头排列，趋势健康（非恐慌场景）")

    # 交易参数（仅 S4）
    trade = None
    if state == "S4" and sc is not None:
        stop = round(sc.Low * 0.995, 2)
        risk = (last.Close - stop) / last.Close
        trade = {"entry_ref": round(last.Close, 2), "stop": stop, "risk_pct": round(risk * 100, 2),
                 "invalid": f"日收盘 < {stop} 或 放量跌破 EMA20",
                 "size_hint": "单笔风险 ≤ 组合 0.5%；次新/山寨再打 5 折"}
    res.update({"state": state, "sc_date": str(sc_idx.date()) if sc_idx is not None else None,
                "sc_low": round(float(sc.Low), 2) if sc is not None else None,
                "sc_high": round(float(sc.High), 2) if sc is not None else None, "trade": trade})
    return res

# ---------------- 主流程 ----------------
def run():
    prev_state = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            prev_state = json.load(f).get("states", {})

    is_crypto = lambda t: t["type"].startswith("crypto")
    items = [t for t in CFG["watchlist"] if MODE == "all" or (MODE == "crypto") == is_crypto(t)]

    cache = {}
    def get(ticker):
        if ticker in cache: return cache[ticker]
        df, src = None, None
        if ticker in HL_MAP:                       # HYPE：先走 Hyperliquid 原生源
            df, src = fetch_hyperliquid(HL_MAP[ticker])
        if df is None:
            df, src = fetch_yf(ticker)
        if df is None and ticker in CG_MAP:
            df, src = fetch_coingecko(CG_MAP[ticker])
        if ticker in CRYPTO_TICKERS:               # 所有加密源统一丢弃未收盘的当日 K
            df = drop_open_candle(df)
        cache[ticker] = (df, src); return cache[ticker]

    # 宏观
    macro = {}
    for m in CFG["macro"]:
        df, src = get(m["ticker"])
        if df is not None:
            d = enrich(df); l = d.iloc[-1]
            macro[m["name"]] = {"value": round(float(l.Close), 2), "chg5d": round((l.Close / d.Close.iloc[-6] - 1) * 100, 2),
                                "above_ema20": bool(l.Close > l.ema20), "date": str(d.index[-1].date())}
        else:
            macro[m["name"]] = {"missing": True}
    fng = fetch_fng() if MODE != "us" else None
    funding = fetch_hl_funding() if MODE != "us" else {}

    rows, missing = [], []
    for it in items:
        df, src = get(it["ticker"])
        if df is None:
            missing.append(it["ticker"]); rows.append({"ticker": it["ticker"], "name": it["name"], "state": "NA"}); continue
        d = enrich(df); l = d.iloc[-1]
        has_ohlc = "无OHLC" not in (src or "")
        bench_df = get(it["bench"])[0] if it.get("bench") else None
        diag = diagnose(d, it, has_ohlc)
        row = {
            "ticker": it["ticker"], "name": it["name"], "type": it["type"], "cluster": it["cluster"],
            "date": str(d.index[-1].date()), "source": src,
            "close": round(float(l.Close), 2), "chg1d": round((l.Close / d.Close.iloc[-2] - 1) * 100, 2),
            "vol_ratio": round(float(l.vol_ratio), 2), "upper_shadow": round(float(l.upper_shadow) * 100, 0) if has_ohlc else None,
            "dd60": round(float(l.dd60) * 100, 1), "ext_ema5": round((l.Close / l.ema5 - 1) * 100, 1),
            "ema_bull": bool(l.ema5 > l.ema10 > l.ema20), "above_ema20": bool(l.Close > l.ema20),
            "rs5": rel_strength(d, bench_df, 5), "rs20": rel_strength(d, bench_df, 20),
            "bars": int(len(d)), "failure_conditions": CFG["failure_conditions"].get(it["ticker"], []),
            **diag,
        }
        # 结构性提示只在“有事发生”(非 S0/S5) 时才进 flags，避免每天占据④板块
        if row["state"] not in ("S0", "S5"):
            if it["type"] == "newlisting" and len(d) < 120:
                row["flags"].append(f"次新股：仅 {len(d)} 根日线，均量基准不稳，阈值未经检验")
            if it["type"] == "crypto_alt":
                row["flags"].append("山寨币：禁止情绪抄底，仅 S4 后半仓")
        row["prev_state"] = prev_state.get(it["ticker"], "S0")
        row["changed"] = row["prev_state"] != row["state"]
        rows.append(row)

    # 保存状态
    states = {**prev_state, **{r["ticker"]: r["state"] for r in rows if r["state"] != "NA"}}
    payload = {"run_utc": NOW.isoformat(), "run_bjt": BJT.strftime("%Y-%m-%d %H:%M"), "mode": MODE,
               "macro": macro, "fng": fng, "funding": funding, "rows": rows, "missing": missing, "states": states}
    with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(DOCS, f"latest_{MODE}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)

    build_html(payload)
    msg = build_telegram(payload)
    changed = any(r.get("changed") for r in rows)
    has_s4 = any(r["state"] in ("S4",) for r in rows)
    if CFG["notify"].get("always_send_brief", True) or changed or has_s4:
        send_telegram(msg)
    print(msg)

# ---------------- 输出 ----------------
def macro_summary(p):
    m, fng = p["macro"], p.get("fng")
    parts = []
    vix = m.get("VIX") or {}
    if vix.get("value") is not None:
        lvl = "恐慌" if vix["value"] >= 30 else "警戒" if vix["value"] >= 22 else "平静"
        parts.append(f"VIX {vix['value']}（{lvl}）")
    if fng and fng.get("value") is not None:
        chg = f"（7日 {fng['chg7d']:+d}）" if fng.get("chg7d") is not None else ""
        parts.append(f"加密恐贪 {fng['value']} {fng.get('label','')}{chg}")
    for k, v in m.items():
        if k == "VIX" or v.get("value") is None:
            continue
        parts.append(f"{k} {v['value']} ({v['chg5d']:+.1f}%/5d)")
    miss = [k for k, v in m.items() if v.get("missing")]
    if miss:
        parts.append("缺失：" + ",".join(miss))
    return " | ".join(parts) if parts else "宏观数据缺失"

def risk_regime(p):
    m, fng = p["macro"], p.get("fng")
    score = 0
    vix = (m.get("VIX") or {}).get("value")
    if vix is not None:
        score += 2 if vix >= 30 else 1 if vix >= 22 else 0
    if fng and fng.get("value") is not None:
        score += 2 if fng["value"] <= 20 else 1 if fng["value"] <= 30 else 0
    smh = m.get("SMH") or m.get("半导体ETF") or {}
    if smh.get("value") is not None and not smh.get("above_ema20", True):
        score += 1
    return "风险偏好恶化 → 个体信号胜率下降，降仓或等确认" if score >= 3 else "风险偏好中性" if score >= 1 else "风险偏好正常"

# ---------------- 规则化解读（全部由阈值生成，不含主观判断） ----------------
def macro_read(p):
    """把五个宏观指标放在一起读：利率端 / 美元 / 半导体 / 波动率与情绪 → 一句结论"""
    m, fng = p["macro"], p.get("fng")
    vix = (m.get("VIX") or {}).get("value")
    tnx = m.get("US10Y") or {}
    dxy = m.get("DXY") or {}
    smh = m.get("SMH") or {}
    out = []
    # 利率
    if tnx.get("value") is not None:
        v, c = tnx["value"], tnx["chg5d"]
        lvl = "异常高位" if v >= 4.75 else "偏高" if v >= 4.25 else "正常区间"
        spd = "且5日快速上行" if c >= 3 else "且5日快速回落" if c <= -3 else "5日变化不大"
        out.append(f"利率：10Y {v}% {lvl}，{spd}")
    # 美元 与 利率同向？
    if dxy.get("value") is not None and tnx.get("value") is not None:
        same = (dxy["chg5d"] > 0) == (tnx["chg5d"] > 0)
        if abs(dxy["chg5d"]) < 0.5: out.append("美元：横盘，无增量信息")
        elif same and dxy["chg5d"] > 0: out.append("美元：随利率走强，属利率驱动（非避险驱动）")
        elif dxy["chg5d"] > 0: out.append("美元：利率未升而美元走强 → 避险买盘，警惕")
        else: out.append("美元：走弱，对加密与成长股偏友好")
    # 半导体
    if smh.get("value") is not None:
        c, ab = smh["chg5d"], smh.get("above_ema20", True)
        if c <= -5: out.append(f"半导体：5日 {c:+.1f}% 领跌{'，且已跌破EMA20' if not ab else '，仍在EMA20上方'} → 引擎在熄火")
        elif c <= -2: out.append(f"半导体：5日 {c:+.1f}% 偏弱{'，跌破EMA20' if not ab else ''}")
        elif c >= 3: out.append(f"半导体：5日 {c:+.1f}% 领涨，风险偏好在扩张")
        else: out.append(f"半导体：5日 {c:+.1f}% 平稳")
    # 波动率 + 情绪
    if vix is not None:
        if vix >= 30: out.append(f"波动率：VIX {vix} 恐慌区，SC（抛售高潮）可能正在发生，开始逐日盯量")
        elif vix >= 22: out.append(f"波动率：VIX {vix} 警戒区，保护需求上升")
        else: out.append(f"波动率：VIX {vix} 平静，期权市场未定价风险 → 不会有 SC")
    if fng and fng.get("value") is not None:
        v, c = fng["value"], fng.get("chg7d")
        trend = "" if c is None else "，7日转差" if c <= -10 else "，7日转好" if c >= 10 else ""
        zone = "极度恐惧（加密 SC 常见区）" if v <= 20 else "恐惧" if v <= 30 else "贪婪" if v >= 70 else "中性"
        out.append(f"加密情绪：{v} {zone}{trend}")
    # 综合结论
    stress = 0
    if tnx.get("value") is not None and tnx["value"] >= 4.75: stress += 1
    if tnx.get("chg5d") is not None and tnx["chg5d"] >= 3: stress += 1
    if smh.get("chg5d") is not None and smh["chg5d"] <= -5: stress += 1
    if smh.get("value") is not None and not smh.get("above_ema20", True): stress += 1
    calm = (vix is not None and vix < 22) and (not fng or fng.get("value", 50) > 30)
    if stress >= 2 and calm:
        concl = "引线已点、炸药未响：利率/半导体端有压力，但 VIX 与情绪尚未反应。不做任何事，每天盯 10Y 与 SMH。"
    elif stress >= 2 and not calm:
        concl = "压力已传导到波动率/情绪端 → 进入观察 SC 的窗口，逐日看量，仍不抄底。"
    elif stress <= 1 and calm:
        concl = "环境平静，无便宜筹码。趋势健康的标的不追，等待。"
    else:
        concl = "波动率/情绪先于基本面走坏，多为短期扰动；看是否演化为 SC，勿预判。"
    return "；".join(out) + "。\n结论：" + concl

def row_read(r):
    """每个标的一句话：状态 + 该做什么，全部按规则生成"""
    s, vr, rs = r["state"], r["vol_ratio"], r.get("rs20")
    if s == "NA": return "数据缺失，不判定"
    if s == "S0":
        if r["ema_bull"]: t = "多头排列，趋势健康；此处没有便宜筹码，不追"
        elif not r["above_ema20"]: t = f"已跌破 EMA20 但回撤 {r['dd60']}% 未到压力线，观望"
        else: t = "震荡区，无信号"
    elif s == "S1":
        t = f"回撤 {r['dd60']}% 触线 → 只做功课：逐条核对失效条件；"
        t += "量比" + (f"{vr}x 已见放量，盯是否形成 SC（抛售高潮）" if vr >= 1.5 else f"{vr}x 未见恐慌量，SC 未出现，不动手")
    elif s == "S2":
        t = f"SC 候选已出现（{r.get('sc_date')}），等缩量回踩 ST（二次测试）；此处禁止买入"
    elif s == "S3":
        t = "回踩缩量、未破 SC 低点 → 供应衰竭迹象；等放量收复 SC 高点（SOS）才算确认"
    elif s == "S4":
        tr = r.get("trade") or {}
        t = f"量价确认 → 可按风险预算试探，止损 {tr.get('stop')}（风险 {tr.get('risk_pct')}%）；先核失效条件"
    elif s == "S5":
        t = "确认后趋势延续，持有不加仓"
    elif s == "S9":
        t = "放量跌破 SC 低点 → 供应未出清，持仓止损，计数归零"
    else: t = ""
    if rs is not None and s in ("S0", "S1", "S5"):
        if rs >= 5: t += f"；RS20 {rs:+.1f}% 明显强于基准"
        elif rs <= -5: t += f"；RS20 {rs:+.1f}% 明显弱于基准，若有 SC 优先级靠后"
    if r.get("flags"): t += "；⚠ " + r["flags"][0]
    return t

def funding_read(p):
    """资金费率解读：年化 + 与价格方向的组合含义"""
    fund = p.get("funding") or {}
    if not fund: return None
    px = {"BTC": "BTC-USD", "HYPE": "HYPE32196-USD"}
    chg = {r["ticker"]: r.get("chg1d", 0) for r in p["rows"] if r["state"] != "NA"}
    L = []
    for k, v in fund.items():
        f8 = v["funding_8h"] * 100; apr = f8 * 3 * 365
        c1 = chg.get(px.get(k), 0) or 0
        if f8 >= 0.03: lvl = "多头极度拥挤"
        elif f8 >= 0.01: lvl = "多头偏拥挤"
        elif f8 <= -0.01: lvl = "空头付费（空头拥挤）"
        elif f8 < 0: lvl = "略偏空"
        else: lvl = "中性"
        if f8 >= 0.01 and c1 < -2: mean = "价跌而多头仍付费加杠杆 → 多头被套未认输，下跌未完成"
        elif f8 <= -0.01 and c1 < -2: mean = "价跌且空头付费 → 空头拥挤，若出现 SC 易有轧空反弹"
        elif f8 <= -0.01 and c1 > 0: mean = "价涨空头付费 → 轧空进行中，不追"
        elif f8 >= 0.03: mean = "杠杆过热，随时可能清算式回落"
        else: mean = "杠杆不拥挤，费率无增量信息"
        L.append(f"{k} {f8:.4f}%/8h（年化约 {apr:.1f}%）{lvl} → {mean}")
    return L

def build_telegram(p):
    rows = [r for r in p["rows"] if r["state"] != "NA"]
    L = [f"<b>带血筹码雷达 · {p['mode'].upper()}</b>  {p['run_bjt']} 北京时间", macro_summary(p), f"<i>{risk_regime(p)}</i>",
         f"<b>宏观解读：</b>{macro_read(p)}", ""]
    ch = [r for r in rows if r["changed"]]
    L.append("<b>① 发生了什么变化</b>")
    L += [f"• {r['ticker']} {STATE_NAME[r['prev_state']]} → <b>{STATE_NAME[r['state']]}</b>" for r in ch] or ["• 无状态变化"]
    s4 = [r for r in rows if r["state"] == "S4"]
    L.append("\n<b>② 量价确认候选（S4）</b>")
    if s4:
        for r in s4:
            t = r["trade"]
            L.append(f"• <b>{r['ticker']}</b> {r['close']} 量比{r['vol_ratio']}x | 止损 {t['stop']} (风险 {t['risk_pct']}%) | {t['invalid']}")
            L.append(f"  失效条款自查：{' / '.join(r['failure_conditions'][:2])}")
    else: L.append("• 无。恐慌≠买点，继续等。")
    research = [r for r in rows if r["state"] in ("S1", "S2", "S3")]
    L.append("\n<b>③ 值得进一步研究</b>")
    if research:
        for r in research:
            rs = f" RS20 {r['rs20']:+.1f}%" if r["rs20"] is not None else ""
            L.append(f"• {r['ticker']} [{STATE_NAME[r['state']]}] {r['close']} 回撤{r['dd60']}% 量比{r['vol_ratio']}x{rs}")
            for n in r["notes"][:2]: L.append(f"  – {n}")
            L.append(f"  → {row_read(r)}")
    else: L.append("• 无")
    wait = [r for r in rows if r["state"] in ("S9",) or r["flags"]]
    L.append("\n<b>④ 应该继续等 / 警示</b>")
    if wait:
        for r in wait:
            L.append(f"• {r['ticker']} [{STATE_NAME[r['state']]}] " + "；".join(r["flags"][:2] or r["notes"][:1]))
    else: L.append("• 无")
    L.append("\n<b>⑤ 各标的一句话</b>")
    for r in rows:
        if r["state"] in ("S1", "S2", "S3", "S4", "S9"): continue   # 上面板块已详述
        L.append(f"• {r['ticker']} [{STATE_NAME[r['state']]}] {row_read(r)}")
    if p["missing"]: L.append(f"\n⚠ 数据缺失：{', '.join(p['missing'])}")
    fr = funding_read(p)
    if fr:
        L.append("\n<b>资金费率</b>")
        L += [f"• {x}" for x in fr]
    L.append("\n<i>阈值未经历史检验，仅作观察提醒。买入需另符合风险预算。</i>")
    return "\n".join(L)

def send_telegram(text):
    # .strip() 防止 Secret 粘贴时带入换行（曾导致 URL 被截断、静默失败）
    tok = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not tok or not chat:
        print(f"[TG] 未读到凭据 token={'有' if tok else '空'} chat_id={'有' if chat else '空'}"); return
    for i in range(0, len(text), 3900):
        try:
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                              json={"chat_id": chat, "text": text[i:i+3900], "parse_mode": "HTML",
                                    "disable_web_page_preview": True}, timeout=20)
            print(f"[TG] status={r.status_code} resp={r.text[:200]}")
        except Exception as e:
            print(f"[TG] 发送异常: {e}")

def build_html(p):
    color = {"S0": "bg-gray-100", "S1": "bg-yellow-100", "S2": "bg-orange-100", "S3": "bg-amber-100",
             "S4": "bg-green-200 font-bold", "S5": "bg-emerald-100", "S9": "bg-red-200", "NA": "bg-gray-300"}
    tr = ""
    for r in p["rows"]:
        if r["state"] == "NA":
            tr += f"<tr class='bg-gray-300'><td class='p-2'>{r['ticker']}</td><td colspan='12' class='p-2'>数据缺失</td></tr>"; continue
        us = f"{r['upper_shadow']:.0f}%" if r["upper_shadow"] is not None else "N/A"
        rs = f"{r['rs20']:+.1f}%" if r["rs20"] is not None else "—"
        notes = "<br>".join(r["notes"] + [f"<span class='text-red-700'>{x}</span>" for x in r["flags"]])
        trade = f"止损 {r['trade']['stop']} / 风险 {r['trade']['risk_pct']}%" if r["trade"] else ""
        fc = "<br>".join(f"☐ {x}" for x in r["failure_conditions"])
        tr += f"""<tr class='{color[r['state']]} border-b align-top'>
<td class='p-2'><b>{r['ticker']}</b><br><span class='text-xs text-gray-500'>{r['name']}</span></td>
<td class='p-2'>{STATE_NAME[r['state']]}{' ⬆' if r['changed'] else ''}</td>
<td class='p-2 text-right'>{r['close']}<br><span class='text-xs'>{r['chg1d']:+.2f}%</span></td>
<td class='p-2 text-right'>{r['vol_ratio']}x</td><td class='p-2 text-right'>{us}</td>
<td class='p-2 text-right'>{r['dd60']}%</td><td class='p-2 text-right'>{rs}</td>
<td class='p-2 text-center'>{'多头' if r['ema_bull'] else '—'}<br><span class='text-xs'>{'>EMA20' if r['above_ema20'] else '<EMA20'}</span></td>
<td class='p-2 text-xs'>{notes}</td>
<td class='p-2 text-xs text-slate-800'>{row_read(r)}</td>
<td class='p-2 text-xs'>{trade}</td>
<td class='p-2 text-xs text-gray-600'>{fc}</td>
<td class='p-2 text-xs text-gray-400'>{r['date']}<br>{r['source']}</td></tr>"""
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>带血筹码雷达</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-50 p-4 text-sm">
<h1 class="text-2xl font-bold">带血筹码雷达</h1>
<p class="text-gray-600">运行：{p['run_bjt']} 北京时间 · 模式 {p['mode']} · <a class="underline" href="state.json">JSON</a></p>
<div class="my-3 p-3 bg-white rounded shadow"><b>宏观：</b>{macro_summary(p)}<br><b>环境判定：</b>{risk_regime(p)}
<div class="mt-2 pt-2 border-t text-slate-700"><b>宏观解读：</b>{macro_read(p).replace(chr(10), '<br>')}</div>
{('<div class="mt-2 pt-2 border-t text-slate-700"><b>资金费率：</b><br>' + '<br>'.join(funding_read(p)) + '</div>') if funding_read(p) else ''}</div>
<div class="overflow-x-auto bg-white rounded shadow"><table class="min-w-full">
<thead class="bg-slate-800 text-white text-xs"><tr>
<th class="p-2 text-left">标的</th><th class="p-2 text-left">状态</th><th class="p-2">收盘/日涨跌</th><th class="p-2">量比</th>
<th class="p-2">上影</th><th class="p-2">60日回撤</th><th class="p-2">RS20</th><th class="p-2">均线</th>
<th class="p-2 text-left">量价证据</th><th class="p-2 text-left">解读</th><th class="p-2 text-left">交易参数(S4)</th><th class="p-2 text-left">预写失效条件</th><th class="p-2">数据</th>
</tr></thead><tbody>{tr}</tbody></table></div>
<div class="mt-4 text-xs text-gray-500">
<p>状态机：S0 常态 → S1 压力观察（只研究）→ S2 SC候选 → S3 测试中 → <b>S4 确认（才允许试探仓）</b> → S5 趋势；S9 放量破 SC 低点 = 供应未出清。</p>
<p>所有阈值未经历史检验，仅作观察提醒。恐慌触发研究，量价确认触发交易。</p>
{'<p class="text-red-600">数据缺失：' + ', '.join(p['missing']) + '</p>' if p['missing'] else ''}
</div></body></html>"""
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f: f.write(html)

if __name__ == "__main__":
    run()
