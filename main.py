import os, math, asyncio
from threading import Thread
from datetime import datetime, timezone
from typing import Tuple, Dict, Any

import ccxt
import pandas as pd
import numpy as np
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

# ========= ENV =========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TIMEFRAME_DEFAULT = os.getenv("TIMEFRAME", "15m")
TZ = os.getenv("TZ", "Asia/Ho_Chi_Minh")

VALID_TF = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"}

# ========= KEEP-ALIVE (Render/UptimeRobot) =========
def start_keep_alive():
    try:
        from keep_alive import run_server
        Thread(target=run_server, daemon=True).start()
    except Exception:
        # không có keep_alive.py cũng không sao
        pass

# ========= CCXT MEXC =========
def mexc_client():
    # Dùng mexc (không phải mexc3) + defaultType=swap để trade Futures USDT
    return ccxt.mexc({
        "options": {"defaultType": "swap"},
        "enableRateLimit": True,
    })

def symbol_usdt_perp(base: str) -> str:
    # Chuẩn hoá sang format FUTURES USDT: BASE/USDT:USDT
    return f"{base.upper()}/USDT:USDT"

# ========= CHỈ BÁO NHẸ =========
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    down = (-d).clip(lower=0.0)
    rs = up.rolling(n).mean() / (down.rolling(n).mean() + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    f = ema(close, fast)
    s = ema(close, slow)
    m = f - s
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ========= LẤY DỮ LIỆU + PHÂN TÍCH =========
def fetch_ohlcv(base: str, tf: str, limit: int = 300) -> pd.DataFrame:
    ex = mexc_client()
    sym = symbol_usdt_perp(base)
    ohlcv = ex.fetch_ohlcv(sym, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema12"] = ema(d["close"], 12)
    d["ema26"] = ema(d["close"], 26)
    m, s, h = macd(d["close"])
    d["macd"], d["macd_sig"], d["macd_hist"] = m, s, h
    d["rsi"] = rsi(d["close"])
    d["atr"] = atr(d)
    v = d["volume"]
    d["vol_z"] = (v - v.rolling(50).mean()) / (v.rolling(50).std() + 1e-9)
    return d.dropna()

def simple_ai_score(row: Dict[str, Any]) -> Tuple[int, str]:
    """Điểm ‘AI’ đơn giản dựa trên trend EMA + MACD + RSI + Volume."""
    score = 50
    reasons = []
    # Xu hướng
    if row["ema12"] > row["ema26"]:
        score += 15; reasons.append("EMA12>EMA26 (uptrend)")
    else:
        score -= 15; reasons.append("EMA12<EMA26 (downtrend)")
    # MACD histogram
    if row["macd_hist"] > 0:
        score += 15; reasons.append("MACD_hist > 0 (bullish momentum)")
    else:
        score -= 15; reasons.append("MACD_hist < 0 (bearish momentum)")
    # RSI
    if row["rsi"] > 55:
        score += 10; reasons.append("RSI>55 (mạnh)")
    elif row["rsi"] < 45:
        score -= 10; reasons.append("RSI<45 (yếu)")
    # Volume bất thường
    if row["vol_z"] > 1.0:
        score += 10; reasons.append("Volume tăng bất thường")
    elif row["vol_z"] < -1.0:
        reasons.append("Volume giảm")
    score = max(0, min(100, score))
    return score, "; ".join(reasons)

def make_targets(entry: float, atrv: float, side: str) -> Tuple[float,float,float]:
    # TP/SL đơn giản dựa ATR
    tp1 = entry + (1.5 * atrv if side == "LONG" else -1.5 * atrv)
    tp2 = entry + (2.5 * atrv if side == "LONG" else -2.5 * atrv)
    sl  = entry - (1.0 * atrv if side == "LONG" else -1.0 * atrv)
    return tp1, tp2, sl

def analyze(base: str, tf: str) -> Dict[str, Any]:
    df = fetch_ohlcv(base, tf, limit=350)
    d = enrich(df)
    row = d.iloc[-1].to_dict()

    # Phán side cơ bản
    side = "LONG" if (row["ema12"] > row["ema26"] and row["macd_hist"] > 0 and row["rsi"] >= 48) else "SHORT"

    score, why = simple_ai_score(row)
    tp1, tp2, sl = make_targets(row["close"], row["atr"], side)

    # funding rate (MEXC có thể không expose cho ccxt ở mọi symbol—cố gắng lấy, lỗi thì bỏ qua)
    funding_txt = "-"
    try:
        ex = mexc_client()
        fr = ex.fetch_funding_rate(symbol_usdt_perp(base))
        if "fundingRate" in fr and fr["fundingRate"] is not None:
            funding_txt = f"{float(fr['fundingRate']):.5f}"
    except Exception:
        pass

    return {
        "base": base.upper(),
        "tf": tf,
        "side": side,
        "price": float(row["close"]),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "sl": float(sl),
        "ema12": float(row["ema12"]),
        "ema26": float(row["ema26"]),
        "rsi": float(row["rsi"]),
        "macd_hist": float(row["macd_hist"]),
        "vol_z": float(row["vol_z"]),
        "atr": float(row["atr"]),
        "score": int(score),
        "why": why,
        "funding": funding_txt,
    }

# ========= PARSE INPUT =========
def parse_symbol_tf(text: str, default_tf: str) -> Tuple[str, str]:
    parts = (text or "").strip().lower().split()
    if not parts:
        raise ValueError("Bạn chưa nhập coin.")
    base = parts[0].upper()
    tf = default_tf
    if len(parts) > 1 and parts[1].lower() in VALID_TF:
        tf = parts[1].lower()
    return base, tf

def fmt(x, nd=6):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

# ========= TELEGRAM =========
async def cmd_start(update, ctx):
    await update.message.reply_text(
        "🤖 Bot đã sẵn sàng.\n"
        "• Gõ coin: `btc` hoặc `op 15m`\n"
        "• Bot sẽ phân tích: Hướng, Entry/TP/SL, EMA/RSI/MACD, Volume, Funding (nếu có), AI Score.\n"
        "• Khung thời gian hỗ trợ: 1m/3m/5m/15m/30m/1h/2h/4h/6h/12h/1d."
    )

async def handle_text(update, ctx):
    text = (update.message.text or "").strip()
    try:
        base, tf = parse_symbol_tf(text, TIMEFRAME_DEFAULT)
        r = analyze(base, tf)

        msg = (
            f"📊 {r['base']} — TF: {r['tf']}\n"
            f"➡️ Xu hướng: {r['side']}\n"
            f"💰 Entry: {fmt(r['price'])} USDT\n"
            f"🎯 TP1: {fmt(r['tp1'])} | TP2: {fmt(r['tp2'])}\n"
            f"🛡 SL: {fmt(r['sl'])}\n"
            f"📈 EMA12/26: {fmt(r['ema12'])}/{fmt(r['ema26'])}\n"
            f"📉 RSI: {fmt(r['rsi'],2)} | MACD_hist: {fmt(r['macd_hist'],5)}\n"
            f"📦 VolZ: {fmt(r['vol_z'],2)} | ATR: {fmt(r['atr'],5)}\n"
            f"🏷 Funding: {r['funding']}\n"
            f"🤖 AI Score: {r['score']}%\n"
            f"🔎 Lý do: {r['why']}"
        )
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Lỗi: {e}")

# ========= MAIN =========
def main():
    start_keep_alive()

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN trong Environment của Render.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot đang chạy (polling)…")
    app.run_polling(allowed_updates=None)

if __name__ == "__main__":
    main()
