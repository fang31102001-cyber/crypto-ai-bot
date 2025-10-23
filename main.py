import os
from threading import Thread
from datetime import datetime
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

# ========= ENV =========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TIMEFRAME_DEFAULT = os.getenv("TIMEFRAME", "15m")

# ========= PARSE COIN + TIMEFRAME =========
VALID_TF = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"}

def parse_symbol_tf(text: str, default_tf: str) -> tuple[str, str]:
    parts = (text or "").strip().lower().split()
    if not parts:
        raise ValueError("Bạn chưa nhập coin.")
    base = parts[0].upper()
    tf = default_tf
    if len(parts) > 1 and parts[1].lower() in VALID_TF:
        tf = parts[1].lower()
    return base, tf

# ========= KẾT NỐI MEXC & CHỈ BÁO =========
import ccxt
import pandas as pd
import numpy as np

def mexc_client():
    return ccxt.mexc3({
        "options": {"defaultType": "swap"},
        "enableRateLimit": True,
    })

def normalize_symbol(base):
    return f"{base.upper()}/USDT:USDT"

def fetch_ohlcv(base, timeframe, limit=200):
    ex = mexc_client()
    symbol = normalize_symbol(base)
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100/(1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df, n=14):
    df["prev_close"] = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["prev_close"]).abs(),
        (df["low"] - df["prev_close"]).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def analyze_coin(base, timeframe):
    df = fetch_ohlcv(base, timeframe)
    df = df.dropna()
    
    df["ema12"] = ema(df["close"], 12)
    df["ema26"] = ema(df["close"], 26)
    df["rsi"] = rsi(df["close"])
    macd_line, signal_line, hist = macd(df["close"])
    df["macd_hist"] = hist
    df["atr"] = atr(df)

    last = df.iloc[-1]
    price = last["close"]
    atr_val = last["atr"]

    side = "LONG" if last["ema12"] > last["ema26"] and last["macd_hist"] > 0 else "SHORT"
    if side == "LONG":
        tp1 = price + atr_val * 1.5
        tp2 = price + atr_val * 2.5
        sl = price - atr_val * 1.0
    else:
        tp1 = price - atr_val * 1.5
        tp2 = price - atr_val * 2.5
        sl = price + atr_val * 1.0

    return {
        "side": side,
        "price": price,
        "ema12": last["ema12"],
        "ema26": last["ema26"],
        "rsi": last["rsi"],
        "macd_hist": last["macd_hist"],
        "atr": atr_val,
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl
    }

# ========= LỆNH BẮT ĐẦU =========
async def cmd_start(update, ctx):
    await update.message.reply_text(
        "🤖 Bot đã sẵn sàng phân tích kỹ thuật.\n"
        "- Gõ coin: `btc` hoặc `sol 15m` để xem phân tích.\n"
        "- Bao gồm EMA, RSI, MACD, ATR, TP/SL."
    )

# ========= XỬ LÝ TIN NHẮN =========
async def handle_text(update, ctx):
    try:
        text = (update.message.text or "").strip()
        base, tf = parse_symbol_tf(text, TIMEFRAME_DEFAULT)
        result = analyze_coin(base, tf)

        msg = (
            f"📊 *{base}/{tf.upper()}* — Xu hướng: *{result['side']}*\n"
            f"💵 Giá hiện tại: `{result['price']:.4f}`\n"
            f"🎯 TP1: `{result['tp1']:.4f}` | TP2: `{result['tp2']:.4f}`\n"
            f"⛔ SL: `{result['sl']:.4f}`\n"
            f"📈 EMA12/EMA26: `{result['ema12']:.4f}` / `{result['ema26']:.4f}`\n"
            f"📊 RSI: `{result['rsi']:.2f}` | MACD Hist: `{result['macd_hist']:.5f}`\n"
            f"📏 ATR: `{result['atr']:.5f}`\n"
            f"✅ (Bước tiếp theo: thêm dòng tiền + AI Score + tự động quét)"
        )

        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Lỗi: {e}")

# ========= MAIN =========
def main():
    from keep_alive import run_server
    Thread(target=run_server, daemon=True).start()

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN trong Environment của Render.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot đang chạy (polling)…")
    app.run_polling(allowed_updates=None)

if __name__ == "__main__":
    main()
