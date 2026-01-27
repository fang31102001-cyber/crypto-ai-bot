# main.py
import os, math, json, time, random, logging
from threading import Thread
from datetime import datetime, timezone, timedelta
from typing import Tuple

import ccxt
import pandas as pd
import numpy as np
import pytz
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.error import Conflict

# ================== LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("bot")

# ================== ENV & DEFAULTS ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TZ                 = os.getenv("TZ", "Asia/Ho_Chi_Minh")
TIMEFRAME_DEFAULT  = os.getenv("TIMEFRAME", "15m")

AUTO_SCAN          = os.getenv("AUTO_SCAN", "true").lower() == "true"
SCAN_INTERVAL_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "3600"))
ALERT_THRESHOLD    = int(os.getenv("ALERT_THRESHOLD", "80"))
MIN_VOLZ           = float(os.getenv("MIN_VOLZ", "2"))

MARKET_TYPE        = os.getenv("MARKET_TYPE", "swap")
QUOTE              = os.getenv("QUOTE", "USDT").upper()

LABEL_TP_PCT       = float(os.getenv("LABEL_TP_PCT", "0.004"))
LABEL_SL_PCT       = float(os.getenv("LABEL_SL_PCT", "0.004"))

CHAT_ID            = int(os.getenv("CHAT_ID", "5335165612"))

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TRADES_PATH  = os.path.join(DATA_DIR, "trades.json")
PENDING_PATH = os.path.join(DATA_DIR, "pending.json")

if not os.path.exists(TRADES_PATH):
    with open(TRADES_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

if not os.path.exists(PENDING_PATH):
    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f)


# ================== KEEP ALIVE ==================
def start_keep_alive():
    try:
        from keep_alive import run_server
        Thread(target=run_server, daemon=True).start()
        print("✅ keep_alive started", flush=True)
    except Exception as e:
        print("⚠️ keep_alive failed:", repr(e), flush=True)

# ================== Helpers ==================
VALID_TF = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"}

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
def cooldown_ok(symbol: str, side: str) -> bool:
    key = f"{symbol}_{side}"
    now = time.time()
    last = LAST_SIGNAL_TIME.get(key, 0)
    return (now - last) > COOLDOWN_MINUTES * 60

        
def in_trading_hours(
    start_hour: int = 8,
    end_hour: int = 22,
    tz_name: str = "Asia/Ho_Chi_Minh"
) -> bool:
    """
    Kiểm tra có đang trong giờ trade hay không
    Mặc định: 08:00 -> 22:00 giờ VN
    """
    try:
        import pytz
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
    except Exception:
        now = datetime.now()

    hour = now.hour
    return start_hour <= hour < end_hour


# ================== CCXT (MEXC Swap) ==================
EX = ccxt.mexc({
    "options": {"defaultType": MARKET_TYPE},
    "enableRateLimit": True,
})

def symbol_usdt_perp(base: str) -> str:
    return f"{base.upper()}/{QUOTE}:{QUOTE}"

# ================== Indicators ==================
def ema(s, n): 
    return s.ewm(span=n, adjust=False).mean()
    
def get_htf_trend(base: str) -> str:
    """
    Xác định xu hướng khung 1H để lọc trade 15m
    UP    -> chỉ được LONG
    DOWN  -> chỉ được SHORT
    SIDE  -> bỏ qua
    """
    df_htf = fetch_ohlcv(base, "1h", limit=300)

    # EMA50 & EMA200 chuẩn
    df_htf["ema50"] = ema(df_htf["close"], 50)
    df_htf["ema200"] = ema(df_htf["close"], 200)

    df_htf = df_htf.dropna()
    if len(df_htf) < 5:
        return "SIDE"

    ema50 = df_htf["ema50"].iloc[-1]
    ema200 = df_htf["ema200"].iloc[-1]

    if ema50 > ema200:
        return "UP"
    elif ema50 < ema200:
        return "DOWN"
    else:
        return "SIDE"

def rsi(close, n=14):
    d = close.diff()
    up, down = d.clip(lower=0), (-d).clip(lower=0)
    rs = up.rolling(n).mean() / (down.rolling(n).mean() + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    f, s = ema(close, fast), ema(close, slow)
    m = f - s
    sig = ema(m, signal)
    h = m - sig
    return m, sig, h

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ================== Market Structure ==================
def detect_bos(df: pd.DataFrame, lookback: int = 20) -> str:
    high = df["high"]
    low = df["low"]

    swing_high = high.rolling(lookback).max()
    swing_low = low.rolling(lookback).min()

    last_close = df["close"].iloc[-1]

    if last_close > swing_high.iloc[-2]:
        return "BOS_UP"
    if last_close < swing_low.iloc[-2]:
        return "BOS_DOWN"
    return "NO_BOS"

def break_retest_ok(df: pd.DataFrame, side: str, lookback: int = 20, tol: float = 0.002) -> bool:
    high = df["high"]
    low = df["low"]

    swing_high = high.rolling(lookback).max()
    swing_low = low.rolling(lookback).min()

    last_close = df["close"].iloc[-1]

    if side == "LONG":
        level = swing_high.iloc[-2]
        return abs(last_close - level) / level <= tol

    if side == "SHORT":
        level = swing_low.iloc[-2]
        return abs(last_close - level) / level <= tol

    return False

def detect_strong_bos(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    BOS mạnh cho futures:
    - Close phá swing
    - Body nến lớn (>=60%)
    - Volume tăng
    """
    if len(df) < lookback + 2:
        return "NO_BOS"

    high = df["high"]
    low = df["low"]

    swing_high = high.rolling(lookback).max()
    swing_low = low.rolling(lookback).min()

    last = df.iloc[-1]

    o = last["open"]
    h = last["high"]
    l = last["low"]
    c = last["close"]

    body = abs(c - o)
    range_ = h - l
    if range_ == 0:
        return "NO_BOS"

    body_ratio = body / range_

    vol = last["volume"]
    vol_ma = df["volume"].rolling(20).mean().iloc[-2]

    # BOS UP
    if (
        c > swing_high.iloc[-2] and
        body_ratio >= 0.6 and
        vol > vol_ma and
        c > o
    ):
        return "BOS_UP"

    # BOS DOWN
    if (
        c < swing_low.iloc[-2] and
        body_ratio >= 0.6 and
        vol > vol_ma and
        c < o
    ):
        return "BOS_DOWN"

    return "NO_BOS"

def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Phát hiện quét thanh khoản:
    SWEEP_UP   -> quét đỉnh
    SWEEP_DOWN -> quét đáy
    NO_SWEEP
    """

    if len(df) < lookback + 5:
        return "NO_SWEEP"

    high = df["high"]
    low = df["low"]

    swing_high = high.rolling(lookback).max()
    swing_low = low.rolling(lookback).min()

    prev_high = swing_high.iloc[-3]
    prev_low = swing_low.iloc[-3]

    last_high = high.iloc[-2]
    last_low = low.iloc[-2]
    last_close = df["close"].iloc[-1]

    # Sweep đáy rồi bật lên
    if last_low < prev_low and last_close > prev_low:
        return "SWEEP_DOWN"

    # Sweep đỉnh rồi rơi xuống
    if last_high > prev_high and last_close < prev_high:
        return "SWEEP_UP"

    return "NO_SWEEP"

def ema_pullback_ok(row: dict, side: str, tol: float = 0.003) -> bool:
    price = row["close"]
    ema50 = row["ema50"]

    if side == "LONG":
        return price >= ema50 and abs(price - ema50) / ema50 <= tol

    if side == "SHORT":
        return price <= ema50 and abs(price - ema50) / ema50 <= tol

    return False



# ================== Candlestick patterns ==================
def detect_pattern(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    op, cp = float(prev["open"]), float(prev["close"])

    rng = h - l
    if rng <= 0:
        return "-"

    body = abs(c - o)

    if body < rng * 0.3 and c > o:
        return "Hammer 🟢"
    if body < rng * 0.3 and c < o:
        return "Inverted Hammer 🔴"
    if c > op and o < cp and (c - o) > rng * 0.5:
        return "Bullish Engulfing 💚"
    if c < op and o > cp and (o - c) > rng * 0.5:
        return "Bearish Engulfing ❤️"
    if body < rng * 0.05:
        return "Doji ⚪"
    return "-"

# ================== Data ==================
def fetch_ohlcv(base: str, tf: str, limit: int = 300) -> pd.DataFrame:
    sym = symbol_usdt_perp(base)
    try:
        data = EX.fetch_ohlcv(sym, timeframe=tf, limit=limit)
    except Exception:
        sym2 = f"{base.upper()}/{QUOTE}"
        data = EX.fetch_ohlcv(sym2, timeframe=tf, limit=limit)

    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["ema12"] = ema(d["close"], 12)
    d["ema26"] = ema(d["close"], 26)
    d["ema50"] = ema(d["close"], 50)

    m, s, h = macd(d["close"])
    d["macd"], d["macd_sig"], d["macd_hist"] = m, s, h

    d["rsi"] = rsi(d["close"])
    d["atr"] = atr(d)

    v = d["volume"]
    d["vol_z"] = (v - v.rolling(50).mean()) / (v.rolling(50).std() + 1e-9)

    d["pattern"] = "-"
    if len(d) > 2:
        d.loc[d.index[-1], "pattern"] = detect_pattern(d)

    d = d.dropna()
    if len(d) < 5:
        raise ValueError("Không đủ dữ liệu để phân tích.")
    return d


# ================== Analysis ==================
def analyze(base: str, tf: str) -> dict:
    # ===== TIME FILTER (08h-22h VN) =====
    if not in_trading_hours(8, 22, TZ):
        return {"skip": True, "reason": "Out of trading hours (08-22)"}
        
    df = enrich(fetch_ohlcv(base, tf, limit=300))
    row = df.iloc[-1].to_dict()
    # ===== ATR FILTER =====
    atr_ratio = row["atr"] / row["close"]

    if atr_ratio < 0.002:
        return {"skip": True, "reason": "ATR too low"}
    
    #  HTF trend filter (1H)
    htf_trend = get_htf_trend(base)
    if htf_trend == "SIDE":
        return {"skip": True, "reason": "HTF sideways"}

    # ===== STRONG BOS (15m) =====
    bos = detect_bos(df)
    if bos == "NO_BOS":
        return {"skip": True, "reason": "Weak / fake BOS"}

    # Xác định hướng trade
    side = "LONG" if bos == "BOS_UP" else "SHORT"

    # ===== COOLDOWN =====
    if not cooldown_ok(base, side):
        return {"skip": True, "reason": "Cooldown"}

    # ===== BREAK & RETEST =====
    if not break_retest_ok(df, side):
        return {"skip": True, "reason": "No break & retest"}

    # ===== HTF DIRECTION LOCK =====
    if htf_trend == "UP" and side != "LONG":
        return {"skip": True, "reason": "HTF UP only LONG"}

    if htf_trend == "DOWN" and side != "SHORT":
        return {"skip": True, "reason": "HTF DOWN only SHORT"}

    # ===== VOLUME CONFIRM =====
    if abs(row["vol_z"]) < MIN_VOLZ:
        return {"skip": True, "reason": "Weak volume"}
    # ===== RSI FILTER =====
    rsi_val = row["rsi"]

    if side == "LONG" and rsi_val > 70:
        return {"skip": True, "reason": "RSI overbought"}

    if side == "SHORT" and rsi_val < 30:
        return {"skip": True, "reason": "RSI oversold"}
    # ===== EMA50 PULLBACK =====
    if not ema_pullback_ok(row, side):
        return {"skip": True, "reason": "No EMA50 pullback"}

    
    # 3. TP / SL theo ATR
    entry = float(row["close"])
    atrv = float(row["atr"])
    tp = entry + (2.0 * atrv if side == "LONG" else -2.0 * atrv)
    sl = entry - (1.0 * atrv if side == "LONG" else -1.0 * atrv)

    # 4. Lưu trade để theo dõi WIN / LOSE
    trade = {
        "time": datetime.utcnow().isoformat(),
        "base": base,
        "tf": tf,
        "side": side,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "status": "OPEN"
    }

    with open(TRADES_PATH, "r+", encoding="utf-8") as f:
        trades = json.load(f)
        trades.append(trade)
        f.seek(0)
        json.dump(trades, f, indent=2)
    LAST_SIGNAL_TIME[f"{base}_{side}"] = time.time()

    return {
        "base": base.upper(),
        "tf": tf,
        "side": side,
        "price": entry,
        "tp": tp,
        "sl": sl,
        "bos": bos
    }
def update_trades_and_learn():
    with open(TRADES_PATH, "r+", encoding="utf-8") as f:
        trades = json.load(f)

    changed = False

    for t in trades:
        if t["status"] != "OPEN":
            continue

        df = fetch_ohlcv(t["base"], t["tf"], limit=5)
        high = df["high"].iloc[-1]
        low = df["low"].iloc[-1]

        if t["side"] == "LONG":
            if high >= t["tp"]:
                t["status"] = "WIN"
                changed = True
            elif low <= t["sl"]:
                t["status"] = "LOSE"
                changed = True

        if t["side"] == "SHORT":
            if low <= t["tp"]:
                t["status"] = "WIN"
                changed = True
            elif high >= t["sl"]:
                t["status"] = "LOSE"
                changed = True

    if changed:
        with open(TRADES_PATH, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2)

# ================== Telegram ==================
async def cmd_start(update, ctx):
    await update.message.reply_text(
        "🤖 Bot AI Futures đã sẵn sàng!\n"
        "• Gõ coin: btc hoặc sol 15m\n"
        "• Bot tự động gửi tín hiệu mỗi 1h khi có sóng mạnh\n"
    )

async def handle_text(update, ctx):
    text = (update.message.text or "").strip()
    try:
        update_trades_and_learn()
        base, tf = parse_symbol_tf(text, TIMEFRAME_DEFAULT)
        r = analyze(base, tf)

        if r.get("skip"):
            await update.message.reply_text(f"⏭️ Bỏ qua: {r['reason']}")
            return

        msg = (
            f"📊 {r['base']} ({r['tf']})\n"
            f"Hướng: {r['side']} | BOS: {r['bos']}\n"
            f"Entry: {fmt(r['price'])}\n"
            f"TP: {fmt(r['tp'])} | SL: {fmt(r['sl'])}\n"
        )
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"⚠️ Lỗi phân tích: {e}")


# ================== Auto scan ==================
async def auto_scan(ctx):
    # ===== TIME FILTER (08h-22h VN) =====
    if not in_trading_hours(8, 22, TZ):
        return
    chat_id = int(ctx.job.data["chat_id"])
    update_trades_and_learn()
    top = ["BTC","ETH","SOL","WLD","XRP","TON","ARB","LINK","PEPE","SUI"]

    for coin in top:
        try:
            r = analyze(coin, TIMEFRAME_DEFAULT)

            if r.get("skip"):
                continue

            if not r.get("skip"):
                msg = (
                    f"🔥 Tín hiệu mạnh — {r['base']}/USDT ({r['tf']})\n"
                    f"Hướng: {r['side']} | BOS: {r['bos']}\n"
                    f"Entry: {fmt(r['price'])}\n"
                    f"TP: {fmt(r['tp'])} | SL: {fmt(r['sl'])}\n"
                )

                await ctx.application.bot.send_message(
                    chat_id=chat_id,
                    text=msg
                )

        except Exception as e:
            log.warning("scan fail %s: %r", coin, e)


# ================== Run ==================
def _seconds_to_next_hour(tz_name: str) -> int:
    try:
        import pytz
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
    except Exception:
        now = datetime.now()

    nxt = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
    delta = int((nxt - now).total_seconds())
    return max(10, delta)

async def post_init(app):
    # dọn webhook/pending trước khi polling
    await app.bot.delete_webhook(drop_pending_updates=True)
    me = await app.bot.get_me()
    print(f"✅ Telegram OK: @{me.username} (id={me.id})", flush=True)

async def on_error(update, ctx):
    err = ctx.error
    if isinstance(err, Conflict):
        print("⚠️ Conflict: có instance khác đang polling token này.", flush=True)
    else:
        print("⚠️ Error:", repr(err), flush=True)

def main():
    start_keep_alive()

    print("AUTO_SCAN =", AUTO_SCAN)

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN trong Environment Variables.")

    # retry loop để không chết vì Conflict (Render restart/deploy chồng)
    while True:
        try:
            app = (
                ApplicationBuilder()
                .token(TELEGRAM_BOT_TOKEN)
                .post_init(post_init)
                .build()
            )

            app.add_handler(CommandHandler("start", cmd_start))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
            app.add_error_handler(on_error)

            if AUTO_SCAN:
                first = _seconds_to_next_hour(TZ)
                app.job_queue.run_repeating(
                    auto_scan,
                    interval=SCAN_INTERVAL_SEC,
                    first=first,
                    data={"chat_id": CHAT_ID},
                )

            print("🤖 Bot polling start...", flush=True)
            app.run_polling(drop_pending_updates=True, allowed_updates=None)
            break

        except Conflict:
            wait = 20 + random.randint(0, 20)
            print(f"⚠️ Conflict getUpdates -> đợi {wait}s rồi chạy lại...", flush=True)
            time.sleep(wait)

if __name__ == "__main__":
    main()
