# main.py
import os, json, time, random, logging
from threading import Thread
from datetime import datetime, timedelta
from typing import Tuple

import ccxt
import pandas as pd
import pytz
from telegram.ext import ApplicationBuilder, CommandHandler
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
SCAN_INTERVAL_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "60"))
MIN_VOLZ           = float(os.getenv("MIN_VOLZ", "0.1"))

MARKET_TYPE        = os.getenv("MARKET_TYPE", "swap")
QUOTE              = os.getenv("QUOTE", "USDT").upper()

LABEL_TP_PCT       = float(os.getenv("LABEL_TP_PCT", "0.004"))
LABEL_SL_PCT       = float(os.getenv("LABEL_SL_PCT", "0.004"))
MODE = "HYBRID"
use_big_money = True
LAST_GLOBAL_SIGNAL = 0

CHAT_IDS = [
    5335165612,
    5895497001
]
LAST_SIGNAL_TIME = {}
ACTIVE_TRADES = {}
LAST_BAR_SIGNAL = {}
COOLDOWN_MINUTES = 30


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


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

def fmt(x):
    try:
        x = float(x)
        if x < 0.001:
            return f"{x:.8f}"
        elif x < 1:
            return f"{x:.6f}"
        else:
            return f"{x:.4f}"
    except:
        return str(x)
        
def cooldown_ok(symbol: str, side: str) -> bool:
    key = f"{symbol}_{side}"
    now = time.time()
    last = LAST_SIGNAL_TIME.get(key, 0)
    return (now - last) > COOLDOWN_MINUTES * 60

        
def in_trading_hours(start_hour=7, end_hour=22, tz_name="Asia/Ho_Chi_Minh"):
    try:
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
    except Exception:
        now = datetime.now()

    return start_hour <= now.hour < end_hour


# ================== CCXT (MEXC Swap) ==================
EX = ccxt.mexc({
    "options": {"defaultType": MARKET_TYPE},
    "enableRateLimit": True,
    "timeout": 30000
})

def symbol_usdt_perp(base: str) -> str:
    return f"{base.upper()}/{QUOTE}:{QUOTE}"

# ================== Indicators ==================
def ema(s, n): 
    return s.ewm(span=n, adjust=False).mean()
    
def ema_slope(series, length=3):

    if len(series) < length + 1:
        return 0

    return series.iloc[-1] - series.iloc[-length]
    
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
    
def strong_breakout_candle(df):

    last = df.iloc[-1]

    body = abs(last["close"] - last["open"])
    candle_range = last["high"] - last["low"]

    if candle_range == 0:
        return False

    body_ratio = body / candle_range

    if body_ratio > 0.6:
        return True

    return False
    
def detect_true_breakout(df):

    last = df.iloc[-1]

    body = abs(last["close"] - last["open"])
    rng = last["high"] - last["low"]

    if rng == 0:
        return False

    body_ratio = body / rng

    vol = last["volume"]
    vol_ma = df["volume"].rolling(20).mean().iloc[-2]

    if body_ratio > 0.6 and vol > vol_ma * 1.3:
        return True

    return False
    
def detect_compression(df):

    # cần đủ dữ liệu
    if len(df) < 30:
        return False

    atr_now = df["atr"].iloc[-1]
    atr_avg = df["atr"].rolling(20).mean().iloc[-1]

    ema_spread = abs(df["ema12"].iloc[-1] - df["ema26"].iloc[-1]) / df["close"].iloc[-1]

    # volatility thấp + EMA siết
    if atr_now < atr_avg * 0.7 and ema_spread < 0.002:
        return True

    return False
def detect_accumulation(df):

    if len(df) < 30:
        return False

    recent = df.tail(20)

    high = recent["high"].max()
    low = recent["low"].min()

    range_ratio = (high - low) / low

    avg_vol = recent["volume"].mean()
    last_vol = df["volume"].iloc[-1]

    # giá nén + volume tăng
    if range_ratio < 0.04 and last_vol > avg_vol * 1.2:
        return True

    return False
    
def detect_volume_accumulation(df):

    if len(df) < 40:
        return False

    recent = df.tail(30)

    # giá biến động nhỏ
    price_range = (recent["high"].max() - recent["low"].min()) / recent["low"].min()

    # volume đang tăng
    vol_old = recent["volume"].iloc[:15].mean()
    vol_new = recent["volume"].iloc[15:].mean()

    # ATR vẫn thấp
    atr_recent = recent["atr"].mean()
    price = recent["close"].iloc[-1]

    atr_ratio = atr_recent / price

    if price_range < 0.05 and vol_new > vol_old * 1.3 and atr_ratio < 0.01:
        return True

    return False
    
def detect_absorption(df):

    if len(df) < 20:
        return False

    recent = df.tail(10)

    price_range = (recent["high"].max() - recent["low"].min()) / recent["close"].iloc[-1]

    vol_mean = recent["volume"].mean()
    vol_last = recent["volume"].iloc[-1]

    # volume lớn nhưng giá không đi
    if price_range < 0.01 and vol_last > vol_mean * 1.5:
        return True

    return False
def detect_momentum_expansion(df):

    if len(df) < 20:
        return False

    atr_now = df["atr"].iloc[-1]
    atr_old = df["atr"].iloc[-10]

    if atr_now > atr_old * 1.3:
        return True

    return False


def detect_volume_trend(df):

    if len(df) < 15:
        return False

    vols = df["volume"].tail(5)

    if vols.is_monotonic_increasing:
        return True

    return False
def detect_volatility_expansion(df):

    if len(df) < 20:
        return False

    atr_now = df["atr"].iloc[-1]
    atr_prev = df["atr"].iloc[-5]

    vol_now = df["volume"].iloc[-1]
    vol_ma = df["volume"].rolling(20).mean().iloc[-2]

    price_move = abs(df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]

    # ATR đang mở rộng + volume tăng + giá chưa chạy nhiều
    if atr_now > atr_prev * 1.25 and vol_now > vol_ma * 1.2 and price_move < 0.02:
        return True

    return False
    
def detect_pre_pump(df):

    if len(df) < 20:
        return False

    recent = df.tail(10)

    # volume tăng dần
    vol_trend = recent["volume"].iloc[-1] > recent["volume"].iloc[-5]

    # ATR đang tăng
    atr_now = df["atr"].iloc[-1]
    atr_prev = df["atr"].iloc[-5]

    atr_expand = atr_now > atr_prev * 1.15

    # giá chưa chạy nhiều
    move = abs(df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]

    if vol_trend and atr_expand and move < 0.02:
        return True

    return False
    
def detect_fake_breakout(df):

    if len(df) < 5:
        return False

    last = df.iloc[-1]

    open_price = last["open"]
    close_price = last["close"]
    high = last["high"]
    low = last["low"]

    body = abs(close_price - open_price)
    full_range = high - low

    if full_range == 0:
        return False

    body_ratio = body / full_range

    # volume
    vol = last["volume"]
    vol_ma = df["volume"].rolling(20).mean().iloc[-2]

    # điều kiện trap:
    # 1. râu dài (body nhỏ)
    # 2. volume lớn bất thường
    if body_ratio < 0.4 and vol > vol_ma * 1.5:
        return True

    return False
    
def detect_early_trend(df):

    if len(df) < 30:
        return None

    recent = df.tail(10)

    # EMA bắt đầu dốc
    ema12_slope = df["ema12"].iloc[-1] - df["ema12"].iloc[-4]

    # Volume tăng
    vol_now = df["volume"].iloc[-1]
    vol_prev = df["volume"].iloc[-5]

    vol_up = vol_now > vol_prev * 1.2

    # ATR mở rộng
    atr_now = df["atr"].iloc[-1]
    atr_prev = df["atr"].iloc[-5]

    atr_up = atr_now > atr_prev * 1.15

    # Giá chưa chạy nhiều (rất quan trọng)
    move = abs(df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]

    if move > 0.02:
        return None

    # LONG
    if ema12_slope > 0 and vol_up and atr_up:
        return "LONG"

    # SHORT
    if ema12_slope < 0 and vol_up and atr_up:
        return "SHORT"

    return None
    
def detect_breakout(df):

    high = df["high"].rolling(20).max().iloc[-2]
    low = df["low"].rolling(20).min().iloc[-2]

    close = df["close"].iloc[-1]

    if close > high:
        return "LONG"

    if close < low:
        return "SHORT"

    return None

# ================== Market Structure ==================
def detect_bos(df: pd.DataFrame, lookback: int = 8) -> str:
    high = df["high"]
    low = df["low"]

    swing_high = high.rolling(lookback).max()
    swing_low = low.rolling(lookback).min()

    prev_close = df["close"].iloc[-2]
    last_close = df["close"].iloc[-1]

    # phá đỉnh
    if last_close > swing_high.iloc[-2] and last_close > prev_close:
        return "BOS_UP"

    # phá đáy
    if last_close < swing_low.iloc[-2] and last_close < prev_close:
        return "BOS_DOWN"

    return "NO_BOS"


def break_retest_ok(df: pd.DataFrame, side: str, lookback: int = 15, tol: float = 0.03) -> bool:

    last = df.iloc[-1]

    high = df["high"]
    low = df["low"]

    swing_high = high.rolling(lookback).max()
    swing_low = low.rolling(lookback).min()

    close = last["close"]
    openp = last["open"]
    highp = last["high"]
    lowp = last["low"]

    if side == "LONG":

        level = swing_high.iloc[-2]

        near = abs(close - level) / level <= tol
        wick = lowp < level and close > level

        return near or wick

    if side == "SHORT":

        level = swing_low.iloc[-2]

        near = abs(close - level) / level <= tol
        wick = highp > level and close < level

        return near or wick

    return False
    
def confirm_entry_pro(df, side):

    if len(df) < 20:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ===== pullback =====
    if side == "LONG":
        pullback = last["low"] <= df["ema12"].iloc[-1]
        confirm = last["close"] > prev["high"]
    else:
        pullback = last["high"] >= df["ema12"].iloc[-1]
        confirm = last["close"] < prev["low"]

    # ===== volume =====
    vol_ok = last["volume"] > df["volume"].rolling(20).mean().iloc[-2]

    # ===== RSI =====
    rsi_val = df["rsi"].iloc[-1]

    if side == "LONG":
        rsi_ok = 45 < rsi_val < 70
    else:
        rsi_ok = 30 < rsi_val < 55

    return pullback and confirm and vol_ok and rsi_ok
    
def detect_strong_bos(df: pd.DataFrame, lookback: int = 12) -> str:
    if len(df) < lookback + 2:
        return "NO_BOS"

    high = df["high"]
    low = df["low"]

    swing_high = high.rolling(lookback).max()
    swing_low = low.rolling(lookback).min()

    last = df.iloc[-1]

    o = last["open"]
    c = last["close"]
    v = last["volume"]

    vol_ma = df["volume"].rolling(20).mean().iloc[-2]

    if c > swing_high.iloc[-2] and c > o and v >= vol_ma * 0.8:
        return "BOS_UP"

    if c < swing_low.iloc[-2] and c < o and v >= vol_ma * 0.8:
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

    last_high = high.iloc[-1]
    last_low = low.iloc[-1]
    last_close = df["close"].iloc[-1]

    # Sweep đáy rồi bật lên
    if last_low < prev_low and last_close > prev_low:
        return "SWEEP_DOWN"

    # Sweep đỉnh rồi rơi xuống
    if last_high > prev_high and last_close < prev_high:
        return "SWEEP_UP"

    return "NO_SWEEP"

def detect_liquidity_grab(df):

    if len(df) < 10:
        return None

    prev = df.iloc[-2]
    last = df.iloc[-1]

    # grab phía dưới (bullish)
    if last["low"] < prev["low"] and last["close"] > prev["low"]:
        return "LONG"

    # grab phía trên (bearish)
    if last["high"] > prev["high"] and last["close"] < prev["high"]:
        return "SHORT"

    return None
    
def detect_order_block(df):

    if len(df) < 10:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(prev["close"] - prev["open"])
    candle_range = prev["high"] - prev["low"]

    # strong bullish candle trước đó
    if prev["close"] > prev["open"] and body > candle_range * 0.6:

        # giá hiện tại retest vùng đó
        if last["low"] <= prev["close"] <= last["high"]:
            return "LONG"

    # strong bearish candle
    if prev["close"] < prev["open"] and body > candle_range * 0.6:

        if last["low"] <= prev["open"] <= last["high"]:
            return "SHORT"

    return None
    
def detect_whale_flow(ob):

    try:

        if not ob["bids"] or not ob["asks"]:
            return None

        bid_volume = sum([b[1] for b in ob["bids"][:5]])
        ask_volume = sum([a[1] for a in ob["asks"][:5]])

        if bid_volume > ask_volume * 1.4:
            return "LONG"

        if ask_volume > bid_volume * 1.4:
            return "SHORT"

        return None

    except:
        return None
        
def detect_liquidity_vacuum(ob):

    try:

        bid_total = sum([b[1] for b in ob["bids"][:10]])
        ask_total = sum([a[1] for a in ob["asks"][:10]])

        spread = ob["asks"][0][0] - ob["bids"][0][0]

        if bid_total < ask_total * 0.7 or ask_total < bid_total * 0.7:

            if spread / ob["bids"][0][0] > 0.001:
                return True

        return False

    except:
        return False
        
def detect_liquidity_pressure(ob):

    try:

        bids = ob["bids"][:10]
        asks = ob["asks"][:10]

        if not bids or not asks:
            return None

        bid_volume = sum([b[1] for b in bids])
        ask_volume = sum([a[1] for a in asks])

        best_bid = bids[0][0]
        best_ask = asks[0][0]

        spread = best_ask - best_bid

        if bid_volume > ask_volume * 1.6 and spread / best_bid < 0.0008:
            return "LONG"

        if ask_volume > bid_volume * 1.6 and spread / best_bid < 0.0008:
            return "SHORT"

        return None

    except:
        return None
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
def dynamic_atr_threshold(atr_ratio: float) -> float:

    if atr_ratio >= 0.01:
        return 0.0015

    elif atr_ratio >= 0.006:
        return 0.001

    elif atr_ratio >= 0.003:
        return 0.0008

    else:
        return 0.0005
def detect_market_condition(df):

    if len(df) < 50:
        return "NORMAL"

    atr_now = df["atr"].iloc[-1]
    atr_avg = df["atr"].rolling(20).mean().iloc[-1]

    vol_now = df["volume"].iloc[-1]
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]

    ema12 = df["ema12"].iloc[-1]
    ema26 = df["ema26"].iloc[-1]

    trend_strength = abs(ema12 - ema26) / df["close"].iloc[-1]

    if atr_now > atr_avg * 1.2 and vol_now > vol_avg * 1.2 and trend_strength > 0.003:
        return "STRONG"

    return "WEAK"


def detect_big_money_flow(df):

    if len(df) < 50:
        return False

    recent = df.tail(30)

    vol_old = recent["volume"].iloc[:15].mean()
    vol_new = recent["volume"].iloc[15:].mean()

    vol_increase = vol_new > vol_old * 1.4

    high = recent["high"].max()
    low = recent["low"].min()

    price_range = (high - low) / low

    atr_now = df["atr"].iloc[-1]
    price = df["close"].iloc[-1]

    atr_ratio = atr_now / price

    if vol_increase and price_range < 0.05 and atr_ratio < 0.008:
        return True

    return False
    
def detect_smart_money(df):

    if len(df) < 50:
        return False

    recent = df.tail(30)

    # volume tăng mạnh nhưng giá đi ngang
    vol_old = recent["volume"].iloc[:15].mean()
    vol_new = recent["volume"].iloc[15:].mean()

    vol_up = vol_new > vol_old * 1.5

    # giá bị nén cực mạnh
    high = recent["high"].max()
    low = recent["low"].min()

    range_ratio = (high - low) / low

    # ATR cực thấp
    atr_ratio = df["atr"].iloc[-1] / df["close"].iloc[-1]

    if vol_up and range_ratio < 0.03 and atr_ratio < 0.007:
        return True

    return False

def manage_trailing(base, df):

    if base not in ACTIVE_TRADES:
        return None

    trade = ACTIVE_TRADES[base]
    side = trade["side"]
    entry = trade["entry"]

    old_sl = trade["sl"]

    price = df["close"].iloc[-1]
    atrv = df["atr"].iloc[-1]

    profit = abs(price - entry)

    # hòa vốn
    if profit > 3 * atrv:
        trade["sl"] = entry

    # khóa lợi nhuận
    if profit > 5 * atrv:
        if side == "LONG":
            trade["sl"] = price - atrv
        else:
            trade["sl"] = price + atrv

    # chỉ update khi SL thay đổi đủ lớn
    if abs(trade["sl"] - old_sl) / old_sl > 0.003:
        return trade

    return None

def analyze(base: str, tf: str, df=None, manual=False) -> dict:

    if df is None:
        df = enrich(fetch_ohlcv(base, tf, limit=300))
        
    market = detect_market_condition(df)
    side = None
    early = detect_early_trend(df)

    if MODE == "SNIPER":
        if not early:
            return {"skip": True, "reason": "Not sniper"}

    if MODE == "SWING":
        early = None

    if early:
        side = early
   
    row = df.iloc[-1].to_dict()
    price_move = abs(df["close"].iloc[-1] - df["close"].iloc[-3]) / df["close"].iloc[-3]

    if price_move > 0.06:
        return {"skip": True, "reason": "Too late"}

    last_bar = df["ts"].iloc[-1]

    now = pd.Timestamp.utcnow()

    # chỉ cho trade khi nến gần đóng
    if (now - last_bar).total_seconds() < 300:
        return {"skip": True, "reason": "Candle not mature"}

    last_bar = df["ts"].iloc[-1]

    # chỉ trade khi có nến mới
    if base in LAST_BAR_SIGNAL:
        if str(LAST_BAR_SIGNAL[base]) == str(last_bar):
            return {"skip": True, "reason": "Same candle"}
        

    # ===== ATR FILTER =====
    atr_ratio = row["atr"] / row["close"]

    min_atr = dynamic_atr_threshold(atr_ratio)

    if not manual and atr_ratio < min_atr:
        return {"skip": True, "reason": "ATR too low"}

    # ===== EARLY BREAKOUT DETECTOR =====
    compression = detect_compression(df)
    accumulation = detect_accumulation(df)
    volume_accum = detect_volume_accumulation(df)
    absorption = detect_absorption(df)
    momentum = detect_momentum_expansion(df)
    vol_trend = detect_volume_trend(df)
    vol_expand = detect_volatility_expansion(df)
    pre_pump = detect_pre_pump(df)
    big_money = detect_big_money_flow(df)
    smart_money = detect_smart_money(df)
    # ===== FILTER DÒNG TIỀN (BẮT BUỘC) =====
    if use_big_money:
        if not (pre_pump or volume_accum or absorption):
            return {"skip": True, "reason": "No real money flow"}
            
    if early:
        if not (pre_pump or volume_accum or absorption):
            return {"skip": True, "reason": "Early but no confirmation"}
    
    if compression:
        breakout = detect_breakout(df)

        if breakout:
            side = breakout

    sweep = detect_liquidity_sweep(df)
    grab = detect_liquidity_grab(df)
    order_block = detect_order_block(df)
    log.info(f"{base} SWEEP={sweep}")
    symbol = symbol_usdt_perp(base)

    orderbook = {"bids": [], "asks": []}

    try:
        orderbook = EX.fetch_order_book(symbol, limit=10)
    except:
        pass
    if orderbook["bids"] and orderbook["asks"]:

        spread = (orderbook["asks"][0][0] - orderbook["bids"][0][0]) / orderbook["bids"][0][0]

        if spread > 0.002:
            return {"skip": True, "reason": "Spread too high"}
    whale = detect_whale_flow(orderbook)
    vacuum = detect_liquidity_vacuum(orderbook)
    pressure = detect_liquidity_pressure(orderbook)

    bos = detect_strong_bos(df)
    true_break = detect_true_breakout(df)
    
    # nếu breakout rồi thì bỏ (đu giá)
    if true_break and not pre_pump:
        return {"skip": True, "reason": "Late breakout"}

    if bos == "NO_BOS":
        bos = detect_bos(df)
    fake = detect_fake_breakout(df)

    if fake:
        return {"skip": True, "reason": "Fake breakout trap"}

    # ===== nếu có breakout sớm thì dùng nó =====
    if side is None:

        # 🚀 EARLY ENTRY
        if (pre_pump or volume_accum or absorption) and momentum:

            if df["ema12"].iloc[-1] > df["ema26"].iloc[-1]:
                side = "LONG"
            else:
                side = "SHORT"

        else:
            if bos == "NO_BOS":
                return {"skip": True, "reason": "No market structure"}

            side = "LONG" if bos == "BOS_UP" else "SHORT"
        
    if not confirm_entry_pro(df, side):

        if not (pre_pump or volume_accum):
            return {"skip": True, "reason": "Bad entry timing"}
    
    # Trend continuation filter
    if side == "LONG" and row["ema12"] < row["ema26"]:
        return {"skip": True, "reason": "Weak bullish momentum"}

    if side == "SHORT" and row["ema12"] > row["ema26"]:
        return {"skip": True, "reason": "Weak bearish momentum"}

    ema12_slope = ema_slope(df["ema12"])
    ema26_slope = ema_slope(df["ema26"])
        
    # ===== HTF TREND FILTER =====
    trend = get_htf_trend(base)
    # ===== SWING FILTER (BẮT BUỘC) =====
    if MODE == "SWING":
        if side == "LONG" and trend == "DOWN":
            return {"skip": True, "reason": "Against HTF trend"}

        if side == "SHORT" and trend == "UP":
            return {"skip": True, "reason": "Against HTF trend"}

    # 🚀 CHẶN NGƯỢC TREND TUYỆT ĐỐI
    if side == "LONG" and trend != "UP":
        return {"skip": True, "reason": "Against HTF trend"}

    if side == "SHORT" and trend != "DOWN":
        return {"skip": True, "reason": "Against HTF trend"}
        
    # EMA trend filter
    if side == "LONG" and row["close"] < row["ema50"]:
        return {"skip": True, "reason": "Below EMA50"}

    if side == "SHORT" and row["close"] > row["ema50"]:
        return {"skip": True, "reason": "Above EMA50"}
        
    # ===== COOLDOWN =====
    if not manual and not cooldown_ok(base, side):
        return {"skip": True, "reason": "Cooldown"}
    # ===== VOLUME CONFIRM =====
    if not manual and abs(row["vol_z"]) < MIN_VOLZ:
        return {"skip": True, "reason": "Weak volume"}
    # ===== RSI FILTER =====
    rsi_val = row["rsi"]

    if side == "LONG" and rsi_val > 80:
        return {"skip": True, "reason": "RSI overbought"}

    if side == "SHORT" and rsi_val < 20:
        return {"skip": True, "reason": "RSI oversold"}
    # ===== SIGNAL SCORE =====
    score = 0
    pump_score = 0
    
    if bos != "NO_BOS" and not strong_breakout_candle(df):
        score -= 10
    # ===== EARLY PUMP SIGNAL =====
    if pre_pump:
        score += 35
        pump_score += 40
        
    if smart_money:
        score += 40
        pump_score += 50  
    
    if true_break:
        score += 25
        pump_score += 20
    if absorption:
        score += 25
        pump_score += 20
        
    if volume_accum:
        score += 30
        pump_score += 25
        
    if pressure == side:
        score += 25
        pump_score += 25
        
    if momentum:
        score += 15
        pump_score += 10

    if vol_trend:
        score += 15
        pump_score += 15

    if vol_expand:
        score += 20
        pump_score += 20

    if vacuum:
        score += 30
        pump_score += 20

    if accumulation:
        score += 25
        pump_score += 20

    if whale == side:
        score += 25
        pump_score += 20

    # ===== SMART MONEY STRUCTURE =====
    if grab == side:
        score += 15

    if order_block == side:
        score += 20

    if bos != "NO_BOS":
        score += 30

    if sweep != "NO_SWEEP":
        score += 20

    # ===== MOMENTUM =====
    if abs(row["vol_z"]) > MIN_VOLZ:
        score += 20
        pump_score += 15

    if side == "LONG" and row["close"] > row["ema50"]:
        score += 20

    if side == "SHORT" and row["close"] < row["ema50"]:
        score += 20

    if 30 < row["rsi"] < 70:
        score += 10

    # ===== FILTER CUỐI =====
    if MODE == "SNIPER" and score < 65:
        return {"skip": True, "reason": f"Low sniper {score}"}

    if MODE == "SWING" and score < 80:
        return {"skip": True, "reason": f"Low swing {score}"}

    if MODE == "HYBRID":

        if market == "WEAK" and score < 80:
            return {"skip": True, "reason": f"Weak market {score}"}

        if market == "STRONG" and score < 70:
            return {"skip": True, "reason": f"Low hybrid {score}"}
        
    log.info(f"SIGNAL {base} {side} SCORE={score} PUMP={pump_score} BOS={bos}")
    
    # 3. TP / SL theo ATR
    entry = float(row["close"])
    atrv = float(row["atr"])

    if MODE == "SNIPER":
        tp = entry + (2.0 * atrv if side == "LONG" else -2.0 * atrv)
        sl = entry - (1.0 * atrv if side == "LONG" else -1.0 * atrv)

    elif MODE == "SWING":
        tp = entry + (4.0 * atrv if side == "LONG" else -4.0 * atrv)
        sl = entry - (1.8 * atrv if side == "LONG" else -1.8 * atrv)

    else:  # HYBRID
        if side == "LONG":
            tp = entry + 3.0 * atrv
            sl = entry - 1.3 * atrv
        else:
            tp = entry - 3.0 * atrv
            sl = entry + 1.3 * atrv
        
    LAST_SIGNAL_TIME[f"{base}_{side}"] = time.time()
    LAST_BAR_SIGNAL[base] = df["ts"].iloc[-1]
    ACTIVE_TRADES[base] = {
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp": tp
    }
    return {
        "base": base.upper(),
        "tf": tf,
        "side": side,
        "price": entry,
        "tp": tp,
        "sl": sl,
        "bos": bos,
        "pump_score": pump_score
    }

   
# ================== Telegram ==================
async def cmd_start(update, ctx):
    await update.message.reply_text(
         "🤖 Bot Futures AI đã chạy!\n"
         "Bot tự động quét thị trường và gửi tín hiệu mạnh nhất."
    )


# ================== Auto scan ==================
async def auto_scan(ctx):

    chat_ids = ctx.job.data["chat_ids"]

    try:
        tickers = EX.fetch_tickers()
    except Exception as e:
        log.warning(f"fetch_tickers error: {e}")
        return

    volumes = {}

    for s, t in tickers.items():

        if "/USDT:USDT" not in s:
            continue

        base = s.split("/")[0]

        if base in ["USDC","USDT","USDP","TUSD","BUSD"]:
            continue

        vol = t.get("quoteVolume")

        if vol:
            volumes[base] = vol

    sorted_coins = sorted(volumes, key=volumes.get, reverse=True)

    top = sorted_coins[:15]

    log.info(f"Scanning coins: {top}")

    signals = []


    for coin in top:
        
        try:
            df = enrich(fetch_ohlcv(coin, TIMEFRAME_DEFAULT, 100))
        except Exception as e:
            log.warning(f"{coin} fetch error: {e}")
            continue

        try:
            r = analyze(coin, TIMEFRAME_DEFAULT, df=df, manual=False)
        except Exception as e:
            log.warning(f"{coin} analyze error: {e}")
            continue
            
        trade = manage_trailing(coin, df)

        try:

            if r.get("skip"):
                log.info(f"{coin} skip: {r['reason']}")
                continue

            signals.append(r)

        except Exception as e:
            log.warning("scan fail %s: %r", coin, e)

    global LAST_GLOBAL_SIGNAL

    if not signals:
        return

    if time.time() - LAST_GLOBAL_SIGNAL < 600:
        return

    LAST_GLOBAL_SIGNAL = time.time()
    
    # chọn duy nhất 1 tín hiệu mạnh nhất
    best = max(signals, key=lambda x: x["pump_score"])

    msg = (
        f"🚀 TOP 1 SIGNAL — {best['base']}/USDT ({best['tf']})\n"
        f"Hướng: {best['side']} | BOS: {best['bos']}\n"
        f"Entry: {fmt(best['price'])}\n"
        f"TP: {fmt(best['tp'])} | SL: {fmt(best['sl'])}\n"
    )

    for cid in chat_ids:
        await ctx.application.bot.send_message(
            chat_id=cid,
            text=msg
        )

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

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_error_handler(on_error)

    if AUTO_SCAN:
        first = _seconds_to_next_hour(TZ)
        app.job_queue.run_repeating(
            auto_scan,
            interval=SCAN_INTERVAL_SEC,
            first=first,
            data={"chat_ids": CHAT_IDS},
        )

    print("🤖 Bot polling start...", flush=True)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
