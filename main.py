import os, math, asyncio, json, time
from threading import Thread
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List

import ccxt
import pandas as pd
import numpy as np
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# ================== ENV & DEFAULTS ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TZ                 = os.getenv("TZ", "Asia/Ho_Chi_Minh")
TIMEFRAME_DEFAULT  = os.getenv("TIMEFRAME", "15m")

# Auto scan & filters
AUTO_SCAN          = os.getenv("AUTO_SCAN", "true").lower() == "true"
SCAN_INTERVAL_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "30"))
ALERT_THRESHOLD    = int(os.getenv("ALERT_THRESHOLD", "80"))   # AI score min
MIN_VOLZ           = float(os.getenv("MIN_VOLZ", "2"))         # volume z-score min
MIN_ATR_PCT        = float(os.getenv("MIN_ATR_PCT", "0.25"))   # atr% min
OI_DELTA_PCT       = float(os.getenv("OI_DELTA_PCT", "4"))     # (để dành nâng cấp OI)
FUNDING_MAX        = float(os.getenv("FUNDING_MAX", "0.02"))   # ±0.02% max (nếu lấy được)
TF_ALIGN_REQ       = int(os.getenv("TF_ALIGN", "2"))           # số TF đồng thuận tối thiểu
TF_SET             = [x.strip() for x in os.getenv("TF_SET", "5m,15m,1h").split(",") if x.strip()]
MAX_SIGNALS_PER_HR = int(os.getenv("MAX_SIGNALS_PER_HOUR", "5"))

# Universe
EXCHANGE           = os.getenv("EXCHANGE", "MEXC")
MARKET_TYPE        = os.getenv("MARKET_TYPE", "swap")          # chỉ futures
QUOTE              = os.getenv("QUOTE", "USDT").upper()
AUTO_COIN_TOP      = os.getenv("AUTO_COIN_TOP", "true").lower() == "true"
COIN_LIST_RAW      = os.getenv("COIN_LIST", "")
TOP_LIMIT          = int(os.getenv("TOP_LIMIT", "20"))

# AI tự học thị trường (auto-label)
AUTO_LEARN_MARKET  = os.getenv("AUTO_LEARN_MARKET", "true").lower() == "true"
LABEL_HORIZON_BARS = int(os.getenv("LABEL_HORIZON_BARS", "5"))
LABEL_TP_PCT       = float(os.getenv("LABEL_TP_PCT", "0.004"))   # 0.4%
LABEL_SL_PCT       = float(os.getenv("LABEL_SL_PCT", "0.004"))   # 0.4%

# Storage
DATA_DIR    = "data"
MEMO_PATH   = os.path.join(DATA_DIR, "memory.json")
PENDING_PATH= os.path.join(DATA_DIR, "pending.json")
os.makedirs(DATA_DIR, exist_ok=True)

# ================== KEEP-ALIVE WEB (Render/UptimeRobot) ==================
def start_keep_alive():
    try:
        from keep_alive import run_server
        Thread(target=run_server, daemon=True).start()
    except Exception:
        pass

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

def _tf_to_minutes(tf: str) -> int:
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 60 * 24
    return 15

# ================== CCXT MEXC (Futures/Swap) ==================
def mexc_client():
    return ccxt.mexc({
        "options": {"defaultType": MARKET_TYPE},  # swap = futures perp
        "enableRateLimit": True,
    })

def symbol_usdt_perp(base: str) -> str:
    return f"{base.upper()}/{QUOTE}:{QUOTE}"

# ================== Indicators ==================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    down = (-d).clip(lower=0.0)
    rs = up.rolling(n).mean() / (down.rolling(n).mean() + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    f = ema(close, fast)
    s = ema(close, slow)
    m = f - s
    sig = ema(m, signal)
    h = m - sig
    return m, sig, h

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ================== Data fetch & enrich ==================
def fetch_ohlcv(base: str, tf: str, limit: int = 300) -> pd.DataFrame:
    ex = mexc_client()
    sym = symbol_usdt_perp(base)
    data = ex.fetch_ohlcv(sym, timeframe=tf, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
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

# ================== Online AI (auto-learn) ==================
def _load_json(path, default):
    try:
        if os.path.exists(path):
            return json.load(open(path, "r"))
    except Exception:
        pass
    return default

def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"))

class OnlineAI:
    # 6 features: trend, macd_hist, rsi_c, volz, atrp, bias
    def __init__(self, path):
        self.path = path
        self.w = np.zeros(6, dtype=float)
        self.lr = 0.03
        self._load()
    def _load(self):
        obj = _load_json(self.path, {})
        self.w = np.array(obj.get("w", self.w.tolist()), dtype=float)
        self.lr = float(obj.get("lr", self.lr))
    def _save(self):
        _save_json(self.path, {"w": self.w.tolist(), "lr": self.lr})
    def _feat(self, row):
        trend = (row["ema12"] - row["ema26"]) / (abs(row["ema26"]) + 1e-9)
        macd_h = row["macd_hist"]
        rsi_c  = (row["rsi"] - 50) / 50.0
        volz   = np.tanh(row["vol_z"] / 3.0)
        atrp   = row["atr"] / max(row["close"], 1e-9)
        bias   = 1.0
        return np.array([trend, macd_h, rsi_c, volz, atrp, bias], dtype=float)
    def score(self, row) -> int:
        x = self._feat(row)
        z = float(np.dot(self.w, x))
        p = 1 / (1 + math.exp(-z))
        return int(round(p * 100))
    def learn(self, row, label: int):
        x = self._feat(row)
        z = float(np.dot(self.w, x))
        p = 1 / (1 + math.exp(-z))
        grad = (p - label) * x
        self.w -= self.lr * grad
        self._save()

AI = OnlineAI(MEMO_PATH)

def _pending_key(base, tf): return f"{base.upper()}_{tf}"

def _append_pending_sample(base, tf, row):
    if not AUTO_LEARN_MARKET: return
    pend = _load_json(PENDING_PATH, {})
    key = _pending_key(base, tf)
    item = {
        "ts": datetime.utcnow().isoformat(),
        "close": float(row["close"]),
        "ema12": float(row["ema12"]),
        "ema26": float(row["ema26"]),
        "rsi": float(row["rsi"]),
        "macd_hist": float(row["macd_hist"]),
        "vol_z": float(row["vol_z"]),
        "atr": float(row["atr"]),
    }
    arr = pend.get(key, [])
    arr.append(item)
    pend[key] = arr[-200:]  # giữ tối đa 200 mẫu chờ
    _save_json(PENDING_PATH, pend)

def _process_pending_auto_learn(base, tf):
    if not AUTO_LEARN_MARKET: return 0, 0
    pend = _load_json(PENDING_PATH, {})
    key = _pending_key(base, tf)
    arr = pend.get(key, [])
    if not arr: return 0, 0

    try:
        dnow = enrich(fetch_ohlcv(base, tf, limit=120))
        cur = float(dnow.iloc[-1]["close"])
    except Exception:
        return 0, 0

    learned = kept = 0
    new_arr = []
    for it in arr:
        px0 = float(it["close"])
        ret = (cur - px0) / max(px0, 1e-9)
        label = None
        if ret >= LABEL_TP_PCT: label = 1
        elif ret <= -LABEL_SL_PCT: label = 0

        if label is None:
            new_arr.append(it)
            kept += 1
        else:
            AI.learn(it, label)
            learned += 1

    pend[key] = new_arr
    _save_json(PENDING_PATH, pend)
    return learned, kept

# ================== Scoring / Targets ==================
def simple_ai_score(row: Dict[str, Any]) -> Tuple[int, str]:
    score = 50
    reasons = []
    if row["ema12"] > row["ema26"]:
        score += 15; reasons.append("EMA12>EMA26")
    else:
        score -= 15; reasons.append("EMA12<EMA26")
    if row["macd_hist"] > 0:
        score += 15; reasons.append("MACD_hist>0")
    else:
        score -= 15; reasons.append("MACD_hist<0")
    if row["rsi"] > 55:
        score += 10; reasons.append("RSI>55")
    elif row["rsi"] < 45:
        score -= 10; reasons.append("RSI<45")
    if row["vol_z"] > 1.0:
        score += 10; reasons.append("VolZ>1")
    score = int(np.clip(score, 0, 100))
    return score, "; ".join(reasons)

def make_targets(entry: float, atrv: float, side: str) -> Tuple[float,float,float]:
    tp1 = entry + (1.5 * atrv if side == "LONG" else -1.5 * atrv)
    tp2 = entry + (2.5 * atrv if side == "LONG" else -2.5 * atrv)
    sl  = entry - (1.0 * atrv if side == "LONG" else -1.0 * atrv)
    return tp1, tp2, sl

# ================== Analyze one coin ==================
def analyze(base: str, tf: str) -> Dict[str, Any]:
    # học các mẫu cũ nếu đủ điều kiện
    _process_pending_auto_learn(base, tf)

    df = fetch_ohlcv(base, tf, limit=350)
    d  = enrich(df)
    row = d.iloc[-1].to_dict()

    # side cơ bản
    side = "LONG" if (row["ema12"] > row["ema26"] and row["macd_hist"] > 0 and row["rsi"] >= 48) else "SHORT"

    # AI score cơ bản + online AI
    base_score, why1 = simple_ai_score(row)
    online_score = AI.score(row)
    score = int(round(0.5 * base_score + 0.5 * online_score))
    why = f"{why1}; online={online_score}%"

    tp1, tp2, sl = make_targets(row["close"], row["atr"], side)

    atr_pct = float(row["atr"]) / max(float(row["close"]), 1e-9) * 100.0

    # funding (best effort)
    funding_txt = "-"
    try:
        ex = mexc_client()
        fr = ex.fetch_funding_rate(symbol_usdt_perp(base))
        val = fr.get("fundingRate")
        if val is not None:
            funding_txt = f"{float(val):.4%}"
    except Exception:
        pass

    # ghi mẫu hiện tại để lần sau auto-label
    _append_pending_sample(base, tf, row)

    return {
        "base": base.upper(), "tf": tf, "side": side,
        "price": float(row["close"]),
        "tp1": float(tp1), "tp2": float(tp2), "sl": float(sl),
        "ema12": float(row["ema12"]), "ema26": float(row["ema26"]),
        "rsi": float(row["rsi"]), "macd_hist": float(row["macd_hist"]),
        "vol_z": float(row["vol_z"]), "atr": float(row["atr"]),
        "atr_pct": atr_pct, "score": int(score),
        "why": why, "funding": funding_txt,
    }

# ================== Multi-TF alignment ==================
def multi_tf_alignment(base: str, side_ref: str, tfs: List[str]) -> Tuple[int, List[str]]:
    agree = 0
    marks = []
    for tf in tfs:
        try:
            df = enrich(fetch_ohlcv(base, tf, limit=160))
            row = df.iloc[-1]
            side = "LONG" if (row["ema12"] > row["ema26"] and row["macd_hist"] > 0 and row["rsi"] >= 48) else "SHORT"
            marks.append(f"{tf}:{'↑' if side=='LONG' else '↓'}")
            if side == side_ref:
                agree += 1
        except Exception:
            marks.append(f"{tf}:—")
    return agree, marks

# ================== Universe (top coins) ==================
def fetch_top_bases() -> List[str]:
    if not AUTO_COIN_TOP and COIN_LIST_RAW:
        return [x.strip().upper() for x in COIN_LIST_RAW.split(",") if x.strip()]
    try:
        ex = mexc_client()
        ex.load_markets()
        # ưu tiên theo số lượng market (fallback nếu thiếu volume)
        cands = []
        for s, info in ex.markets.items():
            if info.get("type") != "swap": continue
            if str(info.get("quote", "")).upper() != QUOTE: continue
            cands.append(info["base"])
        # unique + cắt TOP_LIMIT
        out, seen = [], set()
        for b in cands:
            if b not in seen:
                out.append(b); seen.add(b)
            if len(out) >= TOP_LIMIT: break
        if out: return out
    except Exception:
        pass
    # fallback list
    return ["BTC","ETH","SOL","OP","ARB","WLD","XRP","DOGE","TON","ADA","AVAX","LINK","APT","SUI","SEI","MANTA","FTM","FIL","LDO","PEPE"][:TOP_LIMIT]

# ================== Alert logic ==================
_last_alert_ts: Dict[str, float] = {}
_alert_sent_window: List[float] = []

ALERT_COOLDOWN = 60 * 10  # 10 phút không lặp lại/coin

def _signals_in_last_hour() -> int:
    now = time.time()
    # dọn cũ
    while _alert_sent_window and now - _alert_sent_window[0] > 3600:
        _alert_sent_window.pop(0)
    return len(_alert_sent_window)

async def maybe_alert(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, base: str, r: Dict[str, Any], tf_marks: List[str]):
    now = time.time()
    if now - _last_alert_ts.get(base, 0) < ALERT_COOLDOWN:
        return
    if _signals_in_last_hour() >= MAX_SIGNALS_PER_HR:
        return

    # bộ lọc an toàn
    if r["score"] < ALERT_THRESHOLD: return
    if r["vol_z"] < MIN_VOLZ: return
    if r["atr_pct"] < MIN_ATR_PCT: return
    # funding filter nếu lấy được
    try:
        if r["funding"] != "-" and abs(float(r["funding"].strip("%"))/100.0) > FUNDING_MAX:
            return
    except Exception:
        pass

    msg = (
        f"🔥 STRONG SIGNAL — {r['base']}/USDT (swap)\n"
        f"TF: {r['tf']}  |  Multi-TF: {'  '.join(tf_marks)}\n"
        f"Hướng: {r['side']}\n"
        f"Entry: {fmt(r['price'])}\n"
        f"TP1/TP2: {fmt(r['tp1'])} / {fmt(r['tp2'])}\n"
        f"SL: {fmt(r['sl'])}\n"
        f"EMA12/26: {fmt(r['ema12'])}/{fmt(r['ema26'])}\n"
        f"RSI: {fmt(r['rsi'],2)}  |  MACD_hist: {fmt(r['macd_hist'],5)}\n"
        f"VolZ: {fmt(r['vol_z'],2)}  |  ATR: {fmt(r['atr'],5)} ({fmt(r['atr_pct'],2)}%)  |  Funding: {r['funding']}\n"
        f"AI Score: {r['score']}%\n"
        f"🔎 Lý do: {r['why']}"
    )
    await ctx.application.bot.send_message(chat_id=chat_id, text=msg)
    _last_alert_ts[base] = now
    _alert_sent_window.append(now)

# ================== Scan job ==================
async def scan_job(ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = ctx.job.data["chat_id"]
    bases = fetch_top_bases()
    for base in bases:
        try:
            r = analyze(base, TIMEFRAME_DEFAULT)
            agree, marks = multi_tf_alignment(base, r["side"], TF_SET)
            if agree >= TF_ALIGN_REQ:
                await maybe_alert(ctx, chat_id, base, r, marks)
        except Exception as e:
            # im lặng để tránh spam log
            continue

# ================== Telegram Commands ==================
HELP = (
    "Lệnh:\n"
    "• Gõ coin: `btc` | `sol 5m` | `op 1h`\n"
    "• /status  — xem cấu hình & trạng thái\n"
    "• /auto on|off  — bật/tắt quét tự động\n"
    "• /set threshold <N>  — đổi ngưỡng AI Score\n"
    "• /scan  — quét ngay một vòng (thay vì đợi chu kỳ)"
)

async def cmd_start(upd, ctx):
    await upd.message.reply_text("🤖 Bot AI Futures đã sẵn sàng.\n" + HELP)
    # gắn job scan cho chat này nếu đang bật auto
    if AUTO_SCAN:
        ctx.job_queue.run_repeating(
            scan_job, interval=SCAN_INTERVAL_SEC, first=5,
            data={"chat_id": upd.effective_chat.id}
        )

async def cmd_status(upd, ctx):
    msg = (
        f"Chế độ: {'Auto-scan ✅' if AUTO_SCAN else 'Thủ công'}\n"
        f"TF mặc định: {TIMEFRAME_DEFAULT}\n"
        f"TF đồng thuận: {', '.join(TF_SET)}  (yêu cầu ≥ {TF_ALIGN_REQ})\n"
        f"Ngưỡng AI Score: {ALERT_THRESHOLD}\n"
        f"Min VolZ: {MIN_VOLZ} | Min ATR%: {MIN_ATR_PCT}\n"
        f"Funding max: ±{FUNDING_MAX*100:.2f}%\n"
        f"Chu kỳ quét: {SCAN_INTERVAL_SEC}s | TopLimit: {TOP_LIMIT}\n"
        f"AUTO_LEARN_MARKET: {'On' if AUTO_LEARN_MARKET else 'Off'}\n"
        f"Signals/h max: {MAX_SIGNALS_PER_HR}\n"
        f"Sàn: {EXCHANGE} {MARKET_TYPE.upper()} {QUOTE}"
    )
    await upd.message.reply_text(msg)

async def cmd_auto(upd, ctx):
    global AUTO_SCAN
    parts = (upd.message.text or "").strip().split()
    if len(parts) >= 2 and parts[1].lower() in ("on","off"):
        AUTO_SCAN = (parts[1].lower() == "on")
        await upd.message.reply_text(f"AUTO_SCAN = {AUTO_SCAN}")
        if AUTO_SCAN:
            ctx.job_queue.run_repeating(
                scan_job, interval=SCAN_INTERVAL_SEC, first=3,
                data={"chat_id": upd.effective_chat.id}
            )
    else:
        await upd.message.reply_text("Dùng: /auto on | /auto off")

async def cmd_set(upd, ctx):
    global ALERT_THRESHOLD
    parts = (upd.message.text or "").strip().split()
    if len(parts) == 3 and parts[1].lower() == "threshold":
        try:
            ALERT_THRESHOLD = int(parts[2])
            await upd.message.reply_text(f"ALERT_THRESHOLD = {ALERT_THRESHOLD}")
        except:
            await upd.message.reply_text("Giá trị không hợp lệ.")
    else:
        await upd.message.reply_text("Dùng: /set threshold 80")

async def cmd_scan(upd, ctx):
    await upd.message.reply_text("Đang quét…")
    await scan_job(ctx)

# --------- text handler (manual analysis) ----------
async def handle_text(upd, ctx):
    text = (upd.message.text or "").strip()
    try:
        base, tf = parse_symbol_tf(text, TIMEFRAME_DEFAULT)
        r = analyze(base, tf)
        agree, marks = multi_tf_alignment(base, r["side"], TF_SET)
        msg = (
            f"📊 {r['base']} — TF: {r['tf']} | Multi-TF: {'  '.join(marks)}\n"
            f"Hướng: {r['side']}\n"
            f"Entry: {fmt(r['price'])}\n"
            f"TP1/TP2: {fmt(r['tp1'])} / {fmt(r['tp2'])}  |  SL: {fmt(r['sl'])}\n"
            f"EMA12/26: {fmt(r['ema12'])}/{fmt(r['ema26'])}\n"
            f"RSI: {fmt(r['rsi'],2)} | MACD_hist: {fmt(r['macd_hist'],5)}\n"
            f"VolZ: {fmt(r['vol_z'],2)} | ATR: {fmt(r['atr'],5)} ({fmt(r['atr_pct'],2)}%)\n"
            f"Funding: {r['funding']}\n"
            f"AI Score: {r['score']}%  —  Lý do: {r['why']}\n"
            f"({agree}/{len(TF_SET)} TF đồng thuận)"
        )
        await upd.message.reply_text(msg)
    except Exception as e:
        await upd.message.reply_text(f"⚠️ Lỗi: {e}")

# ================== App start ==================
def main():
    start_keep_alive()

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN trong Environment.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("auto",    cmd_auto))
    app.add_handler(CommandHandler("set",     cmd_set))
    app.add_handler(CommandHandler("scan",    cmd_scan))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot starting…")
    app.run_polling(allowed_updates=None)

if __name__ == "__main__":
    main()
