# main.py
import os, math, json, time, random, logging
from threading import Thread
from datetime import datetime, timezone, timedelta
from typing import Tuple

import ccxt
import pandas as pd
import numpy as np
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

CHAT_ID            = int(os.getenv("CHAT_ID", "7992112548"))

DATA_DIR     = "data"
MEMO_PATH    = os.path.join(DATA_DIR, "memory.json")
PENDING_PATH = os.path.join(DATA_DIR, "pending.json")
os.makedirs(DATA_DIR, exist_ok=True)

# create memory.json if missing
if not os.path.exists(MEMO_PATH):
    with open(MEMO_PATH, "w", encoding="utf-8") as f:
        json.dump({"w": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "lr": 0.03, "memory": []}, f, indent=2)

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

# ================== CCXT (MEXC Swap) ==================
EX = ccxt.mexc({
    "options": {"defaultType": MARKET_TYPE},
    "enableRateLimit": True,
})

def symbol_usdt_perp(base: str) -> str:
    return f"{base.upper()}/{QUOTE}:{QUOTE}"

# ================== Indicators ==================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

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

# ================== Online AI ==================
class OnlineAI:
    def __init__(self, path: str):
        self.path = path
        self.w = np.zeros(6, dtype=float)
        self.lr = 0.03
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                self.w = np.array(obj.get("w", self.w.tolist()), dtype=float)
                self.lr = float(obj.get("lr", self.lr))
            except Exception:
                pass

    def _save(self):
        obj = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                obj = {}
        obj["w"] = self.w.tolist()
        obj["lr"] = float(self.lr)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def _feat(self, row: dict) -> np.ndarray:
        trend = (row["ema12"] - row["ema26"]) / (abs(row["ema26"]) + 1e-9)
        macd_h = float(row["macd_hist"])
        rsi_c = (float(row["rsi"]) - 50.0) / 50.0
        volz = float(np.tanh(float(row["vol_z"]) / 3.0))
        atrp = float(row["atr"]) / max(float(row["close"]), 1e-9)
        bias = 1.0
        return np.array([trend, macd_h, rsi_c, volz, atrp, bias], dtype=float)

    def _sigmoid(self, z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def score(self, row: dict) -> int:
        x = self._feat(row)
        z = float(np.dot(self.w, x))
        p = self._sigmoid(z)
        return int(round(p * 100))

    def learn(self, row: dict, label: int):
        x = self._feat(row)
        z = float(np.dot(self.w, x))
        p = self._sigmoid(z)
        grad = (p - label) * x
        self.w -= self.lr * grad
        self._save()

AI = OnlineAI(MEMO_PATH)

# ================== Analysis ==================
def make_targets(entry: float, atrv: float, side: str):
    tp1 = entry + (1.5 * atrv if side == "LONG" else -1.5 * atrv)
    tp2 = entry + (2.5 * atrv if side == "LONG" else -2.5 * atrv)
    sl  = entry - (1.0 * atrv if side == "LONG" else -1.0 * atrv)
    return tp1, tp2, sl

def analyze(base: str, tf: str) -> dict:
    df = enrich(fetch_ohlcv(base, tf, limit=300))
    row = df.iloc[-1].to_dict()

    side = "LONG" if (row["ema12"] > row["ema26"] and row["macd_hist"] > 0 and row["rsi"] > 48) else "SHORT"
    score = AI.score(row)
    tp1, tp2, sl = make_targets(float(row["close"]), float(row["atr"]), side)
    pattern = row.get("pattern", "-")

    return {
        "base": base.upper(), "tf": tf, "side": side, "price": float(row["close"]),
        "tp1": float(tp1), "tp2": float(tp2), "sl": float(sl), "score": int(score),
        "ema12": float(row["ema12"]), "ema26": float(row["ema26"]), "rsi": float(row["rsi"]),
        "macd_hist": float(row["macd_hist"]), "vol_z": float(row["vol_z"]), "atr": float(row["atr"]),
        "pattern": pattern
    }

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
        base, tf = parse_symbol_tf(text, TIMEFRAME_DEFAULT)
        r = analyze(base, tf)
        msg = (
            f"📊 {r['base']} ({r['tf']}) — {r['side']}\n"
            f"Entry: {fmt(r['price'])}\n"
            f"TP1: {fmt(r['tp1'])} | TP2: {fmt(r['tp2'])} | SL: {fmt(r['sl'])}\n"
            f"RSI: {fmt(r['rsi'],2)} | MACD_hist: {fmt(r['macd_hist'],5)}\n"
            f"EMA12/26: {fmt(r['ema12'])}/{fmt(r['ema26'])}\n"
            f"VolZ: {fmt(r['vol_z'],2)} | ATR: {fmt(r['atr'],4)}\n"
            f"Mô hình nến: {r['pattern']}\n"
            f"AI Score: {r['score']}%"
        )
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Lỗi phân tích: {e}")

# ================== Auto scan ==================
async def auto_scan(ctx):
    chat_id = int(ctx.job.data["chat_id"])
    top = ["BTC","ETH","SOL","WLD","XRP","TON","ARB","LINK","PEPE","SUI"]
    for coin in top:
        try:
            r = analyze(coin, TIMEFRAME_DEFAULT)
            if r["score"] >= ALERT_THRESHOLD and r["vol_z"] > MIN_VOLZ:
                msg = (
                    f"🔥 Tín hiệu mạnh — {r['base']}/USDT ({r['tf']})\n"
                    f"Hướng: {r['side']} | Giá: {fmt(r['price'])}\n"
                    f"TP1/TP2: {fmt(r['tp1'])}/{fmt(r['tp2'])} | SL: {fmt(r['sl'])}\n"
                    f"RSI: {fmt(r['rsi'],2)} | MACD_hist: {fmt(r['macd_hist'],5)}\n"
                    f"EMA12/26: {fmt(r['ema12'])}/{fmt(r['ema26'])}\n"
                    f"VolZ: {fmt(r['vol_z'],2)} | ATR: {fmt(r['atr'],5)}\n"
                    f"Nến: {r['pattern']} | AI Score: {r['score']}%"
                )
                await ctx.application.bot.send_message(chat_id=chat_id, text=msg)
        except Exception as e:
            log.warning("scan fail %s: %r", coin, e)

# ================== Drive Sync (tự tắt nếu invalid_grant) ==================
import io, threading
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

_drive_enabled = True  # sẽ tắt nếu invalid_grant

def _has_drive_env() -> bool:
    return all([
        os.getenv("GOOGLE_REFRESH_TOKEN"),
        os.getenv("GOOGLE_CLIENT_ID"),
        os.getenv("GOOGLE_CLIENT_SECRET"),
    ])

def google_creds():
    return Credentials(
        None,
        refresh_token=os.getenv("GOOGLE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=[
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive.metadata.readonly",
        ],
    )

def _is_invalid_grant(err: Exception) -> bool:
    s = repr(err).lower()
    return "invalid_grant" in s or "bad request" in s

def sync_ai_memory_to_drive():
    global _drive_enabled
    if not _drive_enabled:
        return
    try:
        if not os.path.exists(MEMO_PATH):
            return

        with open(MEMO_PATH, "r", encoding="utf-8") as f:
            memory_data = json.load(f)

        memory_data["_last_synced_utc"] = datetime.now(timezone.utc).isoformat()

        with open("AI_memory.json", "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)

        creds = google_creds()
        service = build("drive", "v3", credentials=creds)

        media = MediaFileUpload("AI_memory.json", mimetype="application/json")
        resp = service.files().list(q="name='AI_memory.json'", spaces="drive").execute()

        if resp.get("files"):
            file_id = resp["files"][0]["id"]
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            meta = {"name": "AI_memory.json"}
            service.files().create(body=meta, media_body=media, fields="id").execute()

    except Exception as e:
        if _is_invalid_grant(e):
            _drive_enabled = False
            print("⛔ Google Drive invalid_grant -> TẮT sync Drive (cần cấp lại refresh token).", flush=True)
        else:
            print("⚠️ Drive Sync Error:", repr(e), flush=True)

def load_ai_memory_from_drive():
    global _drive_enabled
    if not _drive_enabled:
        return None
    try:
        creds = google_creds()
        service = build("drive", "v3", credentials=creds)
        results = service.files().list(q="name='AI_memory.json'", spaces="drive").execute()
        files = results.get("files", [])
        if not files:
            return None

        file_id = files[0]["id"]
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        fh.seek(0)
        data = json.load(fh)

        with open(MEMO_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if "w" in data:
            AI.w = np.array(data["w"], dtype=float)

        print("✅ Đã tải AI_memory.json từ Drive.", flush=True)
        return data

    except Exception as e:
        if _is_invalid_grant(e):
            _drive_enabled = False
            print("⛔ Google Drive invalid_grant -> TẮT sync Drive (cần cấp lại refresh token).", flush=True)
        else:
            print("⚠️ Lỗi khi tải AI_memory.json:", repr(e), flush=True)
        return None

def auto_backup_loop(interval_hours=3):
    def loop():
        while True:
            try:
                sync_ai_memory_to_drive()
                if _drive_enabled:
                    print(f"🕒 Drive sync ({datetime.now().strftime('%H:%M:%S')})", flush=True)
            except Exception as e:
                print("⚠️ Lỗi auto backup:", repr(e), flush=True)
            time.sleep(interval_hours * 3600)
    threading.Thread(target=loop, daemon=True).start()

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

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN trong Environment Variables.")

    if _has_drive_env():
        load_ai_memory_from_drive()
        auto_backup_loop(3)
    else:
        print("ℹ️ Thiếu Google Drive ENV -> bỏ qua sync.", flush=True)

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
