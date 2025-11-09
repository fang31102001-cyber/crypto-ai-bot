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
SCAN_INTERVAL_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "3600"))   # quét mỗi 1h
ALERT_THRESHOLD    = int(os.getenv("ALERT_THRESHOLD", "80"))       # AI score min
MIN_VOLZ           = float(os.getenv("MIN_VOLZ", "2"))
MIN_ATR_PCT        = float(os.getenv("MIN_ATR_PCT", "0.25"))
FUNDING_MAX        = float(os.getenv("FUNDING_MAX", "0.02"))
TF_ALIGN_REQ       = int(os.getenv("TF_ALIGN", "2"))
TF_SET             = [x.strip() for x in os.getenv("TF_SET", "5m,15m,1h").split(",") if x.strip()]
MAX_SIGNALS_PER_HR = int(os.getenv("MAX_SIGNALS_PER_HOUR", "5"))

# Universe
EXCHANGE           = os.getenv("EXCHANGE", "MEXC")
MARKET_TYPE        = os.getenv("MARKET_TYPE", "swap")
QUOTE              = os.getenv("QUOTE", "USDT").upper()
TOP_LIMIT          = int(os.getenv("TOP_LIMIT", "20"))

# AI tự học thị trường (auto-label)
AUTO_LEARN_MARKET  = os.getenv("AUTO_LEARN_MARKET", "true").lower() == "true"
LABEL_TP_PCT       = float(os.getenv("LABEL_TP_PCT", "0.004"))
LABEL_SL_PCT       = float(os.getenv("LABEL_SL_PCT", "0.004"))

# Storage
DATA_DIR    = "data"
MEMO_PATH   = os.path.join(DATA_DIR, "memory.json")
PENDING_PATH= os.path.join(DATA_DIR, "pending.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Tự tạo file memory.json nếu chưa tồn tại để tránh lỗi khi sync / khi AI load
if not os.path.exists(MEMO_PATH):
    with open(MEMO_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "w": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "lr": 0.03,
                "memory": []
            },
            f,
            indent=2
        )

# ================== KEEP ALIVE ==================
def start_keep_alive():
    try:
        from keep_alive import run_server
        Thread(target=run_server, daemon=True).start()
    except:
        pass
# ================== Helper & Indicators ==================
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

# ================== CCXT MEXC (Futures/Swap) ==================
def mexc_client():
    return ccxt.mexc({
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
    m = f - s; sig = ema(m, signal); h = m - sig
    return m, sig, h
def atr(df, n=14):
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ================== Candlestick patterns ==================
def detect_pattern(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    o,h,l,c = last["open"], last["high"], last["low"], last["close"]
    op,cp = prev["open"], prev["close"]

    body = abs(c - o)
    wick = h - l
    upwick = h - max(c,o)
    lowwick = min(c,o) - l

    if body < wick * 0.3 and c > o: return "Hammer 🟢"
    if body < wick * 0.3 and c < o: return "Inverted Hammer 🔴"
    if c > op and o < cp and (c - o) > body * 0.8: return "Bullish Engulfing 💚"
    if c < op and o > cp and (o - c) > body * 0.8: return "Bearish Engulfing ❤️"
    if abs(c - o) < body * 0.05: return "Doji ⚪"
    return "-"

# ================== Enrich Data ==================
def fetch_ohlcv(base, tf, limit=300):
    ex = mexc_client()
    sym = symbol_usdt_perp(base)
    data = ex.fetch_ohlcv(sym, timeframe=tf, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def enrich(df):
    d = df.copy()
    d["ema12"] = ema(d["close"],12)
    d["ema26"] = ema(d["close"],26)
    m,s,h = macd(d["close"]); d["macd"],d["macd_sig"],d["macd_hist"] = m,s,h
    d["rsi"] = rsi(d["close"])
    d["atr"] = atr(d)
    v = d["volume"]
    d["vol_z"] = (v - v.rolling(50).mean()) / (v.rolling(50).std() + 1e-9)
    d["pattern"] = d.apply(lambda row: "-", axis=1)
    if len(d) > 2:
        d.loc[d.index[-1], "pattern"] = detect_pattern(d)
    return d.dropna()

# ================== Online AI ==================
class OnlineAI:
    def __init__(self, path):
        self.path = path
        self.w = np.zeros(6, dtype=float)
        self.lr = 0.03
        self._load()
    def _load(self):
        if os.path.exists(self.path):
            try:
                obj = json.load(open(self.path, "r"))
                self.w = np.array(obj.get("w", self.w.tolist()), dtype=float)
                self.lr = float(obj.get("lr", self.lr))
            except: pass
    def _save(self):
        json.dump({"w": self.w.tolist(),"lr":self.lr}, open(self.path,"w"))
    def _feat(self, row):
        trend = (row["ema12"] - row["ema26"]) / (abs(row["ema26"])+1e-9)
        macd_h = row["macd_hist"]; rsi_c = (row["rsi"]-50)/50.0
        volz = np.tanh(row["vol_z"]/3.0); atrp = row["atr"]/max(row["close"],1e-9)
        bias = 1.0
        return np.array([trend,macd_h,rsi_c,volz,atrp,bias])
    def score(self,row):
        x=self._feat(row); z=float(np.dot(self.w,x)); p=1/(1+math.exp(-z))
        return int(round(p*100))
    def learn(self,row,label:int):
        x=self._feat(row); z=float(np.dot(self.w,x)); p=1/(1+math.exp(-z))
        grad=(p-label)*x; self.w-=self.lr*grad; self._save()

AI = OnlineAI(MEMO_PATH)
# ================== PHÂN TÍCH & HỌC ==================
def make_targets(entry, atrv, side):
    tp1 = entry + (1.5 * atrv if side == "LONG" else -1.5 * atrv)
    tp2 = entry + (2.5 * atrv if side == "LONG" else -2.5 * atrv)
    sl  = entry - (1.0 * atrv if side == "LONG" else -1.0 * atrv)
    return tp1, tp2, sl

def analyze(base, tf):
    df = enrich(fetch_ohlcv(base, tf, limit=300))
    row = df.iloc[-1].to_dict()
    side = "LONG" if (row["ema12"] > row["ema26"] and row["macd_hist"] > 0 and row["rsi"] > 48) else "SHORT"
    score = AI.score(row)
    tp1,tp2,sl = make_targets(row["close"], row["atr"], side)
    pattern = row.get("pattern","-")

    return {
        "base": base.upper(), "tf": tf, "side": side, "price": row["close"],
        "tp1": tp1, "tp2": tp2, "sl": sl, "score": score,
        "ema12": row["ema12"], "ema26": row["ema26"], "rsi": row["rsi"],
        "macd_hist": row["macd_hist"], "vol_z": row["vol_z"], "atr": row["atr"],
        "pattern": pattern
    }

# ================== TELEGRAM BOT ==================
async def cmd_start(update, ctx):
    await update.message.reply_text(
        "🤖 Bot AI Futures đã sẵn sàng!\n"
        "• Gõ coin: `btc` hoặc `sol 15m`\n"
        "• Bot tự động gửi tín hiệu mỗi 1h khi có sóng mạnh 📈📉\n"
        "• AI học từ thị trường thực tế, phân tích mô hình nến, dòng tiền, RSI, MACD, Volume\n"
        "• Khi có tín hiệu mạnh: Bot gửi ngay"
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

# ================== TỰ ĐỘNG GỬI TÍN HIỆU MẠNH ==================
async def auto_scan(ctx):
    chat_id = ctx.job.data["chat_id"]
    top = ["BTC","ETH","SOL","WLD","XRP","TON","ARB","LINK","PEPE","SUI"]
    for coin in top:
        try:
            r = analyze(coin, TIMEFRAME_DEFAULT)
            if r["score"] >= 80 and r["vol_z"] > 2:
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
        except Exception:
            continue

# ================== KHỞI CHẠY BOT ==================
def main():
    start_keep_alive()

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN trong Environment Variables.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # chạy tự động đúng mỗi giờ (00 phút)
    from datetime import time
    for hour in range(24):
        app.job_queue.run_daily(auto_scan, time=time(hour, 0), data={"chat_id": 7992112548})

    print("🤖 Bot đang chạy và quét đúng mỗi giờ (00 phút)...")
    app.run_polling(allowed_updates=None)

if __name__ == "__main__":
    main()
# ========== AI MEMORY SYNC (Google Drive) ==========
import json, threading, io
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

def google_creds():
    """Khởi tạo thông tin xác thực Google Drive"""
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

def sync_ai_memory_to_drive():
    """Đồng bộ file AI_memory.json lên Google Drive"""
    try:
        creds = google_creds()
        service = build("drive", "v3", credentials=creds)

        data = {
            "updated": datetime.utcnow().isoformat(),
            "learning": {
                "trend_model": "EMA+RSI+Volume",
                "last_signal": "Short OP 15m",
                "ai_score": "tăng độ chính xác",
            },
        }

        with open("AI_memory.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        media = MediaFileUpload("AI_memory.json", mimetype="application/json")
        resp = service.files().list(q="name='AI_memory.json'", spaces="drive").execute()

        if len(resp.get("files", [])) > 0:
            file_id = resp["files"][0]["id"]
            service.files().update(fileId=file_id, media_body=media).execute()
            print("✅ Đã cập nhật AI_memory.json lên Google Drive.")
        else:
            meta = {"name": "AI_memory.json"}
            service.files().create(body=meta, media_body=media, fields="id").execute()
            print("✅ Đã tạo file AI_memory.json mới trên Google Drive.")
    except Exception as e:
        print("⚠️ Drive Sync Error:", e)

def load_ai_memory_from_drive():
    """Tải lại dữ liệu AI_memory.json từ Google Drive khi bot khởi động"""
    try:
        creds = google_creds()
        service = build("drive", "v3", credentials=creds)
        results = service.files().list(q="name='AI_memory.json'", spaces="drive").execute()
        files = results.get("files", [])
        if not files:
            print("⚠️ Không tìm thấy file AI_memory.json trên Google Drive.")
            return None

        file_id = files[0]["id"]
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        data = json.load(fh)
        print("✅ Đã tải AI_memory.json từ Google Drive.")
        return data
    except Exception as e:
        print("⚠️ Lỗi khi tải AI_memory.json:", e)
        return None

def auto_backup_loop(interval_hours=3):
    """Tự động đồng bộ trí nhớ AI lên Drive định kỳ"""
    def loop():
        while True:
            try:
                sync_ai_memory_to_drive()
                print(f"🕒 Tự động đồng bộ Drive hoàn tất ({datetime.now().strftime('%H:%M:%S')})")
            except Exception as e:
                print("⚠️ Lỗi auto backup:", e)
            time.sleep(interval_hours * 3600)

    threading.Thread(target=loop, daemon=True).start()

# 🔹 Gọi song song khi bot chạy
threading.Thread(target=sync_ai_memory_to_drive, daemon=True).start()
auto_backup_loop(3)

# ✅ Kiểm tra trí nhớ cũ nếu có
ai_memory = load_ai_memory_from_drive()
if ai_memory:
    print("🧠 Trí nhớ AI trước đó:", ai_memory.get("learning", {}))
else:
    print("🧠 Không có trí nhớ cũ — bắt đầu mới.")
