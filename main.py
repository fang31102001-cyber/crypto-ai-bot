# ================== main_v2_7_mexc_ai_deepflow_validator.py ==================
import os, math, asyncio, json, time, io, threading
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# ========== ENV ==============
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
QUOTE = "USDT"
MARKET_TYPE = "swap"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

MEMO_PATH = os.path.join(DATA_DIR, "memory.json")
AI_MEMORY_PATH = "AI_memory.json"

# ========== CLIENT ============
def mexc_client():
    return ccxt.mexc({
        "options": {"defaultType": MARKET_TYPE},
        "enableRateLimit": True
    })

def symbol(base): return f"{base}/{QUOTE}:{QUOTE}"

# ========== INDICATORS =========
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(close, n=14):
    d = close.diff()
    up, down = d.clip(lower=0), (-d).clip(lower=0)
    rs = up.rolling(n).mean() / (down.rolling(n).mean() + 1e-9)
    return 100 - (100 / (1 + rs))
def macd(close, fast=12, slow=26, signal=9):
    f, s = ema(close, fast), ema(close, slow)
    m = f - s; sig = ema(m, signal)
    return m, sig, m - sig
def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_close = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_close).abs(), (l-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def detect_pattern(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    o, h, l, c = last["open"], last["high"], last["low"], last["close"]
    op, cp = prev["open"], prev["close"]
    body = abs(c - o)
    if abs(c - o) < (h - l) * 0.05: return "Doji"
    if c > op and o < cp: return "Bullish Engulfing"
    if c < op and o > cp: return "Bearish Engulfing"
    if c > o and (h - c) < (c - l) * 0.3: return "Hammer"
    if c < o and (c - l) < (h - c) * 0.3: return "Inverted Hammer"
    return "-"

# ========== ENRICH ===========
def fetch_ohlcv(base, tf="15m", limit=300):
    ex = mexc_client()
    data = ex.fetch_ohlcv(symbol(base), tf, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def enrich(df):
    d = df.copy()
    d["ema12"], d["ema26"] = ema(d["close"],12), ema(d["close"],26)
    m, s, h = macd(d["close"]); d["macd_hist"] = h
    d["rsi"] = rsi(d["close"])
    d["atr"] = atr(d)
    d["vol_z"] = (d["volume"] - d["volume"].rolling(50).mean()) / (d["volume"].rolling(50).std() + 1e-9)
    if len(d) > 2: d.loc[d.index[-1], "pattern"] = detect_pattern(d)
    return d.dropna()

# ========== AI CORE ===========
class OnlineAI:
    def __init__(self, path):
        self.path = path
        self.w = np.zeros(6)
        self.lr = 0.03
        self.memory = []
        self._load()
    def _load(self):
        if os.path.exists(self.path):
            try:
                obj = json.load(open(self.path))
                self.w = np.array(obj.get("w", self.w.tolist()))
                self.memory = obj.get("memory", [])
            except: pass
    def _save(self):
        json.dump({"w": self.w.tolist(), "memory": self.memory[-500:]}, open(self.path,"w"))
    def feat(self, row):
        trend = (row["ema12"] - row["ema26"]) / (abs(row["ema26"])+1e-9)
        macd_h = row["macd_hist"]; rsi_c = (row["rsi"]-50)/50
        volz = np.tanh(row["vol_z"]/3); atrp = row["atr"]/max(row["close"],1)
        return np.array([trend, macd_h, rsi_c, volz, atrp, 1.0])
    def score(self, row):
        x=self.feat(row); z=float(np.dot(self.w,x)); p=1/(1+math.exp(-z))
        return int(round(p*100))
    def learn(self,row,label:int):
        x=self.feat(row); z=float(np.dot(self.w,x)); p=1/(1+math.exp(-z))
        grad=(p-label)*x; self.w-=self.lr*grad
        self.memory.append({"t":time.time(),"label":label,"score":p})
        self._save()
    def auto_validate(self,row,side,old_price):
        if not old_price: return
        diff=(row["close"]-old_price)/old_price
        if side=="LONG": label=1 if diff>0.002 else 0
        else: label=1 if diff<-0.002 else 0
        self.learn(row,label)

AI=OnlineAI(MEMO_PATH)

# ========== ANALYZE ===========
def analyze(base, tf="15m", old_price=None, old_side=None):
    df=enrich(fetch_ohlcv(base,tf))
    row=df.iloc[-1].to_dict()
    side="LONG" if (row["ema12"]>row["ema26"] and row["macd_hist"]>0 and row["rsi"]>48) else "SHORT"
    score=AI.score(row)
    AI.auto_validate(row,old_side,old_price)
    atrv=row["atr"]; entry=row["close"]
    tp1=entry+(1.5*atrv if side=="LONG" else -1.5*atrv)
    tp2=entry+(2.5*atrv if side=="LONG" else -2.5*atrv)
    sl=entry-(1.0*atrv if side=="LONG" else -1.0*atrv)
    return {"base":base,"side":side,"price":entry,"tp1":tp1,"tp2":tp2,"sl":sl,
            "rsi":row["rsi"],"macd":row["macd_hist"],"ema12":row["ema12"],"ema26":row["ema26"],
            "volz":row["vol_z"],"atr":row["atr"],"pattern":row.get("pattern","-"),"score":score}

# ========== TELEGRAM ==========
async def cmd_start(update,ctx):
    await update.message.reply_text("Bot AI DeepFlow v2.7 đã sẵn sàng — tự học, tự xác minh tín hiệu.\nNhập coin ví dụ: `btc` hoặc `sol 15m`")

async def handle_text(update,ctx):
    txt=(update.message.text or "").strip()
    parts=txt.split()
    base=parts[0].upper(); tf=parts[1] if len(parts)>1 else "15m"
    r=analyze(base,tf)
    msg=(f"{r['base']} ({tf}) - {r['side']}\n"
         f"Entry: {r['price']:.4f}\nTP1: {r['tp1']:.4f} | TP2: {r['tp2']:.4f} | SL: {r['sl']:.4f}\n"
         f"RSI: {r['rsi']:.2f} | MACD_hist: {r['macd']:.5f}\nEMA12/26: {r['ema12']:.2f}/{r['ema26']:.2f}\n"
         f"VolZ: {r['volz']:.2f} | ATR: {r['atr']:.5f}\n"
         f"Pattern: {r['pattern']}\nAI Score: {r['score']}%")
    await update.message.reply_text(msg)

async def auto_scan(ctx):
    chat_id=ctx.job.data["chat_id"]
    coins=["BTC","ETH","SOL","XRP","LINK","TON","ARB","OP","SUI","WLD","MATIC","DOGE","AVAX","ADA","APT","SEI","NEAR","TRX","ATOM","PEPE"]
    for c in coins:
        try:
            r=analyze(c,"15m")
            if r["score"]>=80 and r["volz"]>2:
                msg=(f"{r['base']} ({r['side']}) | Giá {r['price']:.2f}\n"
                     f"TP1/TP2: {r['tp1']:.2f}/{r['tp2']:.2f} | SL: {r['sl']:.2f}\n"
                     f"RSI {r['rsi']:.1f} | MACD_hist {r['macd']:.5f}\n"
                     f"VolZ {r['volz']:.2f} | ATR {r['atr']:.5f}\n"
                     f"Nến: {r['pattern']} | AI Score {r['score']}%")
                await ctx.application.bot.send_message(chat_id=chat_id,text=msg)
        except: continue

# ========== GOOGLE DRIVE ==========
def google_creds():
    return Credentials(None,
        refresh_token=os.getenv("GOOGLE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=["https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive.metadata.readonly"])

def sync_ai_memory_to_drive():
    try:
        creds=google_creds(); service=build("drive","v3",credentials=creds)
        media=MediaFileUpload(MEMO_PATH,mimetype="application/json")
        resp=service.files().list(q="name='AI_memory.json'",spaces="drive").execute()
        if len(resp.get("files",[]))>0:
            file_id=resp["files"][0]["id"]; service.files().update(fileId=file_id,media_body=media).execute()
            print("Đã cập nhật AI_memory.json lên Drive.")
        else:
            service.files().create(body={"name":"AI_memory.json"},media_body=media,fields="id").execute()
            print("Tạo mới AI_memory.json trên Drive.")
    except Exception as e:
        print("Drive Sync Error:",e)

def auto_backup_loop():
    def loop():
        while True:
            sync_ai_memory_to_drive()
            time.sleep(3600)
    threading.Thread(target=loop,daemon=True).start()

# ========== MAIN ==========
def main():
    from keep_alive import run_server
    threading.Thread(target=run_server,daemon=True).start()

    app=ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,handle_text))

    from datetime import time as t
    for h in range(0,24,0):
        app.job_queue.run_repeating(auto_scan,interval=1800,first=10,data={"chat_id":7992112548})
    auto_backup_loop()
    print("Bot DeepFlow v2.7 đang chạy (quét mỗi 30 phút)...")
    app.run_polling(allowed_updates=None)

if __name__=="__main__":
    main()
