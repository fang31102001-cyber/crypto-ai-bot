import os, csv, math, time, json, asyncio, logging, requests
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timezone
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from config import *

logging.basicConfig(level=logging.INFO)
os.makedirs(DATA_DIR, exist_ok=True)

# ===== MEXC Futures helpers =====
class MexcF:
    def __init__(self):
        self.ex = ccxt.mexc({'options': {'defaultType':'swap'}, 'enableRateLimit':True})
        self.ex.load_markets()
    def ensure_symbol(self, base):
        for s in (f"{base}/USDT:USDT", f"{base}/USDT"):
            if s in self.ex.markets: return s
        target = base+"USDT"
        for m in self.ex.markets:
            if target == m.replace("/","").replace(":USDT",""): return m
        raise ValueError(f"{base} không có trên MEXC Futures")
    def ohlcv(self, base, tf, limit=400):
        sym = self.ensure_symbol(base)
        rows = self.ex.fetch_ohlcv(sym, timeframe=tf, limit=limit)
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["ts"]=pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df
    def trades(self, base, limit=200):
        try:
            sym = self.ensure_symbol(base)
            return self.ex.fetch_trades(sym, limit=limit)
        except: return []
    def orderbook_imbalance(self, base, depth=10):
        try:
            sym = self.ensure_symbol(base)
            ob = self.ex.fetch_order_book(sym, depth)
            b=sum(x[1] for x in ob.get("bids",[])[:depth]); a=sum(x[1] for x in ob.get("asks",[])[:depth])
            d=(b+a) or 1e-9; return (b-a)/d
        except: return 0.0
    def funding(self, base):
        try:
            sym = self.ensure_symbol(base)
            r = self.ex.fetch_funding_rate(sym)
            fr = r.get("fundingRate", r.get("info",{}).get("fundingRate"))
            return float(fr) if fr is not None else None
        except: return None

mexcf = MexcF()

# ===== Indicators =====
def ema(s,n): return s.ewm(span=n, adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=(-d).clip(lower=0)
    rs=up.rolling(n).mean()/(dn.rolling(n).mean()+1e-9); return 100-(100/(1+rs))
def macd(close,fast=12,slow=26,sig=9):
    f,s=ema(close,fast),ema(close,slow); m=f-s; sg=ema(m,sig); h=m-sg; return m,sg,h
def atr(df,n=14):
    h,l,c=df["high"],df["low"],df["close"]; pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); return tr.rolling(n).mean()
def enrich(df):
    df=df.copy()
    df["ema12"]=ema(df["close"],12); df["ema26"]=ema(df["close"],26)
    m,sg,h=macd(df["close"]); df["macd"],df["macd_sig"],df["macd_hist"]=m,sg,h
    df["rsi"]=rsi(df["close"]); df["atr"]=atr(df)
    v=df["volume"]; df["vol_z"]=(v-v.rolling(50).mean())/(v.rolling(50).std()+1e-9)
    return df.dropna()

# ===== Online AI (nhẹ, tự học) =====
class OnlineAI:
    def __init__(self, memo_path):
        self.path=memo_path; self.lr=0.03
        self.state={"w":{}, "lr":self.lr}  # per-coin weights
        self._load()
    def _load(self):
        if os.path.exists(self.path):
            try: self.state=json.load(open(self.path))
            except: pass
    def _save(self):
        json.dump(self.state, open(self.path,"w"))
    def _feat(self,row):
        trend=(row["ema12"]-row["ema26"])/(abs(row["ema26"])+1e-9)
        macd_h=row["macd_hist"]; rsi_c=(row["rsi"]-50)/50.0
        volz=np.tanh(row["vol_z"]/3.0); atrp=row["atr"]/max(row["close"],1e-9); bias=1.0
        return np.array([trend, macd_h, rsi_c, volz, atrp, bias], dtype=float)
    def _get_w(self, base):
        w = self.state["w"].get(base, [0.0]*6)
        return np.array(w, dtype=float)
    def _set_w(self, base, w):
        self.state["w"][base]=w.tolist(); self._save()
    def score(self, base, row):
        w=self._get_w(base); x=self._feat(row); z=float(np.dot(w,x)); p=1/(1+math.exp(-z))
        return int(round(p*100)), p
    def learn(self, base, row, label:int):
        w=self._get_w(base); x=self._feat(row); z=float(np.dot(w,x)); p=1/(1+math.exp(-z))
        w = w - self.lr*((p-label)*x)
        self._set_w(base, w)

AI = OnlineAI(MEMO_PATH)

def fmt(x,nd=6):
    try: return f"{float(x):.{nd}f}"
    except: return str(x)

def make_targets(entry, atr, side):
    tp1=entry+(1.5*atr if side=="LONG" else -1.5*atr)
    tp2=entry+(2.5*atr if side=="LONG" else -2.5*atr)
    sl =entry-(1.0*atr if side=="LONG" else -1.0*atr)
    return tp1,tp2,sl

def analyze(base, tf=DEFAULT_TF):
    df=mexcf.ohlcv(base, tf, 400)
    dfe=enrich(df); row=dfe.iloc[-1].to_dict()
    score,_=AI.score(base, row)
    side="LONG" if (row["ema12"]>row["ema26"] and row["macd_hist"]>0 and row["rsi"]>48) else "SHORT"
    tp1,tp2,sl=make_targets(row["close"], row["atr"], side)

    newfile = not os.path.exists(LOG_PATH)
    with open(LOG_PATH,"a",newline="") as f:
        w=csv.writer(f)
        if newfile: w.writerow(["ts","base","tf","side","price","score","tp1","tp2","sl"])
        w.writerow([datetime.utcnow().isoformat(), base.upper(), tf, side, row["close"], score, tp1, tp2, sl])

    return {
        "base":base.upper(),"tf":tf,"side":side,"price":row["close"],
        "tp1":tp1,"tp2":tp2,"sl":sl,"score":score,
        "ema12":row["ema12"],"ema26":row["ema26"],"rsi":row["rsi"],
        "macd_hist":row["macd_hist"],"vol_z":row["vol_z"]
    }

# ===== TOP20 =====
def fetch_top_volume(limit=20):
    url="https://contract.mexc.com/api/v1/contract/ticker"
    try:
        js=requests.get(url,timeout=8).json().get("data",[])
        rows=[]
        for it in js:
            sym=str(it.get("symbol",""))
            if not sym.endswith("_USDT"): continue
            base=sym.replace("_USDT","")
            turnover=float(it.get("turnover24h",0.0))
            rows.append((base,turnover))
        rows.sort(key=lambda x: x[1], reverse=True)
        return [b for b,_ in rows[:limit]]
    except:
        return ["BTC","ETH","SOL","XRP","DOGE","TON","ADA","AVAX","LINK","ARB",
                "OP","APT","SUI","PEPE","SEI","FIL","FTM","MANTA","WLD","LDO"][:limit]

def coin_list():
    v=COIN_LIST.strip().upper()
    if v=="TOP20": return fetch_top_volume(20)
    if v=="ALL":
        bases=set()
        for m,info in mexcf.ex.markets.items():
            if info.get("type")=="swap" and str(info.get("quote","")).upper()=="USDT":
                b=str(info.get("base","")).upper()
                if b: bases.add(b)
        return sorted(bases)
    return [x.strip().upper() for x in COIN_LIST.split(",") if x.strip()]

COINS = coin_list()

# ===== Telegram =====
HELP=("Gõ coin: `wld` | `btc 5m`\nGõ `learn win` hoặc `learn loss` khi lệnh rõ kết quả để AI tự học.\n")
user_tf={}
last_alert={}
ALERT_COOLDOWN=300
alert_chat_id=None

async def cmd_start(update, ctx):
    global alert_chat_id
    alert_chat_id = update.effective_chat.id
    await update.message.reply_text("✅ Bot AI đã sẵn sàng.\n"+HELP, parse_mode="Markdown")

async def cmd_help(update, ctx):
    await update.message.reply_text(HELP, parse_mode="Markdown")

async def cmd_tf(update, ctx):
    if not ctx.args:
        await update.message.reply_text("Dùng: /tf 5m | 15m | 1h | 4h | 1d"); return
    tf = update.message.text.strip().split()[1].lower()
    user_tf[update.effective_user.id]=tf
    await update.message.reply_text(f"🕒 Khung mặc định phiên: **{tf}**", parse_mode="Markdown")

async def handle_text(update, ctx):
    text=(update.message.text or "").strip().lower()
    if text in ("learn win","learn loss"):
        try:
            df=pd.read_csv(LOG_PATH); last=df.iloc[-1]
            dfe=enrich(mexcf.ohlcv(last["base"], last["tf"], 400))
            row=dfe.iloc[-1].to_dict()
            AI.learn(last["base"], row, 1 if text.endswith("win") else 0)
            await update.message.reply_text("🧠 AI đã học từ lệnh gần nhất.")
        except Exception as e:
            await update.message.reply_text(f"Không học được: {e}")
        return
    parts=text.split()
    if not parts:
        await update.message.reply_text("Nhập coin: `btc` hoặc `sol 15m`."); return
    base=parts[0].upper()
    tf=parts[1] if len(parts)>1 else user_tf.get(update.effective_user.id, DEFAULT_TF)
    try:
        r=analyze(base, tf)
        msg=(f"🧠 {r['base']} ({r['tf']}) — {'🟢 LONG' if r['side']=='LONG' else '🔴 SHORT'}\n"
             f"Giá: {fmt(r['price'])}\nTP1: {fmt(r['tp1'])} | TP2: {fmt(r['tp2'])}\nSL: {fmt(r['sl'])}\n"
             f"AI Score: {r['score']}%\nEMA12/26: {fmt(r['ema12'])}/{fmt(r['ema26'])} | "
             f"RSI: {fmt(r['rsi'],2)} | MACD_hist: {fmt(r['macd_hist'],5)} | VolZ: {fmt(r['vol_z'],2)}")
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"Lỗi phân tích {base}: {e}")

def compute_combo_score(base):
    try:
        trades=mexcf.trades(base, 200)
        b=s=0.0
        for t in trades:
            amt=float(t.get("amount",0))
            if t.get("side")=="buy": b+=amt
            else: s+=amt
        tbr=(b/(b+s)) if (b+s)>0 else 0.0
        ob = mexcf.orderbook_imbalance(base, 10)
        df1=enrich(mexcf.ohlcv(base, "1m", 60))
        volz=float(df1["vol_z"].iloc[-1])
        fr=mexcf.funding(base)
        score=0.0
        score += max(0,(tbr-0.60))*60
        score += max(0,ob)*50
        if volz>=2.5: score+=30
        if fr is not None and fr<0: score += min(12,60000*abs(fr))
        return int(np.clip(score,0,100)), tbr, ob, volz, fr, float(df1['close'].iloc[-1])
    except Exception as e:
        logging.warning(f"score {base} err: {e}")
        return 0,0,0,0,None,0.0

def fmt_alert(base,res):
    score,tbr,ob,volz,fr,price = res
    return (f"🔔 ALERT {base}/USDT — Tín hiệu sớm ({score}%)\n"
            f"• Taker Buy Ratio: {tbr:.2f}\n• Orderbook Imb.: {ob:.2f}\n"
            f"• Volume z: {volz:.2f}\n• Funding: {('—' if fr is None else f'{fr:.4%}')}\n"
            f"💰 Giá: {fmt(price)} USDT\n→ Gõ tên coin để nhận Entry/TP/SL")

scan_i=0; last_alert_ts={}; ALERT_COOLDOWN=300

async def scan_top(context: ContextTypes.DEFAULT_TYPE):
    global scan_i
    coins = COINS
    if not coins: return
    n=len(coins); start=scan_i; end=min(start+BATCH_SIZE,n); batch=coins[start:end]
    scan_i = 0 if end>=n else end
    if context.job and context.job.chat_id:
        chat_id = context.job.chat_id
    else:
        chat_id = None
    now=time.time()
    for base in batch:
        res=compute_combo_score(base); score=res[0]
        if score>=ALERT_THRESHOLD and now-last_alert_ts.get(base,0)>=ALERT_COOLDOWN and chat_id:
            await context.bot.send_message(chat_id=chat_id, text=fmt_alert(base,res))
            last_alert_ts[base]=now

async def main():
    if not TELEGRAM_BOT_TOKEN: raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("tf",    cmd_tf))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # chạy scan sau khi /start (job gắn chat_id sau khi có /start)
    app.job_queue.run_repeating(scan_top, interval=SCAN_INTERVAL_SEC, first=10, chat_id=None)

    logging.info(f"Coins={len(COINS)} | batch={BATCH_SIZE} | scan={SCAN_INTERVAL_SEC}s")
    await app.run_polling(allowed_updates=None)

if __name__ == "__main__":
    asyncio.run(main())
