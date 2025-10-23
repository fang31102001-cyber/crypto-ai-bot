import os
from threading import Thread
from datetime import datetime
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

# ========= ENV =========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TIMEFRAME_DEFAULT = os.getenv("TIMEFRAME", "15m")

# ========= HỖ TRỢ PARSE COIN + TIMEFRAME =========
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

# ========= LỆNH BẮT ĐẦU =========
async def cmd_start(update, ctx):
    await update.message.reply_text(
        "🤖 Bot đã sẵn sàng.\n"
        "- Gõ coin: `btc` hoặc `sol 15m`.\n"
        "- Hệ thống đã nhận dạng nhiều khung thời gian (1m,5m,15m,1h,...).\n"
        "- Bước tiếp theo sẽ thêm phân tích AI."
    )

# ========= XỬ LÝ TIN NHẮN =========
async def handle_text(update, ctx):
    try:
        text = (update.message.text or "").strip()
        base, tf = parse_symbol_tf(text, TIMEFRAME_DEFAULT)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        await update.message.reply_text(
            f"✅ Đã nhận: {base} — TF: {tf}\n"
            f"[{now} UTC]\n"
            f"(Bot đang ở chế độ nền tảng. Sắp thêm phân tích kỹ thuật + AI.)"
        )
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
