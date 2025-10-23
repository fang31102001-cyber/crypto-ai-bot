import os
from threading import Thread
from datetime import datetime
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TIMEFRAME_DEFAULT = os.getenv("TIMEFRAME", "15m")

async def cmd_start(update, ctx):
    await update.message.reply_text(
        "🤖 Bot đã sẵn sàng.\n"
        "- Gõ coin: `btc` hoặc `sol 15m`.\n"
        "- Xong bước nền tảng sẽ thêm AI quét tự động & tự học."
    )

async def handle_text(update, ctx):
    text = (update.message.text or "").strip().lower()
    parts = text.split()
    if not parts:
        await update.message.reply_text("Nhập coin như: `btc` hoặc `sol 15m`"); return
    base = parts[0].upper()
    tf = parts[1] if len(parts) > 1 else TIMEFRAME_DEFAULT
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    await update.message.reply_text(f"[{now} UTC] Nhận: {base} / TF: {tf}\n(Bot chạy OK)")

def main():
    # bật web để UptimeRobot ping
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
