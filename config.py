import os

TELEGRAM_BOT_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DEFAULT_TF           = os.environ.get("DEFAULT_TF", "15m")
SCAN_INTERVAL_SEC    = int(os.environ.get("SCAN_INTERVAL_SEC", "30"))
ALERT_THRESHOLD      = int(os.environ.get("ALERT_THRESHOLD", "75"))
BATCH_SIZE           = int(os.environ.get("BATCH_SIZE", "25"))
COIN_LIST            = os.environ.get("COIN_LIST", "TOP20")  # ALL | TOP20 | "BTC,ETH,WLD"
TZ                   = os.environ.get("TZ", "Asia/Ho_Chi_Minh")

DATA_DIR   = "./data"
MEMO_PATH  = os.path.join(DATA_DIR, "memory.json")
LOG_PATH   = os.path.join(DATA_DIR, "signals_log.csv")
