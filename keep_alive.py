# keep_alive.py
import os
import logging
from flask import Flask

# Giảm log của werkzeug để không spam request
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)

@app.get("/")
def home():
    return "Bot AI đang hoạt động.", 200

def run_server():
    """
    Hàm này sẽ được gọi từ main.py.
    Render yêu cầu bind đúng PORT trong env, nếu không có thì dùng 10000.
    """
    port = int(os.environ.get("PORT", "10000"))

    # threaded=True giúp xử lý ping ổn định hơn.
    # use_reloader=False tránh tạo process phụ.
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
