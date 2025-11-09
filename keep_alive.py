# keep_alive.py
import os
import logging
from flask import Flask

# Tắt bớt log của Flask để không spam "GET /"
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot AI đang hoạt động."

def run_server():
    """
    Hàm này sẽ được gọi từ main.py.
    Render yêu cầu bind đúng PORT trong env, nếu không có thì dùng 10000.
    """
    port = int(os.environ.get("PORT", "10000"))
    # debug=False, use_reloader=False để không tạo thêm process/threaad phụ
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
