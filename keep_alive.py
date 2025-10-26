import logging
from flask import Flask

# Tắt log GET / HTTP/1.1 200 -
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ AI Futures Bot đang online và chạy ổn định 24/7!"

def run_server():
    app.run(host='0.0.0.0', port=10000)
