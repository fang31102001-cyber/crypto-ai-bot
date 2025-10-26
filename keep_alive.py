from flask import Flask
from threading import Thread
import logging

# Tắt toàn bộ log mặc định của Flask để tránh spam GET /
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask('')

@app.route('/')
def home():
    return "Bot AI DeepFlow v2.7 is alive."

def run():
    # Không in log ra console
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()
