from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/")
def root():
    return jsonify(status="ok")

def run_server():
    from waitress import serve
    import os
    port = int(os.getenv("PORT", "10000"))
    serve(app, host="0.0.0.0", port=port)
