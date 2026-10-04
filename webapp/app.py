"""Flask server for the web app. The page sends a random per-browser id in the
X-Session-Id header, so every visitor plays their own game (see sessions.py).
The API logic itself lives in api.py."""
import re
import threading

from flask import Flask, jsonify, render_template, request

from webapp import api
from webapp.game import GameSession
from webapp.sessions import SessionStore

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024  # requests are tiny JSON bodies

SESSION_ID_RE = re.compile(r"[A-Za-z0-9-]{16,64}")

_sessions = SessionStore(GameSession)
# Each request is microseconds of work, so one lock is the simplest way to keep the
# session store (and, with TTT_PERSIST, the shared agents) safe across threads.
_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/<name>", methods=["POST"])
def api_call(name):
    path = f"/api/{name}"
    if path not in api.ROUTES:
        return jsonify({"error": "not found"}), 404
    session_id = request.headers.get("X-Session-Id", "")
    if not SESSION_ID_RE.fullmatch(session_id):
        return jsonify({"error": "missing or invalid session id"}), 400

    data = request.get_json(silent=True)
    with _lock:
        status, body = api.handle(path, data, _sessions.get(session_id))
    return jsonify(body), status
