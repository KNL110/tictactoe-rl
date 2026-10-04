"""Flask routes for the web app. The page sends a random per-browser id in the
X-Session-Id header, so every visitor plays their own game (see sessions.py)."""
import re
import threading

from flask import Flask, jsonify, render_template, request

from tictactoe.models import MODEL_PATHS
from webapp.game import GameSession
from webapp.sessions import SessionStore

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024  # requests are tiny JSON bodies

SESSION_ID_RE = re.compile(r"[A-Za-z0-9-]{16,64}")

_sessions = SessionStore(GameSession)
# Each request is microseconds of work, so one lock is the simplest way to keep the
# session store (and, with TTT_PERSIST, the shared agents) safe across threads.
_lock = threading.Lock()


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


@app.errorhandler(ApiError)
def handle_api_error(err):
    return jsonify({"error": err.message}), err.status


def request_data():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def current_session():
    session_id = request.headers.get("X-Session-Id", "")
    if not SESSION_ID_RE.fullmatch(session_id):
        raise ApiError("missing or invalid session id")
    return _sessions.get(session_id)


def agent_kind(data):
    kind = data.get("agent", "q")
    if kind not in MODEL_PATHS:
        raise ApiError("unknown agent")
    return kind


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    data = request_data()
    kind = agent_kind(data)
    fresh = bool(data.get("fresh", False))
    default_epsilon = 0.3 if fresh else 0.0
    try:
        epsilon = max(0.0, min(1.0, float(data.get("epsilon", default_epsilon))))
    except (TypeError, ValueError):
        epsilon = default_epsilon

    with _lock:
        sess = current_session()
        result = sess.new_game(
            kind,
            fresh,
            learn=bool(data.get("learn", True)),
            human_first=bool(data.get("human_first", True)),
            epsilon=epsilon,
        )
        return jsonify(sess.to_json(*result))


@app.route("/api/reset_fresh", methods=["POST"])
def reset_fresh():
    kind = agent_kind(request_data())
    with _lock:
        current_session().reset_fresh(kind)
    return jsonify({"ok": True})


@app.route("/api/move", methods=["POST"])
def move():
    cell = request_data().get("cell")
    if not isinstance(cell, int) or isinstance(cell, bool) or not 0 <= cell <= 8:
        raise ApiError("cell must be 0-8")

    with _lock:
        sess = current_session()
        if sess.env is None or sess.done:
            raise ApiError("No game in progress. Press New Game.", 409)
        if not sess.is_human_turn():
            raise ApiError("not your turn")
        if sess.env.board[cell] != 0:
            raise ApiError("cell occupied")
        return jsonify(sess.to_json(*sess.human_move(cell)))
