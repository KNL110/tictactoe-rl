"""The JSON API as plain functions, with no web framework involved.

Shared by the Flask server (app.py) and the static GitHub Pages build, where this
same code runs inside the visitor's browser via Pyodide (see static/pyodide_api.js).
"""
from tictactoe.models import MODEL_PATHS


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def agent_kind(data):
    kind = data.get("agent", "q")
    if kind not in MODEL_PATHS:
        raise ApiError("unknown agent")
    return kind


def new_game(sess, data):
    kind = agent_kind(data)
    fresh = bool(data.get("fresh", False))
    default_epsilon = 0.3 if fresh else 0.0
    try:
        epsilon = max(0.0, min(1.0, float(data.get("epsilon", default_epsilon))))
    except (TypeError, ValueError):
        epsilon = default_epsilon

    result = sess.new_game(
        kind,
        fresh,
        learn=bool(data.get("learn", True)),
        human_first=bool(data.get("human_first", True)),
        epsilon=epsilon,
    )
    return sess.to_json(*result)


def reset_fresh(sess, data):
    sess.reset_fresh(agent_kind(data))
    return {"ok": True}


def move(sess, data):
    cell = data.get("cell")
    if not isinstance(cell, int) or isinstance(cell, bool) or not 0 <= cell <= 8:
        raise ApiError("cell must be 0-8")
    if sess.env is None or sess.done:
        raise ApiError("No game in progress. Press New Game.", 409)
    if not sess.is_human_turn():
        raise ApiError("not your turn")
    if sess.env.board[cell] != 0:
        raise ApiError("cell occupied")
    return sess.to_json(*sess.human_move(cell))


ROUTES = {
    "/api/new_game": new_game,
    "/api/reset_fresh": reset_fresh,
    "/api/move": move,
}


def handle(path, data, sess):
    """Run one API call against `sess` (a GameSession). Returns (status, body)."""
    if path not in ROUTES:
        return 404, {"error": "not found"}
    if not isinstance(data, dict):
        data = {}
    try:
        return 200, ROUTES[path](sess, data)
    except ApiError as err:
        return err.status, {"error": err.message}
