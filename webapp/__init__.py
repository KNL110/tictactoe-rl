"""Browser-based Tic-Tac-Toe against the trained agents.

  app       Flask routes (the JSON API the page calls)
  game      one visitor's game: board, agents, per-move learning
  sessions  in-memory store that keeps each visitor's game separate

Run locally with `python -m webapp`, or in production with
`gunicorn -w 1 webapp.app:app` (one worker: games live in process memory).
"""
