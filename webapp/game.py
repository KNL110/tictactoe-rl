"""One visitor's game: the board, their agents, and the per-move learning.

Each session gets its own agents, so one visitor's games (and learning) never
affect another's. The pretrained tables are loaded from disk once and copied per
session; nothing is ever written back unless TTT_PERSIST=1 is set (local use only,
see serve.sh), in which case the pretrained agents are shared and saved after every
learning game, like train_vs_human.py.
"""
import os

from tictactoe import models
from tictactoe.env import TicTacToeEnv, available_actions
from tictactoe.learning import make_learner

PERSIST = os.environ.get("TTT_PERSIST") == "1"

_base_tables = {}  # kind -> pretrained table as loaded from disk; never mutated
_shared_agents = {}  # kind -> pretrained agent shared by all sessions (PERSIST only)


def _base_table(kind):
    if kind not in _base_tables:
        _base_tables[kind] = models.load_table(kind)
    return _base_tables[kind]


def pretrained_agent(kind):
    if PERSIST:
        if kind not in _shared_agents:
            _shared_agents[kind] = models.new_agent(kind, table=_base_table(kind))
        return _shared_agents[kind]
    return models.new_agent(kind, table=models.copy_table(_base_table(kind)))


def board_json(board):
    symbols = {1: "X", -1: "O", 0: ""}
    return [symbols[c] for c in board]


class GameSession:
    def __init__(self):
        self.agents = {}  # (kind, fresh) -> agent, built on first use
        self.env = None
        self.kind = "q"
        self.fresh = False
        self.learn = True
        self.human_side = 1
        self.learner = None
        self.done = True
        self.tally = {"you": 0, "draw": 0, "agent": 0}

    def agent(self):
        """fresh=True gets a blank agent that only ever learns from this visitor's games."""
        key = (self.kind, self.fresh)
        if key not in self.agents:
            self.agents[key] = models.new_agent(self.kind) if self.fresh else pretrained_agent(self.kind)
        return self.agents[key]

    def reset_fresh(self, kind):
        self.agents.pop((kind, True), None)

    def new_game(self, kind, fresh, learn, human_first, epsilon):
        """Returns (agent_action, winner, done); the agent opens if the human goes second."""
        self.kind, self.fresh, self.learn = kind, fresh, learn
        self.human_side = 1 if human_first else -1
        agent = self.agent()
        agent.epsilon = epsilon
        self.learner = make_learner(kind, agent) if learn else None
        self.env = TicTacToeEnv()
        self.done = False

        if self.human_side == -1:
            return self._agent_move()
        return None, 0, False

    def is_human_turn(self):
        return self.env is not None and not self.done and self.env.player == self.human_side

    def human_move(self, cell):
        """Plays the human's cell, then the agent's reply. Returns (agent_action, winner, done)."""
        winner, done = self._play(cell, by_agent=False)
        if done:
            return None, winner, done
        return self._agent_move()

    def _agent_move(self):
        board, mover = self.env.board, self.env.player
        action = self.agent().choose_action(board, mover, available_actions(board))
        winner, done = self._play(action, by_agent=True)
        return action, winner, done

    def _play(self, action, by_agent):
        board, mover = self.env.board, self.env.player
        next_board, winner, done = self.env.step(action)
        if self.learner is not None:
            update = self.learner.agent_moved if by_agent else self.learner.opponent_moved
            update(board, mover, action, next_board, winner, done)
        if done:
            self._finish(winner)
        return winner, done

    def _finish(self, winner):
        self.done = True
        self.tally[self.winner_label(winner)] += 1
        if self.learn and PERSIST and not self.fresh:
            models.save_agent(self.kind, self.agent())

    def winner_label(self, winner):
        if winner == 0:
            return "draw"
        return "you" if winner == self.human_side else "agent"

    def to_json(self, agent_action, winner, done):
        return {
            "board": board_json(self.env.board),
            "human_symbol": "X" if self.human_side == 1 else "O",
            "agent_move": agent_action,
            "done": done,
            "winner": self.winner_label(winner) if done else None,
            "tally": self.tally,
            "states_learned": len(models.table_of(self.agent())),
        }
