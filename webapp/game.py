"""One visitor's game: the board, their agents, and the per-move learning.

Each session gets its own agents, so one visitor's games (and learning) never
affect another's. The pretrained tables are loaded from disk once and copied per
session; nothing is ever written back unless TTT_PERSIST=1 is set (local use only,
see serve.sh), in which case the pretrained agents are shared and saved after every
learning game, like train_vs_human.py.
"""
import os
from typing import Literal, TypedDict

from tictactoe import models
from tictactoe.env import Action, Board, Player, TicTacToeEnv, available_actions
from tictactoe.learning import OnlineLearner, make_learner
from tictactoe.models import AgentKind, LearningAgent, Table

type Outcome = Literal["you", "draw", "agent"]
type MoveResult = tuple[Action | None, Player, bool]  # (agent's move, winner, done)

PERSIST = os.environ.get("TTT_PERSIST") == "1"

_base_tables: dict[AgentKind, Table] = {}  # pretrained tables as loaded from disk; never mutated
_shared_agents: dict[AgentKind, LearningAgent] = {}  # pretrained agents shared by all sessions (PERSIST only)


class GameState(TypedDict):
    board: list[str]
    human_symbol: str
    agent_move: Action | None
    done: bool
    winner: Outcome | None
    tally: dict[Outcome, int]
    states_learned: int


def _base_table(kind: AgentKind) -> Table:
    if kind not in _base_tables:
        _base_tables[kind] = models.load_table(kind)
    return _base_tables[kind]


def pretrained_agent(kind: AgentKind) -> LearningAgent:
    if PERSIST:
        if kind not in _shared_agents:
            _shared_agents[kind] = models.new_agent(kind, table=_base_table(kind))
        return _shared_agents[kind]
    return models.new_agent(kind, table=models.copy_table(_base_table(kind)))


def board_json(board: Board) -> list[str]:
    symbols = {1: "X", -1: "O", 0: ""}
    return [symbols[c] for c in board]


class GameSession:
    def __init__(self) -> None:
        self.agents: dict[tuple[AgentKind, bool], LearningAgent] = {}  # (kind, fresh) -> agent, built on first use
        self.env = TicTacToeEnv()
        self.kind: AgentKind = "q"
        self.fresh = False
        self.learn = True
        self.human_side: Player = 1
        self.learner: OnlineLearner | None = None
        self.done = True  # no game in progress until new_game()
        self.tally: dict[Outcome, int] = {"you": 0, "draw": 0, "agent": 0}

    def agent(self) -> LearningAgent:
        """fresh=True gets a blank agent that only ever learns from this visitor's games."""
        key = (self.kind, self.fresh)
        if key not in self.agents:
            self.agents[key] = models.new_agent(self.kind) if self.fresh else pretrained_agent(self.kind)
        return self.agents[key]

    def reset_fresh(self, kind: AgentKind) -> None:
        self.agents.pop((kind, True), None)

    def new_game(self, kind: AgentKind, fresh: bool, learn: bool, human_first: bool, epsilon: float) -> MoveResult:
        """Starts a game; the agent opens if the human goes second."""
        self.kind, self.fresh, self.learn = kind, fresh, learn
        self.human_side = 1 if human_first else -1
        agent = self.agent()
        agent.epsilon = epsilon
        self.learner = make_learner(agent) if learn else None
        self.env = TicTacToeEnv()
        self.done = False

        if self.human_side == -1:
            return self._agent_move()
        return None, 0, False

    def is_human_turn(self) -> bool:
        return not self.done and self.env.player == self.human_side

    def human_move(self, cell: Action) -> MoveResult:
        """Plays the human's cell, then the agent's reply."""
        winner, done = self._play(cell, by_agent=False)
        if done:
            return None, winner, done
        return self._agent_move()

    def _agent_move(self) -> MoveResult:
        board, mover = self.env.board, self.env.player
        action = self.agent().choose_action(board, mover, available_actions(board))
        winner, done = self._play(action, by_agent=True)
        return action, winner, done

    def _play(self, action: Action, by_agent: bool) -> tuple[Player, bool]:
        board, mover = self.env.board, self.env.player
        next_board, winner, done = self.env.step(action)
        if self.learner is not None:
            update = self.learner.agent_moved if by_agent else self.learner.opponent_moved
            update(board, mover, action, next_board, winner, done)
        if done:
            self._finish(winner)
        return winner, done

    def _finish(self, winner: Player) -> None:
        self.done = True
        self.tally[self.winner_label(winner)] += 1
        if self.learn and PERSIST and not self.fresh:
            models.save_agent(self.kind, self.agent())

    def winner_label(self, winner: Player) -> Outcome:
        if winner == 0:
            return "draw"
        return "you" if winner == self.human_side else "agent"

    def to_json(self, agent_action: Action | None, winner: Player, done: bool) -> GameState:
        return {
            "board": board_json(self.env.board),
            "human_symbol": "X" if self.human_side == 1 else "O",
            "agent_move": agent_action,
            "done": done,
            "winner": self.winner_label(winner) if done else None,
            "tally": self.tally,
            "states_learned": len(models.table_of(self.agent())),
        }
