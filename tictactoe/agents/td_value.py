import random
from collections import defaultdict

from tictactoe.env import Action, Board, Player, apply_move
from tictactoe.utils import canonical_state

type VTable = dict[Board, float]


def default_value() -> float:
    return 0.5  # unknown states assumed a coin flip


class TDValueAgent:
    """Sutton & Barto-style state-value learner (their Chapter 1 tic-tac-toe example).

    V(s) estimates, for an "afterstate" s (the board immediately after a move,
    seen from that mover's own perspective), the probability that the mover who
    just produced it goes on to win the game (with a draw counted as 0.5).

    Self-play trick: for a zero-sum game with a shared draw value of 0.5,
    P(mover wins) + P(opponent wins) + P(draw) = 1 implies
        V(mover's afterstate) == 1 - V(opponent's very next afterstate)
    so a single TD(0) update using "1 - V(next afterstate)" as the bootstrap
    target lets one table learn from both sides of the game.
    """

    def __init__(self, alpha: float = 0.3, epsilon: float = 0.2) -> None:
        self.alpha = alpha
        self.epsilon = epsilon
        self.V: defaultdict[Board, float] = defaultdict(default_value)

    def _afterstates(self, board: Board, player: Player, actions: list[Action]) -> dict[Action, Board]:
        return {a: canonical_state(apply_move(board, a, player), player) for a in actions}

    def choose_action(self, board: Board, player: Player, actions: list[Action], greedy: bool = False) -> Action:
        afterstates = self._afterstates(board, player, actions)
        if not greedy and random.random() < self.epsilon:
            return random.choice(actions)
        best_v = max(self.V[s] for s in afterstates.values())
        best_actions = [a for a, s in afterstates.items() if self.V[s] == best_v]
        return random.choice(best_actions)

    def update(self, state: Board, target: float) -> None:
        self.V[state] += self.alpha * (target - self.V[state])
