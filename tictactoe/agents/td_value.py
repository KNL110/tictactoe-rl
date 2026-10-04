import random
from collections import defaultdict

from tictactoe.utils import canonical_state


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

    def __init__(self, alpha=0.3, epsilon=0.2):
        self.alpha = alpha
        self.epsilon = epsilon
        self.V = defaultdict(lambda: 0.5)  # unknown states assumed a coin flip

    def _afterstates(self, board, player, actions):
        results = {}
        for a in actions:
            next_board = list(board)
            next_board[a] = player
            results[a] = canonical_state(tuple(next_board), player)
        return results

    def choose_action(self, board, player, actions, greedy=False):
        afterstates = self._afterstates(board, player, actions)
        if not greedy and random.random() < self.epsilon:
            return random.choice(actions)
        best_v = max(self.V[s] for s in afterstates.values())
        best_actions = [a for a, s in afterstates.items() if self.V[s] == best_v]
        return random.choice(best_actions)

    def update(self, state, target):
        self.V[state] += self.alpha * (target - self.V[state])
