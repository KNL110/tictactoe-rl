import random
from collections import defaultdict

import numpy as np

from tictactoe.utils import canonical_state


class QLearningAgent:
    """Tabular Q-learning, trained via self-play.

    States are stored canonically (from the mover's own perspective, see
    utils.canonical_state), so ONE table Q[state][action] serves both X and O:
    it always answers "how good is action `a` for whoever is about to move".
    """

    def __init__(self, alpha=0.3, gamma=1.0, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Optimistic default: an unvisited action looks like a draw (0.5), not a loss (0).
        # With a zero default, an under-explored action permanently looks worse than a
        # mediocre-but-sampled one, so the agent stops revisiting it once epsilon decays
        # and never corrects the estimate — this alone was the difference between the
        # agent plateauing at ~90% draws vs. a perfect opponent and reaching 100%.
        self.Q = defaultdict(lambda: np.full(9, 0.5))

    def choose_action(self, board, player, actions, greedy=False):
        state = canonical_state(board, player)
        if not greedy and random.random() < self.epsilon:
            return random.choice(actions)
        qs = self.Q[state]
        best_q = max(qs[a] for a in actions)
        best_actions = [a for a in actions if qs[a] == best_q]
        return random.choice(best_actions)

    def update(self, state, action, target):
        current = self.Q[state][action]
        self.Q[state][action] = current + self.alpha * (target - current)
