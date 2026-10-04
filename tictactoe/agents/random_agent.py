import random


class RandomAgent:
    """Baseline opponent that picks a uniformly random legal move."""

    def choose_action(self, board, player, actions, greedy=False):
        return random.choice(actions)
