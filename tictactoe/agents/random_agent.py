import random

from tictactoe.env import Action, Board, Player


class RandomAgent:
    """Baseline opponent that picks a uniformly random legal move."""

    def choose_action(self, board: Board, player: Player, actions: list[Action], greedy: bool = False) -> Action:
        return random.choice(actions)
