from typing import Protocol

from tictactoe.env import Action, Board, Player


class Agent(Protocol):
    """Anything that can pick a move: learned agents and benchmark opponents alike."""

    def choose_action(self, board: Board, player: Player, actions: list[Action], greedy: bool = False) -> Action: ...
