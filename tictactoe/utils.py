import pickle
from pathlib import Path
from typing import Any

from tictactoe.env import Board, Player


def canonical_state(board: Board, player: Player) -> Board:
    """View the board from `player`'s perspective: player's own pieces become 1,
    opponent's become -1. This lets a single table serve both X and O in self-play."""
    return tuple(player * c for c in board)


def save_pickle(obj: object, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
