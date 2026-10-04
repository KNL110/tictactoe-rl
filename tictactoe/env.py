"""Tic-Tac-Toe environment.

Board is a length-9 tuple. Cells hold 1 (player X), -1 (player O), or 0 (empty).
Players alternate starting with X (1).
"""

type Board = tuple[int, ...]
type Player = int  # 1 (X) or -1 (O)
type Action = int  # cell index 0-8

WIN_LINES: list[tuple[int, int, int]] = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]

EMPTY_BOARD: Board = (0,) * 9


def check_winner(board: Board) -> tuple[Player, bool]:
    """Return 1 if X won, -1 if O won, 0 if draw/ongoing, and whether the game is over."""
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1, True
        if s == -3:
            return -1, True
    if 0 not in board:
        return 0, True  # draw
    return 0, False  # ongoing


def available_actions(board: Board) -> list[Action]:
    return [i for i, v in enumerate(board) if v == 0]


def apply_move(board: Board, action: Action, player: Player) -> Board:
    """The board after `player` plays `action` (boards are immutable tuples)."""
    next_board = list(board)
    next_board[action] = player
    return tuple(next_board)


class TicTacToeEnv:
    def __init__(self) -> None:
        self.board: Board = EMPTY_BOARD
        self.player: Player = 1

    def reset(self) -> Board:
        self.board = EMPTY_BOARD
        self.player = 1  # X moves first
        return self.board

    def step(self, action: Action) -> tuple[Board, Player, bool]:
        """Play `action` for the current player. Returns (board, winner, done)."""
        if self.board[action] != 0:
            raise ValueError(f"Illegal move: cell {action} is occupied")
        self.board = apply_move(self.board, action, self.player)

        winner, done = check_winner(self.board)
        self.player *= -1  # switch turns
        return self.board, winner, done

    def render(self) -> None:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for r in range(3):
            rows.append(" ".join(symbols[self.board[r * 3 + c]] for c in range(3)))
        print("\n".join(rows))
