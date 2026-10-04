"""Tic-Tac-Toe environment.

Board is a length-9 tuple. Cells hold 1 (player X), -1 (player O), or 0 (empty).
Players alternate starting with X (1).
"""

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def check_winner(board):
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


def available_actions(board):
    return [i for i, v in enumerate(board) if v == 0]


class TicTacToeEnv:
    def __init__(self):
        self.board = None
        self.player = None
        self.reset()

    def reset(self):
        self.board = (0,) * 9
        self.player = 1  # X moves first
        return self.board

    def step(self, action):
        if self.board[action] != 0: # type: ignore
            raise ValueError(f"Illegal move: cell {action} is occupied")
        new_board = list(self.board) # type: ignore
        new_board[action] = self.player # type: ignore
        self.board = tuple(new_board)

        winner, done = check_winner(self.board)
        self.player *= -1  # type: ignore # switch turns
        return self.board, winner, done

    def render(self):
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for r in range(3):
            rows.append(" ".join(symbols[self.board[r * 3 + c]] for c in range(3))) # type: ignore
        print("\n".join(rows))
