"""Terminal board rendering and move input, shared by play.py and train_vs_human.py."""
from tictactoe.env import available_actions

CELL = {1: "X", -1: "O", 0: " "}


def render(board):
    def cell(i):
        return CELL[board[i]] if board[i] != 0 else str(i)

    rows = [" | ".join(cell(r * 3 + c) for c in range(3)) for r in range(3)]
    print(("\n" + "-" * 11 + "\n").join(rows))


def human_move(board):
    actions = available_actions(board)
    while True:
        raw = input(f"Your move {actions}: ").strip()
        if raw.isdigit() and int(raw) in actions:
            return int(raw)
        print("Invalid move, try again.")
