import random

from tictactoe.env import Action, Board, Player, apply_move, available_actions, check_winner

_MEMO: dict[tuple[Board, Player], tuple[int, list[Action]]] = {}


def _negamax(board: Board, player: Player) -> tuple[int, list[Action]]:
    """Return (value, best_actions) for `player` to move, value in {-1, 0, 1}
    from `player`'s perspective. Memoized since the state space is tiny (~5.5k states)."""
    key = (board, player)
    if key in _MEMO:
        return _MEMO[key]

    winner, done = check_winner(board)
    if done:
        value = 0 if winner == 0 else (1 if winner == player else -1)
        _MEMO[key] = (value, [])
        return _MEMO[key]

    best_value = -2
    best_actions: list[Action] = []
    for a in available_actions(board):
        opp_value, _ = _negamax(apply_move(board, a, player), -player)
        value = -opp_value
        if value > best_value:
            best_value = value
            best_actions = [a]
        elif value == best_value:
            best_actions.append(a)

    result = (best_value, best_actions)
    _MEMO[key] = result
    return result


class MinimaxAgent:
    """Plays perfectly (never loses). Used as a benchmark opponent, not for training
    — it never explores, so an agent trained only against it would overfit to one style
    of (perfect) play instead of learning to punish mistakes."""

    def choose_action(self, board: Board, player: Player, actions: list[Action], greedy: bool = False) -> Action:
        _, best_actions = _negamax(board, player)
        return random.choice(best_actions)
