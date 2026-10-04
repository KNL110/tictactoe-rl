"""Exhaustive exploitability check: can *any* opponent ever beat a trained agent?

Evaluating against minimax only tests one (perfect) style of play. This instead
walks the full game tree: every possible opponent move, and every move the greedy
agent might pick (it breaks ties at random), from both sides. Tic-Tac-Toe is small
enough that this takes well under a second.
"""
from tictactoe.agents.q_learning import QLearningAgent
from tictactoe.env import EMPTY_BOARD, Action, Board, Player, apply_move, available_actions, check_winner
from tictactoe.models import LearningAgent
from tictactoe.utils import canonical_state


def greedy_moves(agent: LearningAgent, board: Board, player: Player) -> list[Action]:
    """Every move the greedy agent might pick in this position."""
    actions = available_actions(board)
    values: dict[Action, float]
    if isinstance(agent, QLearningAgent):
        qs = agent.Q[canonical_state(board, player)]
        values = {a: float(qs[a]) for a in actions}
    else:
        values = {a: agent.V[canonical_state(apply_move(board, a, player), player)] for a in actions}
    best = max(values.values())
    return [a for a, v in values.items() if v == best]


def find_losses(agent: LearningAgent, agent_side: Player) -> tuple[dict[str, int], list[list[Action]]]:
    """Count every possible game's outcome with the agent playing `agent_side`
    (1 = X, moves first; -1 = O). Returns the counts plus every move sequence
    (cells 0-8, in order) where the agent loses."""
    results = {"win": 0, "draw": 0, "loss": 0}
    losing_lines: list[list[Action]] = []

    def walk(board: Board, player: Player, line: list[Action]) -> None:
        winner, done = check_winner(board)
        if done:
            outcome = "draw" if winner == 0 else ("win" if winner == agent_side else "loss")
            results[outcome] += 1
            if outcome == "loss":
                losing_lines.append(line)
            return
        moves = greedy_moves(agent, board, player) if player == agent_side else available_actions(board)
        for a in moves:
            walk(apply_move(board, a, player), -player, line + [a])

    walk(EMPTY_BOARD, 1, [])
    return results, losing_lines


def report(agent: LearningAgent, name: str) -> int:
    """Print the check for both sides; returns the total number of losing games."""
    total = 0
    for agent_side, label in ((1, "agent first"), (-1, "opponent first")):
        results, losing_lines = find_losses(agent, agent_side)
        total += results["loss"]
        print(f"{name} ({label}): {sum(results.values())} possible games, "
              f"agent W/D/L = {results['win']}/{results['draw']}/{results['loss']}"
              + (f", e.g. losing line {losing_lines[0]}" if losing_lines else ""))
    return total


if __name__ == "__main__":
    # python -m tictactoe.verify: check both saved models
    from tictactoe.models import AGENT_NAMES, load_agent

    for kind, name in AGENT_NAMES.items():
        report(load_agent(kind), name)
