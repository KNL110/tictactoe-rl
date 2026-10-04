"""Play interactively against a trained agent."""
import argparse

from tictactoe.console import human_move, render
from tictactoe.env import TicTacToeEnv, available_actions
from tictactoe.models import AGENT_NAMES, load_agent, parse_kind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["q", "td"], default="q")
    parser.add_argument("--first", choices=["human", "agent"], default="human")
    args = parser.parse_args()

    kind = parse_kind(args.agent)
    assert kind is not None  # guaranteed by argparse choices
    agent = load_agent(kind)
    print(f"Playing against the {AGENT_NAMES[kind]} agent (greedy, fully trained).")

    human_player = 1 if args.first == "human" else -1
    print(f"You are {'X' if human_player == 1 else 'O'}. Cells are numbered 0-8 left-to-right, top-to-bottom.\n")

    env = TicTacToeEnv()
    env.reset()
    while True:
        render(env.board)
        actions = available_actions(env.board)
        mover = env.player
        if mover == human_player:
            action = human_move(env.board)
        else:
            action = agent.choose_action(env.board, mover, actions, greedy=True)
            print(f"Agent plays {action}")
        _, winner, done = env.step(action)
        print()
        if done:
            render(env.board)
            if winner == 0:
                print("\nDraw!")
            elif winner == human_player:
                print("\nYou win!")
            else:
                print("\nAgent wins!")
            break


if __name__ == "__main__":
    main()
