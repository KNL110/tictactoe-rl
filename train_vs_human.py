"""Play against a trained agent while it keeps learning from your games.

Unlike train.py (pure self-play), this pits the agent against a live human
opponent and updates its table after every move from the actual games played.

By default it continues from the already-mastered self-play model, so it plays
a strong game while still silently correcting itself if you find and repeatedly
exploit a weak spot. Pass --fresh to instead train a blank agent purely from
your games — note that will play weakly for a long time: self-play needed
roughly a million games to reach perfect play, and a human can only supply a
handful per sitting, so a fresh agent mostly demonstrates the learning
mechanics rather than reaching mastery in one session.
"""
import argparse

from tictactoe.console import human_move, render
from tictactoe.env import Action, Board, Player, TicTacToeEnv
from tictactoe.learning import make_learner
from tictactoe.models import AGENT_NAMES, MODEL_PATHS, load_agent, new_agent, parse_kind, save_agent
from tictactoe.training import play_episode_vs_human


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["q", "td"], default="q")
    parser.add_argument("--fresh", action="store_true", help="start from a blank agent instead of the pretrained one")
    parser.add_argument("--first", choices=["human", "agent"], default="human")
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="agent exploration rate while playing you; default is 0.0 (pretrained) or 0.3 (--fresh)",
    )
    args = parser.parse_args()

    kind = parse_kind(args.agent)
    assert kind is not None  # guaranteed by argparse choices
    epsilon = args.epsilon if args.epsilon is not None else (0.3 if args.fresh else 0.0)
    agent = new_agent(kind, epsilon) if args.fresh else load_agent(kind, epsilon)
    agent_name = AGENT_NAMES[kind]

    print(f"Training the {agent_name} agent live against you "
          f"({'starting from scratch' if args.fresh else 'continuing from the pretrained model'}, epsilon={epsilon}).")
    print("It updates its table after every move. Ctrl-C any time to stop; progress is saved after every game.\n")

    human_side = 1 if args.first == "human" else -1
    tally: dict[str, int] = {"you": 0, "draw": 0, "agent": 0}
    game_num = 0

    def on_move(mover: Player, action: Action, board: Board, winner: Player, done: bool) -> None:
        if mover != human_side:
            print(f"Agent plays {action}")
            render(board)

    def get_human_move(board: Board, actions: list[Action]) -> Action:
        render(board)
        return human_move(board)

    try:
        while True:
            game_num += 1
            print(f"\n=== Game {game_num} — you are {'X' if human_side == 1 else 'O'} ===")
            learner = make_learner(agent)
            winner = play_episode_vs_human(TicTacToeEnv(), learner, -human_side, get_human_move, on_move=on_move)

            if winner == 0:
                tally["draw"] += 1
                print("Result: draw")
            elif winner == human_side:
                tally["you"] += 1
                print("Result: you win!")
            else:
                tally["agent"] += 1
                print("Result: agent wins!")
            print(f"Tally — you: {tally['you']}  draws: {tally['draw']}  agent: {tally['agent']}")

            save_agent(kind, agent)

            if input("Play again? [Y/n] ").strip().lower() == "n":
                break
            human_side *= -1  # alternate sides for balanced training coverage

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        save_agent(kind, agent)
        print(f"Saved {agent_name} agent to saved_models/{MODEL_PATHS[kind].name}")
        print(f"Final tally — you: {tally['you']}  draws: {tally['draw']}  agent: {tally['agent']}")


if __name__ == "__main__":
    main()
