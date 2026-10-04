"""Train both agents from scratch via self-play, then save the models and training
curves. The training logic itself lives in tictactoe/training.py."""
import argparse

from tictactoe.models import ROOT, save_agent
from tictactoe.plotting import plot_history
from tictactoe.training import train
from tictactoe.verify import report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-games", type=int, default=300)
    args = parser.parse_args()

    histories = {}
    q_agent, histories["q"] = train("q", args.episodes, args.eval_every, args.eval_games)
    td_agent, histories["td"] = train("td", args.episodes, args.eval_every, args.eval_games)

    save_agent("q", q_agent)
    save_agent("td", td_agent)
    print("Saved trained agents to saved_models/")

    plot_history(histories, ROOT / "plots" / "training_curves.png")

    print("\nExhaustive check (every possible opponent move):")
    if report(q_agent, "Q-learning") + report(td_agent, "TD value"):
        print("Some lines still beat an agent; patch them with `python finetune.py --agent <q|td>`.")
