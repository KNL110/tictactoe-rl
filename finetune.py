"""Patch weak spots in a trained agent: fine-tune it from random openings, then
check every possible game (tictactoe/verify.py). The model is only saved if no
opponent can beat it anymore and it still never loses to minimax."""
import argparse
import random

from tictactoe.agents.minimax import MinimaxAgent
from tictactoe.models import AGENT_NAMES, load_agent, parse_kind, save_agent
from tictactoe.training import evaluate, finetune
from tictactoe.verify import report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["q", "td"], default="q")
    parser.add_argument("--episodes", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=None, help="random seed, for a reproducible run")
    args = parser.parse_args()

    kind = parse_kind(args.agent)
    assert kind is not None  # guaranteed by argparse choices
    if args.seed is not None:
        random.seed(args.seed)
    name = AGENT_NAMES[kind]
    agent = load_agent(kind)

    print("Before:")
    if report(agent, name) == 0:
        print("No opponent can beat it already; nothing to fix.")
        return

    print(f"\nFine-tuning for {args.episodes:,} episodes from random openings...")
    finetune(agent, args.episodes)

    print("\nAfter:")
    losses = report(agent, name)
    vs_minimax = evaluate(agent, MinimaxAgent(), 300)
    print(f"{name} vs minimax W/D/L = {vs_minimax['win']:.2f}/{vs_minimax['draw']:.2f}/{vs_minimax['loss']:.2f}")

    if losses or vs_minimax["loss"]:
        print("\nStill beatable, so the saved model was NOT changed. Try again with more --episodes.")
        raise SystemExit(1)
    save_agent(kind, agent)
    print("\nUnbeatable now. Saved to saved_models/.")


if __name__ == "__main__":
    main()
