"""Training-curve plots (matplotlib is only needed here, so it's imported lazily)."""


def plot_history(histories, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    colors = {"win": "tab:green", "draw": "tab:blue", "loss": "tab:red"}

    for col, (kind, title) in enumerate([("q", "Q-learning"), ("td", "TD value learning")]):
        history = histories[kind]
        for row, opponent_key in enumerate(["vs_random", "vs_minimax"]):
            ax = axes[row][col]
            for outcome in ["win", "draw", "loss"]:
                ax.plot(
                    history["episode"],
                    [h[outcome] for h in history[opponent_key]],
                    label=outcome,
                    color=colors[outcome],
                )
            opponent_name = "random" if opponent_key == "vs_random" else "minimax (perfect)"
            ax.set_title(f"{title} vs {opponent_name}")
            ax.set_ylabel("rate")
            ax.set_ylim(-0.02, 1.02)
            if row == 1:
                ax.set_xlabel("training episodes")
            ax.legend(loc="center right", fontsize=8)
            ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved training curves to {path}")
