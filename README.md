# Tic-Tac-Toe RL

Two tabular reinforcement learning agents, Q-learning and TD(0) value learning,
that learn to play Tic-Tac-Toe perfectly through self-play. Neither agent uses
labeled data or a hand-coded strategy; both learn from game outcomes alone.

**Live demo:** <https://knl110.github.io/tictactoe-rl/>

## Results

After 1,000,000 self-play episodes, both agents draw every game against a perfect
minimax opponent, the best achievable result in a game that is a forced draw under
optimal play.

![training curves](plots/training_curves.png)

- **Top row:** results against a random opponent. The win rate climbs toward 100%.
- **Bottom row:** results against a perfect minimax opponent. Winning is impossible
  here, so the meaningful signal is the loss rate falling to zero as draws take over.

Results against minimax only cover the lines that minimax happens to play. An
exhaustive check (see [Verification](#verification)) confirms that no sequence of
opponent moves beats either saved agent, from either side of the board.

## Agents

### Q-learning (`tictactoe/agents/q_learning.py`)

Learns `Q(s, a)`, the expected outcome of playing move `a` in position `s`. This is
classic off-policy, value-based RL, the same family as DQN, with a table in place
of a neural network. A table is tractable here because Tic-Tac-Toe has only about
4,500 distinct positions (from the mover's perspective).

### TD(0) value learning (`tictactoe/agents/td_value.py`)

The Tic-Tac-Toe example from Chapter 1 of Sutton & Barto's *Reinforcement Learning:
An Introduction*. It learns `V(s)` for afterstates (the board immediately after a
move): the probability that the player who just moved goes on to win, with a draw
counted as 0.5. It picks the move whose resulting afterstate has the highest value.

### Benchmark opponents

`tictactoe/agents/random_agent.py` (uniformly random moves) and
`tictactoe/agents/minimax.py` (memoized negamax search, perfect play) are used
only to evaluate the learned agents during training.

## Techniques

**Canonical states.** Both agents view the board from the mover's perspective
(`tictactoe/utils.py`): the mover's pieces are `+1` and the opponent's are `-1`. A
single table therefore serves both X and O, and every self-play game trains both
sides.

**Negamax-style bootstrapping.** In a zero-sum game, a position that is good for the
opponent is bad for the mover. Both agents bootstrap from the opponent's very next
position instead of waiting a full turn:

- Q-learning target: `-max_a' Q(next_state, a')`
- TD target: `1 - V(next_afterstate)`, valid because
  `P(win) + P(loss) + P(draw) = 1` and a draw is worth 0.5 to both players.

**Optimistic initialization.** Unvisited entries start at 0.5 ("assume a draw")
rather than 0. With a zero default, an under-explored move permanently looks worse
than any mediocre move already tried, so it stops being revisited once exploration
decays. This change raised Q-learning's draw rate against minimax from about 90% to 100%.

**Exploration and learning-rate schedules.** Training starts exploratory
(`epsilon=0.5`, `alpha=0.3`) and anneals toward pure exploitation (`epsilon=0`,
`alpha=0.02`), so early training covers the state space broadly and late training
settles into a stable policy.

**Exploring starts.** `finetune.py` continues training a saved agent from random
opening positions (see below).

## Verification

`tictactoe/verify.py` walks the complete game tree: every possible opponent move,
combined with every move the greedy agent might choose when it breaks a tie at
random, with the agent playing both X and O.

This check found a weakness in the originally trained Q-learning agent that the
minimax evaluation had missed. After an edge opening (X on 5), one tie-break line
led the agent into a position where X could create a fork and win: 4 losing games
out of 2,744. Self-play rarely opens on an edge, so those positions were barely
visited, and an unvisited position looks like a draw under the optimistic 0.5
default.

The TD agent had no such gap. Its afterstate table needs about 5,500 values instead
of about 16,000 position-move pairs, and each afterstate can be reached by several
move orders, so its experience generalizes across them.

`finetune.py` repaired the Q-learning agent by continuing training from random
openings (exploring starts), with exploration and learning rate annealed back to
zero. It saves the model only when the exhaustive check finds no losing line and
the agent still never loses to minimax. Both saved agents now pass:

| Agent | Agent plays X | Agent plays O |
| --- | --- | --- |
| Q-learning | 1,244 possible games, 0 losses | 2,770 possible games, 0 losses |
| TD value | 83 possible games, 0 losses | 486 possible games, 0 losses |

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The code is fully type-annotated and checked with strict mypy; ruff handles linting:

```bash
uv run ruff check .
uv run mypy .
```

## Usage

```bash
uv run train.py                         # train both agents from scratch (a few minutes); saves models and plot
uv run play.py --agent q --first human  # play the Q-learning agent in the terminal
uv run play.py --agent td --first agent # play the TD agent, agent moves first
uv run train_vs_human.py --agent q      # play the agent while it keeps learning from the games
uv run -m tictactoe.verify              # exhaustive check of both saved agents
uv run finetune.py --agent q            # repair weak spots found by the check
```

### Learning from human games

`train_vs_human.py` applies the same update rules as self-play, with the human's
moves in place of the self-play opponent. It continues from the pretrained model by
default (`--fresh` starts from a blank table), saves the table after every game,
and alternates sides between games. When a human move ends the game, it also
corrects the agent's previous move directly from the observed outcome, since a
human supplies far fewer games than self-play.

A fresh agent trained this way remains weak: self-play needed on the order of a
million games to reach perfect play, while a human supplies a few dozen per session.

## Web app

A browser version, playable on a phone, offers both agents, a choice of first
player, pretrained or blank ("fresh") agents, optional learning during play, and an
adjustable exploration rate.

```bash
uv run -m webapp   # Flask dev server on http://<lan-ip>:5000
./serve.sh         # same, plus a temporary public Cloudflare tunnel URL
```

Each browser session has its own game and its own copy of the agents, so concurrent
visitors do not interfere, and learning during web games is kept in memory without
modifying `saved_models/`. `serve.sh` sets `TTT_PERSIST=1`, which instead saves
learning from local games back to the models.

### Deployment

The public demo is a static site on GitHub Pages. `build_static.py` packages the
page with the same Python game code (`webapp/api.py`, `webapp/game.py`,
`tictactoe/`) and the saved models, and [Pyodide](https://pyodide.org) runs that
code in the visitor's browser, so no server is required.
`.github/workflows/pages.yml` rebuilds and publishes the site on every push to `main`.

```bash
uv run build_static.py && python -m http.server -d site   # local preview on port 8000
```

The `Dockerfile` packages the Flask app for server hosting (e.g. Render). It runs
gunicorn with a single worker, since games are held in process memory, and listens
on `$PORT`.

## Project structure

| Path | Purpose |
| --- | --- |
| `tictactoe/env.py` | Game rules: board, win detection, step/reset |
| `tictactoe/utils.py` | State canonicalization, pickle helpers |
| `tictactoe/agents/` | Q-learning, TD(0), random, and minimax agents |
| `tictactoe/models.py` | Saved-model paths, loading and saving agents |
| `tictactoe/training.py` | Self-play episodes, training, fine-tuning, evaluation, episodes against a human |
| `tictactoe/learning.py` | Per-move learning updates against a human (CLI and web app) |
| `tictactoe/verify.py` | Exhaustive game-tree check |
| `tictactoe/plotting.py` | Training curves |
| `tictactoe/console.py` | Terminal board rendering and move input |
| `train.py`, `finetune.py`, `play.py`, `train_vs_human.py` | Command-line entry points |
| `webapp/api.py` | JSON API as plain functions, shared by Flask and the static build |
| `webapp/app.py`, `webapp/game.py`, `webapp/sessions.py` | Flask routes, per-visitor game, session store |
| `webapp/templates/index.html` | Browser UI |
| `build_static.py`, `static/` | Static GitHub Pages build (Pyodide) |
| `serve.sh` | Local server with a temporary public tunnel |
| `Dockerfile` | Container image for server hosting |
| `saved_models/` | Trained Q and V tables |

## Possible extensions

- Replace the tabular `Q`/`V` with a small neural network and compare sample efficiency.
- Larger boards (e.g. 4x4) where exhaustive search is impractical.
- A SARSA agent (on-policy) for comparison with Q-learning's off-policy updates.
