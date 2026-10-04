"""Where trained tables live on disk, and how to turn them back into agents."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from tictactoe.agents.q_learning import QLearningAgent
from tictactoe.agents.td_value import TDValueAgent
from tictactoe.utils import load_pickle, save_pickle

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS = {
    "q": ROOT / "saved_models" / "q_agent.pkl",
    "td": ROOT / "saved_models" / "td_agent.pkl",
}
AGENT_NAMES = {"q": "Q-learning", "td": "TD value"}


def new_agent(kind, epsilon=0.0, table=None):
    """A blank agent, or one starting from `table` (a saved Q/V dict). The table's
    values are used as-is, not copied — pass copy_table(...) if it must stay untouched."""
    if kind == "q":
        agent = QLearningAgent(epsilon=epsilon)
        if table is not None:
            agent.Q = defaultdict(lambda: np.full(9, 0.5), table)
    elif kind == "td":
        agent = TDValueAgent(epsilon=epsilon)
        if table is not None:
            agent.V = defaultdict(lambda: 0.5, table)
    else:
        raise ValueError(kind)
    return agent


def load_table(kind):
    """The saved table for `kind`, or an empty one if it hasn't been trained yet."""
    path = MODEL_PATHS[kind]
    return load_pickle(path) if path.exists() else {}


def load_agent(kind, epsilon=0.0):
    return new_agent(kind, epsilon, load_table(kind))


def table_of(agent):
    return agent.Q if hasattr(agent, "Q") else agent.V


def copy_table(table):
    """Deep enough copy that updates can't leak back: Q-values are numpy arrays
    updated in place, so each one needs its own copy."""
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in table.items()}


def save_agent(kind, agent):
    save_pickle(dict(table_of(agent)), MODEL_PATHS[kind])
