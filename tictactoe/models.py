"""Where trained tables live on disk, and how to turn them back into agents."""
from collections import defaultdict
from pathlib import Path
from typing import Literal, overload

import numpy as np

from tictactoe.agents.q_learning import QLearningAgent, QTable, default_q_values
from tictactoe.agents.td_value import TDValueAgent, VTable, default_value
from tictactoe.utils import load_pickle, save_pickle

type AgentKind = Literal["q", "td"]
type LearningAgent = QLearningAgent | TDValueAgent
type Table = QTable | VTable

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS: dict[AgentKind, Path] = {
    "q": ROOT / "saved_models" / "q_agent.pkl",
    "td": ROOT / "saved_models" / "td_agent.pkl",
}
AGENT_NAMES: dict[AgentKind, str] = {"q": "Q-learning", "td": "TD value"}


def parse_kind(value: object) -> AgentKind | None:
    """`value` as an AgentKind if it names one (e.g. from a request), else None."""
    if value == "q":
        return "q"
    if value == "td":
        return "td"
    return None


@overload
def new_agent(kind: Literal["q"], epsilon: float = 0.0, table: Table | None = None) -> QLearningAgent: ...
@overload
def new_agent(kind: Literal["td"], epsilon: float = 0.0, table: Table | None = None) -> TDValueAgent: ...
@overload
def new_agent(kind: AgentKind, epsilon: float = 0.0, table: Table | None = None) -> LearningAgent: ...
def new_agent(kind: AgentKind, epsilon: float = 0.0, table: Table | None = None) -> LearningAgent:
    """A blank agent, or one starting from `table` (a saved Q/V dict). The table's
    values are used as-is, not copied — pass copy_table(...) if it must stay untouched."""
    if kind == "q":
        q_agent = QLearningAgent(epsilon=epsilon)
        if table is not None:
            q_agent.Q = defaultdict(default_q_values, table)  # type: ignore[arg-type]
        return q_agent
    td_agent = TDValueAgent(epsilon=epsilon)
    if table is not None:
        td_agent.V = defaultdict(default_value, table)  # type: ignore[arg-type]
    return td_agent


def kind_of(agent: LearningAgent) -> AgentKind:
    return "q" if isinstance(agent, QLearningAgent) else "td"


def load_table(kind: AgentKind) -> Table:
    """The saved table for `kind`, or an empty one if it hasn't been trained yet."""
    path = MODEL_PATHS[kind]
    table: Table = load_pickle(path) if path.exists() else {}
    return table


def load_agent(kind: AgentKind, epsilon: float = 0.0) -> LearningAgent:
    return new_agent(kind, epsilon, load_table(kind))


def table_of(agent: LearningAgent) -> Table:
    return agent.Q if isinstance(agent, QLearningAgent) else agent.V


def copy_table(table: Table) -> Table:
    """Deep enough copy that updates can't leak back: Q-values are numpy arrays
    updated in place, so each one needs its own copy."""
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in table.items()}  # type: ignore[return-value]


def save_agent(kind: AgentKind, agent: LearningAgent) -> None:
    save_pickle(dict(table_of(agent)), MODEL_PATHS[kind])
