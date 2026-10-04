"""Self-play training for two tabular RL agents on Tic-Tac-Toe:

  1. Q-learning       (agents/q_learning.py)
  2. TD(0) value learning, Sutton & Barto style (agents/td_value.py)

Both train purely by playing against themselves (one policy controls both X and O),
and are periodically evaluated (greedy, no exploration) against a random opponent
and a perfect minimax opponent so you can watch them approach optimal play:
  - vs random:  win rate should climb toward ~100%
  - vs minimax: loss rate should fall to 0% (a perfect agent can never be beaten,
    but a perfect Tic-Tac-Toe player still only draws minimax, never wins)
"""
import random
from collections.abc import Callable
from typing import TypedDict

from tictactoe.agents import Agent
from tictactoe.agents.minimax import MinimaxAgent
from tictactoe.agents.q_learning import QLearningAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.agents.td_value import TDValueAgent
from tictactoe.env import Action, Board, Player, TicTacToeEnv, available_actions
from tictactoe.learning import OnlineLearner
from tictactoe.models import AgentKind, LearningAgent, new_agent
from tictactoe.utils import canonical_state

type EvalResult = dict[str, float]  # "win" / "draw" / "loss" -> rate
type HumanMoveFn = Callable[[Board, list[Action]], Action]
type MoveCallback = Callable[[Player, Action, Board, Player, bool], None]


class History(TypedDict):
    episode: list[int]
    vs_random: list[EvalResult]
    vs_minimax: list[EvalResult]


def epsilon_schedule(episode: int, total_episodes: int, start: float = 0.5, end: float = 0.0) -> float:
    frac = min(episode / (0.9 * total_episodes), 1.0)
    return start + frac * (end - start)


def alpha_schedule(episode: int, total_episodes: int, start: float = 0.3, end: float = 0.02) -> float:
    frac = min(episode / total_episodes, 1.0)
    return start + frac * (end - start)


def random_opening(env: TicTacToeEnv, n_moves: int) -> bool:
    """Exploring start: play `n_moves` uniformly random moves (not learned from).
    Returns True if that already ended the game."""
    for _ in range(n_moves):
        _, _, done = env.step(random.choice(available_actions(env.board)))
        if done:
            return True
    return False


def play_q_learning_episode(env: TicTacToeEnv, agent: QLearningAgent, opening_moves: int = 0) -> None:
    """One self-play episode with a negamax-style Q-learning update.

    Q(s, a) is the value of action a to whoever is about to move. After a
    non-terminal move, the very next decision belongs to the opponent, and in
    a zero-sum game "good for the opponent" == "bad for me" — so the update
    bootstraps off -max(Q(opponent's resulting state)) immediately, one ply
    ahead, instead of waiting two plies for this player's own next turn.
    This mirrors the 1-V(next) trick in the TD value agent and converges far
    more cleanly than the two-ply version (which doubles the max-bootstrap
    bias and was noisier in practice).

    `opening_moves` starts the episode from a random position (see random_opening).
    """
    env.reset()
    if random_opening(env, opening_moves):
        return
    while True:
        mover = env.player
        board = env.board
        actions = available_actions(board)
        state = canonical_state(board, mover)
        action = agent.choose_action(board, mover, actions)
        next_board, winner, done = env.step(action)

        if done:
            reward = 1.0 if winner == mover else 0.5  # can't be a loss on your own move
            agent.update(state, action, reward)
            return
        else:
            next_mover = -mover
            next_state = canonical_state(next_board, next_mover)
            next_actions = available_actions(next_board)
            next_qs = agent.Q[next_state]
            target = -agent.gamma * max(next_qs[a] for a in next_actions)
            agent.update(state, action, target)


def play_td_episode(env: TicTacToeEnv, agent: TDValueAgent, opening_moves: int = 0) -> None:
    """One self-play episode with single-ply TD(0) updates via the 1-V(next) trick
    (see the docstring in agents/td_value.py for why this is valid)."""
    env.reset()
    if random_opening(env, opening_moves):
        return
    prev_state: Board | None = None

    while True:
        mover = env.player
        board = env.board
        actions = available_actions(board)
        action = agent.choose_action(board, mover, actions)
        next_board, winner, done = env.step(action)
        state = canonical_state(next_board, mover)

        if done:
            reward = 1.0 if winner == mover else 0.5
            agent.update(state, reward)
            if prev_state is not None:
                agent.update(prev_state, 1.0 - reward)
            return
        else:
            if prev_state is not None:
                agent.update(prev_state, 1.0 - agent.V[state])
            prev_state = state


def play_self_play_episode(env: TicTacToeEnv, agent: LearningAgent, opening_moves: int = 0) -> None:
    if isinstance(agent, QLearningAgent):
        play_q_learning_episode(env, agent, opening_moves)
    else:
        play_td_episode(env, agent, opening_moves)


def play_episode_vs_human(
    env: TicTacToeEnv,
    learner: OnlineLearner,
    agent_side: Player,
    human_move_fn: HumanMoveFn,
    on_move: MoveCallback | None = None,
) -> Player:
    """Like the self-play episodes, but one side's actions come from `human_move_fn`,
    and only the agent's moves are learned from (see tictactoe/learning.py).
    Returns the winner (0 for a draw)."""
    env.reset()
    agent = learner.agent

    while True:
        mover = env.player
        board = env.board
        actions = available_actions(board)

        if mover == agent_side:
            action = agent.choose_action(board, mover, actions)
        else:
            action = human_move_fn(board, actions)

        next_board, winner, done = env.step(action)
        if on_move is not None:
            on_move(mover, action, next_board, winner, done)

        if mover == agent_side:
            learner.agent_moved(board, mover, action, next_board, winner, done)
        else:
            learner.opponent_moved(board, mover, action, next_board, winner, done)

        if done:
            return winner


def evaluate(agent: Agent, opponent: Agent, n_games: int = 200) -> EvalResult:
    """Greedy (no exploration) evaluation: `agent` plays n_games/2 as X and n_games/2 as O."""
    env = TicTacToeEnv()
    results = {"win": 0, "draw": 0, "loss": 0}

    for i in range(n_games):
        agent_side = 1 if i % 2 == 0 else -1
        env.reset()
        while True:
            actions = available_actions(env.board)
            mover = env.player
            if mover == agent_side:
                action = agent.choose_action(env.board, mover, actions, greedy=True)
            else:
                action = opponent.choose_action(env.board, mover, actions, greedy=True)
            _, winner, done = env.step(action)
            if done:
                if winner == 0:
                    results["draw"] += 1
                elif winner == agent_side:
                    results["win"] += 1
                else:
                    results["loss"] += 1
                break

    total = sum(results.values())
    return {k: v / total for k, v in results.items()}


def train(agent_kind: AgentKind, episodes: int, eval_every: int, eval_games: int) -> tuple[LearningAgent, History]:
    agent = new_agent(agent_kind)
    random_opponent = RandomAgent()
    minimax_opponent = MinimaxAgent()
    history: History = {"episode": [], "vs_random": [], "vs_minimax": []}

    for ep in range(1, episodes + 1):
        agent.epsilon = epsilon_schedule(ep, episodes)
        agent.alpha = alpha_schedule(ep, episodes)
        play_self_play_episode(TicTacToeEnv(), agent)

        if ep % eval_every == 0 or ep == episodes:
            vs_random = evaluate(agent, random_opponent, eval_games)
            vs_minimax = evaluate(agent, minimax_opponent, eval_games)
            history["episode"].append(ep)
            history["vs_random"].append(vs_random)
            history["vs_minimax"].append(vs_minimax)
            print(
                f"[{agent_kind}] ep {ep:>6} | eps={agent.epsilon:.3f} | "
                f"vs random  W/D/L = {vs_random['win']:.2f}/{vs_random['draw']:.2f}/{vs_random['loss']:.2f}  | "
                f"vs minimax W/D/L = {vs_minimax['win']:.2f}/{vs_minimax['draw']:.2f}/{vs_minimax['loss']:.2f}"
            )

    return agent, history


def finetune(agent: LearningAgent, episodes: int, max_opening_moves: int = 6) -> LearningAgent:
    """Keep training an already-trained agent from random starting positions.

    Plain self-play converges on its own favorite lines, so positions that only
    arise after an odd opening (like an edge first move) can stay barely visited,
    and an unvisited position looks like a draw thanks to the optimistic 0.5
    default. Random openings force those positions to be played out; exploration
    and learning rate anneal to zero again so the fixes settle instead of being
    shaken loose by noise.
    """
    env = TicTacToeEnv()
    for ep in range(1, episodes + 1):
        agent.epsilon = epsilon_schedule(ep, episodes, start=0.3)
        agent.alpha = alpha_schedule(ep, episodes, start=0.2, end=0.01)
        play_self_play_episode(env, agent, opening_moves=random.randint(0, max_opening_moves))
    agent.epsilon = 0.0
    return agent
