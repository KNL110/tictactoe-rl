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
from tictactoe.agents.minimax import MinimaxAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.env import TicTacToeEnv, available_actions
from tictactoe.models import new_agent
from tictactoe.utils import canonical_state


def epsilon_schedule(episode, total_episodes, start=0.5, end=0.0):
    frac = min(episode / (0.9 * total_episodes), 1.0)
    return start + frac * (end - start)


def alpha_schedule(episode, total_episodes, start=0.3, end=0.02):
    frac = min(episode / total_episodes, 1.0)
    return start + frac * (end - start)


def play_q_learning_episode(env, agent):
    """One self-play episode with a negamax-style Q-learning update.

    Q(s, a) is the value of action a to whoever is about to move. After a
    non-terminal move, the very next decision belongs to the opponent, and in
    a zero-sum game "good for the opponent" == "bad for me" — so the update
    bootstraps off -max(Q(opponent's resulting state)) immediately, one ply
    ahead, instead of waiting two plies for this player's own next turn.
    This mirrors the 1-V(next) trick in the TD value agent and converges far
    more cleanly than the two-ply version (which doubles the max-bootstrap
    bias and was noisier in practice).
    """
    env.reset()
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


def play_td_episode(env, agent):
    """One self-play episode with single-ply TD(0) updates via the 1-V(next) trick
    (see the docstring in agents/td_value.py for why this is valid)."""
    env.reset()
    prev_state = None

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


def play_episode_vs_human(env, learner, agent_side, human_move_fn, on_move=None):
    """Like the self-play episodes, but one side's actions come from `human_move_fn`,
    and only the agent's moves are learned from (see tictactoe/learning.py)."""
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


def evaluate(agent, opponent, n_games=200):
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


def train(agent_kind, episodes, eval_every, eval_games):
    if agent_kind == "q":
        play_episode = play_q_learning_episode
    elif agent_kind == "td":
        play_episode = play_td_episode
    else:
        raise ValueError(agent_kind)
    agent = new_agent(agent_kind)

    random_opponent = RandomAgent()
    minimax_opponent = MinimaxAgent()
    history = {"episode": [], "vs_random": [], "vs_minimax": []}

    for ep in range(1, episodes + 1):
        agent.epsilon = epsilon_schedule(ep, episodes)
        agent.alpha = alpha_schedule(ep, episodes)
        play_episode(TicTacToeEnv(), agent)

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
