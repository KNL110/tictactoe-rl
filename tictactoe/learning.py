"""Per-move learning updates for games against an outside opponent (a human).

Self-play (training.py) controls both sides, so its updates live inside one tight
loop. Against a human, moves arrive one at a time (a terminal prompt, a web
request), so these learners keep the little state that carries over between plies
and are fed each move as it happens. Used by train_vs_human.py and the web app.

Both learners only update the agent's own moves. The bootstraps (negamax for
Q-learning, the 1-V(next) trick for TD) don't care who picked the opponent's actual
move, so a human is a drop-in swap for the self-play opponent.
"""
from tictactoe.env import available_actions
from tictactoe.utils import canonical_state


class QOnlineLearner:
    """Same negamax update as self-play Q-learning. Extra: if the human's move ends
    the game, the agent's *previous* move is also corrected with the true observed
    outcome (a Monte-Carlo-style backup) instead of waiting for many repeated
    self-play visits to smooth it out — with only a handful of human games to learn
    from, that patience isn't available."""

    def __init__(self, agent):
        self.agent = agent
        self.pending_agent_move = None  # (state, action) awaiting confirmation/correction

    def agent_moved(self, board, mover, action, next_board, winner, done):
        agent = self.agent
        state = canonical_state(board, mover)
        if done:
            reward = 1.0 if winner == mover else 0.5  # can't be a loss on your own move
            agent.update(state, action, reward)
            self.pending_agent_move = None
        else:
            next_state = canonical_state(next_board, -mover)
            next_qs = agent.Q[next_state]
            target = -agent.gamma * max(next_qs[a] for a in available_actions(next_board))
            agent.update(state, action, target)
            self.pending_agent_move = (state, action)

    def opponent_moved(self, board, mover, action, next_board, winner, done):
        if done and self.pending_agent_move is not None:
            prev_state, prev_action = self.pending_agent_move
            reward = 0.5 if winner == 0 else 0.0  # human drew or won -> agent drew or lost
            self.agent.update(prev_state, prev_action, reward)


class TDOnlineLearner:
    """Same 1-V(next) TD(0) update as self-play, with the human's afterstate as "next"."""

    def __init__(self, agent):
        self.agent = agent
        self.prev_agent_state = None  # agent's last afterstate, awaiting its bootstrap

    def agent_moved(self, board, mover, action, next_board, winner, done):
        state = canonical_state(next_board, mover)
        if done:
            reward = 1.0 if winner == mover else 0.5
            self.agent.update(state, reward)
        self.prev_agent_state = state

    def opponent_moved(self, board, mover, action, next_board, winner, done):
        if self.prev_agent_state is None:
            return
        if done:
            reward = 0.5 if winner == 0 else 0.0
            self.agent.update(self.prev_agent_state, reward)
        else:
            human_after_state = canonical_state(next_board, mover)
            self.agent.update(self.prev_agent_state, 1.0 - self.agent.V[human_after_state])


def make_learner(kind, agent):
    return QOnlineLearner(agent) if kind == "q" else TDOnlineLearner(agent)
