"""An implementation of Q Learning."""
import numpy as np


# take a step and observe the environment variables
next_state, reward, done, info = env.step(action)
# calculate the target: r + γ max_a'(Q[s', a'])
target = reward + gamma * np.argmax(model.predict(next_state))


# the format string for this objects representation
_REPR = """
{}(
    learning_rate={},
    discount_factor={},
    exploration_rate={},
    exploration_decay={},
    episodes={}
)
""".lstrip()


class DeepQAgent(object):
    """
    Algorithm:
        initialize Q[s, a]
        observe initial state s
        while not done:
          select and perform an action a
          observe a reward r and new state s'
          Q[s, a] ←  Q[s, a] + α(r + γ max_a'(Q[s', a']) - Q[s, a])
          s ← s'

    Note:
    -   when α = 1, this reduces to vanilla Bellman Equation
      -   Q[s, a] ← r + γmax_a'(Q[s', a'])
    -   another formulation uses the probabilistic inverse of the
        learning rate as a factor for original Q value:
      -   Q[s, a] ←  (1 - α)Q[s, a] + α(r + γ max_a'(Q[s', a']) - Q[s, a])
    """

    def __init__(self,
                 learning_rate: float,
                 discount_factor: float,
                 exploration_rate: float,
                 exploration_decay: float,
                 episodes: int) -> None:
        """
        Initialize a new Deep Q Agent.

        Args:
            learning_rate: the learning rate, α
            discount_factor: the discount factor, γ
            exploration_rate: the exploration rate, ε
            exploration_decay: the decay factor for exploration rate
            episodes: the number of episodes for the agent to experience

        Returns: None
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.episodes = episodes

    def __repr__(self) -> str:
        """Return a debugging string of this agent."""
        return _REPR.format(*[
            self.__class__.__name__,
            self.learning_rate,
            self.discount_factor,
            self.exploration_rate,
            self.exploration_decay,
            self.episodes
        ])


__all__ = ['DeepQAgent']
