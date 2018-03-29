"""An implementation of Q Learning."""
import numpy as np


# the format string for this objects representation
_REPR_TEMPLATE = """
{}(
    learning_rate={},
    discount_factor={},
    exploration_rate={},
    exploration_decay={},
    episodes={},
    backup_file={}
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
                 episodes: int,
                 backup_file: str=None) -> None:
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
        self.backup_file = backup_file

    def __repr__(self) -> str:
        """Return a debugging string of this agent."""
        return _REPR_TEMPLATE.format(
            self.__class__.__name__,
            self.learning_rate,
            self.discount_factor,
            self.exploration_rate,
            self.exploration_decay,
            self.episodes
            self.backup_file
        )

    def load_model(self) -> None:
        """Load the model of the Q algorithm into memory."""
        raise NotImplementedError('TODO:')

    def save_model(self) -> None:
        """Backup the model of this agent to disk."""
        # if there is no file specified, use the class name
        if self.backup_file is None:
            backup_file = self.__class__.__name__
        # otherwise use the name stored in this object
        else:
            backup_file = self.backup_file
        # save the model of this Q algorithm to disk
        self.model.save(f'backup_file'.h5)

    def step(self):
        """
        TODO: description
        TODO: args and types
        TODO: return type
        TODO: implementation
        """
        # TODO: this is just example code
        # take a step and observe the environment variables
        next_state, reward, done, info = env.step(action)
        # calculate the target: r + γ max_a'(Q[s', a'])
        target = reward + gamma * np.argmax(model.predict(next_state))


# explicitly export the public API of this module
__all__ = ['DeepQAgent']
