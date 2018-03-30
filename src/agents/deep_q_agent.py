"""An implementation of Deep Q-Learning."""
from keras.models import Model
from .agent import Agent


# the format string for this objects representation
_REPR_TEMPLATE = """
{}(
    model={},
    learning_rate={},
    discount_factor={},
    exploration_rate={},
    exploration_decay={},
    episodes={}
)
""".lstrip()


class DeepQAgent(Agent):
    """
    An implementation of Deep Q-Learning.

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
                 model: Model,
                 learning_rate: float=0.001,
                 discount_factor: float=0.99,
                 exploration_rate: float=0.8,
                 exploration_decay: float=0.99,
                 episodes: int=1000) -> None:
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
        # verify model
        if not isinstance(model, Model):
            raise TypeError('model must be of type: keras.models.Model')
        # verify learning_rate
        if not isinstance(learning_rate, float):
            raise TypeError('learning_rate must be of type float')
        if learning_rate < 0:
            raise ValueError('learning_rate must be positive')
        # verify discount_factor
        if not isinstance(discount_factor, float):
            raise TypeError('discount_factor must be of type float')
        if discount_factor < 0:
            raise ValueError('discount_factor must be positive')
        # verify exploration_rate
        if not isinstance(exploration_rate, float):
            raise TypeError('exploration_rate must be of type float')
        if not 0 <= exploration_rate <= 1:
            raise ValueError('exploration_rate must be in [0,1]')
        # verify exploration_decay
        if not isinstance(exploration_decay, float):
            raise TypeError('exploration_decay must be of type float')
        if exploration_decay < 0:
            raise ValueError('exploration_decay must be positive')
        # verify episodes
        if not isinstance(episodes, int):
            raise TypeError('episodes must be of type int')
        if episodes < 0:
            raise ValueError('episodes must be positive')
        # assign args to self
        self.model = model
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.episodes = episodes

    def __repr__(self) -> str:
        """Return a debugging string of this agent."""
        return _REPR_TEMPLATE.format(
            self.__class__.__name__,
            self.model,
            self.learning_rate,
            self.discount_factor,
            self.exploration_rate,
            self.exploration_decay,
            self.episodes,
        )

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


# explicitly define the outward facing API of this module
__all__ = ['DeepQAgent']
