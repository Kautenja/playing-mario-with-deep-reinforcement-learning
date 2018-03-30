"""An implementation of Deep Q-Learning."""
import numpy as np
import cv2
from keras.models import Model
from .agent import Agent
from .replay_queue import ReplayQueue


# the format string for this objects representation
_REPR_TEMPLATE = """
{}(
    model={},
    env={},
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
                 env,
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

        # TODO: validate env type

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
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.episodes = episodes
        # setup other instance members
        # TODO: parameterize the replay queue size (defaults to 20000)
        self.queue = ReplayQueue()

    def __repr__(self) -> str:
        """Return a debugging string of this agent."""
        return _REPR_TEMPLATE.format(
            self.__class__.__name__,
            self.model,
            self.env,
            self.learning_rate,
            self.discount_factor,
            self.exploration_rate,
            self.exploration_decay,
            self.episodes,
        )

    @property
    def num_actions(self) -> int:
        """Return the number of actions for this agent."""
        return self.model.output_shape[1]

    def predict_action(self, frames: list) -> tuple:
        """
        Predict an action from a stack of frames.

        Args:
            frames: the stack of frames to predict Q values from

        Returns:
            a tuple of:
                - the optimal action
                - the Q value for the optimal action

        """
        # predict the values of each action
        actions = self.model.predict(
            frames.reshape((1, *self.model.input_shape[1:])),
            batch_size=1
        )
        # draw a number in [0, 1] and explore if it's less than the
        # exploration rate
        if np.random.random() < self.exploration_rate:
            # select a random action. the output shape of the network implies
            # the action name by index, so use that shape as the upper bound
            optimal_action = np.random.randint(0, self.num_actions)
        else:
            # select the action with the highest estimated score as the
            # optimal action
            optimal_action = np.argmax(actions)

        # return the optimal action and its corresponding Q value
        return optimal_action, actions[0, optimal_action]

    def train(self, s, a, r, d, s2) -> None:
        """
        """
        targets = np.zeros((s.shape[0], self.num_actions))

        for index in range(len(s)):
            targets[index] = self.model.predict(
                s[index].reshape(self.model.input_shape),
                batch_size=1
            )
            a_prime = self.model.predict(
                s2[index].reshape(self.model.input_shape),
                batch_size=1
            )
            targets[i, a[i]] = r[i]
            if not d[i]:
                targets[i, a[i]] += self.discount_factor * np.max(a_prime)

        return self.model.train_on_batch(s, targets)

    def downsample(self, frame):
        """
        Down-sample the given frame from RGB to B&W with a reduced size.

        Args:
            frame: the frame to down-sample

        Returns:
            a down-sample B&W frame

        """
        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (84, 84))

    @property
    def initial_state(self):
        """
        """
        # reset the environment
        state = [self.env.reset()] * 4
        # convert the initial frame to 4 down-sampled frames
        black_buffer = np.array([self.downsample(frame) for frame in state]).T
        return black_buffer
        # sample a random action to start with
        # action = self.env.action_space.sample()
        # TODO: refactor this four out to be parameterized somehow, maybe
        # build_model will have to be passed or something
        # for index in range(4):
        #     next_state, reward, done, info = env.step(action=action)
        # return [self.env.step(action=action) for index in range(4)]

    def next_state(self, action):
        """
        """
        next_state = []
        reward = 0
        # iterate over the number of buffered frames
        for _ in range(4):
            # render the frame
            self.env.render()
            # make the step and observe the state, reward, done
            state, _reward, done, _ = self.env.step(action=action)
            # store the state and reward from this frame
            next_state.append(self.downsample(state))
            reward += _reward
            # TODO: is this necessary?
            reward = reward if not done else -10

        next_state = np.array(next_state).T

        return next_state, reward / 4, done

    def run(self):
        """
        """
        for episode in range(self.episodes):
            # reset the game and get the initial state
            state = self.initial_state
            # the done flag indicating that an episode has ended
            done = False
            # loop until done
            while not done:
                # predict the best action based on the current state
                action, Q = self.predict_action(state)
                # hold the action for the number of frames
                for _ in range(4):
                    # render the frame
                    self.env.render()
                    # make the step and observe the next_state, reward, done
                    next_state, reward, done, _ = self.env.step(action=action)
                    # TODO: is this necessary
                    reward = reward if not done else -10
                    # push this state to the replay queue
                    self.queue.push(state, action, reward, done, next_state)

    # def step(self):
    #     """
    #     TODO: description
    #     TODO: args and types
    #     TODO: return type
    #     TODO: implementation
    #     """
    #     # TODO: this is just example code
    #     # take a step and observe the environment variables
    #     next_state, reward, done, info = env.step(action)
    #     # calculate the target: r + γ max_a'(Q[s', a'])
    #     target = reward + gamma * np.argmax(model.predict(next_state))


# explicitly define the outward facing API of this module
__all__ = ['DeepQAgent']
