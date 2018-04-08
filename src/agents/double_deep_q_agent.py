"""An implementation of Double Deep Q-Learning."""
import numpy as np
import tensorflow as tf
from typing import Callable
from tqdm import tqdm
from keras.optimizers import RMSprop
from src.models import build_deep_mind_model
from src.base import AnnealingVariable
from src.downsamplers import Downsampler
from .replay_queue import ReplayQueue
from .deep_q_agent import DeepQAgent


# the format string for this objects representation
_REPR_TEMPLATE = """
{}(
    env={},
    downsample={},
    replay_memory_size={},
    agent_history_length={},
    discount_factor={},
    update_frequency={},
    optimizer={},
    exploration_rate={},
    null_op_max={},
    loss={},
    image_size={},
    target_update_freq={}
)
""".lstrip()


# the operation for null ops
NULL_OP = 0


class DoubleDeepQAgent(DeepQAgent):
    """The Double Deep Q reinforcement learning algorithm."""

    def __init__(self, env, downsample: Downsampler,
        replay_memory_size: int=1000000,
        agent_history_length: int=4,
        discount_factor: float=0.99,
        update_frequency: int=4,
        optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
        exploration_rate=AnnealingVariable(1.0, 0.1, 1000000),
        null_op_max: int=30,
        loss=tf.losses.huber_loss,
        image_size: tuple=(84, 84),
        target_update_freq: int=10000
    ) -> None:
        """
        Initialize a new Deep Q Agent.

        Args:
            env: the environment to run on
            downsample: the down-sampler for the Gym environment
            agent_history_length: the number of previous frames for the agent
                                  to make new decisions based on
            discount_factor: the discount factor, γ
            update_frequency: the number of actions between updates to the
                              deep Q network
            optimizer: the optimization method to use on the CNN
            exploration_rate: the exploration rate, ε, expected as an
                              AnnealingVariable subclass
            null_op_max: the maximum number of random null ops at the start of
                         each new game. this adds stochastic "human start" to
                         the training and playing process for the agent
            loss: the loss method to use from TF or Keras
            image_size: the size of the images to pass to the CNN
            target_update_freq: the frequency with which to update the target
                                network

        Returns:
            None

        """
        self.env = env
        self.downsample = downsample
        self.queue = ReplayQueue(replay_memory_size)
        self.agent_history_length = agent_history_length
        self.discount_factor = discount_factor
        self.update_frequency = update_frequency
        self.optimizer = optimizer
        self.exploration_rate = exploration_rate
        self.null_op_max = null_op_max
        self.loss = loss
        self.image_size = image_size
        self.target_update_freq = target_update_freq
        # setup the buffer for frames the agent uses to predict on
        self.frame_buffer = np.zeros((*image_size, agent_history_length))
        # build the neural model for estimating Q values
        self.model = build_deep_mind_model(
            image_size=image_size,
            num_frames=agent_history_length,
            num_actions=env.action_space.n,
            loss=loss,
            optimizer=optimizer
        )
        # build the target model for estimating target values
        self.target_model = build_deep_mind_model(
            image_size=image_size,
            num_frames=agent_history_length,
            num_actions=env.action_space.n,
            loss=loss,
            optimizer=optimizer
        )

    def __repr__(self) -> str:
        """Return a debugging string of this agent."""
        return _REPR_TEMPLATE.format(
            self.__class__.__name__,
            self.env,
            self.downsample,
            self.queue.size,
            self.agent_history_length,
            self.discount_factor,
            self.update_frequency,
            self.optimizer,
            self.exploration_rate,
            self.null_op_max,
            self.loss,
            self.image_size,
            self.target_update_freq
        )

    def _replay(self, s, a, r, d, s2) -> float:
        """
        Train the network on a mini-batch of replay data.

        Notes:
            all arguments are arrays that should be of the same size

        Args:
            s: a batch of current states
            a: a batch of actions from each state in s
            r: a batch of reward from each action in a
            d: a batch of terminal flags after each action in a
            s2: the next state from each state, action pair in s, a

        Returns:
            the loss as a result of the training

        """
        # initialize target y values as a matrix of zeros
        y = np.zeros((len(s), self.env.action_space.n))

        # predict Q values for the next state of each memory in the batch and
        # take the maximum value. dont mask any outputs, i.e. use ones
        all_mask = np.ones((len(s), self.env.action_space.n))
        Q = np.max(self.target_model.predict_on_batch([s2, all_mask]), axis=1)
        # terminal states have a Q value of zero by definition
        Q[d] = 0
        # set the y value for each sample to the reward of the selected
        # action plus the discounted Q value
        y[range(y.shape[0]), a] = r + self.discount_factor * Q

        # use an identity of size action space, and index rows from it using
        # the action vector to produce a one-hot matrix representing the mask
        action_mask = np.eye(self.env.action_space.n)[a]
        # train the model on the batch and return the loss. use the mask that
        # disables training for actions that aren't the selected actions.
        return self.model.train_on_batch([s, action_mask], y)

    def train(self,
        frames_to_play: int=50000000,
        batch_size: int=32,
        callback: Callable=None,
    ) -> None:
        """
        Train the network for a number of episodes (games).

        Args:
            frames: the number of frames to play the game for
            batch_size: the size of the replay history batches
            null_op_max: the max number of random null ops at the beginning
                         of an episode to introduce stochasticity
            callback: an optional callback to get updates about the score,
                      loss, discount factor, and exploration rate every
                      episode

        Returns:
            None

        """
        # the progress bar for the operation
        progress = tqdm(total=frames_to_play, unit='frame')
        # loop indefinitely
        while True:
            # the done flag indicating that an episode has ended
            done = False
            # metrics throughout this episode
            score = 0
            loss = 0
            frames = 0
            # reset the game and get the initial state
            state = self._initial_state()
            # perform NOPs randomly
            for k in range(np.random.randint(0, self.null_op_max)):
                state, reward, done = self._next_state(NULL_OP)
            # loop until the episode ends
            while not done:
                # predict the best action based on the current state
                action = self.predict_action(state, self.exploration_rate.value)
                # step the exploration rate forward
                self.exploration_rate.step()
                # fire the action and observe the next state, reward, and flag
                next_state, reward, done = self._next_state(action)
                # add the reward to the cumulative score
                score += reward
                # push the memory onto the replay queue
                self.queue.push(state, action, reward, done, next_state)
                # set the state to the new state
                state = next_state
                # decrement the observation counter
                frames_to_play -= 1
                frames += 1
                # update Q from replay
                if frames_to_play % self.update_frequency == 0:
                    loss += self._replay(*self.queue.sample(size=batch_size))
                # update Target Q from online Q
                if frames_to_play % self.target_update_freq == 0:
                    self.target_model.set_weights(self.model.get_weights())
                # break out if done observing
                if frames_to_play <= 0:
                    # update and close the progress bar
                    progress.update(frames)
                    progress.close()
                    return

            # pass the score to the callback at the end of the episode
            if callable(callback):
                callback(score, loss)
            # update the progress bar
            progress.update(frames)


# explicitly define the outward facing API of this module
__all__ = ['DoubleDeepQAgent']
