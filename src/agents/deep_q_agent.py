"""An implementation of Deep Q-Learning."""
from typing import Callable
import gym
import numpy as np
from tqdm import tqdm
from keras.optimizers import Optimizer
from keras.optimizers import Adam
from src.models import build_deep_q_model
from src.models import build_dueling_deep_q_model
from src.models.losses import huber_loss
from src.base import AnnealingVariable
from src.base import ReplayQueue
from src.base import PrioritizedReplayQueue
from .agent import Agent


# the format string for this objects representation
_REPR_TEMPLATE = """
{}(
    env={},
    render_mode={}
    replay_memory_size={},
    prioritized_experience_replay={},
    discount_factor={},
    update_frequency={},
    optimizer={},
    exploration_rate={},
    loss={},
    target_update_freq={},
    dueling_network={}
)
""".lstrip()


class DeepQAgent(Agent):
    """The Deep Q reinforcement learning algorithm."""

    def __init__(self,
        env: gym.Env,
        render_mode: str=None,
        replay_memory_size: int=750000,
        prioritized_experience_replay: bool=False,
        discount_factor: float=0.99,
        update_frequency: int=4,
        optimizer: Optimizer=Adam(lr=2e-5),
        exploration_rate: AnnealingVariable=AnnealingVariable(1., .1, 1000000),
        loss: Callable=huber_loss,
        target_update_freq: int=10000,
        dueling_network: bool=False,
    ) -> None:
        """
        Initialize a new Deep Q Agent.

        Args:
            env: the environment for the agent to experience
            render_mode: the mode for rendering frames in the OpenAI gym env
                - None: don't render (much faster execution)
                - 'human': render in a window to observe on screen
            replay_memory_size: the number of previous experiences to store
                in the experience replay queue
            prioritized_experience_replay: whether to use prioritized
                experience replay. If False, will use the standard replay
                queue with uniform random sampling
            discount_factor: discount factor, γ, for discounting future reward
            update_frequency: the number of actions between updates to the
                deep Q network from replay memory
            optimizer: the optimization method to use on the CNN gradients
            exploration_rate: the exploration rate, ε, expected as an
                AnnealingVariable subclass for scheduled decay
            loss: the loss method to use at the end of the CNN
            target_update_freq: frequency to update the target network (steps)
            dueling_network: whether to use the dueling architecture

        Returns:
            None

        """
        # setup the Gym environment variables
        super().__init__(env, render_mode)
        # setup the replay queue
        self.prioritized_experience_replay = prioritized_experience_replay
        if prioritized_experience_replay:
            self.queue = PrioritizedReplayQueue(replay_memory_size)
        else:
            self.queue = ReplayQueue(replay_memory_size)
        # setup the Q learning algorithm variables
        self.discount_factor = discount_factor
        self.update_frequency = update_frequency
        self.optimizer = optimizer
        self.exploration_rate = exploration_rate
        self.loss = loss
        self.target_update_freq = target_update_freq
        self.dueling_network = dueling_network
        # build an output mask that lets all action values pass through
        mask_shape = (1, env.action_space.n)
        self.mask = np.ones(mask_shape, dtype=np.float32)
        # use an identity of size action space, to index rows from it using
        # an action vector to produce a one-hot vector masks for training error
        self.action_onehot = np.eye(env.action_space.n, dtype=np.float32)
        # setup the model for predicting Q values
        if dueling_network:
            build_model = build_dueling_deep_q_model
        else:
            build_model = build_deep_q_model
        # build the neural model for estimating Q values
        self.model = build_model(
            image_size=env.observation_space.shape[:2],
            num_frames=env.observation_space.shape[-1],
            num_actions=env.action_space.n,
            loss=loss,
            optimizer=optimizer
        )
        # build the target model for estimating target values
        self.target_model = build_model(
            image_size=env.observation_space.shape[:2],
            num_frames=env.observation_space.shape[-1],
            num_actions=env.action_space.n,
            loss=loss,
            optimizer=optimizer
        )

    def __repr__(self) -> str:
        """Return a debugging string of this agent."""
        return _REPR_TEMPLATE.format(
            self.__class__.__name__,
            self.env,
            repr(self.render_mode),
            self.queue.size,
            self.prioritized_experience_replay,
            self.discount_factor,
            self.update_frequency,
            self.optimizer,
            self.exploration_rate,
            self.loss.__name__,
            self.target_update_freq,
            self.dueling_network
        )

    def _td_error(self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        d: np.ndarray,
        s2: np.ndarray
    ) -> float:
        """
        Calculate the TD-error for a single experience.

        Args:
            s: the current state
            a: the action to get from current state `s` to next state `s2`
            r: the reward resulting from taking action `a` in state `s`
            d: the flag denoting whether the episode ended after action `a`
            s2: the next state from taking action `a` in state `s`

        Returns:
            the TD-error as a result of the experience

        """
        if d:
            # terminal states have a Q value of zero by definition
            Q_t = 0.0
        else:
            # predict Q values for the next state and take the max value.
            Q_t = self.target_model.predict([s2[None, :, :, :], self.mask])

        # calculate the predicted Q value from the current state and action
        Q = self.model.predict([s[None, :, :, :], self.mask])
        # calculate the TD error based on the reward, discounted future
        # reward, and the predicted future reward
        td_error = abs(r + self.discount_factor * np.max(Q_t) - np.max(Q))

        return td_error

    def _remember(self,
        s: np.ndarray,
        a: int,
        r: int,
        d: bool,
        s2: np.ndarray,
    ) -> None:
        """
        Push an experience onto the replay queue.

        Args:
            s: the current state
            a: the action to get from current state `s` to next state `s2`
            r: the reward resulting from taking action `a` in state `s`
            d: the flag denoting whether the episode ended after action `a`
            s2: the next state from taking action `a` in state `s`

        Returns:
            None

        """
        if self.prioritized_experience_replay:
            # calculate the priority of the experience based on the TD error
            priority = self._td_error(s, a, r, d, s2)
            self.queue.push(s, a, r, d, s2, priority=priority)
        else:
            self.queue.push(s, a, r, d, s2)

    def _replay(self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        d: np.ndarray,
        s2: np.ndarray
    ) -> float:
        """
        Train the network on a mini-batch of replay data.

        Args:
            s: a batch of current states
            a: a batch of actions from each state in s
            r: a batch of reward from each action in a
            d: a batch of terminal flags after each action in a
            s2: a batch of next states from each state-action pair in s, a

        Returns:
            the loss as a result of the training

        """
        # initialize target y values
        y = np.zeros((len(s), self.env.action_space.n), dtype=np.float32)

        # predict Q values for the next state of each memory in the batch and
        # take the max value. don't mask any outputs, i.e. use ones
        mask = np.repeat(self.mask, len(s), axis=0)
        Q = np.max(self.target_model.predict_on_batch([s2, mask]), axis=1)
        # terminal states have a Q value of zero by definition
        Q[d] = 0
        # set the y value for each sample to the reward of the selected
        # action plus the discounted Q value
        y[range(y.shape[0]), a] = r + self.discount_factor * Q

        # train the model on the batch and return the loss. use the mask that
        # disables training for actions that aren't the selected actions.
        return self.model.train_on_batch([s, self.action_onehot[a]], y)

    def observe(self, replay_start_size: int=50000) -> None:
        """
        Observe random moves to initialize the replay memory.

        Args:
            replay_start_size: the number of random observations to make
                i.e. the size to fill the replay memory with to start

        Returns:
            None

        """
        progress = tqdm(total=replay_start_size, unit='frame')
        # loop indefinitely, the loop breaks when the number of
        # frames passed is greater than replay_start_size
        while replay_start_size > 0:
            # reset the game and get the initial state
            state = self._initial_state()
            # the done flag indicating that an episode has ended
            done = False
            # loop until done
            while not done:
                # sample a random action to perform
                action = self.env.action_space.sample()
                # perform action and observe the reward and next state
                next_state, reward, done = self._next_state(action)
                # push the memory onto the replay queue
                self._remember(state, action, reward, done, next_state)
                # set the state to the new state
                state = next_state
                # decrement the observation counter
                replay_start_size -= 1
                # update the progress bar
                progress.update(1)

        progress.close()

    def predict(self, frames: np.ndarray, exploration_rate: float) -> int:
        """
        Predict an action from a stack of frames.

        Args:
            frames: the stack of frames to predict Q values from
            exploration_rate: the exploration rate for epsilon greedy selection

        Returns:
            the predicted optimal action based on the frames

        """
        if np.random.random() < exploration_rate:
            # select a random action and return it
            return self.env.action_space.sample()
        # reshape the frames to pass through the loss network
        frames = frames[np.newaxis, :, :, :]
        # predict the values of each action
        actions = self.model.predict([frames, self.mask])
        # return the action with the highest estimated future reward
        return np.argmax(actions)

    def train(self,
        frames_to_play: int=50000000,
        batch_size: int=32,
        callback: Callable=None,
    ) -> None:
        """
        Train the network for a number of episodes (games).

        Args:
            frames_to_play: the number of frames to play the game for
            batch_size: the size of the replay history batches
            callback: an optional callback to get updates about the score,
                      loss, discount factor, and exploration rate every
                      episode

        Returns:
            None

        """
        # the progress bar for the operation
        progress = tqdm(total=frames_to_play, unit='frame')
        progress.set_postfix(score='?', loss='?')

        while frames_to_play > 0:
            done = False
            score = 0
            loss = 0
            frames = 0
            # reset the game and get the initial state
            state = self._initial_state()

            while not done:
                # predict the best action based on the current state
                action = self.predict(state, self.exploration_rate.value)
                # step the exploration rate forward
                self.exploration_rate.step()
                # fire the action and observe the next state, reward, and flag
                next_state, reward, done = self._next_state(action)
                score += reward
                # push the memory onto the replay queue
                self._remember(state, action, reward, done, next_state)
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

            # pass the score to the callback at the end of the episode
            if callable(callback):
                callback(self, score, loss)
            # update the progress bar
            progress.set_postfix(score=score, loss=loss)
            progress.update(frames)

        progress.close()

    def play(self, games: int=100, exploration_rate: float=0.05) -> np.ndarray:
        """
        Run the agent without training for the given number of games.

        Args:
            games: the number of games to play
            exploration_rate: the epsilon for epsilon greedy exploration

        Returns:
            an array of scores, one for each game

        """
        # the progress bar for the operation
        progress = tqdm(range(games), unit='game')
        progress.set_postfix(score='?')

        # a list to keep track of the scores
        scores = np.zeros(games)
        # iterate over the number of games
        for game in progress:
            done = False
            score = 0
            # reset the game and get the initial state
            state = self._initial_state()

            while not done:
                # predict the best action based on the current state
                action = self.predict(state, exploration_rate)
                # hold the action for the number of frames
                next_state, reward, done = self._next_state(action)
                score += reward
                # set the state to the new state
                state = next_state
            # push the score onto the history
            scores[game] = score
            # update the progress bar
            progress.set_postfix(score=score)
            progress.update(1)

        progress.close()

        return scores


# explicitly define the outward facing API of this module
__all__ = [DeepQAgent.__name__]
