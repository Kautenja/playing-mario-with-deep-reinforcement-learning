"""An implementation of Deep Q-Learning."""
import numpy as np
from typing import Callable
from tqdm import tqdm
from pygame.time import Clock
from keras.optimizers import Adam
from src.models import build_deep_mind_model
from src.models.losses import huber_loss
from src.base import AnnealingVariable
from src.downsamplers import Downsampler
from .agent import Agent
from .replay_queue import ReplayQueue


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
    null_op={},
    loss={},
    image_size={},
    render_mode={}
)
""".lstrip()


class DeepQAgent(Agent):
    """The Deep Q reinforcement learning algorithm."""

    def __init__(self, env, downsample: Downsampler,
        replay_memory_size: int=250000,
        agent_history_length: int=4,
        discount_factor: float=0.99,
        update_frequency: int=4,
        optimizer=Adam(lr=2e-5),
        exploration_rate=AnnealingVariable(1.0, 0.1, 1000000),
        null_op_max: int=30,
        null_op: int=0,
        loss: Callable=huber_loss,
        image_size: tuple=(84, 84),
        render_mode: str='human',
    ) -> None:
        """
        Initialize a new Deep Q Agent.

        Args:
            env: the environment to run on
            downsample: the down-sampler for the Gym environment
            agent_history_length: the number of previous frames for the agent
                                  to make new decisions based on. this will
                                  set the number of filters in the CNN
            discount_factor: the discount factor, γ, for discounting future
                             reward
            update_frequency: the number of actions between updates to the
                              deep Q network from replay memory
            optimizer: the optimization method to use on the CNN gradients
            exploration_rate: the exploration rate, ε, expected as an
                              AnnealingVariable subclass for scheduled decay
            null_op_max: the maximum number of random null ops at the start of
                         each new game. the agent performs null operations at
                         the beginning of training and validation episodes to
                         emulate a stochastic "human" start
            null_op: the value indicating a null operation for null_op_max
            loss: the loss method to use at the end of the CNN
            image_size: the size of the images to pass to the CNN
            render_mode: the mode for rendering frames in the OpenAI gym env
                         -   'human': render in the emulator (default)
                         -   'rgb_array': render in the backend and return a
                                          numpy array (server/Jupyter)

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
        self.null_op = null_op
        self.loss = loss
        self.image_size = image_size
        self.render_mode = render_mode
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

    def __repr__(self) -> str:
        """Return an executable Python string to reproduce self."""
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
            self.null_op,
            self.loss,
            self.image_size,
            repr(self.render_mode)
        )

    def _initial_state(self) -> np.ndarray:
        """Reset the environment and return the initial state."""
        # reset the environment
        frame = self.env.reset()
        # render this frame in the emulator
        self.env.render(mode=self.render_mode)

        # down-sample the frame to B&W and cropped to playable area
        frame = self.downsample(frame, self.image_size)[:, :, np.newaxis]
        # reset the frame buffer with the initial state repeated as necessary
        self.frame_buffer = np.repeat(frame, self.agent_history_length, axis=2)
        # return the frame buffer as the state

        return self.frame_buffer

    def _next_state(self, action: int) -> tuple:
        """
        Return the next state based on the given action.

        Args:
            action: the action to perform for some frames

        Returns:
            a tuple of:
                - the next state
                - the reward as a result of the action
                - the terminal flag

        """
        # make the step and observe the state, reward, done flag
        state, reward, done, _ = self.env.step(action=action)
        # render this frame in the emulator
        self.env.render(mode=self.render_mode)

        # down-sample the state and convert it to the expected shape
        state = self.downsample(state, self.image_size)[:, :, np.newaxis]
        # add the state to the frame buffer
        self.frame_buffer = np.concatenate((self.frame_buffer, state), axis=2)
        # remove the last frame in the frame buffer
        self.frame_buffer = self.frame_buffer[:, :, 1:]

        # assign a negative reward if terminal state
        reward = -1.0 if done else reward
        # clip the reward based on its sign. i.e. clip in [-1, 0, 1]
        reward = np.sign(reward)

        return self.frame_buffer, reward, done

    def predict_action(self,
        frames: np.ndarray,
        exploration_rate: float
    ) -> int:
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
        else:
            # reshape the frames to pass through the loss network
            frames = frames[np.newaxis, :, :, :]
            # build an output mask that lets all action values pass through
            mask = np.ones((self.agent_history_length, self.env.action_space.n))
            # predict the values of each action
            actions = self.model.predict([frames, mask], batch_size=1)
            # return the action with the highest estimated future reward
            return np.argmax(actions)

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
        while True:
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
                self.queue.push(state, action, reward, done, next_state)
                # set the state to the new state
                state = next_state
                # decrement the observation counter
                replay_start_size -= 1
                # update the progress bar
                progress.update(1)
                # break out if done observing
                if replay_start_size <= 0:
                    progress.close()
                    return

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
            s2: the next state from each state, action pair in s, a

        Returns:
            the loss as a result of the training

        """
        # initialize target y values
        y = np.zeros((len(s), self.env.action_space.n))

        # predict Q values for the next state of each memory in the batch and
        # take the maximum value. dont mask any outputs, i.e. use ones
        all_mask = np.ones((len(s), self.env.action_space.n))
        Q = np.max(self.model.predict_on_batch([s2, all_mask]), axis=1)
        # terminal states have a Q value of zero by definition
        Q[d] = 0
        # set the y value for each sample to the reward of the selected
        # action plus the discounted Q value
        y[range(y.shape[0]), a] = r + self.discount_factor * Q

        # use an identity of size action space, and index rows from it using
        # the action vector to produce a one-hot matrix representing the mask
        # for actions in a
        action_mask = np.eye(self.env.action_space.n)[a]
        # train the model on the batch and return the loss. use the mask that
        # disables training for actions that aren't the selected actions.
        return self.model.train_on_batch([s, action_mask], y)

    def train(self,
        frames_to_play: int=10000000,
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
        # loop indefinitely, the loop ends with a break when the
        # frame counter passes the provided number of frames
        while True:
            done = False
            score = 0
            loss = 0
            frames = 0
            # reset the game and get the initial state
            state = self._initial_state()
            # perform NOPs randomly
            for k in range(np.random.randint(0, self.null_op_max)):
                state, reward, done = self._next_state(self.null_op)

            while not done:
                # predict the best action based on the current state
                action = self.predict_action(state, self.exploration_rate.value)
                # step the exploration rate forward
                self.exploration_rate.step()
                # fire the action and observe the next state, reward, and done
                next_state, reward, done = self._next_state(action)
                score += reward
                # push the memory onto the replay queue
                self.queue.push(state, action, reward, done, next_state)
                # set the state to the new state
                state = next_state
                # decrement the observation counter
                frames_to_play -= 1
                frames += 1
                # update Q from replay every self.update_frequency
                if frames_to_play % self.update_frequency == 0:
                    loss += self._replay(*self.queue.sample(size=batch_size))
                # break out if done training
                if frames_to_play <= 0:
                    progress.update(frames)
                    progress.close()
                    return

            # pass the score to the callback at the end of the episode
            if callable(callback):
                callback(score, loss)
            # update the progress bar
            progress.update(frames)

    def play(self,
        games: int=100,
        exploration_rate: float=0.05,
        fps: int=None
    ) -> np.ndarray:
        """
        Run the agent without training for a number of games.

        Args:
            games: the number of games to play
            exploration_rate: the epsilon for epsilon greedy exploration
            fps: the frame-rate to limit game play to
                - if None, the frame-rate will not be limited (i.e infinite)

        Returns:
            an array of scores, one for each game

        """
        # initialize a clock to keep the frame-rate bounded
        clock = Clock()
        # a list to keep track of the scores
        scores = np.zeros(games)
        # iterate over the number of games
        for game in tqdm(range(games), unit='game'):
            done = False
            score = 0
            # reset the game and get the initial state
            state = self._initial_state()
            # perform NOPs randomly
            for k in range(np.random.randint(0, self.null_op_max)):
                state, _, _ = self._next_state(self.null_op)

            while not done:
                # predict the best action based on the current state
                action = self.predict_action(state, exploration_rate)
                # hold the action for the number of frames
                next_state, reward, done = self._next_state(action)
                score += reward
                # set the state to the new state
                state = next_state
                # bound the frame rate if there is an fps provided
                if fps is not None:
                    clock.tick(fps)
            # push the score onto the history
            scores[game] = score

        return scores


# explicitly define the outward facing API of this module
__all__ = ['DeepQAgent']
