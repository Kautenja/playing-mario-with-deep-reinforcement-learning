"""An implementation of Deep Q-Learning."""
import numpy as np
from tqdm import tqdm
from pygame.time import Clock
from .agent import Agent


class RandomAgent(Agent):
    """An agent that behaves randomly."""

    # the representation template for the __repr__ method of this object
    REPR = '{}(env={})'

    def __init__(self, env) -> None:
        """
        Initialize a new random Agent.

        Args:
            env: the environment to run on

        Returns:
            None

        """
        self.env = env

    def __repr__(self) -> str:
        """Return a debugging string of this agent."""
        return self.REPR.format(self.__class__.__name__, self.env)

    def _initial_state(self) -> np.ndarray:
        """Reset the environment and return the initial state."""
        # reset the environment
        frame = self.env.reset()
        # render this frame in the emulator
        self.env.render()

        return frame

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
        self.env.render()

        # assign a negative reward if terminal state
        reward = -1.0 if done else reward
        # clip the reward based on its sign. i.e. clip in [-1, 0, 1]
        reward = np.sign(reward)

        return state, reward, done

    def play(self, games: int=30, fps: int=None) -> np.ndarray:
        """
        Run the agent.

        Args:
            games: the number of games to play
            fps: the frame-rate to limit game play to
                - if None, the frame-rate will not be limited (i.e infinite)

        Returns:
            an array of scores

        """
        # initialize a clock to keep the frame-rate
        clock = Clock()
        # a list to keep track of the scores
        scores = np.zeros(games)
        # iterate over the number of games
        for game in tqdm(range(games), unit='game'):
            # reset the game and get the initial state
            _ = self._initial_state()
            # the done flag indicating that a game has ended
            done = False
            score = 0
            # loop until done
            while not done:
                # pick a random action
                action = self.env.action_space.sample()
                # hold the action for the number of frames
                _, reward, done = self._next_state(action)
                score += reward
                # bound the frame rate if there is an fps provided
                if fps is not None:
                    clock.tick(fps)
            # push the score onto the history
            scores[game] = score

        return scores


# explicitly define the outward facing API of this module
__all__ = ['RandomAgent']
