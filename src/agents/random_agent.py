"""An implementation of Deep Q-Learning."""
import numpy as np
from tqdm import tqdm
from pygame.time import Clock
from .agent import Agent


class RandomAgent(Agent):
    """An agent that behaves randomly."""

    def play(self, games: int=100, fps: int=None) -> np.ndarray:
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
            done = False
            score = 0
            # reset the game and get the initial state
            _ = self._initial_state()

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
