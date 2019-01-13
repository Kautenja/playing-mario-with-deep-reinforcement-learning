"""An abstract base class for reinforcement agents."""
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Agent(object):
    """An abstract base class for building reinforcement agents."""

    def __init__(self, env: gym.Env, render_mode: str=None) -> None:
        """
        Create a new abstract reinforcement agent.

        Args:
            env: the environment for the agent to experience
            render_mode: the mode for rendering frames in the OpenAI gym env
                - None: don't render (much faster execution)
                - 'human': render in a window to observe on screen
        Returns:
            None

        """
        self.env = env
        self.render_mode = render_mode

    def __str__(self) -> str:
        """Return a human readable string representation of this object."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return a debugging string of this agent."""
        return '{}(env={}, render_mode={})'.format(
            self.__class__.__name__,
            self.env,
            self.render_mode
        )

    def _initial_state(self) -> np.ndarray:
        """
        Reset the environment and return the initial state.

        Returns:
            the initial state of the game

        """
        # reset the environment and get the initial state
        state = self.env.reset()
        # render the state if a render_mode exists
        if self.render_mode is not None:
            self.env.render(mode=self.render_mode)

        return state

    def _next_state(self, action: int) -> tuple:
        """
        Return the next state based on the given action.

        Args:
            action: the action to perform for some frames

        Returns:
            a tuple of:
                - the next state
                - the reward as a result of the action
                - a flag determining end of episode

        """
        # perform the action and observe the next state, reward, and done flag
        state, reward, done, _ = self.env.step(action=action)
        # render the state if a render_mode exists
        if self.render_mode is not None:
            self.env.render(mode=self.render_mode)

        return state, reward, done

    @property
    def episode_rewards(self) -> pd.DataFrame:
        """Return the episodic scores and losses."""
        # collect the game scores, actual scores from the reward cache wrapper,
        # not mutated, clipped, or whatever rewards that the agent sees
        scores = [pd.Series(self.env.unwrapped.episode_rewards)]
        scores = pd.concat(scores, axis=1)
        scores.columns = ['Score']
        scores.index.name = 'Episode'

        return scores

    def plot_episode_rewards(self, basename: str=None) -> None:
        """
        Plot the results of a series of episodes and save them to disk.

        Args:
            env: the env with a RewardCacheWrapper to extract data from
            results_dir: the directory to store the plots and data in
            filename: the name of the file to use for the plots and data

        Returns:
            None

        """
        # get the scores
        scores = self.episode_rewards
        # write the scores and a histogram visualization to disk
        if basename is not None:
            scores.to_csv('{}.csv'.format(basename))
        axis = scores['Score'].hist()
        axis.set_title('Histogram of Scores')
        axis.set_ylabel('Number of Episodes')
        axis.set_xlabel('Score')
        if basename is not None:
            plt.savefig('{}.pdf'.format(basename))


# explicitly define the outward facing API of this module
__all__ = [Agent.__name__]
