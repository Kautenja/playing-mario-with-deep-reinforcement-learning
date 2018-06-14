"""An abstract base class for reinforcement agents."""
import gym
import numpy as np


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


# explicitly define the outward facing API of this module
__all__ = [Agent.__name__]
