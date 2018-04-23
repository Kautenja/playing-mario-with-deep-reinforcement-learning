"""A gym wrapper for penalizing deaths."""
import gym


class PenalizeDeathEnv(gym.Wrapper):
    """a wrapper that penalizes deaths, without terminating episodes."""

    def __init__(self, env, penalty: int=-1) -> None:
        """
        Initialize a new death penalizing environment wrapper.

        Args:
            env: the environment to wrap
            penalty: the penalty for losing a life

        Returns:
            None

        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.penalty = penalty

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        # check if its less than the last step, i.e. a death occurred. and
        # set the reward to the penalty if so
        # TODO: should `lives > 0` be constrained?
        reward = self.penalty if lives < self.lives else reward
        self.lives = lives

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        # reset the lives counter
        self.lives = self.env.unwrapped.ale.lives()
        return obs


# explicitly specify the external API of this module
__all__ = [PenalizeDeathEnv.__name__]
