"""A gym wrapper for caching rewards."""
import gym


class RewardCacheEnv(gym.Wrapper):
    """a wrapper that caches rewards of episodes."""

    def __init__(self, env) -> None:
        """
        Initialize a reward caching environment.
        Args:
            env: the environment to wrap
        Returns:
            None
        """
        super().__init__(env)
        self._rewards = []

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done:
            print(info['distance'])
            self._rewards.append(info['distance'])
        return state, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# explicitly specify the external API of this module
__all__ = [RewardCacheEnv.__name__]
