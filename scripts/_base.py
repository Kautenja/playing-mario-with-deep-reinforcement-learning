"""A command line interface to train Tiramisu vision models."""
import datetime
import os
import shutil
import gym
import gym_super_mario_bros
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv, wrap as nes_py_wrap
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def experiment_dir(output_dir: str, env_id: str, agent_name: str) -> str:
    """
    Setup an experiment directory.

    Args:
        output_dir: the output directory to create the experiment directory in
        env_id: the ID of the environment being tested
        agent_name: the string name of the agent being tested

    Returns:

    """
    # get the time right now
    now = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
    # join the directories into a single path
    output_dir = os.path.join(output_dir, env_id, agent_name, now)
    # if the directory exists already, delete it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # make the output directory
    os.makedirs(output_dir)

    return output_dir


def build_env(env: str, monitor_dir: str=None) -> 'gym.Env':
    """
    Build an environment for an agent.

    Args:
        env: the name of the environment to make
        monitor_dir: the directory to output monitor data to

    Returns:
        a gym.Env based on the given env ID with the appropriate wrappers

    """
    env = gym_super_mario_bros.make(env)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    if monitor_dir is not None:
        env = gym.wrappers.Monitor(env, monitor_dir, force=True)
    env = nes_py_wrap(env)

    return env


# explicitly define the outward facing API of this module
__all__ = [
    build_env.__name__,
    experiment_dir.__name__,
]
