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
    # if the directory doesn't exist, make it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    return output_dir


def build_env(env, build_agent: 'Agent', output_dir: str, monitor: bool=False):
    """Train the agent."""
    # setup the output directory
    output_dir = experiment_dir(output_dir, env, build_agent.__name__)
    # create the environment
    env = gym_super_mario_bros.make(env)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = nes_py_wrap(env)
    # if monitoring is enabled, setup the monitor for the environment
    if monitor:
        monitor_dir = os.path.join(output_dir, 'monitor')
        env = gym.wrappers.Monitor(env, monitor_dir, force=True)

    return env, output_dir


__all__ = [build_env.__name__,]
