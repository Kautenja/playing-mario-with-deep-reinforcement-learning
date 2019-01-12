"""A command line interface to train Tiramisu vision models."""
import datetime
import os
import shutil
import gym
import gym_super_mario_bros
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv, wrap as nes_py_wrap
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import _top_level
from src.agents import RandomAgent
from src.training import stochastic
from src.utils import log_
from src.cli import args_and_groups
from src.cli import build_argparser


# build the command line argument parser
PARSER, AGENT, TRAIN, PLAY = build_argparser()


# get the arguments from the argument parser
ARGS, ARG_GROUPS = args_and_groups(PARSER)


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


def getattr_and_call(agent_: object, method_: str, groups) -> any:
    """
    Lookup a method in an agent and call it with command line arguments.

    Args:
        agent_: the agent to lookup a method in
        method_: the method to lookup in the agent
        groups: the argument groups from argparser to get keyword args from

    Returns:
        the return of the provided method if the method exists, else None

    """
    # if the agent doesn't have this feature, return None
    if not hasattr(agent_, method_):
        return None
    # if the method isn't feature in the command line args, call with defaults
    if method_ not in groups:
        return getattr(agent_, method_)()
    # get the method and call it with the given arguments
    return getattr(agent_, method_)(**vars(groups[method_]))


@stochastic(ARGS.seed, ARGS.gpu)
def main():
    """Train the model."""
    # setup the output directory
    output_dir = experiment_dir(ARGS.output_dir, ARGS.env, RandomAgent.__name__)
    log_('Experiment Directory', output_dir)
    # create the environment
    env = gym_super_mario_bros.make(ARGS.env)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = nes_py_wrap(env)
    # if monitoring is enabled, setup the monitor for the environment
    if ARGS.monitor:
        monitor_dir = os.path.join(output_dir, 'monitor')
        env = gym.wrappers.Monitor(env, monitor_dir, force=True)
        log_('Monitor Directory', monitor_dir)
    # create a the agent
    agent_kwargs = vars(ARG_GROUPS['agent'])
    agent_kwargs['env'] = env
    agent = RandomAgent(**vars(ARG_GROUPS['agent']))
    log_('Agent', agent)
    # get the observe, train, and play methods and call them
    try:
        getattr_and_call(agent, 'observe', ARG_GROUPS)
        getattr_and_call(agent, 'train', ARG_GROUPS)
        getattr_and_call(agent, 'play', ARG_GROUPS)
    except KeyboardInterrupt:
        env.close()


if __name__ == '__main__':
    main()
