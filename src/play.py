"""Methods for playing environments with agents."""
import os
import sys
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
from gym.wrappers import Monitor
import gym_tetris
import gym_super_mario_bros
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv, wrap as nes_py_wrap
from gym_super_mario_bros.actions import (
    SIMPLE_MOVEMENT,
    COMPLEX_MOVEMENT,
    RIGHT_ONLY,
)


def play(
    results_dir: str,
    is_monitor: bool=True
) -> None:
    """
    Play an environment with a certain agent.

    Args:
        results_dir: the directory containing results of a training session
        is_monitor: whether to wrap the environment with a monitor

    Returns:
        None

    """
    try:
        env_id = results_dir.split('/')[-3]
    except IndexError:
        raise ValueError('invalid results directory: {}'.format(results_dir))

    # set up the weights file
    weights_file = '{}/weights.h5'.format(results_dir)
    # make sure the weights exist
    if not os.path.exists(weights_file):
        raise OSError('weights file not found: {}'.format(weights_file))

    # these are long to import and train is only ever called once during
    # an execution lifecycle. import here to save early execution time
    from src.environment.atari import build_atari_environment
    from src.agents import DeepQAgent

    # TODO: replace this logic using an internal `wrap` command
    if 'Tetris' in env_id:
        env = gym_tetris.make(env_id)
        env = gym_tetris.wrap(env, clip_rewards=False)
    elif 'SuperMarioBros' in env_id:
        env = gym_super_mario_bros.make(env_id)
        env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
        env = nes_py_wrap(env)
    else:
        env = build_atari_environment(env_id, is_validation=True)

    # wrap the environment with a monitor if enabled
    if is_monitor:
        env = Monitor(env, '{}/monitor_play'.format(results_dir), force=True)

    # build the agent without any replay memory since we're just playing, load
    # the trained weights, and play some games
    agent = DeepQAgent(env, replay_memory_size=0)
    agent.model.load_weights(weights_file)
    agent.target_model.load_weights(weights_file)

    try:
        agent.play()
    except KeyboardInterrupt:
        env.close()
        sys.exit(0)

    # collect the game scores, actual scores from the reward cache wrapper,
    # not mutated, clipped, or whatever rewards that the agent sees
    scores = pd.concat([pd.Series(env.unwrapped.episode_rewards)], axis=1)
    scores.columns = ['Score']
    scores.index.name = 'Episode'
    print(scores.describe())
    # write the scores and a histogram visualization to disk
    scores.to_csv('{}/final_scores.csv'.format(results_dir))
    ax = scores['Score'].hist()
    ax.set_title('Histogram of Scores')
    ax.set_ylabel('Number of Episodes')
    ax.set_xlabel('Score')
    plt.savefig('{}/final_scores.pdf'.format(results_dir))

    env.close()


def play_random(
    env_id: str,
    output_dir: str,
    is_monitor: bool=False
) -> None:
    """
    Run a uniformly random agent in the given environment.

    Args:
        env_id: the ID of the environment to play
        output_dir: the base directory to store results into
        is_monitor: whether to wrap the environment with a monitor

    Returns:
        None

    """
    # setup the output directory with a timestamped directory
    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    output_dir = '{}/{}/Random/{}'.format(output_dir, env_id, now)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('writing results to {}'.format(repr(output_dir)))

    # these are long to import and train is only ever called once during
    # an execution lifecycle. import here to save early execution time
    from src.environment.atari import build_atari_environment
    from src.agents import RandomAgent

    # TODO: replace this logic using an internal `wrap` command
    if 'Tetris' in env_id:
        env = gym_tetris.make(env_id)
        env = gym_tetris.wrap(env, clip_rewards=False)
    elif 'SuperMarioBros' in env_id:
        env = gym_super_mario_bros.make(env_id)
        env = gym_super_mario_bros.wrap(env, clip_rewards=False)
    else:
        env = build_atari_environment(env_id)

    # wrap the environment with a monitor if enabled
    if is_monitor:
        env = Monitor(env, '{}/monitor_random'.format(output_dir), force=True)

    # initialize a random agent on the environment and play a validation batch
    agent = RandomAgent(env)
    agent.play()

    # collect the game scores, actual scores from the reward cache wrapper,
    # not mutated, clipped, or whatever rewards that the agent sees
    scores = pd.concat([pd.Series(env.unwrapped.episode_rewards)], axis=1)
    scores.columns = ['Score']
    scores.index.name = 'Episode'
    print(scores.describe())
    # write the scores and a histogram visualization to disk
    scores.to_csv('{}/random_scores.csv'.format(output_dir))
    ax = scores['Score'].hist()
    ax.set_title('Histogram of Scores')
    ax.set_ylabel('Number of Episodes')
    ax.set_xlabel('Score')
    plt.savefig('{}/random_scores.pdf'.format(output_dir))

    env.close()


# explicitly define the outward facing API of this module
__all__ = [play.__name__, play_random.__name__]
