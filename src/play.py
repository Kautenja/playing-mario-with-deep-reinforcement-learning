"""Methods for playing environments with agents."""
import os
import sys
from datetime import datetime
import gym
import pandas as pd
from matplotlib import pyplot as plt
from .setup_env import setup_env


def plot_results(env: gym.Env, results_dir: str, filename: str) -> None:
    """
    Plot the results of a series of episodes and save them to disk.

    Args:
        env: the env with a RewardCacheWrapper to extract data from
        results_dir: the directory to store the plots and data in
        filename: the name of the file to use for the plots and data

    Returns:
        None

    """
    # collect the game scores, actual scores from the reward cache wrapper,
    # not mutated, clipped, or whatever rewards that the agent sees
    scores = pd.concat([pd.Series(env.unwrapped.episode_rewards)], axis=1)
    scores.columns = ['Score']
    scores.index.name = 'Episode'
    print(scores.describe())
    # write the scores and a histogram visualization to disk
    scores.to_csv('{}/{}.csv'.format(results_dir, filename))
    axis = scores['Score'].hist()
    axis.set_title('Histogram of Scores')
    axis.set_ylabel('Number of Episodes')
    axis.set_xlabel('Score')
    plt.savefig('{}/{}.pdf'.format(results_dir, filename))


def play(results_dir: str, monitor: bool=False) -> None:
    """
    Play an environment with a certain agent.

    Args:
        results_dir: the directory containing results of a training session
        monitor: whether to monitor the operation

    Returns:
        None

    """
    try:
        env_id = list(filter(None, results_dir.split('/')))[-3]
    except IndexError:
        raise ValueError('invalid results directory: {}'.format(results_dir))

    # set up the weights file
    weights_file = '{}/weights.h5'.format(results_dir)
    # make sure the weights exist
    if not os.path.exists(weights_file):
        raise OSError('weights file not found: {}'.format(weights_file))

    # these are long to import and train is only ever called once during
    # an execution life-cycle. import here to save early execution time
    from src.agents import DeepQAgent

    # build the environment
    monitor_dir = '{}/monitor_play'.format(results_dir) if monitor else None
    env = setup_env(env_id, monitor_dir)
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

    # plot the results and save data to disk
    plot_results(env, results_dir, 'result_play')

    env.close()


def play_random(env_id: str, output_dir: str, monitor: bool=False) -> None:
    """
    Run a uniformly random agent in the given environment.

    Args:
        env_id: the ID of the environment to play
        output_dir: the base directory to store results into
        monitor: whether to monitor the operation

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
    # an execution life-cycle. import here to save early execution time
    from src.agents import RandomAgent

    # build the environment
    monitor_dir = '{}/monitor_random'.format(output_dir) if monitor else None
    env = setup_env(env_id, monitor_dir)
    # initialize a random agent on the environment and play a validation batch
    agent = RandomAgent(env)
    agent.play()

    # plot the results and save data to disk
    plot_results(env, output_dir, 'result_random')

    env.close()


# explicitly define the outward facing API of this module
__all__ = [play.__name__, play_random.__name__]
