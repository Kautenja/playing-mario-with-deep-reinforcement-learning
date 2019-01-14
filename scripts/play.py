"""A simple script to play an environment with a trained agent."""
import os
import sys
from _base import build_env
from _top_level import top_level
top_level()
from src import agents


def play(results_dir: str) -> None:
    """
    Play an environment with a certain agent.

    Args:
        results_dir: the directory containing results of a training session

    Returns:
        None
    """
    try:
        # break the directory into components
        components = list(filter(None, results_dir.split('/')))
        # get the environment ID and agent name from the components list
        env_id = components[-3]
        agent_class_name = components[-2]
    except IndexError:
        raise ValueError('invalid results directory: {}'.format(results_dir))

    # set up the weights file
    weights_file = os.path.join(results_dir, 'weights.h5')
    # make sure the weights exist
    if not os.path.exists(weights_file):
        raise OSError('weights file not found: {}'.format(weights_file))

    # build the environment
    env = build_env(env_id, monitor_dir=os.path.join(results_dir, 'play'))

    # try to find the agent class in the agents module
    try:
        agent_class = getattr(agents, agent_class_name)
    except AttributeError:
        msg = 'agent class "{}" not found in agents module'
        raise ValueError(msg.format(agent_class_name))
    # create the agent
    agent = agent_class(env)
    # load the weights
    agent.model.load_weights(weights_file)

    # play the environment
    try:
        agent.play()
    except KeyboardInterrupt:
        pass

    # plot the results and save data to disk
    agent.plot_episode_rewards(basename=os.path.join(results_dir, 'play'))
    # close the environment
    env.close()


if __name__ == '__main__':
    play(sys.argv[1])
