"""Usage:

    python play.py <results directory>
"""
import os
import sys
import gym
from gym.wrappers import Monitor
from src.agents import (
    DeepQAgent,
    DoubleDeepQAgent
)
from src.downsamplers import (
    downsample_pong,
    downsample_breakout,
    downsample_space_invaders
)


# a mapping of string names to agents
agents = {
    'DeepQAgent': DeepQAgent,
    'DoubleDeepQAgent': DoubleDeepQAgent,
}

# down-samplers for each game
downsamplers = {
    'Pong': downsample_pong,
    'Breakout': downsample_breakout,
    'SpaceInvaders': downsample_space_invaders,
}


# load variables from the command line
try:
    exp_directory = sys.argv[1]
    dirs = exp_directory.split('/')
    agent_name = dirs[-1]
    game = dirs[-2]
except IndexError:
    print(__doc__)
    sys.exit(-1)
# set up the weights file
weights_file = '{}/weights.h5'.format(exp_directory)

# make sure the weights exist
if not os.path.exists(weights_file):
    print('{} not found!'.format(weights_file))
    sys.exit(-1)


# build the environment
env = gym.make('{}-v4'.format(game))
env = Monitor(env, '{}/monitor'.format(exp_directory), force=True)
# build the agent without any replay memory (not needed to play from model)
agent = agents[agent_name](env, downsamplers[game],
    replay_memory_size=0,
    render_mode='rgb_array',
)
# load the weights
agent.model.load_weights(weights_file)
# play some games
agent.play()
