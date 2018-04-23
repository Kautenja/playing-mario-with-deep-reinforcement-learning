"""Usage:

    python play.py <results directory>
"""
import os
import sys
import pandas as pd
from gym.wrappers import Monitor
from src.agents import DeepQAgent, A3CAgent
from src.environment.atari import build_atari_environment


# a mapping of string names to agents
agents = {
    DeepQAgent.__name__: DeepQAgent,
    A3CAgent.__name__: A3CAgent,
}


# load variables from the command line
try:
    exp_directory = sys.argv[1]
    dirs = exp_directory.split('/')
    agent_name = dirs[-2]
    game = dirs[-3]
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
env, r_cache = build_atari_environment(game, is_validation=True)
# env = Monitor(env, '{}/monitor_play'.format(exp_directory), force=True)
# build the agent without any replay memory (not needed to play from model)
agent = agents[agent_name](env, replay_memory_size=0)
# load the weights
agent.model.load_weights(weights_file)
# play some games
agent.play()
# collect the game scores
scores = pd.Series(r_cache._rewards)
scores.to_csv('{}/final_validation.csv'.format(exp_directory))
# print some stats
print('min ', scores.min())
print('mean ', scores.mean())
print('max ', scores.max())
