"""Usage:

    python train.py <agent name> <game name> <results directory>
"""
import os
import sys
import gym
import pandas as pd
from src.util import BaseCallback
from src.agents import DeepQAgent, DoubleDeepQAgent
from src.environment.atari import build_atari_environment


# a mapping of string names to agents
agents = {
    'DeepQAgent': DeepQAgent,
    'DoubleDeepQAgent': DoubleDeepQAgent,
}


# load variables from the command line
try:
    agent_name = sys.argv[1]
    game = sys.argv[2]
    exp_directory = sys.argv[3]
except IndexError:
    print(__doc__)
    sys.exit(-1)

# validate the agent name
if agent_name not in agents.keys():
    print('invalid agent')
    sys.exit(-1)

# validate the game
if game not in downsamplers.keys():
    print('invalid game')
    sys.exit(-1)

# setup the experiment directory
exp_directory = '{}/{}/{}'.format(exp_directory, game, agent_name)
if not os.path.exists(exp_directory):
    os.makedirs(exp_directory)
# set up the weights file
weights_file = '{}/weights.h5'.format(exp_directory)



# build the environment
env = build_atari_environment(game)
# build the agent
agent = agents[agent_name](env)

# capture some metrics before training
print('playing games for initial metrics')
initial = pd.Series(agent.play())
print(initial.describe())
initial.to_csv('{}/initial.csv'.format(exp_directory))

# observe some frames using random movement
print('observing frames before training')
agent.observe()
# train the agent
try:
    print('beginning training')
    callback = BaseCallback()
    agent.train(callback=callback)
except KeyboardInterrupt:
    print('canceled training')
# save the training results
scores = pd.Series(callback.scores)
scores.to_csv('{}/scores.csv'.format(exp_directory))
losses = pd.Series(callback.losses)
losses.to_csv('{}/losses.csv'.format(exp_directory))
# save the weights to disk
agent.model.save_weights(weights_file, overwrite=True)

# capture some metrics after training
print('playing games for final metrics')
final = pd.Series(agent.play())
print(final.describe())
final.to_csv('{}/final.csv'.format(exp_directory))

# close the environment to perform necessary cleanup
env.close()
