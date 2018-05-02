"""Usage:

    python dddqn_train.py <game name> <results directory>
"""
import os
import sys
import datetime
import pandas as pd


# load variables from the command line
try:
    game_name = sys.argv[1]
    output_dir = sys.argv[2]
except IndexError:
    print(__doc__)
    sys.exit(-1)


# setup the output directory
now = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
output_dir = '{}/{}/DeepQAgent/{}'.format(output_dir, game_name, now)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print('writing results to {}'.format(repr(output_dir)))
weights_file = '{}/weights.h5'.format(output_dir)


# load these module after command line argument checking because TensorFlow
# is slow to load and generates some warning outputs
from src.environment.atari import build_atari_environment
from src.environment.nes import build_nes_environment
from src.agents import DeepQAgent
from src.util import BaseCallback


# check if we need to load the NES environment
if 'SuperMarioBros' in game_name:
    env = build_nes_environment(game_name)
# default to the Atari environment
else:
    env = build_atari_environment(game_name)
# wrap the environment with a monitor
# env = Monitor(env, '{}/monitor_train'.format(output_dir), force=True)


# build the agent
agent = DeepQAgent(env)
# write some info about the agent's hyper-parameters to disk
with open('{}/agent.py'.format(output_dir), 'w') as agent_file:
    agent_file.write(repr(agent))


# observe frames to fill the replay memory
agent.observe()


# train the agent
try:
    callback = BaseCallback(weights_file)
    agent.train(frames_to_play=1e6, callback=callback)
except KeyboardInterrupt:
    print('canceled training')


# save the training results
scores = pd.Series(callback.scores)
scores.to_csv('{}/scores.csv'.format(output_dir))
losses = pd.Series(callback.losses)
losses.to_csv('{}/losses.csv'.format(output_dir))
# save the weights to disk
agent.model.save_weights(weights_file, overwrite=True)


# close the environment to perform necessary cleanup
env.close()
