"""Usage:

    python dddqn_train.py <game name> <results directory>
"""
import os
import sys
import datetime
import pandas as pd
import gym_tetris


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


# load these after command line arg checking bc tensorflow is slow to load
# and generates some warning output
from src.environment.atari import build_atari_environment
from src.environment.nes import build_nes_environment
from src.agents import DeepQAgent
from src.util import BaseCallback


# check if this is the Tetris environment
if 'Tetris' in game_name:
    env = gym_tetris.make(game_name)
    env = gym_tetris.wrap(env, clip_rewards=False)
# check if we need to load the NES environment
elif 'SuperMarioBros' in game_name:
    env = build_nes_environment(game_name)
# default to the Atari environment
else:
    env = build_atari_environment(game_name)
# wrap the environment with a monitor
# env = Monitor(env, '{}/monitor_train'.format(output_dir), force=True)


# build the agent
agent = DeepQAgent(env, replay_memory_size=int(7.5e5))
# write some info about the agent's hyperparameters to disk
with open('{}/agent.py'.format(output_dir), 'w') as agent_file:
    agent_file.write(repr(agent))


# observe frames to fill the replay memory
agent.observe()


# train the agent
try:
    callback = BaseCallback(weights_file)
    agent.train(frames_to_play=int(2.5e6), callback=callback)
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
