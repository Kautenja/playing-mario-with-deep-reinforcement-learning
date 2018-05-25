"""Usage:

    python play_random.py <game name> <results directory>
"""
import os
import sys
from datetime import datetime
import pandas as pd
from gym.wrappers import Monitor
import gym_tetris


# load variables from the command line
try:
    game_name = sys.argv[1]
    output_dir = sys.argv[2]
except IndexError:
    print(__doc__)
    sys.exit(-1)
# setup the output directory with a timestamped directory
now = datetime.today().strftime('%Y-%m-%d_%H-%M')
output_dir = '{}/{}/Random/{}'.format(output_dir, game_name, now)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print('writing results to {}'.format(repr(output_dir)))


# load these after command line arg checking bc tensorflow is slow to load
# and generates some warning output
from src.environment.atari import build_atari_environment
from src.environment.nes import build_nes_environment
from src.agents import RandomAgent

if 'Tetris-v0' == game_name:
    env = gym_tetris.make('Tetris-v0')
    env = gym_tetris.wrap(env)
# check if we need to load the NES environment
elif 'SuperMarioBros' in game_name:
    env = build_nes_environment(game_name)
# default to the Atari environment
else:
    env = build_atari_environment(game_name, is_validation=True)
# wrap the environment with a monitor
env = Monitor(env, '{}/monitor'.format(output_dir), force=True)


# initialize a random agent on the environment and play a validation batch
agent = RandomAgent(env)
agent.play()


# save the scores from the validation games in the output directory
scores = pd.Series(env.unwrapped.episode_rewards)
scores.to_csv('{}/final_scores.csv'.format(output_dir))
# print some stats
print('min ', scores.min())
print('mean ', scores.mean())
print('max ', scores.max())

env.close()
