"""Usage:

    python validate_random.py <game name> <results directory>
"""
import os
import sys
from datetime import datetime
import pandas as pd
from gym.wrappers import Monitor


# load variables from the command line
try:
    game_name = sys.argv[1]
    output_dir = sys.argv[2]
except IndexError:
    print(__doc__)
    sys.exit(-1)
# setup the output directory with a timestamped directory
now = datetime.today().strftime('%Y-%m-%d_%H-%M')
output_dir = '{}/{}/{}'.format(output_dir, game_name, now)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# load these after command line arg checking bc tensorflow is slow to load
from src.environment.atari import build_atari_environment
from src.environment.nes import build_nes_environment
from src.agents import RandomAgent


# check if we need to load the NES environment
if 'SuperMarioBros' in game_name:
    env, r_cache = build_nes_environment(game_name)
# default to the Atari environment
else:
    env, r_cache = build_atari_environment(game_name, is_validation=True)
# wrap the environment with a monitor
env = Monitor(env, '{}/monitor'.format(output_dir), force=True)


# initialize a random agent on the environment and play a validation batch
agent = RandomAgent(env)
agent.play(games=3)


# save the scores from the validation games in the output directory
scores = pd.Series(r_cache._rewards)
scores.to_csv('{}/final_scores.csv'.format(output_dir))
