"""Usage:

    python dddqn_play.py <results directory>
"""
import os
import sys
import pandas as pd
from matplotlib import pyplot as plt


# load variables from the command line
try:
    output_dir = sys.argv[1]
    # these will have the form .../<game_name>/<method>/<timestamp>
    game_name = output_dir.split('/')[-3]
except IndexError:
    print(__doc__)
    sys.exit(-1)


# set up the weights file
weights_file = '{}/weights.h5'.format(output_dir)
# make sure the weights exist
if not os.path.exists(weights_file):
    print('{} not found!'.format(weights_file))
    sys.exit(-1)


# load these after command line arg checking bc tensorflow is slow to load
# and generates some warning output deprecations and such that clog std out
from src.environment.atari import build_atari_environment
from src.environment.nes import build_nes_environment
from src.agents import DeepQAgent


# check if we need to load the NES environment
if 'SuperMarioBros' in game_name:
    monitor_dir = '{}/monitor_play'.format(output_dir)
    env = build_nes_environment(game_name, monitor_dir=monitor_dir)
# default to the Atari environment
else:
    env = build_atari_environment(game_name, is_validation=True)
    from gym.wrappers import Monitor
    env = Monitor(env, '{}/monitor_play'.format(output_dir), force=True)


# build the agent without any replay memory since we're just playing, load
# the trained weights, and play some games
agent = DeepQAgent(env, replay_memory_size=0)
agent.model.load_weights(weights_file)
agent.target_model.load_weights(weights_file)
agent.play()


# collect the game scores, actual scores from the reward cache wrapper, not
# mutated, clipped, or whatever rewards that the agent sees
scores = pd.concat([pd.Series(env.unwrapped.episode_rewards)], axis=1)
scores.columns = ['Score']
scores.index.name = 'Episode'
print(scores.describe())
# write the scores and a histogram visualization to disk
scores.to_csv('{}/final_scores.csv'.format(output_dir))
ax = scores['Score'].hist()
ax.set_title('Histogram of Scores')
ax.set_ylabel('Number of Episodes')
ax.set_xlabel('Score')
plt.savefig('{}/final_scores.pdf'.format(output_dir))


env.close()
