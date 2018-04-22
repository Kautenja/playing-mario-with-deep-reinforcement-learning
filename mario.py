exp_directory = 'results'
agent_name = 'DeepQAgent'
game_name = 'SuperMarioBros-1-1'
render_mode='rgb_array'

import os
import datetime
from multiprocessing import Lock
import pandas as pd
from gym.wrappers import Monitor

import logging
logger = logging.getLogger('gym')
logger.setLevel(50)

from src.agents import DeepQAgent, A3CAgent
from src.util import BaseCallback, JupyterCallback
from src.environment.nes import build_nes_environment


agents = {
    DeepQAgent.__name__: DeepQAgent,
    A3CAgent.__name__: A3CAgent,
}


# setup the experiment directory
now = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
exp_directory = '{}/{}/{}/{}'.format(exp_directory, game_name, agent_name, now)
if not os.path.exists(exp_directory):
    os.makedirs(exp_directory)


plot_dir = '{}/plots'.format(exp_directory)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


env = build_nes_environment(game_name)
env.configure(lock=Lock())
env = Monitor(env, '{}/monitor'.format(exp_directory), force=True)
print(env.observation_space.shape)

from src.base import AnnealingVariable
agent = agents[agent_name](env, render_mode=render_mode,
        exploration_rate=AnnealingVariable(0.1, 0.1, 1)
    )
agent.model.load_weights('weights.h5')
agent.target_model.load_weights('weights.h5')
agent


# write some info about the agent to disk
with open('{}/agent.py'.format(exp_directory), 'w') as agent_file:
    agent_file.write(repr(agent))


# initial = agent.play(games=5)
# initial = pd.Series(initial)
# initial.to_csv('{}/initial.csv'.format(exp_directory))
# print(initial.describe())


# agent.observe()


callback = BaseCallback()
agent.train(callback=callback, frames_to_play=2500000)


# save the training results
scores = pd.Series(callback.scores)
scores.to_csv('{}/scores.csv'.format(exp_directory))
losses = pd.Series(callback.losses)
losses.to_csv('{}/losses.csv'.format(exp_directory))


final = agent.play(games=5)
final = pd.Series(final)
final.to_csv('{}/final.csv'.format(exp_directory))
print(final.describe())


agent.model.save_weights('{}/weights.h5'.format(exp_directory), overwrite=True)
