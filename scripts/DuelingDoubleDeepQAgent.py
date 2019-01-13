
# coding: utf-8

# In[ ]:


import os
import gym
from _base import build_env, experiment_dir
import _top_level
from _top_level import top_level
top_level()


# In[ ]:


from src.agents import DeepQAgent
from src.utils import BaseCallback
from src.utils import seed


# In[ ]:


# set the random number seed
seed(1)
# make the output directory
OUTPUT_DIR = experiment_dir('../results', 'SuperMarioBros-4-4-v0', DeepQAgent.__name__)
OUTPUT_DIR


# In[ ]:


# create a file to save the weights to
WEIGHTS_FILE = '{}/weights.h5'.format(OUTPUT_DIR)


# In[ ]:


# create an agent
agent = DeepQAgent(build_env('SuperMarioBros-4-4-v0'))
agent


# In[ ]:


agent.observe()


# In[ ]:


# create a callback for the training procedure
callback = BaseCallback(WEIGHTS_FILE)
# train the agent with given parameters and the callback
agent.train(frames_to_play=10e6, callback=callback)
# save the weights to disk after the training procedure
agent.model.save_weights(WEIGHTS_FILE, overwrite=True)


# In[ ]:


callback.export('{}/training'.format(OUTPUT_DIR))


# In[ ]:


# if monitoring is enabled, setup the monitor for the environment
agent.env = gym.wrappers.Monitor(agent.env, os.path.join(OUTPUT_DIR, 'monitor'), force=True)


# In[ ]:


df = agent.play()


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


agent.plot_episode_rewards(os.path.join(OUTPUT_DIR, 'episode_rewards'))


#
