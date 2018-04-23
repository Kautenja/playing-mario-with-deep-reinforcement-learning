from multiprocessing import Lock
import pandas as pd
from gym.wrappers import Monitor
from src.environment.nes import build_nes_environment


env = build_nes_environment('SuperMarioBros-1-1')
env.configure(lock=Lock())


scores = []

for game in range(100):
    env.reset()
    done = False
    score = 0
    while not done:
        _, reward, done, _ = env.step(env.action_space.sample())
        env.render()
    scores.append(score)

scores = pd.Series(scores)
scores.to_csv('results/SMB_random.csv')
