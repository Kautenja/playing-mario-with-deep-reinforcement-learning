"""A package providing an Open.ai Gym interface to Super Mario Bros."""
from gym.envs.registration import register
from .nes_env import NesEnv, MetaNesEnv
from .super_mario_bros import SuperMarioBrosEnv, MetaSuperMarioBrosEnv, SMB_LEVELS


# register the full scale, 32 level environment
register(
    id='SuperMarioBros-v0',
    entry_point='super_mario:MetaSuperMarioBrosEnv',
    max_episode_steps=9999999,
    reward_threshold=32000,
    kwargs={
        'average_over': 3,
        'passing_grade': 600,
        'min_tries_for_avg': 3
    },
    nondeterministic=True,
)


# iterate over the levels and register each one
for (world_number, level_number, area_number, max_distance) in SMB_LEVELS:
    level = (world_number - 1) * 4 + (level_number - 1)
    register(
        id='SuperMarioBros-{}-{}-v0'.format(world_number, level_number),
        entry_point='super_mario:SuperMarioBrosEnv',
        max_episode_steps=10000,
        reward_threshold=(max_distance - 40),
        kwargs={ 'level': level },
        # Seems to be non-deterministic about 5% of the time
        nondeterministic=True,
    )
