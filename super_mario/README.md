# NES - Super Mario Bros

Environments for playing Super Mario Bros with Open.ai Gym.

## Usage

```
import gym
# import the SMB gym module to register the environments
import gym_super_mario_bros
env = gym.make('SuperMarioBros-1-1-v0')
env.reset()
```

**NOTE** FCEUX will launch when upon a call to `reset()`

# Single Level

This environment allows you to play the individual levels from Super Mario
Bros.

## Environments

-   There are 32 environments available, with the following syntax.
    -   world_number is a number between 1 and 8
    -   level_number is a number between 1 and 4

```
SuperMarioBros-<world_number>-<level_number>-v0
```

and

```
SuperMarioBros-<world_number>-<level_number>-Tiles-v0
```

#### Example

```
SuperMarioBros-6-1-v0
```

# Full Game

This environment allows you to play the original Super Mario Bros all the way
through towards a goal of 32,000 points.

```
SuperMarioBros-v0
```

## Scoring

-   Each level score has been standardized on a scale of 0 to 1,000
-   The passing score for a level is 990 (99th percentile)
-   A bonus of 1,600 (50 * 32 levels) is given if all levels are passed
-   The score for a level is the average of the last 3 tries
-   If there has been less than 3 tries for a level, the missing tries will
    have a score of 0 (e.g. if you score 1,000 on the first level on your
    first try, your level score will be (1,000 + 0 + 0) / 3 = 333.33)
-   The total score is the sum of the level scores, plus the bonus if you passed all levels.

#### Example

Given a list of tries:

-   Level 0: 500
-   Level 0: 750
-   Level 0: 800
-   Level 0: 1,000
-   Level 1: 100
-   Level 1: 200

| Level            | Score                          |
|:-----------------|:-------------------------------|
| 0                | [1,000 + 800 + 750] / 3 = 850  |
| 1                | [200 + 100 + 0] / 3 = 100      |
| 2-8              | 0                              |
| Completion Bonus | 0                              |
|                  |                                |
| **Total**        | 950                            |


## Advancement

To unlock the next level, you must achieve a level score (avg of last 3
tries) of at least 600 (i.e. passing 60% of the last level)

### Changing Levels

#### Manual

-   `obs, reward, is_finished, info = env.step(action)`
-   if `is_finished` is true, you can call
    `env.change_level(level_number)` to change to an unlocked level
-   level_number is a number from 0 to 31
-   you can see
    -   the current level with `info["level"]`
    -   the list of level score with `info["scores"]`
    -   the list of locked levels with `info["locked_levels"]`
    -   your total score with `info["total_reward"]`

```python
import gym
env = gym.make('meta-SuperMarioBros-v0')
env.reset()
total_score = 0
while total_score < 32000:
    action = [0] * 6
    obs, reward, is_finished, info = env.step(action)
    env.render()
    total_score = info["total_reward"]
    if is_finished:
        env.change_level(level_you_want)
```

#### Automatic

if you don't call change_level() and the level is finished, the system will
automatically select the unlocked level with the lowest level score (which is
likely to be the last unlocked level)

```python
import gym
env = gym.make('meta-SuperMarioBros-v0')
env.reset()
total_score = 0
while total_score < 32000:
    action = [0] * 6
    obs, reward, is_finished, info = env.step(action)
    env.render()
    total_score = info["total_reward"]
```


# Observation Space

Environments will return a 256x224 array representation of the screen, where
each square contains red, blue, and green value (RGB).

# Action Space

The NES controller is composed of 6 buttons (Up, Left, Down, Right, A, B).
The step function expects an array of booleans indicating which buttons to
hold of:

-   First Item -  Up
-   Second Item - Left
-   Third Item -  Down
-   Fourth Item - Right
-   Fifth Item -  A
-   Sixth Item -  B

#### Example

action = [0, 0, 0, 1, 1, 0] would activate right (4th element), and A (5th
element).

## Discrete Action Space

Discrete action spaces are often easier to work with. `gym_super_mario_bros`
provides a wrapper to convert the environment to discrete actions space:

```
from gym_super_mario_bros.wrappers import ToDiscreteActionSpaceEnv
env = ToDiscreteActionSpaceEnv(env)
```

# Gameplay

-   The game will automatically close if Mario dies or shortly after the
    flagpole is touched
-   The game will only accept inputs after the timer has started to decrease
    (i.e. it will automatically move through the menus and animations)
-   The total reward is the distance on the x axis.

## `info` dict

The following variables are available in the info dict returned with each step
(`_, _, _, info = env.step(a)`).

| Variable Name   | Description                                                             |
|:----------------|:------------------------------------------------------------------------|
| `distance`      | Total distance from the start (x-axis)                                  |
| `life`          | Number of lives Mario has (3 if Mario is alive, 0 is Mario is dead)     |
| `score`         | The current score                                                       |
| `coins`         | The current number of coins                                             |
| `time`          | The current time left                                                   |
| `player_status` | Indicates if Mario is small (0), big (1), or can shoot fireballs (2+)   |
| `ignore`        | Added with a value of True if the game is stuck and is terminated early |

-   A value of -1 indicates that the value is unknown


## Game is Stuck

After 20 seconds, the stuck game will be automatically closed, and `step()`
will return `done=True` with an `info` dictionary containing `ignore=True`.
You can simply check if the ignore key is in the info dictionary, and ignore
that specific episode.
