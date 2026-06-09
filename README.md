# Playing Super Mario Bros. With Deep Reinforcement Learning

Using Deep-Q style agents to play Super Mario Bros.

![DDQN-SMB-1-4](https://user-images.githubusercontent.com/2184469/113493396-8e6d3080-94a4-11eb-8e4c-956c277ac76f.gif)

## Installation

Python 3.13 or 3.14 is required. The modern package surface is `mario_rl`.
Legacy Keras/TensorFlow code remains in the repository for reference, but it is
not part of the default runtime install path.

### Umbrella Editable Checkout

From `gym-nes/playing-mario-with-deep-reinforcement-learning`:

```shell
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ../nes-py -e ../gym-super-mario-bros
python -m pip install -e .
python -m pip check
python -m unittest discover .
./main.sh unittest
```

This installs `nes-py` and `gym-super-mario-bros` from the umbrella checkout so
imports resolve to the active submodules instead of older PyPI wheels.

### PyPI Dependency Path

From a standalone checkout outside the umbrella repository:

```shell
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip check
python -m unittest discover .
```

At the last dependency refresh, PyPI exposed `gym-super-mario-bros` 7.4.0 and
`nes-py` 8.2.1, while the umbrella checkout carried compatible 8.0.0 and 9.0.0
development releases. The dependency ranges accept both paths.

## Tests

```shell
python -m unittest discover .
./main.sh unittest
```

## Modern Environments

The Gymnasium environment surface lives under `mario_rl.envs`:

```python
from mario_rl.envs import make_env

env = make_env("SuperMarioBros1-1-v0", render_mode="rgb_array", seed=123)
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

The default preprocessing output is a channel-first grayscale frame stack with
shape `(4, 84, 84)` and `uint8` dtype. `make_env` accepts `right_only`,
`simple`, and `complex` action-set names, explicit preprocessing keyword
arguments, or a `MarioEnvConfig` object.

## Legacy Scripts

The original `python . -m train`, `python . -m random`, and `python . -m play`
entrypoints still live under `src/`. They use the old Gym/Keras assumptions and
will be ported by the later PyTorch and Gymnasium migration specs.
