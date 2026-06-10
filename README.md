# Playing Super Mario Bros. With Deep Reinforcement Learning

Using Deep-Q style agents to play Super Mario Bros.

![DDQN-SMB-1-4](https://user-images.githubusercontent.com/2184469/113493396-8e6d3080-94a4-11eb-8e4c-956c277ac76f.gif)

## Installation

Python 3.13 or 3.14 is required. The supported package surface is `mario_rl`.
The old Keras/TensorFlow `src/` tree is deprecated, reference-only, and
excluded from the runtime install path.

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

## Config-Driven Commands

Packaged YAML configs live under `mario_rl.config` and can be addressed by
name or path:

```shell
python -m mario_rl.config list
python -m mario_rl.config path smb_dqn_fast_dev
python -m mario_rl.train --config smb_dqn_fast_dev --train.accelerator cpu
python -m mario_rl.play --config smb_dqn_fast_dev --eval.checkpoint runs/smb_dqn_fast_dev/checkpoints/fast-dev.ckpt
python -m mario_rl.random --config smb_dqn_fast_dev --env.max_smoke_steps 32
```

`main.sh` exposes the same command surface:

```shell
./main.sh config list
./main.sh train --config smb_dqn_fast_dev --train.accelerator cpu
./main.sh play --config smb_dqn_fast_dev --eval.checkpoint runs/example.ckpt
./main.sh random --config smb_dqn_fast_dev
```

Nested overrides use `--section.field value` syntax. Bare positional
`KEY=VALUE` overrides are intentionally rejected so experiment configuration is
always explicit.

## Lightning DQN Smoke Training

The active training path uses PyTorch Lightning, native PyTorch DQN modules,
uniform replay, and the packaged config tree. Smoke runs write a resolved
config, Lightning CSV logs, train metrics, and a checkpoint under
`runs/<experiment_name>/`.

```shell
./main.sh train --config smb_dqn_fast_dev --train.accelerator cpu
./main.sh play --config smb_dqn_fast_dev --eval.episodes 1 --eval.max_steps 32
```

On Apple Silicon with MPS available:

```shell
./main.sh train --config smb_dqn_fast_dev --train.accelerator mps --train.devices 1
```

On a CUDA host:

```shell
MARIO_RL_RUN_CUDA_SMOKE=1 ./main.sh train --config smb_dqn_fast_dev --train.accelerator gpu --train.devices 1
```

The play command defaults to the smoke checkpoint path for the selected config.
Pass `--eval.checkpoint PATH` to evaluate a specific Lightning checkpoint.
Training shows Lightning progress by default. Pass
`--trainer.enable_progress_bar false` for quiet/headless runs.

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

## Deprecated `src` Tree

The original Gym/Keras implementation under `src/` is retained only as
historical reference while the PyTorch migration settles. It is not installed,
tested, or supported as a runtime API. Use `mario_rl`, `python -m mario_rl.*`,
or the `./main.sh` commands above for all active training, evaluation, and
random-rollout workflows.
