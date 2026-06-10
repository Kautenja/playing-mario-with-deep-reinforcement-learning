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

The runtime dependency set targets `gym-super-mario-bros` 9.1.x and `nes-py`
9.x. The umbrella checkout is the preferred development path because it keeps
the native emulator and Mario wrapper pinned to matching local submodules.

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

## MacBook Trainability Gate

`verify-macbook` is the local laptop gate for the modern PyTorch path. It runs
the fast unit subset, then a bounded real Super Mario Bros. training job, then
evaluates the checkpoint it just produced:

```shell
./main.sh verify-macbook
```

The default gate uses `smb_dqn_macbook_gate`, a small real-environment config
with 40x40 frame stacks, eight training steps, one eight-step evaluation
episode, deterministic seeds, no rendering, and no video output. The command
always verifies CPU. In `--device auto` mode it also runs the MPS gate when
PyTorch reports Apple Silicon MPS availability, otherwise it prints a clear
skip. A targeted MPS check can be run directly:

```shell
./main.sh verify-macbook --device mps
```

Each device run writes `macbook-gate-summary.json` under the experiment
directory with wall time, environment frames per second, optimizer steps per
second, peak memory when available, Python/PyTorch versions, the selected
environment ID, artifact paths, and performance-budget warnings. The initial
budget is intentionally conservative and warning-based: train under 180 seconds,
eval under 90 seconds, total under 300 seconds, at least 0.25 environment FPS,
and at least 0.02 optimizer steps per second. Use `--strict-budget` to make
these warnings fail the gate.

For repeatable profiling without the unit/eval stages, run the benchmark mode:

```shell
./main.sh verify-macbook --benchmark-only --device cpu
```

Benchmark mode measures a bounded random environment rollout plus the same mini
optimization pass so later specs can compare environment stepping and optimizer
throughput against the saved summary.

## Lightning DQN Smoke Training

The active training path uses PyTorch Lightning, native PyTorch DQN modules,
uniform replay, and the packaged config tree. Smoke runs write a resolved
config, Lightning CSV and TensorBoard logs, train metrics, and a checkpoint under
`runs/<experiment_name>/`.

```shell
./main.sh train --config smb_dqn_fast_dev --train.accelerator cpu
tensorboard --logdir runs/smb_dqn_fast_dev/logs/tensorboard
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

## Task Conditioning

`mario_rl.tasks` and `mario_rl.envs` expose the `gym-super-mario-bros` 9.1
task metadata surface, including `MarioTask`, `available_tasks`,
`task_for_env_id`, and `smb3_stage_matrix`. `TaskFeatureEncoder` converts a
registered or custom environment ID into one dense `float32` feature vector:

- one-hot `game_family`, canonical `task_id`, and `rom_mode` vocabularies;
- normalized `world` and `stage` numeric fields plus present/absent flags;
- binary `single_stage` and `validated` flags.

Alias IDs such as `SuperMarioBros1-1-v0` encode to the canonical task ID
`SuperMarioBros-1-1-v0`. Unknown custom IDs use the explicit `<unknown>`
categorical bucket with zeroed numeric fields, so feature encoding does not
require constructing a ROM-backed environment.

Task conditioning is disabled by default to preserve existing smoke behavior
and pixel-only checkpoints. Use the opt-in config when training a conditioned
DQN:

```shell
./main.sh train --config smb_dqn_task_conditioned_fast_dev
./main.sh play --config smb_dqn_task_conditioned_fast_dev
```

## Modern Environments

The Gymnasium environment surface lives under `mario_rl.envs`:

```python
from mario_rl.envs import make_env

env = make_env("SuperMarioBros-1-1-v0", render_mode="rgb_array", seed=123)
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

The default preprocessing output is a channel-first grayscale frame stack with
shape `(4, 84, 84)` and `uint8` dtype. `make_env` accepts `right_only`,
`simple`, and `complex` action-set names, explicit preprocessing keyword
arguments, or a `MarioEnvConfig` object.

The default single-stage config uses the canonical
`SuperMarioBros-1-1-v0` ID from the 9.x environment surface. The
separator-free alias `SuperMarioBros1-1-v0` is still accepted by
`gym-super-mario-bros` 9.1.0 for compatibility, and Mario RL re-exports task
metadata helpers for curriculum or smoke selection:

```python
from mario_rl.envs import available_env_ids, choose_stage_env_id

env_ids = available_env_ids(game_family="smb1", single_stage=True)
env_id = choose_stage_env_id(seed=123)
```

The old `SuperMarioBrosRandomStages-*` IDs were removed upstream in
`gym-super-mario-bros` 9.0.0. Use `choose_stage_env_id` for seeded stage
selection, or pass any registered 9.x ID directly, including
`SuperMarioBros2USA-v0`, `SuperMarioBros2USA-<world>-<stage>-v0`,
`SuperMarioBros3-v0`, and `SuperMarioBros3-1-1-v0`.

## Deprecated `src` Tree

The original Gym/Keras implementation under `src/` is retained only as
historical reference while the PyTorch migration settles. It is not installed,
tested, or supported as a runtime API. Use `mario_rl`, `python -m mario_rl.*`,
or the `./main.sh` commands above for all active training, evaluation, and
random-rollout workflows.
