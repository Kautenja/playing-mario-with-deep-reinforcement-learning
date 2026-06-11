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
python -m mario_rl.collect_demo --config smb_ppo_imitation_fast_dev
python -m mario_rl.imitation --config smb_ppo_imitation_fast_dev
python -m mario_rl.train --config smb_ppo_macro_fast_dev --trainer.enable_progress_bar false
python -m mario_rl.play --config smb_dqn_fast_dev --eval.checkpoint runs/smb_dqn_fast_dev/checkpoints/fast-dev.ckpt
python -m mario_rl.eval_matrix --config smb_dqn_eval_matrix_fast_dev
python -m mario_rl.random --config smb_dqn_fast_dev --env.max_smoke_steps 32
```

`main.sh` exposes the same command surface:

```shell
./main.sh config list
./main.sh train --config smb_dqn_fast_dev --train.accelerator cpu
./main.sh collect-demo --config smb_ppo_imitation_fast_dev
./main.sh pretrain --config smb_ppo_imitation_fast_dev
./main.sh train --config smb_ppo_macro_fast_dev --trainer.enable_progress_bar false
./main.sh play --config smb_dqn_fast_dev --eval.checkpoint runs/example.ckpt
./main.sh eval-matrix --config smb_dqn_eval_matrix_fast_dev
./main.sh random --config smb_dqn_fast_dev
```

Nested overrides use `--section.field value` syntax. Bare positional
`KEY=VALUE` overrides are intentionally rejected so experiment configuration is
always explicit.

## Imitation Pretraining

Local demonstrations live under `data/imitation/`, which is ignored by git.
Place one or more `.npz` segment files there, or point
`imitation.data_dir` at another local directory. Do not commit demonstrations,
videos, ROMs, or generated datasets.

Collect a local keyboard demonstration with:

```shell
./main.sh collect-demo --config smb_ppo_imitation_fast_dev --episodes 3
```

Controls are printed at startup. The collector stores only preprocessed pixels,
integer action labels, terminal flags, and metadata; it does not store RAM,
reward, info dictionaries, task metadata, object maps, or tile maps.

Each `.npz` file is a pixel-only episode or segment with these arrays:

| Field | Shape | Notes |
| --- | --- | --- |
| `observations` | `(steps, channels, height, width)` | `uint8` channel-first stacked pixels matching `replay.state_shape`. |
| `actions` | `(steps,)` | Integer labels in the configured action space. |
| `terminated`, `truncated` | `(steps,)` | Boolean Gymnasium episode flags. |
| `episode_boundaries` | `(steps,)` | Optional boolean boundary markers; required only when terminal flags are omitted. |

The file must also contain a `metadata` JSON string field, or a same-stem
`.json` sidecar, with:

```json
{
  "env_id": "SuperMarioBros-1-1-v0",
  "action_set": "complex",
  "action_count": 12,
  "macro_actions": false,
  "macro_action_set": "conservative",
  "pixel_profile": "grayscale_84",
  "observation_shape": [4, 84, 84],
  "image_size": [84, 84],
  "frame_stack": 4,
  "channel_first": true,
  "source_notes": "optional human-readable provenance"
}
```

The loader rejects mismatched action counts, action sets, macro-action settings,
pixel profiles, channel counts, image sizes, and frame stacks. It also rejects
RAM, `info`, reward fields, task features, object maps, and tile maps in
demonstration files. Behavior cloning feeds only the pixel observation tensor to
the recurrent PPO policy; task-conditioned policies receive zero task features
during pretraining so no task metadata enters the imitation input.

Run a pretrain job after adding local files:

```shell
./main.sh pretrain --config smb_ppo_imitation_fast_dev
```

The command writes `imitation-metrics.json` with cross-entropy loss, validation
accuracy, dataset sizes, action histogram, and checkpoint path. Continue PPO
training from the behavior-cloning checkpoint with:

```shell
./main.sh train --config smb_ppo_fast_dev \
  --train.checkpoint_path runs/smb_ppo_imitation_fast_dev/checkpoints/imitation-pretrain.ckpt \
  --trainer.enable_progress_bar false
```

## Action Abstractions

Macro actions are disabled by default. Set `env.macro_actions: true` with a
Joypad action set such as `complex` to expose named deterministic sequences of
existing Joypad action indices to the policy. Native NES action mode
(`env.action_set: nes`) remains available for 256-action experiments and is not
combined with macro actions.

The packaged `smb_ppo_macro_fast_dev` config uses the conservative macro set.
It keeps every primitive Joypad action as a one-step macro, then adds named
movement options: `run_right`, `short_jump`, `full_jump`, `run_jump`,
`hold_left`, `crouch` when the selected Joypad set exposes down, and `wait`.
Train, play, eval-matrix, and random payloads include the base action set, base
action count, active macro set, active macro action count, and the resolved
index/button sequence for each macro.

Frame skip is applied inside each macro step. With `frame_skip: 4`, a macro
sequence of eight Joypad indices can advance up to 32 emulator frames; if the
underlying environment terminates or truncates early, the macro stops
immediately and `frames_skipped` reports only the executed aggregate. Reward
totals, unclipped/clipped reward diagnostics, and reward component sums are
aggregated across the executed macro sequence using the same rules as the
frame-skip wrapper.

## MacBook Trainability Gate

`verify-macbook` is the local laptop gate for the modern PyTorch path. It runs
the fast unit subset, then a bounded real Super Mario Bros. training job, then
evaluates the checkpoint it just produced:

```shell
./main.sh verify-macbook
```

The default gate uses `smb_ppo_fast_dev`, a small vectorized recurrent-PPO
real-environment config with two rollout environments, deterministic seeds, no
video output, a bounded training pass, and one short single-policy evaluation
episode. The command always verifies CPU. In `--device auto` mode it also runs
the MPS gate when PyTorch reports Apple Silicon MPS availability, otherwise it
prints a clear skip. A targeted MPS check can be run directly:

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

The previous DQN gate remains available when explicitly requested:

```shell
./main.sh verify-macbook --config smb_dqn_macbook_gate --experiment-prefix smb_dqn_macbook_gate
```

## Pixel Observation Profiles

Policies receive an image tensor only. RAM, `info` dictionaries, reward
components, object maps, and tile maps are not fed into observation tensors.
Task metadata remains separate from the pixel tensor and is used only by the
existing sampler, metrics, artifact, and task-conditioning paths.

Packaged configs declare `env.pixel_profile` so preprocessing choices are
visible in resolved configs and train artifacts:

| Profile | Configs | Shape | Use |
| --- | --- | --- | --- |
| `grayscale_84` | `smb_ppo_fast_dev`, `smb_dqn_fast_dev`, `smb_dqn_prioritized_fast_dev`, CPU/MPS DQN configs | `(4, 84, 84)` | Fastest smoke tests and the default MacBook gate. |
| `rgb_balanced_90x96` | `smb_ppo_rgb_fast_dev` | `(12, 90, 96)` | Laptop RGB smoke runs that preserve NES aspect ratio. |
| `rgb_high_fidelity_120x128` | `smb_ppo_rgb_high_fidelity` | `(12, 120, 128)` | Longer RGB experiments where throughput and memory headroom are acceptable. |

The RGB profiles use four stacked RGB frames, so `model.input_channels` resolves
to `3 * frame_stack`. The grayscale profile resolves to `1 * frame_stack`.
`replay.state_shape` is derived from the same preprocessing settings; explicit
mismatches fail during config load instead of allocating the wrong replay or PPO
rollout shape.

Approximate per-observation storage with `uint8` samples is 28 KiB for
`grayscale_84`, 101 KiB for `rgb_balanced_90x96`, and 180 KiB for
`rgb_high_fidelity_120x128`. RGB improves color fidelity for enemies, blocks,
backgrounds, and powerups, but it increases convolution cost and PPO rollout
memory. Use the grayscale profile for fast regression gates, the balanced RGB
profile before committing to a color experiment, and the high-fidelity profile
only after the balanced run shows acceptable local throughput.

```shell
./main.sh train --config smb_ppo_rgb_fast_dev --trainer.enable_progress_bar false
```

## Recurrent Actor-Critic Smoke Training

The recommended path for all-game policy training is the recurrent
actor-critic trainer selected with `train.algorithm: ppo` and
`model.architecture: recurrent_actor_critic`. It uses the same task metadata,
reward transform, and metrics artifacts as DQN, but trains an on-policy
policy/value model with task conditioning, GRU memory, generalized advantage
estimation, clipped PPO losses, entropy regularization, and gradient clipping.

```shell
./main.sh train --config smb_ppo_fast_dev --train.accelerator cpu
tensorboard --logdir runs/smb_ppo_fast_dev/logs/tensorboard
```

The packaged `smb_ppo_fast_dev` config is intentionally tiny: it runs short
CPU rollouts with `ppo.num_envs: 2` against the local editable Mario
environment and writes the usual resolved config, checkpoint, TensorBoard logs,
`train-metrics.csv`, and structured `train-metrics.json` artifacts. Raise
`ppo.num_envs` to four or more for longer CPU throughput runs after the
two-environment gate is stable on the target laptop. Use this path as the
starting point for multi-game task suites and native NES action-space
experiments.

## Curiosity Exploration

Random Network Distillation is available as an opt-in PPO exploration bonus
under the `exploration` config section. It uses the next channel-first pixel
observation returned by `env.step(...)` as its only input, trains a predictor
network toward a frozen target network, and adds the scaled intrinsic reward to
the transformed PPO training reward. RAM, `info`, task IDs, reward components,
progress labels, and auxiliary targets are not inputs to the curiosity reward.

The default RND settings are disabled. When enabled, intrinsic rewards are
clipped and scaled before entering GAE, with optional warmup scaling and
observation/reward normalization controls. Training artifacts keep environment
reward, transformed reward, intrinsic reward, and total PPO training reward
separate in CSV, JSON, TensorBoard, and Lightning CSV logs.

```shell
./main.sh train --config smb_ppo_rnd_fast_dev --trainer.enable_progress_bar false
```

## Snapshot Curriculum

The snapshot curriculum uses the public `nes-py` opaque snapshot API:
`NESEnv.dump_state()` captures native emulator state and `NESEnv.load_state(...)`
restores it later in the same Python process. Snapshot objects are not
serialized to disk. Training artifacts write metadata only, including snapshot
ID, environment ID, ROM SHA-256 compatibility key, action set, seed lineage,
progress, episode step, task ID, tags, and rank score. ROM bytes, native
snapshot bytes, and copied observations are intentionally excluded.

Snapshots are valid only for a compatible environment stack: same configured
environment ID, same action set, same wrapper stack and observation shape, same
ROM fingerprint, and compatible installed `nes-py`/`gym-super-mario-bros`
versions. Incompatible restores raise a clear snapshot compatibility error or
are skipped during sampling. The current implementation is process-local; a
future durable format must validate emulator version, mapper, ROM hash, and
wrapper compatibility before loading.

Enable sampling with config overrides such as:

```shell
./main.sh train --config smb_ppo_fast_dev --snapshot.enabled true --snapshot.capture_interval_steps 4 --snapshot.sample_probability 0.5
```

Restored episodes still feed only pixel observations into DQN/PPO models.
Task IDs, progress, seed lineage, and snapshot tags are used for sampling,
metrics, and artifacts only. Metrics mark `snapshot_start` episodes separately
and report full-reset clear counts so hard-section starts are not confused with
full-level clears.

## Auxiliary Losses

Auxiliary losses are optional supervised heads on the recurrent actor-critic
GRU state. They are disabled by default and should be enabled only when the
environment `info` stream provides the corresponding labels. The supported
targets are `progress_delta`, `progress_normalized`, `clear`, `death`,
`transformed_reward`, `reward_total_unclipped`, `reward_total_clipped`, and
`game_family`. Missing labels are masked per target, so unavailable fields do
not contribute to the loss.

Use the packaged smoke config to verify the path:

```shell
./main.sh train --config smb_ppo_auxiliary_fast_dev --trainer.enable_progress_bar false
```

Per-target weights live under `auxiliary.weights`; omitted enabled targets use
weight `1.0`. Training logs `train/auxiliary_loss`,
`train/auxiliary_<target>_loss`, and `train/auxiliary_<target>_valid`. A low
valid count means the environment did not provide enough labels for that head,
not that the prediction is good. Keep weights small at first because the
auxiliary total is added directly to the PPO policy/value objective.

## Lightning DQN Smoke Training

DQN remains available as a compact off-policy baseline using PyTorch Lightning,
native PyTorch DQN modules, uniform replay by default, and the packaged config
tree. Smoke runs write a resolved config, Lightning CSV and TensorBoard logs,
train metrics, and a checkpoint under `runs/<experiment_name>/`.

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
Training shows Lightning's rich progress bar by default. Pass
`--trainer.progress_bar tqdm` for the classic tqdm indicator, or
`--trainer.enable_progress_bar false` for quiet/headless runs.

Set `replay.prioritized: true` to enable proportional prioritized replay for
DQN. Prioritized batches carry sample indices and normalized importance weights,
the Huber TD loss applies those weights per sample, and priorities are updated
after each optimizer step from absolute TD error plus epsilon. The packaged
smoke config exercises the path:

```shell
./main.sh train --config smb_dqn_prioritized_fast_dev --trainer.enable_progress_bar false
```

Train artifacts include a `replay` section in `train-metrics.json` plus CSV
columns for whether replay was prioritized, alpha/beta, priority updates, and
the latest importance-weight mean. The uniform replay contract remains the
default when `replay.prioritized: false`.

## Task Metrics Artifacts

Training writes the existing one-row `train-metrics.csv` summary plus a
structured `train-metrics.json` artifact. Evaluation/play writes
`eval-metrics.json` using the same accumulator. The JSON payloads include a
`global` summary, `by_game_family` summaries, `by_task` summaries, per-episode
records, and missing-info counters for optional fields that were absent from
Gymnasium `info`.

Emitted task fields are `task_id`, `game_family`, `world`, and `stage`.
Episode records include `episode_return`, `transformed_return`, `raw_return`,
`unclipped_return`, `clipped_return`, `clear`, `death`, `timeout`,
`terminated`, `truncated`, `max_progress`, `final_progress`, and
`reward_component_sums`. Aggregate records include episode/step/frame counts,
return totals and means, clear/death/timeout/truncation counts and rates, max
and final-progress summaries, reward component sums, and
`missing_info_counts`.

Lightning logs the key scalar metrics with stable `train/*` names, including
`train/clear_rate`, `train/death_rate`, `train/truncation_count`,
`train/max_progress`, and `train/final_progress_mean`.

## Evaluation Matrix

`eval-matrix` evaluates a checkpoint across a deterministic task matrix instead
of collapsing all progress into one averaged return:

```shell
./main.sh train --config smb_dqn_eval_matrix_fast_dev
./main.sh eval-matrix --config smb_dqn_eval_matrix_fast_dev
```

Matrix filters live under `evaluation_matrix` and support game families,
single-stage versus full-game tasks, train/eval split selection, validated task
selection, explicit include/exclude environment IDs, `max_tasks`, deterministic
seed expansion, and per-task episode counts. Use train split filters for model
selection smoke runs and eval split filters for held-out reporting. For
all-game reports, leave `game_families` empty and include both full-game and
single-stage configs in separate runs so stage transfer and complete-game
rollouts stay comparable.

Each run writes `eval-matrix-summary.json` and `eval-matrix-episodes.csv` under
the experiment directory. The JSON summary contains global, per-game-family,
and per-task aggregates from the shared metrics accumulator plus the selected
task matrix. The CSV has one row per task, seed, and episode for spreadsheet
inspection. Set `evaluation_matrix.include_smb3_catalog: true` to include the
full 56-stage SMB3 catalog as non-runnable metadata alongside the validated
registered SMB3 stage entries.

Video capture is disabled by default. When
`evaluation_matrix.video_enabled: true`, the runner uses stable task/seed/
episode prefixes such as
`eval-matrix-supermariobros3-1-1-v0-seed-123-episode-0` and only enables
`rgb_array` rendering for those video episodes.

## Reward Transforms

`gym-super-mario-bros` 9.x exposes shaped per-step reward diagnostics in
`info`, including `reward_total_unclipped`, `reward_total_clipped`, and
`reward_components`. Mario RL therefore no longer applies Atari-style sign
clipping by default. Packaged configs use `reward_transform.mode: env`, which
trains on the environment reward unchanged and records raw, unclipped, clipped,
and training rewards in replay diagnostics and train metrics.

Available reward modes are `env`, `sign`, `unclipped`, `clipped`, and
`component_weights`:

```shell
./main.sh train --config smb_dqn_fast_dev --reward_transform.mode sign
./main.sh train --config smb_dqn_fast_dev \
  --reward_transform.mode component_weights \
  --reward_transform.component_weights progress=1,death=0.5
```

Use `sign` only when comparing against the old clipped baseline. `unclipped`
and `clipped` read the matching 9.x info fields, and
`reward_transform.missing_total_policy` controls whether missing totals fail or
fall back to the environment reward. `component_weights` combines named
`reward_components`; `reward_transform.missing_component_policy` controls
whether absent components contribute zero or fail fast.

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

## Task Suites

`TaskSuiteConfig` and `TaskSuite` build deterministic curricula from registered
task metadata without constructing environments. Suites can filter by game
family, single-stage or full-game tasks, train/eval split, validation status,
aliases, explicit environment IDs, worlds, and stages. Sampling chooses a game
family first, using per-family weights, so large SMB1 and Lost Levels catalogs
do not automatically swamp smaller SMB2 USA and SMB3 task sets.

The task-suite smoke config alternates between a tiny validated SMB1/SMB3
surface and switches tasks at episode boundaries:

```shell
./main.sh train --config smb_dqn_task_suite_fast_dev
```

For custom configs, set `task_suite.enabled: true` and leave `env.id` as a
single-task fallback for legacy commands and disabled-suite runs.

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
shape `(4, 84, 84)` and `uint8` dtype. `make_env` accepts `nes`, `right`,
`right_only`, `simple`, and `complex` action-set names, explicit preprocessing
keyword arguments, or a `MarioEnvConfig` object. The packaged configs keep
`action_set: simple` as the default smoke-training policy because it is small
and fast for local iteration; universal all-game training may prefer `nes` for
the full 256-button NES action space or `complex` for a 12-action directional
subset. Packaged configs use `model.num_actions: auto`, so training artifacts
record the concrete action count resolved from the configured action set.

The default single-stage config uses the canonical
`SuperMarioBros-1-1-v0` ID from the 9.x environment surface. The
separator-free alias `SuperMarioBros1-1-v0` is still accepted by
`gym-super-mario-bros` 9.1.0 for compatibility, and Mario RL re-exports task
metadata helpers for curriculum, smoke selection, and catalog reporting:

```python
from mario_rl.envs import TaskSuite, TaskSuiteConfig, available_env_ids

env_ids = available_env_ids(game_family="smb1", single_stage=True)
suite = TaskSuite(TaskSuiteConfig(game_families=("smb1", "smb3"), seed=123))
env_id = suite.task_for_episode(0).env_id
```

The old `SuperMarioBrosRandomStages-*` IDs were removed upstream in
`gym-super-mario-bros` 9.0.0. Use task-suite sampling for seeded stage
selection, or pass any registered 9.x ID directly, including
`SuperMarioBros2USA-v0`, `SuperMarioBros2USA-<world>-<stage>-v0`,
`SuperMarioBros3-v0`, and `SuperMarioBros3-1-1-v0`.

## Deprecated `src` Tree

The original Gym/Keras implementation under `src/` is retained only as
historical reference while the PyTorch migration settles. It is not installed,
tested, or supported as a runtime API. Use `mario_rl`, `python -m mario_rl.*`,
or the `./main.sh` commands above for all active training, evaluation, and
random-rollout workflows.
