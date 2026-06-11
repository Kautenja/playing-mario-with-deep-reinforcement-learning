"""Config-driven Lightning training entrypoint for the PyTorch port."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace

from .config import (
    MarioRLConfig,
    action_space_summary,
    cli,
    pixel_observation_summary,
    with_resolved_model_num_actions,
)
from .exploration import exploration_summary
from .rewards import reward_transform_summary


def run(config: MarioRLConfig, *, env_factory=None) -> int:
    """Run a bounded Lightning training job and write smoke artifacts."""
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

    from mario_rl.lightning import (
        DQNLightningModule,
        PPOLightningModule,
        experiment_paths,
        trainer_accelerator,
        trainer_devices,
        trainer_progress_callbacks,
        write_json,
        write_resolved_config,
        write_train_metrics,
    )

    config = with_resolved_model_num_actions(config)
    paths = experiment_paths(config)
    if config.env.video_enabled and config.env.video_dir is None:
        config = replace(
            config,
            env=replace(
                config.env,
                render_mode=config.env.render_mode or "rgb_array",
                video_dir=str(paths.videos),
            ),
        )
        config = with_resolved_model_num_actions(config)
    write_resolved_config(config, paths.resolved_config)
    if config.trainer.seed is not None:
        seed_everything(config.trainer.seed, workers=True)

    algorithm = _normalized_algorithm(config)
    if algorithm == "dqn":
        module = DQNLightningModule(config, env_factory=env_factory)
    elif algorithm == "ppo":
        module = PPOLightningModule(config, env_factory=env_factory)
    else:
        raise ValueError(f"unsupported training algorithm: {config.train.algorithm!r}")
    csv_logger = CSVLogger(save_dir=str(paths.logs), name="lightning")
    tensorboard_logger = TensorBoardLogger(save_dir=str(paths.logs), name="tensorboard")
    trainer = Trainer(
        accelerator=trainer_accelerator(config),
        devices=trainer_devices(config),
        precision=config.trainer.precision,
        deterministic=config.trainer.deterministic,
        default_root_dir=str(paths.root),
        max_epochs=1,
        max_steps=-1,
        limit_train_batches=int(config.train.max_steps),
        logger=[csv_logger, tensorboard_logger],
        enable_checkpointing=False,
        enable_progress_bar=bool(config.trainer.enable_progress_bar),
        callbacks=trainer_progress_callbacks(config),
        log_every_n_steps=max(1, min(int(config.train.log_interval), int(config.train.max_steps))),
    )
    fit_checkpoint_path = config.train.checkpoint_path
    if fit_checkpoint_path:
        from mario_rl.imitation import (
            is_imitation_checkpoint,
            load_imitation_policy_weights,
        )

        if is_imitation_checkpoint(fit_checkpoint_path):
            load_imitation_policy_weights(module, fit_checkpoint_path)
            fit_checkpoint_path = None
    trainer.fit(module, ckpt_path=fit_checkpoint_path)
    trainer.save_checkpoint(str(paths.checkpoint))

    metrics = module.metrics_summary()
    action_summary = action_space_summary(config)
    pixel_summary = pixel_observation_summary(config)
    reward_summary = reward_transform_summary(config.reward_transform)
    exploration = exploration_summary(config)
    replay = _replay_summary(config, module)
    metrics.update(action_summary)
    metrics.update(reward_summary)
    metrics.update(exploration)
    metrics.update(_flat_replay_metrics(replay))
    metrics_payload = {
        "command": "train",
        "algorithm": algorithm,
        **action_summary,
        "pixel_observation": pixel_summary,
        **reward_summary,
        "replay": replay,
        "exploration": {
            **exploration,
            "intrinsic_reward_total": metrics.get("intrinsic_reward_total", 0.0),
            "intrinsic_reward_mean": metrics.get("intrinsic_reward_mean", 0.0),
            "rnd_raw_error_mean": metrics.get("rnd_raw_error_mean", 0.0),
            "rnd_loss": metrics.get("rnd_loss", 0.0),
            "rnd_predictor_grad_norm": metrics.get(
                "rnd_predictor_grad_norm",
                0.0,
            ),
        },
        "lightning": {
            "global_step": metrics["global_step"],
            "env_frames": metrics["env_frames"],
            "episodes": metrics["episodes"],
            "epsilon": metrics["epsilon"],
            "loss": metrics["loss"],
            "learning_rate": metrics["learning_rate"],
        },
        **module.metrics_payload(include_active=True),
    }
    task_suite_payload = module.task_suite_payload()
    if task_suite_payload is not None:
        metrics_payload["curriculum"] = task_suite_payload
        write_json(paths.curriculum_state, task_suite_payload)
    snapshot_payload = module.snapshot_payload()
    if snapshot_payload is not None:
        metrics_payload["snapshots"] = snapshot_payload
        write_json(paths.snapshot_metadata, snapshot_payload)
    if algorithm == "ppo":
        metrics_payload["ppo"] = {
            "total_loss": metrics.get("loss"),
            "policy_loss": metrics.get("ppo_policy_loss"),
            "value_loss": metrics.get("ppo_value_loss"),
            "entropy": metrics.get("ppo_entropy"),
            "approximate_kl": metrics.get("ppo_approximate_kl"),
            "clip_fraction": metrics.get("ppo_clip_fraction"),
            "num_envs": config.ppo.num_envs,
            "rollout_steps": config.ppo.rollout_steps,
            "minibatch_size": config.ppo.minibatch_size,
            "epochs": config.ppo.epochs,
        }
        metrics_payload["auxiliary"] = {
            "enabled": bool(config.auxiliary.enabled),
            "targets": list(config.auxiliary.targets),
            "loss": metrics.get("auxiliary_loss", 0.0),
            "losses": dict(metrics.get("auxiliary_losses", {})),
            "valid_counts": dict(metrics.get("auxiliary_valid_counts", {})),
        }
    metrics["metrics_payload"] = metrics_payload
    write_train_metrics(paths.train_metrics, metrics)
    write_json(paths.train_metrics_json, metrics_payload)
    print(
        json.dumps(
            {
                "command": "train",
                "algorithm": algorithm,
                **action_summary,
                "pixel_observation": pixel_summary,
                **reward_summary,
                "checkpoint": str(paths.checkpoint),
                "experiment_dir": str(paths.root),
                "metrics": str(paths.train_metrics),
                "metrics_json": str(paths.train_metrics_json),
                "curriculum_state": (
                    str(paths.curriculum_state)
                    if task_suite_payload is not None
                    else None
                ),
                "snapshot_metadata": (
                    str(paths.snapshot_metadata)
                    if snapshot_payload is not None
                    else None
                ),
                "resolved_config": str(paths.resolved_config),
                "tensorboard": str(tensorboard_logger.log_dir),
                "env_frames": metrics["env_frames"],
                "global_step": metrics["global_step"],
                "clear_rate": metrics["clear_rate"],
                "death_rate": metrics["death_rate"],
                "exploration": exploration,
            },
            sort_keys=True,
        )
    )
    return 0


def _normalized_algorithm(config: MarioRLConfig) -> str:
    value = str(getattr(config.train, "algorithm", "dqn")).strip().lower()
    if value in {"dqn", "deep_q", "deep_q_network"}:
        return "dqn"
    if value in {"ppo", "actor_critic", "recurrent_actor_critic"}:
        return "ppo"
    return value


def _replay_summary(config: MarioRLConfig, module) -> dict[str, object]:
    if hasattr(module, "replay_payload"):
        return module.replay_payload()
    return {
        "prioritized": bool(config.replay.prioritized),
        "capacity": int(config.replay.capacity),
        "size": None,
        "batch_size": int(config.replay.batch_size),
        "warmup": int(config.replay.warmup),
        "priority_alpha": float(config.replay.priority_alpha),
        "priority_beta": float(config.replay.priority_beta),
        "priority_epsilon": None,
        "priority_updates": 0,
        "max_priority": None,
        "mean_priority": None,
        "last_importance_weight_mean": None,
        "last_priority_mean": None,
        "sample_dtype": str(config.replay.sample_dtype),
        "state_shape": [int(dimension) for dimension in config.replay.state_shape],
        "store_reward_info": bool(config.replay.store_reward_info),
        "training_updates": None,
    }


def _flat_replay_metrics(replay: dict[str, object]) -> dict[str, object]:
    return {
        "replay_prioritized": replay.get("prioritized"),
        "replay_priority_alpha": replay.get("priority_alpha"),
        "replay_priority_beta": replay.get("priority_beta"),
        "replay_priority_epsilon": replay.get("priority_epsilon"),
        "replay_priority_updates": replay.get("priority_updates"),
        "replay_priority_max": replay.get("max_priority"),
        "replay_priority_mean": replay.get("mean_priority"),
        "replay_importance_weight_mean": replay.get("last_importance_weight_mean"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse training config and execute the training command."""
    return cli(
        argv,
        description="Train a Mario RL experiment from a typed config.",
        runner=run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
