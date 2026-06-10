"""Config-driven Lightning training entrypoint for the PyTorch port."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace

from .config import MarioRLConfig, action_space_summary, cli, with_resolved_model_num_actions
from .rewards import reward_transform_summary


def run(config: MarioRLConfig, *, env_factory=None) -> int:
    """Run a bounded Lightning DQN training job and write smoke artifacts."""
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

    from mario_rl.lightning import (
        DQNLightningModule,
        experiment_paths,
        trainer_accelerator,
        trainer_devices,
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

    module = DQNLightningModule(config, env_factory=env_factory)
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
        log_every_n_steps=max(1, min(int(config.train.log_interval), int(config.train.max_steps))),
    )
    trainer.fit(module, ckpt_path=config.train.checkpoint_path)
    trainer.save_checkpoint(str(paths.checkpoint))

    metrics = module.metrics_summary()
    action_summary = action_space_summary(config)
    reward_summary = reward_transform_summary(config.reward_transform)
    metrics.update(action_summary)
    metrics.update(reward_summary)
    metrics_payload = {
        "command": "train",
        **action_summary,
        **reward_summary,
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
    metrics["metrics_payload"] = metrics_payload
    write_train_metrics(paths.train_metrics, metrics)
    write_json(paths.train_metrics_json, metrics_payload)
    print(
        json.dumps(
            {
                "command": "train",
                **action_summary,
                **reward_summary,
                "checkpoint": str(paths.checkpoint),
                "experiment_dir": str(paths.root),
                "metrics": str(paths.train_metrics),
                "metrics_json": str(paths.train_metrics_json),
                "resolved_config": str(paths.resolved_config),
                "tensorboard": str(tensorboard_logger.log_dir),
                "env_frames": metrics["env_frames"],
                "global_step": metrics["global_step"],
                "clear_rate": metrics["clear_rate"],
                "death_rate": metrics["death_rate"],
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse training config and execute the training command."""
    return cli(
        argv,
        description="Train a Mario DQN experiment from a typed config.",
        runner=run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
