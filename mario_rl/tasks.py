"""Public task metadata and feature encoding API."""
from mario_rl.envs.tasks import (
    MarioTask,
    TaskFeatureEncoder,
    TaskFeatures,
    UNKNOWN_TASK_VALUE,
    available_env_ids,
    available_tasks,
    choose_stage_env_id,
    encode_task_features,
    smb3_stage_matrix,
    task_feature_size,
    task_for_env_id,
    task_for_env_id_or_none,
)


__all__ = [
    "MarioTask",
    "TaskFeatureEncoder",
    "TaskFeatures",
    "UNKNOWN_TASK_VALUE",
    "available_env_ids",
    "available_tasks",
    "choose_stage_env_id",
    "encode_task_features",
    "smb3_stage_matrix",
    "task_feature_size",
    "task_for_env_id",
    "task_for_env_id_or_none",
]
