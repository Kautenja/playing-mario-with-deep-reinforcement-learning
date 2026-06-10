"""Modern Gymnasium environment tools for Mario RL experiments."""
from .actions import ACTION_SETS, get_action_set
from .config import MarioEnvConfig
from .factory import make_env
from .tasks import (
    MarioTask,
    TaskFeatureEncoder,
    TaskFeatures,
    TaskSuite,
    TaskSuiteConfig,
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
from .wrappers import (
    ClipRewardEnv,
    DefaultSeedEnv,
    DownsampleObservationEnv,
    FrameStackEnv,
    MaxFrameskipEnv,
    OpenCVRecordVideoEnv,
)


__all__ = [
    "ACTION_SETS",
    "ClipRewardEnv",
    "DefaultSeedEnv",
    "DownsampleObservationEnv",
    "FrameStackEnv",
    "MarioEnvConfig",
    "MarioTask",
    "MaxFrameskipEnv",
    "OpenCVRecordVideoEnv",
    "TaskFeatureEncoder",
    "TaskFeatures",
    "TaskSuite",
    "TaskSuiteConfig",
    "UNKNOWN_TASK_VALUE",
    "available_env_ids",
    "available_tasks",
    "choose_stage_env_id",
    "encode_task_features",
    "get_action_set",
    "make_env",
    "smb3_stage_matrix",
    "task_feature_size",
    "task_for_env_id",
    "task_for_env_id_or_none",
]
