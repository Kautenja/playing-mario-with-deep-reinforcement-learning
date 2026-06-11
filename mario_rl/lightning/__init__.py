"""Lightning training and evaluation helpers for Mario RL experiments."""
if __name__ == "lightning":
    import importlib.machinery as _importlib_machinery
    import importlib.util as _importlib_util
    import os as _os
    import sys as _sys

    _current_parent = _os.path.dirname(_os.path.dirname(__file__))
    _search_path = [
        path
        for path in _sys.path
        if _os.path.abspath(path or _os.getcwd()) != _os.path.abspath(_current_parent)
    ]
    _spec = _importlib_machinery.PathFinder.find_spec("lightning", _search_path)
    if _spec is None or _spec.loader is None:  # pragma: no cover - environment fault.
        raise ImportError("could not find external lightning package")
    _module = _importlib_util.module_from_spec(_spec)
    _sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    if hasattr(_module, "__path__"):
        _module.__path__.append(_os.path.dirname(__file__))
    globals().update(_module.__dict__)
else:
    from .artifacts import (
        ExperimentPaths,
        checkpoint_path,
        experiment_paths,
        trainer_accelerator,
        trainer_devices,
        write_json,
        write_resolved_config,
        write_train_metrics,
    )
    from .evaluate import evaluate_checkpoint
    from .module import DQNLightningModule
    from .ppo_module import PPOLightningModule

    __all__ = [
        "DQNLightningModule",
        "ExperimentPaths",
        "checkpoint_path",
        "evaluate_checkpoint",
        "experiment_paths",
        "PPOLightningModule",
        "trainer_accelerator",
        "trainer_devices",
        "write_json",
        "write_resolved_config",
        "write_train_metrics",
    ]
