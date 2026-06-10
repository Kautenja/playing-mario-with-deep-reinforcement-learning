"""Local MacBook trainability gate and profiling helpers."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_CONFIG = "smb_dqn_macbook_gate"
DEFAULT_EXPERIMENT_PREFIX = "smb_dqn_macbook_gate"
FAST_UNIT_TESTS = (
    "mario_rl.config.tests.test_config",
    "mario_rl.tests.test_train_cli",
    "mario_rl.tests.test_play_cli",
    "mario_rl.tests.test_verify_macbook",
)


class GateError(RuntimeError):
    """Raised when the MacBook gate cannot verify a required condition."""


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess result with wall-clock timing."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


@dataclass(frozen=True)
class DeviceSelection:
    """One requested device and optional skip reason."""

    name: str
    skipped: bool = False
    reason: str = ""


@dataclass(frozen=True)
class PerformanceBudget:
    """Warning thresholds for the bounded laptop gate."""

    max_train_seconds: float = 180.0
    max_eval_seconds: float = 90.0
    max_total_seconds: float = 300.0
    min_env_fps: float = 0.25
    min_optimizer_steps_per_second: float = 0.02


@dataclass(frozen=True)
class ArtifactCheck:
    """A verified file artifact emitted by train or evaluation commands."""

    key: str
    path: str
    size_bytes: int


@dataclass(frozen=True)
class PerformanceSample:
    """Measured gate throughput for one device run."""

    mode: str
    device: str
    config: str
    experiment_name: str
    env_id: str
    action_set: str
    action_count: int
    train_seconds: float
    eval_seconds: float
    total_seconds: float
    env_frames: int
    optimizer_steps: int
    eval_steps: int
    peak_memory_mib: float | None
    python_version: str
    pytorch_version: str
    random_steps: int | None = None
    random_seconds: float | None = None

    @property
    def env_fps(self) -> float:
        """Environment frames per train wall-clock second."""
        if self.train_seconds <= 0:
            return 0.0
        return float(self.env_frames) / self.train_seconds

    @property
    def optimizer_steps_per_second(self) -> float:
        """Optimizer steps per train wall-clock second."""
        if self.train_seconds <= 0:
            return 0.0
        return float(self.optimizer_steps) / self.train_seconds

    @property
    def random_steps_per_second(self) -> float | None:
        """Random rollout environment steps per second for benchmark mode."""
        if self.random_steps is None or not self.random_seconds or self.random_seconds <= 0:
            return None
        return float(self.random_steps) / self.random_seconds


CommandRunner = Callable[[Sequence[str], float | None], CommandResult]
MpsAvailable = Callable[[], bool]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the MacBook gate."""
    parser = argparse.ArgumentParser(
        prog="python -m mario_rl.verify_macbook",
        description="Run the bounded MacBook trainability gate.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="packaged config or YAML path")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="device to verify; auto runs CPU and MPS only when available",
    )
    parser.add_argument(
        "--experiment-prefix",
        default=DEFAULT_EXPERIMENT_PREFIX,
        help="prefix for generated experiment names",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="override artifact root directory from the config",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="run the env-step and mini-optimization benchmark without unit tests or eval",
    )
    parser.add_argument(
        "--skip-unit-tests",
        action="store_true",
        help="skip the fast unit subset before train/eval",
    )
    parser.add_argument(
        "--strict-budget",
        action="store_true",
        help="turn performance budget warnings into command failures",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per-subprocess timeout in seconds",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable used for child commands",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m mario_rl.verify_macbook``."""
    return run_gate(argv)


def run_gate(
    argv: Sequence[str] | None = None,
    *,
    command_runner: CommandRunner | None = None,
    mps_available_fn: MpsAvailable | None = None,
    stdout=None,
) -> int:
    """Run the MacBook gate, returning a process-style exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    stream = stdout or sys.stdout
    runner = command_runner or run_subprocess
    mps_available = mps_available_fn or is_mps_available

    summary: dict[str, Any] = {
        "command": "verify-macbook",
        "mode": "benchmark" if args.benchmark_only else "gate",
        "config": args.config,
        "requested_device": args.device,
        "devices": [],
        "skips": [],
    }

    try:
        if not args.benchmark_only and not args.skip_unit_tests:
            result = run_unit_tests(
                runner,
                python_executable=args.python_executable,
                timeout=args.timeout,
            )
            summary["unit_tests"] = {
                "elapsed_seconds": result.elapsed_seconds,
                "modules": list(FAST_UNIT_TESTS),
            }

        for selection in select_devices(args.device, mps_available):
            if selection.skipped:
                skip = asdict(selection)
                summary["skips"].append(skip)
                print(f"SKIP {selection.name}: {selection.reason}", file=stream)
                continue
            if args.benchmark_only:
                device_summary = run_benchmark_for_device(args, selection.name, runner)
            else:
                device_summary = run_train_eval_for_device(args, selection.name, runner)
            summary["devices"].append(device_summary)
            print(format_device_summary(device_summary), file=stream)

        print(json.dumps(summary, sort_keys=True), file=stream)
    except GateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def run_unit_tests(
    command_runner: CommandRunner,
    *,
    python_executable: str,
    timeout: float | None,
) -> CommandResult:
    """Run the fast unit subset required before the real train/eval gate."""
    command = [python_executable, "-m", "unittest", *FAST_UNIT_TESTS]
    return run_checked("fast unit tests", command_runner, command, timeout=timeout)


def run_train_eval_for_device(
    args: argparse.Namespace,
    device: str,
    command_runner: CommandRunner,
) -> dict[str, Any]:
    """Run real-environment train and eval for one device."""
    experiment_name = f"{args.experiment_prefix}_{device}"
    train_command = config_command(
        args.python_executable,
        "mario_rl.train",
        args.config,
        device=device,
        experiment_name=experiment_name,
        save_dir=args.save_dir,
    )
    train_result = run_checked(
        f"{device} train",
        command_runner,
        train_command,
        timeout=args.timeout,
    )
    train_payload = parse_json_payload(train_result.stdout, expected_command="train")
    train_artifacts = verify_train_artifacts(train_payload)
    train_metrics = read_train_metrics(Path(train_payload["metrics"]))

    play_command = config_command(
        args.python_executable,
        "mario_rl.play",
        args.config,
        device=device,
        experiment_name=experiment_name,
        save_dir=args.save_dir,
    )
    eval_result = run_checked(
        f"{device} eval",
        command_runner,
        play_command,
        timeout=args.timeout,
    )
    eval_payload = parse_json_payload(eval_result.stdout, expected_command="play")
    eval_artifacts = verify_eval_artifacts(eval_payload)
    eval_metrics = read_json_metrics(Path(eval_payload["metrics_path"]))

    sample = make_performance_sample(
        mode="gate",
        device=device,
        config=args.config,
        experiment_name=experiment_name,
        train_result=train_result,
        eval_result=eval_result,
        train_payload=train_payload,
        eval_payload=eval_payload,
    )
    warnings = performance_warnings(sample, PerformanceBudget())
    if warnings and args.strict_budget:
        raise GateError("performance budget failed:\n" + "\n".join(f"- {item}" for item in warnings))

    summary = {
        "mode": "gate",
        "device": device,
        "experiment_name": experiment_name,
        "train": train_payload,
        "eval": eval_payload,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "artifacts": [asdict(item) for item in (*train_artifacts, *eval_artifacts)],
        "performance": sample_to_dict(sample),
        "budget": asdict(PerformanceBudget()),
        "warnings": warnings,
    }
    summary["summary_path"] = str(write_gate_summary(summary))
    return summary


def run_benchmark_for_device(
    args: argparse.Namespace,
    device: str,
    command_runner: CommandRunner,
) -> dict[str, Any]:
    """Run repeatable env-step and mini-optimization profiling for one device."""
    experiment_name = f"{args.experiment_prefix}_benchmark_{device}"
    random_command = config_command(
        args.python_executable,
        "mario_rl.random",
        args.config,
        device=device,
        experiment_name=experiment_name,
        save_dir=args.save_dir,
    )
    random_result = run_checked(
        f"{device} env-step benchmark",
        command_runner,
        random_command,
        timeout=args.timeout,
    )
    random_payload = parse_json_payload(random_result.stdout, expected_command="random")

    train_command = config_command(
        args.python_executable,
        "mario_rl.train",
        args.config,
        device=device,
        experiment_name=experiment_name,
        save_dir=args.save_dir,
    )
    train_result = run_checked(
        f"{device} optimization benchmark",
        command_runner,
        train_command,
        timeout=args.timeout,
    )
    train_payload = parse_json_payload(train_result.stdout, expected_command="train")
    train_artifacts = verify_train_artifacts(train_payload)
    train_metrics = read_train_metrics(Path(train_payload["metrics"]))

    sample = make_performance_sample(
        mode="benchmark",
        device=device,
        config=args.config,
        experiment_name=experiment_name,
        train_result=train_result,
        eval_result=CommandResult((), 0, "", "", 0.0),
        train_payload=train_payload,
        eval_payload={"total_steps": 0},
        random_result=random_result,
        random_payload=random_payload,
    )
    warnings = performance_warnings(sample, PerformanceBudget())
    if warnings and args.strict_budget:
        raise GateError("performance budget failed:\n" + "\n".join(f"- {item}" for item in warnings))

    summary = {
        "mode": "benchmark",
        "device": device,
        "experiment_name": experiment_name,
        "random": random_payload,
        "train": train_payload,
        "train_metrics": train_metrics,
        "artifacts": [asdict(item) for item in train_artifacts],
        "performance": sample_to_dict(sample),
        "budget": asdict(PerformanceBudget()),
        "warnings": warnings,
    }
    summary["summary_path"] = str(write_gate_summary(summary))
    return summary


def config_command(
    python_executable: str,
    module: str,
    config: str,
    *,
    device: str,
    experiment_name: str,
    save_dir: str | None,
) -> list[str]:
    """Build a config-driven command with stable device and artifact overrides."""
    command = [
        python_executable,
        "-m",
        module,
        "--config",
        config,
        "--experiment_name",
        experiment_name,
        "--train.accelerator",
        device,
        "--trainer.accelerator",
        device,
        "--train.devices",
        "1",
        "--trainer.devices",
        "1",
        "--trainer.enable_progress_bar",
        "false",
    ]
    if save_dir is not None:
        command.extend(["--save_dir", save_dir])
    return command


def run_subprocess(args: Sequence[str], timeout: float | None = None) -> CommandResult:
    """Execute a child command and capture stdout/stderr for JSON parsing."""
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            list(args),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - start
        return CommandResult(
            args=tuple(str(item) for item in args),
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=elapsed,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CommandResult(
            args=tuple(str(item) for item in args),
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            elapsed_seconds=elapsed,
        )


def run_checked(
    phase: str,
    command_runner: CommandRunner,
    args: Sequence[str],
    *,
    timeout: float | None,
) -> CommandResult:
    """Run a command and raise a readable error if it fails."""
    result = command_runner(args, timeout)
    if result.returncode != 0:
        command = " ".join(result.args)
        details = [f"{phase} failed with exit code {result.returncode}: {command}"]
        if result.stdout.strip():
            details.append("stdout:\n" + result.stdout.strip())
        if result.stderr.strip():
            details.append("stderr:\n" + result.stderr.strip())
        raise GateError("\n".join(details))
    return result


def select_devices(requested: str, mps_available_fn: MpsAvailable) -> tuple[DeviceSelection, ...]:
    """Resolve requested device mode into concrete gate runs and skips."""
    if requested == "cpu":
        return (DeviceSelection("cpu"),)
    if requested == "mps":
        if mps_available_fn():
            return (DeviceSelection("mps"),)
        return (DeviceSelection("mps", skipped=True, reason="MPS is not available"),)
    if requested == "auto":
        selections = [DeviceSelection("cpu")]
        if mps_available_fn():
            selections.append(DeviceSelection("mps"))
        else:
            selections.append(DeviceSelection("mps", skipped=True, reason="MPS is not available"))
        return tuple(selections)
    raise ValueError(f"unsupported device mode: {requested!r}")


def is_mps_available() -> bool:
    """Return whether PyTorch reports an available Apple Silicon MPS backend."""
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def parse_json_payload(output: str, *, expected_command: str) -> dict[str, Any]:
    """Parse the last JSON object for the expected command from mixed output."""
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("command") == expected_command:
            return payload
    raise GateError(f"could not find {expected_command!r} JSON payload in command output")


def verify_train_artifacts(payload: dict[str, Any]) -> tuple[ArtifactCheck, ...]:
    """Verify required training files and TensorBoard event output."""
    checks = verify_file_artifacts(payload, ("checkpoint", "metrics", "resolved_config"))
    tensorboard_dir = Path(str(payload.get("tensorboard", ""))).expanduser()
    if not tensorboard_dir.is_dir():
        raise GateError(f"missing tensorboard directory: {tensorboard_dir}")
    if not list(tensorboard_dir.glob("events.out.tfevents.*")):
        raise GateError(f"tensorboard directory has no event files: {tensorboard_dir}")
    return checks


def verify_eval_artifacts(payload: dict[str, Any]) -> tuple[ArtifactCheck, ...]:
    """Verify required evaluation metric output."""
    return verify_file_artifacts(payload, ("metrics_path",))


def verify_file_artifacts(
    payload: dict[str, Any],
    keys: Sequence[str],
) -> tuple[ArtifactCheck, ...]:
    """Verify payload path values are existing nonempty files."""
    checks = []
    for key in keys:
        if key not in payload:
            raise GateError(f"missing artifact key in payload: {key}")
        path = Path(str(payload[key])).expanduser()
        if not path.is_file():
            raise GateError(f"missing artifact for {key}: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise GateError(f"empty artifact for {key}: {path}")
        checks.append(ArtifactCheck(key=key, path=str(path), size_bytes=int(size)))
    return tuple(checks)


def read_train_metrics(path: Path) -> dict[str, Any]:
    """Read the one-row train metrics CSV into typed scalar values."""
    with path.expanduser().open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise GateError(f"train metrics file has no rows: {path}")
    return {key: parse_metric_value(value) for key, value in rows[-1].items()}


def read_json_metrics(path: Path) -> dict[str, Any]:
    """Read a JSON metrics file and require a mapping payload."""
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GateError(f"metrics file must contain a JSON object: {path}")
    return data


def parse_metric_value(value: str) -> int | float | str:
    """Parse numeric CSV values while leaving strings intact."""
    if value == "":
        return value
    try:
        integer = int(value)
    except ValueError:
        pass
    else:
        return integer
    try:
        return float(value)
    except ValueError:
        return value


def make_performance_sample(
    *,
    mode: str,
    device: str,
    config: str,
    experiment_name: str,
    train_result: CommandResult,
    eval_result: CommandResult,
    train_payload: dict[str, Any],
    eval_payload: dict[str, Any],
    random_result: CommandResult | None = None,
    random_payload: dict[str, Any] | None = None,
) -> PerformanceSample:
    """Build the normalized performance sample for summaries and budgets."""
    env_id = configured_env_id(train_payload.get("resolved_config"))
    runtime = runtime_versions()
    return PerformanceSample(
        mode=mode,
        device=device,
        config=config,
        experiment_name=experiment_name,
        env_id=env_id,
        action_set=str(train_payload.get("action_set", "unknown")),
        action_count=int(train_payload.get("action_count", 0)),
        train_seconds=float(train_result.elapsed_seconds),
        eval_seconds=float(eval_result.elapsed_seconds),
        total_seconds=float(train_result.elapsed_seconds + eval_result.elapsed_seconds),
        env_frames=int(train_payload.get("env_frames", 0)),
        optimizer_steps=int(train_payload.get("global_step", 0)),
        eval_steps=int(eval_payload.get("total_steps", 0)),
        peak_memory_mib=peak_memory_mib(),
        python_version=runtime["python"],
        pytorch_version=runtime["pytorch"],
        random_steps=int(random_payload.get("steps", 0)) if random_payload else None,
        random_seconds=float(random_result.elapsed_seconds) if random_result else None,
    )


def configured_env_id(resolved_config: str | None) -> str:
    """Read env.id from the resolved YAML config when available."""
    if not resolved_config:
        return "unknown"
    path = Path(resolved_config).expanduser()
    if not path.is_file():
        return "unknown"
    try:
        import yaml
    except ImportError:
        return "unknown"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return "unknown"
    env = data.get("env", {})
    if not isinstance(env, dict):
        return "unknown"
    return str(env.get("id", "unknown"))


def runtime_versions() -> dict[str, str]:
    """Return Python and PyTorch version strings for the gate summary."""
    try:
        import torch
    except ImportError:
        torch_version = "unavailable"
    else:
        torch_version = str(torch.__version__)
    return {"python": platform.python_version(), "pytorch": torch_version}


def peak_memory_mib() -> float | None:
    """Return process/child peak RSS in MiB when the platform exposes it."""
    try:
        import resource
    except ImportError:
        return None

    usage = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def performance_warnings(
    sample: PerformanceSample,
    budget: PerformanceBudget,
) -> list[str]:
    """Return non-fatal performance budget warnings for one sample."""
    warnings = []
    if sample.train_seconds > budget.max_train_seconds:
        warnings.append(
            f"train took {sample.train_seconds:.2f}s "
            f"(budget {budget.max_train_seconds:.2f}s)"
        )
    if sample.eval_seconds > budget.max_eval_seconds:
        warnings.append(
            f"eval took {sample.eval_seconds:.2f}s "
            f"(budget {budget.max_eval_seconds:.2f}s)"
        )
    if sample.total_seconds > budget.max_total_seconds:
        warnings.append(
            f"total took {sample.total_seconds:.2f}s "
            f"(budget {budget.max_total_seconds:.2f}s)"
        )
    if sample.env_fps < budget.min_env_fps:
        warnings.append(
            f"env throughput {sample.env_fps:.3f} fps "
            f"(budget {budget.min_env_fps:.3f} fps)"
        )
    if sample.optimizer_steps_per_second < budget.min_optimizer_steps_per_second:
        warnings.append(
            f"optimizer throughput {sample.optimizer_steps_per_second:.3f} steps/s "
            f"(budget {budget.min_optimizer_steps_per_second:.3f} steps/s)"
        )
    return warnings


def sample_to_dict(sample: PerformanceSample) -> dict[str, Any]:
    """Convert a sample to JSON-ready values including derived throughput."""
    data = asdict(sample)
    data["env_fps"] = sample.env_fps
    data["optimizer_steps_per_second"] = sample.optimizer_steps_per_second
    if sample.random_steps_per_second is not None:
        data["random_steps_per_second"] = sample.random_steps_per_second
    return data


def format_device_summary(summary: dict[str, Any]) -> str:
    """Return a concise human-readable summary for one device run."""
    performance = summary["performance"]
    parts = [
        f"{summary['mode']} {summary['device']}",
        f"train={performance['train_seconds']:.2f}s",
        f"eval={performance['eval_seconds']:.2f}s",
        f"env_fps={performance['env_fps']:.2f}",
        f"optimizer_steps_per_sec={performance['optimizer_steps_per_second']:.2f}",
        f"env={performance['env_id']}",
        f"actions={performance['action_set']}:{performance['action_count']}",
        f"summary={summary['summary_path']}",
    ]
    if "random_steps_per_second" in performance:
        parts.insert(3, f"random_steps_per_sec={performance['random_steps_per_second']:.2f}")
    if summary.get("warnings"):
        parts.append("warnings=" + str(len(summary["warnings"])))
    return " | ".join(parts)


def write_gate_summary(summary: dict[str, Any]) -> Path:
    """Persist the gate summary next to the experiment artifacts."""
    experiment_dir = Path(str(summary["train"]["experiment_dir"])).expanduser()
    path = experiment_dir / "macbook-gate-summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
