"""MacBook trainability gate tests."""
from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mario_rl.config import load
from mario_rl.verify_macbook import (
    CommandResult,
    GateError,
    PerformanceBudget,
    PerformanceSample,
    build_parser,
    format_device_summary,
    parse_json_payload,
    performance_warnings,
    read_train_metrics,
    run_gate,
    select_devices,
    verify_file_artifacts,
)


class FakeCommandRunner:
    """Create deterministic command results and tiny artifacts for gate tests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args, timeout):
        command = tuple(str(item) for item in args)
        self.calls.append(command)
        module = command[2]
        if module == "unittest":
            return CommandResult(command, 0, "OK\n", "", 0.25)
        if module == "mario_rl.random":
            payload = {
                "command": "random",
                "action_set": "simple",
                "action_count": 7,
                "native_action_space": False,
                "steps": 16,
                "reward": 0.0,
            }
            return CommandResult(command, 0, json.dumps(payload) + "\n", "", 0.5)
        if module == "mario_rl.train":
            payload = self._write_train_payload(command)
            return CommandResult(command, 0, json.dumps(payload) + "\n", "", 2.0)
        if module == "mario_rl.play":
            payload = self._write_eval_payload(command)
            return CommandResult(command, 0, json.dumps(payload) + "\n", "", 1.0)
        return CommandResult(command, 2, "", f"unexpected module: {module}", 0.0)

    def _write_train_payload(self, command: tuple[str, ...]) -> dict[str, object]:
        experiment = _arg_value(command, "--experiment_name")
        save_dir = Path(_arg_value(command, "--save_dir", str(self.root)))
        root = save_dir / experiment
        checkpoints = root / "checkpoints"
        tensorboard = root / "logs" / "tensorboard" / "version_0"
        checkpoints.mkdir(parents=True)
        tensorboard.mkdir(parents=True)
        checkpoint = checkpoints / "macbook-gate.ckpt"
        metrics = root / "train-metrics.csv"
        resolved = root / "resolved-config.yaml"
        event = tensorboard / "events.out.tfevents.fake"
        checkpoint.write_bytes(b"ckpt")
        metrics.write_text(
            "action_set,action_count,native_action_space,global_step,env_frames,"
            "episodes,episode_reward,epsilon,loss,learning_rate\n"
            "simple,7,False,7,32,1,0.0,0.1,0.25,0.00025\n",
            encoding="utf-8",
        )
        resolved.write_text("env:\n  id: SuperMarioBros-1-1-v0\n", encoding="utf-8")
        event.write_text("event", encoding="utf-8")
        return {
            "command": "train",
            "action_set": "simple",
            "action_count": 7,
            "native_action_space": False,
            "checkpoint": str(checkpoint),
            "experiment_dir": str(root),
            "metrics": str(metrics),
            "resolved_config": str(resolved),
            "tensorboard": str(tensorboard),
            "env_frames": 32,
            "global_step": 7,
        }

    def _write_eval_payload(self, command: tuple[str, ...]) -> dict[str, object]:
        experiment = _arg_value(command, "--experiment_name")
        save_dir = Path(_arg_value(command, "--save_dir", str(self.root)))
        root = save_dir / experiment
        metrics = root / "eval-metrics.json"
        payload = {
            "action_set": "simple",
            "action_count": 7,
            "native_action_space": False,
            "checkpoint": str(root / "checkpoints" / "macbook-gate.ckpt"),
            "episode_count": 1,
            "episodes": [{"episode": 0, "reward": 1.0, "steps": 4}],
            "total_reward": 1.0,
            "total_steps": 4,
        }
        metrics.write_text(json.dumps(payload), encoding="utf-8")
        return {"command": "play", "metrics_path": str(metrics), **payload}


class VerifyMacbookCliTest(TestCase):
    """Validate gate CLI parsing and control flow without real training."""

    def test_parser_defaults_to_auto_device_and_gate_config(self):
        args = build_parser().parse_args([])

        self.assertEqual("auto", args.device)
        self.assertEqual("smb_ppo_fast_dev", args.config)
        self.assertFalse(args.benchmark_only)

    def test_select_devices_handles_cpu_mps_and_auto_skip(self):
        self.assertEqual(("cpu",), tuple(item.name for item in select_devices("cpu", lambda: False)))
        self.assertEqual(("mps",), tuple(item.name for item in select_devices("mps", lambda: True)))

        auto = select_devices("auto", lambda: False)
        self.assertEqual("cpu", auto[0].name)
        self.assertTrue(auto[1].skipped)
        self.assertIn("MPS", auto[1].reason)

    def test_gate_runs_unit_train_and_eval_against_fake_runner(self):
        with TemporaryDirectory() as tmpdir:
            runner = FakeCommandRunner(Path(tmpdir))
            output = io.StringIO()

            result = run_gate(
                ["--device", "cpu", "--save-dir", tmpdir],
                command_runner=runner,
                mps_available_fn=lambda: False,
                stdout=output,
            )

            self.assertEqual(0, result)
            self.assertEqual(
                ["unittest", "mario_rl.train", "mario_rl.play"],
                [call[2] for call in runner.calls],
            )
            summary = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual("gate", summary["mode"])
            self.assertEqual("cpu", summary["devices"][0]["device"])
            self.assertTrue(Path(summary["devices"][0]["summary_path"]).is_file())

    def test_benchmark_only_runs_random_and_train_without_units_or_eval(self):
        with TemporaryDirectory() as tmpdir:
            runner = FakeCommandRunner(Path(tmpdir))
            output = io.StringIO()

            result = run_gate(
                ["--benchmark-only", "--device", "cpu", "--save-dir", tmpdir],
                command_runner=runner,
                mps_available_fn=lambda: False,
                stdout=output,
            )

            self.assertEqual(0, result)
            self.assertEqual(["mario_rl.random", "mario_rl.train"], [call[2] for call in runner.calls])
            self.assertIn("random_steps_per_sec", output.getvalue())


class VerifyMacbookArtifactAndMetricsTest(TestCase):
    """Validate artifact checks, metrics parsing, and budget behavior."""

    def test_verify_file_artifacts_requires_nonempty_files(self):
        with TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "artifact.txt"
            artifact.write_text("ok", encoding="utf-8")

            checks = verify_file_artifacts({"artifact": str(artifact)}, ("artifact",))

            self.assertEqual(1, len(checks))
            self.assertEqual("artifact", checks[0].key)
            self.assertGreater(checks[0].size_bytes, 0)

            empty = Path(tmpdir) / "empty.txt"
            empty.touch()
            with self.assertRaises(GateError):
                verify_file_artifacts({"artifact": str(empty)}, ("artifact",))

    def test_train_metrics_parser_coerces_numeric_values(self):
        with TemporaryDirectory() as tmpdir:
            metrics = Path(tmpdir) / "train-metrics.csv"
            metrics.write_text(
                "global_step,env_frames,loss\n7,32,0.5\n",
                encoding="utf-8",
            )

            parsed = read_train_metrics(metrics)

            self.assertEqual(7, parsed["global_step"])
            self.assertEqual(32, parsed["env_frames"])
            self.assertEqual(0.5, parsed["loss"])

    def test_budget_warnings_report_slow_or_empty_training(self):
        sample = PerformanceSample(
            mode="gate",
            device="cpu",
            config="smb_ppo_fast_dev",
            experiment_name="slow",
            env_id="SuperMarioBros-1-1-v0",
            action_set="simple",
            action_count=7,
            train_seconds=10.0,
            eval_seconds=2.0,
            total_seconds=12.0,
            env_frames=1,
            optimizer_steps=0,
            eval_steps=1,
            peak_memory_mib=None,
            python_version="3.x",
            pytorch_version="2.x",
        )

        warnings = performance_warnings(
            sample,
            PerformanceBudget(
                max_train_seconds=1.0,
                max_eval_seconds=1.0,
                max_total_seconds=1.0,
                min_env_fps=1.0,
                min_optimizer_steps_per_second=1.0,
            ),
        )

        self.assertGreaterEqual(len(warnings), 5)
        self.assertTrue(any("optimizer throughput" in item for item in warnings))

    def test_json_payload_parser_ignores_non_json_output(self):
        payload = parse_json_payload(
            "progress\n{\"command\":\"train\",\"global_step\":1}\n",
            expected_command="train",
        )

        self.assertEqual(1, payload["global_step"])


class VerifyMacbookConfigAndSummaryTest(TestCase):
    """Validate the mini config and human summary output."""

    def test_macbook_gate_config_is_bounded_vector_ppo_and_loadable(self):
        config = load("smb_ppo_fast_dev")

        self.assertEqual("SuperMarioBros-1-1-v0", config.env.id)
        self.assertEqual("ppo", config.train.algorithm)
        self.assertEqual("recurrent_actor_critic", config.model.architecture)
        self.assertGreater(config.ppo.num_envs, 1)
        self.assertLessEqual(config.train.max_steps, 2)
        self.assertLessEqual(config.env.max_smoke_steps, 32)
        self.assertEqual((4, 84, 84), config.replay.state_shape)
        self.assertLessEqual(config.ppo.minibatch_size, 4)

    def test_format_device_summary_includes_core_perf_numbers(self):
        summary = {
            "mode": "gate",
            "device": "cpu",
            "summary_path": "runs/example/macbook-gate-summary.json",
            "warnings": [],
            "performance": {
                "train_seconds": 2.0,
                "eval_seconds": 1.0,
                "env_fps": 16.0,
                "optimizer_steps_per_second": 3.5,
                "env_id": "SuperMarioBros-1-1-v0",
                "action_set": "simple",
                "action_count": 7,
            },
        }

        text = format_device_summary(summary)

        self.assertIn("gate cpu", text)
        self.assertIn("env_fps=16.00", text)
        self.assertIn("optimizer_steps_per_sec=3.50", text)
        self.assertIn("actions=simple:7", text)


def _arg_value(command: tuple[str, ...], flag: str, default: str | None = None) -> str:
    if flag not in command:
        if default is None:
            raise AssertionError(f"missing {flag} in {command}")
        return default
    index = command.index(flag)
    return command[index + 1]
