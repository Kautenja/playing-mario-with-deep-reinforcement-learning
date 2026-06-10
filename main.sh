#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -x ".venv/bin/python" ]]; then
    python_bin=".venv/bin/python"
else
    python_bin="python3"
fi

command="${1:-}"

case "$command" in
    unittest)
        shift
        "$python_bin" -m unittest discover . "$@"
        ;;
    config)
        shift
        "$python_bin" -m mario_rl.config "$@"
        ;;
    train)
        shift
        "$python_bin" -m mario_rl.train "$@"
        ;;
    play)
        shift
        "$python_bin" -m mario_rl.play "$@"
        ;;
    eval-matrix)
        shift
        "$python_bin" -m mario_rl.eval_matrix "$@"
        ;;
    random)
        shift
        "$python_bin" -m mario_rl.random "$@"
        ;;
    verify-macbook)
        shift
        "$python_bin" -m mario_rl.verify_macbook "$@"
        ;;
    help|--help|-h)
        "$python_bin" -m mario_rl
        ;;
    "")
        "$python_bin" -m mario_rl
        ;;
    *)
        echo "unknown command: $command" >&2
        exit 2
        ;;
esac
