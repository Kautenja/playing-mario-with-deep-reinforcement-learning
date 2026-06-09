#!/usr/bin/env bash
set -euo pipefail

command="${1:-}"

case "$command" in
    unittest)
        shift
        python3 -m unittest discover . "$@"
        ;;
    config)
        shift
        python3 -m mario_rl.config "$@"
        ;;
    train)
        shift
        python3 -m mario_rl.train "$@"
        ;;
    play)
        shift
        python3 -m mario_rl.play "$@"
        ;;
    random)
        shift
        python3 -m mario_rl.random "$@"
        ;;
    help|--help|-h)
        python3 -m mario_rl
        ;;
    "")
        python3 -m mario_rl
        ;;
    *)
        echo "unknown command: $command" >&2
        exit 2
        ;;
esac
