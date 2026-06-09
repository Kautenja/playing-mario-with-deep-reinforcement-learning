#!/usr/bin/env bash
set -euo pipefail

command="${1:-}"

case "$command" in
    unittest)
        shift
        python3 -m unittest discover . "$@"
        ;;
    "")
        python3 -m mario_rl
        ;;
    *)
        echo "unknown command: $command" >&2
        exit 2
        ;;
esac
