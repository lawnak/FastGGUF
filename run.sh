#!/usr/bin/env sh
# Build (if needed) and run fastgguf on Linux / macOS.
#   ./run.sh chat  model.gguf [ctx]
#   ./run.sh serve model.gguf [model2.gguf ...]      -> http://localhost:8080
set -e
cd "$(dirname "$0")"
[ -x ./fastgguf ] || make fastgguf
mode=${1:-}
[ -n "$mode" ] && shift
case "$mode" in
  chat)  exec ./fastgguf chat "$@" ;;
  serve) exec ./fastgguf serve "$@" "${PORT:-8080}" ;;
  *)     echo "usage: ./run.sh chat model.gguf [ctx] | ./run.sh serve model.gguf [...]"; exit 1 ;;
esac
