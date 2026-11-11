#!/usr/bin/env bash
# Start the user-local Ollama server with settings suited to a 4-core / 8-16 GB laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="${OLLAMA_BIN:-.local/ollama/bin/ollama}"
command -v ollama >/dev/null 2>&1 && BIN="${OLLAMA_BIN:-ollama}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_NUM_PARALLEL=1          # one request at a time: CPU inference does not parallelise usefully
export OLLAMA_MAX_LOADED_MODELS=1     # never hold two models in RAM
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-2m}"
exec "$BIN" serve
