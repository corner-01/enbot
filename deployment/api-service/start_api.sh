#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ENBOT_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

test -f .env || { echo "Missing $PROJECT_ROOT/.env" >&2; exit 1; }
test -x .venv/bin/python || { echo "Missing $PROJECT_ROOT/.venv/bin/python" >&2; exit 1; }

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
curl -fsS --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null || {
  echo "Ollama is not reachable on 127.0.0.1:11434" >&2
  exit 1
}

if command -v redis-cli >/dev/null; then
  redis-cli ping >/dev/null || echo "WARNING: Redis is unavailable; memory will be process-local" >&2
fi

source .venv/bin/activate
exec python -X utf8 main.py
