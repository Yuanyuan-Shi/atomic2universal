#!/usr/bin/env bash
# Start the official GR00T N1.5 policy server (RoboCasa365 fork, scripts/inference_service.py)
# in the policy env for one checkpoint and port. The server prints
# "Server is ready and listening on tcp://..." once the model is loaded. Stop it with Ctrl-C.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash evaluation/serve_policy.sh --port N [--checkpoint DIR]

Serves DIR (default: 'checkpoint' in configs/eval/baseline.yaml, relative to the project root)
with embodiment new_embodiment, data config panda_omron and 4 denoising steps on tcp://*:N.
Environment override: POLICY_ENV (default a2u-policy).
EOF
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLICY_ENV="${POLICY_ENV:-a2u-policy}"
PORT=""
CHECKPOINT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port|--checkpoint)
      if [ "$#" -lt 2 ]; then echo "ERROR: $1 needs a value" >&2; usage >&2; exit 2; fi
      if [ "$1" = "--port" ]; then PORT="$2"; else CHECKPOINT="$2"; fi
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --port N is required and must be a number, got '$PORT'" >&2; usage >&2; exit 2
fi
if [ -z "$CHECKPOINT" ]; then
  CHECKPOINT="$(sed -n 's/^checkpoint:[[:space:]]*\([^[:space:]#]*\).*/\1/p' "$ROOT/configs/eval/baseline.yaml")"
fi
case "$CHECKPOINT" in /*) ;; *) CHECKPOINT="$ROOT/$CHECKPOINT" ;; esac
if [ ! -f "$CHECKPOINT/config.json" ]; then
  echo "ERROR: no config.json in checkpoint '$CHECKPOINT'. Run setup/fetch_checkpoint.sh or pass --checkpoint." >&2
  exit 1
fi
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [ -z "$CONDA" ]; then
  echo "ERROR: conda not found. Activate conda or set CONDA_EXE." >&2; exit 1
fi

echo "Serving $CHECKPOINT on port $PORT from conda env $POLICY_ENV"
exec "$CONDA" run --no-capture-output -n "$POLICY_ENV" python -u \
  "$ROOT/third_party/Isaac-GR00T/scripts/inference_service.py" --server \
  --model-path "$CHECKPOINT" --embodiment-tag new_embodiment --data-config panda_omron \
  --denoising-steps 4 --port "$PORT"
