#!/usr/bin/env bash
# Download the GR00T N1.5 joint SFT checkpoint (target_posttraining/atomic_seen, step 60000).
#
# Uses a throwaway venv (.venv-fetch/) so it never touches the project's conda environments.
# Authentication: the hf CLI reads HF_TOKEN from the environment (optionally loaded from .env)
# or the token cached by `hf auth login` in ~/.cache/huggingface/token.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash setup/fetch_checkpoint.sh [-h|--help]

Downloads robocasa/robocasa365_checkpoints:
  gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000
into checkpoints/sft_joint/ (repo-relative layout preserved), skipping optimizer,
scheduler and RNG state. Re-running resumes or skips files that are already complete.
EOF
}

case "$#" in
  0) ;;
  1)
    case "$1" in
      -h|--help) usage; exit 0 ;;
      *) echo "ERROR: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    ;;
  *) echo "ERROR: unexpected arguments: $*" >&2; usage >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="robocasa/robocasa365_checkpoints"
SUBPATH="gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000"
DEST="$ROOT/checkpoints/sft_joint"
VENV="$ROOT/.venv-fetch"

source "$ROOT/setup/dotenv.sh" && load_dotenv "$ROOT/.env"  # HF_TOKEN if set; unset = anonymous download

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV" || {
    echo "ERROR: could not create venv at $VENV. Check that python3 has the venv module." >&2
    exit 1
  }
fi
"$VENV/bin/pip" install -q -U huggingface_hub || {
  echo "ERROR: pip install of huggingface_hub failed in $VENV. Check network access to PyPI." >&2
  exit 1
}
if [ ! -x "$VENV/bin/hf" ]; then
  echo "ERROR: $VENV/bin/hf not found after install. Check the huggingface_hub version in $VENV." >&2
  exit 1
fi

# Optimizer/scheduler/RNG state is only needed to resume SFT training, not for inference or RL.
# The hf CLI takes one glob per --exclude flag; extra bare words would be parsed as filenames.
"$VENV/bin/hf" download "$REPO" \
  --include "${SUBPATH}/*" \
  --exclude "*optimizer*" \
  --exclude "*scheduler*" \
  --exclude "*rng_state*" \
  --local-dir "$DEST" || {
  echo "ERROR: hf download failed for $REPO. Check network access, the HF token" \
       "(HF_TOKEN or ~/.cache/huggingface/token), and free disk space under $DEST." >&2
  exit 1
}

CKPT="$DEST/$SUBPATH"
if [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint not found at $CKPT. Check the repo path on huggingface.co/$REPO." >&2
  exit 1
fi

missing=0
for required in config.json model.safetensors.index.json experiment_cfg/metadata.json; do
  if [ ! -s "$CKPT/$required" ]; then
    echo "ERROR: missing or empty $CKPT/$required." >&2
    missing=1
  fi
done
if ! compgen -G "$CKPT/*.safetensors" > /dev/null; then
  echo "ERROR: no *.safetensors weight shards in $CKPT." >&2
  missing=1
fi
if [ "$missing" -eq 0 ] && ! "$VENV/bin/python" -I - "$CKPT" <<'PY'
import json, pathlib, sys
checkpoint = pathlib.Path(sys.argv[1])
shards = sorted(set(json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"].values()))
absent = [name for name in shards if not (checkpoint / name).is_file()]
for name in absent:
    print(f"ERROR: shard {name} listed in model.safetensors.index.json is missing.", file=sys.stderr)
sys.exit(1 if absent else 0)
PY
then
  missing=1
fi
if [ "$missing" -ne 0 ]; then
  echo "Check the listing of $SUBPATH on huggingface.co/$REPO and re-run this script." >&2
  exit 1
fi

ls -la "$CKPT"
du -sh "$CKPT"
