#!/usr/bin/env bash
set -euo pipefail

required_vars=(RULER_ROOT DATA_DIR TOKENIZER_PATH TOKENIZER_TYPE MAX_SEQ_LENGTH NUM_SAMPLES)
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing required environment variable: $var"
    echo "Run: source ./scripts/init_env.sh <local_run_name> [max_seq_len] [num_samples] [batch_size] [tokenizer_path] [tokenizer_type]"
    exit 1
  fi
done

SUBSET="${SUBSET:-validation}"

cd "$RULER_ROOT/scripts"

python data/prepare_ruler2.py \
  --save_dir "$DATA_DIR" \
  --subset "$SUBSET" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --tokenizer_type "$TOKENIZER_TYPE" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --num_samples "$NUM_SAMPLES" \
  "$@"
