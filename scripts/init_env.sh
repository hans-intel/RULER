#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced to export environment variables into your current shell."
  echo "Usage: source ./scripts/init_env.sh <local_run_name> [max_seq_len] [num_samples] [batch_size] [tokenizer_path] [tokenizer_type]"
  exit 1
fi

if [[ $# -lt 1 || $# -gt 6 ]]; then
  echo "Usage: source ./scripts/init_env.sh <local_run_name> [max_seq_len] [num_samples] [batch_size] [tokenizer_path] [tokenizer_type]"
  return 1
fi

LOCAL_RUN_NAME="$1"
MAX_SEQ_LENGTH_ARG="${2:-131072}"
NUM_SAMPLES_ARG="${3:-100}"
BATCH_SIZE_ARG="${4:-768}"
TOKENIZER_PATH_ARG="${5:-}"
TOKENIZER_TYPE_ARG="${6:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RULER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_KEY="$(echo "$LOCAL_RUN_NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
MODEL_KEY="$(echo "$MODEL_KEY" | sed -E 's/-([0-9]+k|[0-9]+m)$//')"

if [[ -z "$TOKENIZER_PATH_ARG" || -z "$TOKENIZER_TYPE_ARG" ]]; then
  case "$MODEL_KEY" in
    *gpt-oss-20b*)
      TOKENIZER_PATH_ARG="openai/gpt-oss-20b"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *gpt-oss-120b*)
      TOKENIZER_PATH_ARG="openai/gpt-oss-120b"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *qwen3-next-80b-a3b-instruct*)
      TOKENIZER_PATH_ARG="Qwen/Qwen3-Next-80B-A3B-Instruct"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *qwen3-30b-a3b-instruct*|*qwen3-30b-a3b*)
      TOKENIZER_PATH_ARG="Qwen/Qwen3-30B-A3B-Instruct"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *nvidia-nemotron-3-nano-30b-a3b*|*nemotron-3-nano-30b-a3b*)
      TOKENIZER_PATH_ARG="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *llama-3.1-8b*)
      TOKENIZER_PATH_ARG="meta-llama/Llama-3.1-8B-Instruct"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *llama-3.3-70b*)
      TOKENIZER_PATH_ARG="meta-llama/Llama-3.3-70B-Instruct"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *)
      echo "ERROR: Unable to infer tokenizer from local_run_name='$LOCAL_RUN_NAME' (model key '$MODEL_KEY')."
      echo "Expected names like gpt-oss-120b, qwen3-30b-a3b-instruct, llama-3.3-70b."
      echo "Or pass tokenizer explicitly: source ./scripts/init_env.sh <local_run_name> [max_seq_len] [num_samples] [batch_size] <tokenizer_path> <tokenizer_type>"
      return 1
      ;;
  esac
fi

export WORKDIR="$RULER_ROOT/local_runs/$LOCAL_RUN_NAME"
export DATA_DIR="$WORKDIR/data"
export PRED_DIR="$WORKDIR/pred"

export TOKENIZER_PATH="$TOKENIZER_PATH_ARG"
export TOKENIZER_TYPE="$TOKENIZER_TYPE_ARG"
export MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH_ARG"
export NUM_SAMPLES="$NUM_SAMPLES_ARG"
export BATCH_SIZE="$BATCH_SIZE_ARG"

mkdir -p "$DATA_DIR" "$PRED_DIR"

echo "Initialized RULER2 local environment:"
echo "  RULER_ROOT=$RULER_ROOT"
echo "  WORKDIR=$WORKDIR"
echo "  DATA_DIR=$DATA_DIR"
echo "  PRED_DIR=$PRED_DIR"
echo "  TOKENIZER_PATH=$TOKENIZER_PATH"
echo "  TOKENIZER_TYPE=$TOKENIZER_TYPE"
echo "  MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH"
echo "  NUM_SAMPLES=$NUM_SAMPLES"
echo "  BATCH_SIZE=$BATCH_SIZE"
