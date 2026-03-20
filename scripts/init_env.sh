#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced to export environment variables into your current shell."
  echo "Usage: source ./scripts/init_env.sh <local_run_name> [options]"
  echo "   or: source ./scripts/init_env.sh --run_name <local_run_name> [options]"
  exit 1
fi

show_usage() {
  cat <<'EOF'
Usage:
  source ./scripts/init_env.sh <local_run_name> [options]
  source ./scripts/init_env.sh --run_name <local_run_name> [options]

Options:
  --run_name, -r            Local run name (e.g., gpt-oss-120b-8k)
  --max_seq_length, -l      Total sequence length = input + output (default: auto from run_name + max_output_tokens, else 131072)
  --num_samples, -n         Number of samples (default: 100)
  --batch_size, -b          Batch size (default: 768)
  --tokenizer_path          Tokenizer path/id (optional, auto-infer if omitted)
  --tokenizer_type          Tokenizer type: hf/openai/gemini (optional, auto-infer if omitted)
  --api_base_url            OpenAI-compatible base URL (default: http://127.0.0.1:8123/v1)
  --model_name_or_path      Model id/path for inference (default: tokenizer_path)
  --model                   Alias for --model_name_or_path
  --max_output_tokens       Client-side max output tokens (default: 4096)
  --help, -h                Show this help
EOF
}

LOCAL_RUN_NAME=""
MAX_SEQ_LENGTH_ARG="131072"
MAX_SEQ_LENGTH_SET="0"
NUM_SAMPLES_ARG="100"
BATCH_SIZE_ARG="768"
TOKENIZER_PATH_ARG=""
TOKENIZER_TYPE_ARG=""
API_BASE_URL_ARG="${API_BASE_URL:-http://127.0.0.1:8123/v1}"
MODEL_NAME_OR_PATH_ARG=""
MAX_OUTPUT_TOKENS_ARG="${MAX_OUTPUT_TOKENS:-4096}"

if [[ $# -gt 0 && "$1" != -* ]]; then
  LOCAL_RUN_NAME="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_name|-r)
      LOCAL_RUN_NAME="$2"
      shift 2
      ;;
    --max_seq_length|-l)
      MAX_SEQ_LENGTH_ARG="$2"
      MAX_SEQ_LENGTH_SET="1"
      shift 2
      ;;
    --num_samples|-n)
      NUM_SAMPLES_ARG="$2"
      shift 2
      ;;
    --batch_size|-b)
      BATCH_SIZE_ARG="$2"
      shift 2
      ;;
    --tokenizer_path)
      TOKENIZER_PATH_ARG="$2"
      shift 2
      ;;
    --tokenizer_type)
      TOKENIZER_TYPE_ARG="$2"
      shift 2
      ;;
    --api_base_url)
      API_BASE_URL_ARG="$2"
      shift 2
      ;;
    --model_name_or_path)
      MODEL_NAME_OR_PATH_ARG="$2"
      shift 2
      ;;
    --model)
      MODEL_NAME_OR_PATH_ARG="$2"
      shift 2
      ;;
    --max_output_tokens)
      MAX_OUTPUT_TOKENS_ARG="$2"
      shift 2
      ;;
    --help|-h)
      show_usage
      return 0
      ;;
    *)
      echo "ERROR: Unknown argument '$1'"
      show_usage
      return 1
      ;;
  esac
done

if [[ -z "$LOCAL_RUN_NAME" ]]; then
  echo "ERROR: local_run_name is required."
  show_usage
  return 1
fi

if [[ "$MAX_SEQ_LENGTH_SET" == "0" ]]; then
  CONTEXT_LENGTH_ARG=""
  suffix="$(echo "$LOCAL_RUN_NAME" | tr '[:upper:]' '[:lower:]' | grep -oE '[0-9]+[km]$' || true)"
  if [[ -n "$suffix" ]]; then
    unit="${suffix: -1}"
    value="${suffix%?}"
    if [[ "$unit" == "k" ]]; then
      CONTEXT_LENGTH_ARG="$(( value * 1024 ))"
    elif [[ "$unit" == "m" ]]; then
      CONTEXT_LENGTH_ARG="$(( value * 1024 * 1024 ))"
    fi
    MAX_SEQ_LENGTH_ARG="$(( CONTEXT_LENGTH_ARG + MAX_OUTPUT_TOKENS_ARG ))"
  else
    CONTEXT_LENGTH_ARG="$(( MAX_SEQ_LENGTH_ARG - MAX_OUTPUT_TOKENS_ARG ))"
  fi
else
  CONTEXT_LENGTH_ARG="$(( MAX_SEQ_LENGTH_ARG - MAX_OUTPUT_TOKENS_ARG ))"
fi

if [[ "$CONTEXT_LENGTH_ARG" -lt 1 ]]; then
  echo "ERROR: Invalid length configuration. context_length = max_seq_length - max_output_tokens must be >= 1."
  echo "       current max_seq_length=$MAX_SEQ_LENGTH_ARG, max_output_tokens=$MAX_OUTPUT_TOKENS_ARG"
  return 1
fi

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
    *qwen3-30b*)
      TOKENIZER_PATH_ARG="Qwen/Qwen3-30B-A3B-Instruct-2507"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *qwen3.5-9b*)
      TOKENIZER_PATH_ARG="Qwen/Qwen3.5-9B"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *qwen3.5-35b-a3b*)
      TOKENIZER_PATH_ARG="Qwen/Qwen3.5-35B-A3B"
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
    *granite-4.0-h-small*)
      TOKENIZER_PATH_ARG="ibm-granite/granite-4.0-h-small"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *nemotron-3-nano*)
      TOKENIZER_PATH_ARG="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *kimi-linear*)
      TOKENIZER_PATH_ARG="moonshotai/Kimi-Linear-48B-A3B-Instruct"
      TOKENIZER_TYPE_ARG="hf"
      ;;
    *)
      echo "ERROR: Unable to infer tokenizer from local_run_name='$LOCAL_RUN_NAME' (model key '$MODEL_KEY')."
      echo "Expected names like gpt-oss-120b, qwen3-30b-a3b-instruct, llama-3.3-70b."
      echo "Or pass tokenizer explicitly with --tokenizer_path and --tokenizer_type."
      return 1
      ;;
  esac
fi

if [[ -z "$MODEL_NAME_OR_PATH_ARG" ]]; then
  MODEL_NAME_OR_PATH_ARG="$TOKENIZER_PATH_ARG"
fi

export WORKDIR="$RULER_ROOT/local_runs/$LOCAL_RUN_NAME"
export DATA_DIR="$WORKDIR/data"
export PRED_DIR="$WORKDIR/pred"

export TOKENIZER_PATH="$TOKENIZER_PATH_ARG"
export TOKENIZER_TYPE="$TOKENIZER_TYPE_ARG"
export MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH_ARG"
export NUM_SAMPLES="$NUM_SAMPLES_ARG"
export BATCH_SIZE="$BATCH_SIZE_ARG"
export API_BASE_URL="$API_BASE_URL_ARG"
export MODEL_NAME_OR_PATH="$MODEL_NAME_OR_PATH_ARG"
export MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS_ARG"

mkdir -p "$DATA_DIR" "$PRED_DIR"

echo "Initialized RULER2 local environment:"
echo "  RULER_ROOT=$RULER_ROOT"
echo "  WORKDIR=$WORKDIR"
echo "  DATA_DIR=$DATA_DIR"
echo "  PRED_DIR=$PRED_DIR"
echo "  TOKENIZER_PATH=$TOKENIZER_PATH"
echo "  TOKENIZER_TYPE=$TOKENIZER_TYPE"
echo "  MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH"
echo "  CONTEXT_LENGTH=$CONTEXT_LENGTH_ARG"
echo "  NUM_SAMPLES=$NUM_SAMPLES"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  API_BASE_URL=$API_BASE_URL"
echo "  MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
echo "  MAX_OUTPUT_TOKENS=$MAX_OUTPUT_TOKENS"
