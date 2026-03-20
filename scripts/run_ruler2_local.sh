
if [[ -z "$DATA_DIR" || -z "$PRED_DIR" || -z "$BATCH_SIZE" ]]; then
  echo "ERROR: DATA_DIR, PRED_DIR, and BATCH_SIZE must be set (source scripts/init_env.sh first)."
  exit 1
fi

if [[ -z "$API_BASE_URL" ]]; then
  export API_BASE_URL="http://127.0.0.1:8123/v1"
fi

if [[ -z "$MODEL_NAME_OR_PATH" ]]; then
  echo "ERROR: MODEL_NAME_OR_PATH is not set. Pass --model_name_or_path in scripts/init_env.sh or export it."
  exit 1
fi

for TASK in mk_niah_basic mk_niah_easy mk_niah_medium mk_niah_hard \
            mv_niah_basic mv_niah_easy mv_niah_medium mv_niah_hard \
            qa_basic qa_easy qa_medium qa_hard; do
  TASK_FILE="$DATA_DIR/$TASK/validation.jsonl"
  if [[ ! -f "$TASK_FILE" ]]; then
    TASK_FILE="$DATA_DIR/$TASK/validation.jsonl.gz"
  fi
  if [[ ! -f "$TASK_FILE" ]]; then
    echo "[WARN] Missing prepared data file for task=$TASK: $TASK_FILE (skipping task)"
    continue
  fi

  MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
  python pred/call_api.py \
    --data_dir "$DATA_DIR" \
    --save_dir "$PRED_DIR" \
    --benchmark ruler2 \
    --task "$TASK" \
    --server_type vllm \
    --vllm_endpoint chat \
    --api_base_url "$API_BASE_URL" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --temperature 0.0 \
    --top_p 1.0 \
    --top_k 32 \
    --batch_size ${BATCH_SIZE} \
    --max_output_tokens "${MAX_OUTPUT_TOKENS}" \
    --gzip_output
done
