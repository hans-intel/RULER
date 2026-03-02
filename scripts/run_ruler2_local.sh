
for TASK in mk_niah_basic mk_niah_easy mk_niah_medium mk_niah_hard \
            mv_niah_basic mv_niah_easy mv_niah_medium mv_niah_hard \
            qa_basic qa_easy qa_medium qa_hard; do
  python pred/call_api.py \
    --data_dir "$DATA_DIR" \
    --save_dir "$PRED_DIR" \
    --benchmark ruler2 \
    --task "$TASK" \
    --server_type vllm \
    --vllm_endpoint chat \
    --api_base_url "http://127.0.0.1:8123/v1" \
    --model_name_or_path "/model/gpt-oss-20b-mxfp4" \
    --temperature 0.0 \
    --top_p 1.0 \
    --top_k 32 \
    --batch_size ${BATCH_SIZE}
done
