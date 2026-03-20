# RULER2 Local (No Cluster) Guide

This guide runs RULER2 locally in this repository and assumes your model serving endpoint is already available.

## 0) Environment setup

```bash
cd $USER/Workspace/RULER
python -m venv .venv
source .venv/bin/activate
pip install -r docker/requirements.txt
```

If you use an OpenAI-compatible vLLM endpoint that requires auth:

```bash
export VLLM_API_KEY="<your-key-or-EMPTY>"
```

If your machine must go through a proxy to reach the vLLM endpoint:

```bash
export PROXY_URL="http://proxy.mycorp:8080"
```

## 1) Pick paths and settings

RULER2 in this repo assumes an output budget of **16,384 tokens** per request (from `scripts/data/ruler2/constants.py`).
So with `max_model_len=131072`, the raw no-truncation prompt budget is:

- `131072 - 16384 = 114688`

```bash
export RULER_ROOT=$USER/Workspace/RULER
export WORKDIR=$RULER_ROOT/local_runs/qwen3_14b_128k
export DATA_DIR=$WORKDIR/data
export PRED_DIR=$WORKDIR/pred

export TOKENIZER_PATH="openai/gpt-oss-20b"
export TOKENIZER_TYPE="hf"
export MAX_SEQ_LENGTH=131072
export NUM_SAMPLES=100

mkdir -p "$DATA_DIR" "$PRED_DIR"
```

You can also initialize these variables with:

```bash
source ./scripts/init_env.sh --run_name <local_run_name> [options]

# Example:
source ./scripts/init_env.sh \
  --run_name gpt-oss-120b-8k \
  --max_seq_length 8192 \
  --batch_size 64 \
  --api_base_url http://127.0.0.1:8123/v1 \
  --model_name_or_path /model/gpt-oss-120b-mxfp4
```

Defaults from `scripts/init_env.sh`:
- `max_seq_len=131072`
- `num_samples=100`
- `batch_size=768`
- `api_base_url=http://127.0.0.1:8123/v1`
- `max_output_tokens=4096`
- tokenizer auto-detected from `local_run_name` core model name when not provided.

Auto-detection expects model-like names such as `gpt-oss-120b`, `qwen3-30b-a3b-instruct`, `llama-3.3-70b`.
If no match is found, the script exits with an error and asks for explicit tokenizer args.

## 1.1) How `prepare_ruler2.py` depends on model/tokenizer

`prepare_ruler2.py` is benchmark-task specific, not model-weight specific.

- It always uses the same RULER2 task definitions (`mk_niah_*`, `mv_niah_*`, `qa_*`).
- It does **not** download a different benchmark suite per model.
- Generated sample content and packing can still change when you change:
  - `--tokenizer_path`
  - `--tokenizer_type`
  - `--max_seq_length`
  - `--num_samples`
  - selected task list / random seed

In practice, different tokenizers change token counts and thus context packing, so data should be treated as tokenizer-specific.

### Exact table for `max_model_len=131072` (no prompt truncation)

Assuming output budget = `16384`:

| Target context label | Tokens | Set `MAX_SEQ_LENGTH` | Feasible without truncation |
|---|---:|---:|---|
| 1k | 1024 | 1024 | Yes |
| 2k | 2048 | 2048 | Yes |
| 4k | 4096 | 4096 | Yes |
| 8k | 8192 | 8192 | Yes |
| 16k | 16384 | 16384 | Yes |
| 32k | 32768 | 32768 | Yes |
| 64k | 65536 | 65536 | Yes |
| 128k | 131072 | 114688 | No (131072 prompt is not possible with 16384 output budget on 131072 model context) |

Notes:
- `114688` is the exact raw prompt cap from budget math.
- Chat formatting can add overhead, so if you want extra safety margin, use a slightly smaller value.

## 1.2) Recommended folder organization

`WORKDIR` is just an organization convention; it is not required by code.

Recommended patterns:

- **Separate `WORKDIR` per model+length**
  - Example: `$RULER_ROOT/local_runs/qwen3_30b_128k`
  - Example: `$RULER_ROOT/local_runs/llama33_70b_128k`
- **Reuse `DATA_DIR` only if data-generation inputs are identical**
  - Same tokenizer path/type
  - Same max seq length
  - Same task list and sample count
- **Keep `PRED_DIR` separate per model**
  - Avoid mixing predictions and summary files.

Practical template:

```bash
export WORKDIR=$RULER_ROOT/local_runs/<model_alias>_<length>
export DATA_DIR=$WORKDIR/data
export PRED_DIR=$WORKDIR/pred
```

## 1.3) Tokenizer mapping for common models

Use these values for `TOKENIZER_PATH` / `TOKENIZER_TYPE` in this pipeline:

| Model | TOKENIZER_PATH | TOKENIZER_TYPE |
|---|---|---|
| Qwen3-30B-A3B-Instruct | `Qwen/Qwen3-30B-A3B-Instruct` *(fallback often works: `Qwen/Qwen3-30B-A3B`)* | `hf` |
| Qwen3-Next-80B-A3B-Instruct | `Qwen/Qwen3-Next-80B-A3B-Instruct` | `hf` |
| NVIDIA-Nemotron-3-Nano-30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` *(alt naming exists in some setups)* | `hf` |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B` *(if using instruct serving, use `meta-llama/Llama-3.1-8B-Instruct`)* | `hf` |
| Llama-3.3-70B | `meta-llama/Llama-3.3-70B-Instruct` | `hf` |

## 2) Prepare RULER2 data

```bash
cd "$RULER_ROOT/scripts"
python data/prepare_ruler2.py \
  --save_dir "$DATA_DIR" \
  --subset validation \
  --tokenizer_path "$TOKENIZER_PATH" \
  --tokenizer_type "$TOKENIZER_TYPE" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --num_samples "$NUM_SAMPLES"
```

By default, prepared files are now written as `.jsonl.gz`.
Use `--no-gzip_output` if you need plain `.jsonl` files.

To gzip existing plain `.jsonl` files under all run data folders:

```bash
cd "$RULER_ROOT"
find local_runs -type f -path "*/data/*/*.jsonl" -print0 | xargs -0 -r gzip -f
```

Or run the helper wrapper:

```bash
./scripts/prepare_ruler2_data.sh
```

For small context windows (e.g. 4k), some official RULER2 tasks may be infeasible with default settings
(for example, tasks using `fewshot=5`). `prepare_ruler2.py` now skips those unfit tasks by default.
If you want strict/fail-fast behavior, pass `--no-skip_unfit_tasks` (or `--strict`).

This creates task files like:
- `$DATA_DIR/mk_niah_basic/validation.jsonl.gz`
- `$DATA_DIR/qa_hard/validation.jsonl.gz`

## 3) Run inference against external vLLM endpoint

### Option A: OpenAI-compatible Chat Completions (`/v1/chat/completions`)

```bash
cd "$RULER_ROOT/scripts"
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
    --api_base_url "http://127.0.0.1:8000/v1" \
    --proxy_url "$PROXY_URL" \
    --disable_auth_header \
    --model_name_or_path "Qwen/Qwen3-14B" \
    --temperature 0.0 \
    --top_p 1.0 \
    --top_k 32 \
    --batch_size "$BATCH_SIZE" \
    --max_output_tokens 4096
 done
```

  `--disable_auth_header` is enabled by default now. Use `--no-disable_auth_header` only if your endpoint requires a bearer token header.

### Option B: OpenAI-compatible Completions (`/v1/completions`)

Change only:
- `--vllm_endpoint completions`

### Option C: Legacy local RULER vLLM server (`/generate`)

Change only:
- `--vllm_endpoint native`
- omit `--api_base_url`

## 4) Evaluate

```bash
cd "$RULER_ROOT/scripts"
python eval/evaluate.py \
  --data_dir "$PRED_DIR" \
  --benchmark ruler2
```

Outputs:
- `$PRED_DIR/summary.csv`
- `$PRED_DIR/submission.csv`

### 4.1) Consolidate multiple lengths into one number (Avg / wAvg)

If your runs share a common prefix and differ by length suffix (`-4k`, `-8k`, ...), use:

```bash
cd "$RULER_ROOT"
python scripts/eval/aggregate_ruler2_scores.py \
  --run_prefix gpt-oss-20b \
  --lengths 4k,8k,16k,32k,64k,100k,128k
```

This reads `local_runs/<run_prefix>-<length>/pred/summary.csv` and reports:
- `Avg` (mean across context lengths)
- `wAvg (inc)` (weights increase with length)
- `wAvg (dec)` (weights decrease with length)

## 5) Notes on parity with NeMo-Skills RULER2

- Task set matches the 12 RULER2 tasks used in NeMo-Skills:
  - `mk_niah_*`, `mv_niah_*`, `qa_*`
- Data generation scripts are ported from NeMo-Skills `dataset/ruler2`.
- RULER2 scoring includes fuzzy matching with WER-based fallback and task-specific matching behavior.

## 6) Quick troubleshooting

- If dataset download fails (`hotpotqa`, `mmlu`): retry with stable network and verify `datasets` install.
- If endpoint rejects requests: verify `--api_base_url`, model name, and whether it expects chat or completions schema.
- If generations are cut off: increase `tokens_to_generate` for that task in `scripts/data/ruler2/constants.py` or raise `--max_output_tokens`.
- If your vLLM server runs with a very large `max_model_len` (e.g., 128k), set `--max_output_tokens` (recommended start: `4096`) to prevent runaway/garbage generations from consuming long decode time.
- If you see `max_tokens must be at least 1, got -N`: your prompt is too long for server context.
  - Lower `MAX_SEQ_LENGTH` in data preparation.
  - Or pass `--truncate_prompt_tokens <context_limit>` in `pred/call_api.py` (e.g. `32768`, `131072`, etc).
  - `call_api.py` now has bounded retries (`--request_max_retries`, default `3`) and writes empty predictions after repeated failures instead of looping forever.
