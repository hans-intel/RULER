# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import time
from tqdm import tqdm
from pathlib import Path
import re
import gzip
import shutil

try:
    from data.manifest_utils import read_manifest
except ModuleNotFoundError:
    curr_folder = Path(__file__).resolve().parent
    scripts_root = curr_folder.parent
    if str(scripts_root) not in sys.path:
        sys.path.append(str(scripts_root))
    from data.manifest_utils import read_manifest

SERVER_TYPES = (
    'trtllm',
    'vllm',
    'sglang',
    'openai',
    'gemini',
    'hf',
    'mamba',
)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False,
                    help='Overwrite existing prediction file instead of resuming from existing rows.')
parser.add_argument("--gzip_output", action=argparse.BooleanOptionalAction, default=True,
                    help='Write prediction outputs as .jsonl.gz (default: true).')

# Server
parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--vllm_endpoint", type=str, default='native', choices=['native', 'completions', 'chat'],
                    help='vLLM endpoint mode: native=/generate, completions=/v1/completions, chat=/v1/chat/completions')
parser.add_argument("--api_base_url", type=str, default='',
                    help='Base URL for OpenAI-compatible API. Example: http://127.0.0.1:8000/v1')
parser.add_argument("--api_key_env", type=str, default='VLLM_API_KEY',
                    help='Environment variable that stores API key for OpenAI-compatible endpoint')
parser.add_argument("--disable_auth_header", action=argparse.BooleanOptionalAction, default=True,
                    help='Disable Authorization header for OpenAI-compatible endpoints (default: true).')
parser.add_argument("--proxy_url", type=str, default='',
                    help='Optional HTTP/HTTPS proxy URL for API requests, e.g. http://proxy.mycorp:8080')
parser.add_argument("--truncate_prompt_tokens", type=int, default=0,
                    help='For OpenAI-compatible vLLM endpoints: keep at most this many prompt tokens (0 disables).')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo', 
                    help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')
parser.add_argument("--reasoning_effort", type=str, default='',
                    help='Optional reasoning effort for OpenAI-compatible chat/completions endpoints (e.g., low/medium/high).')

# Inference
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_output_tokens", type=int, default=0,
                    help='Optional hard cap for generation tokens per sample (0 means use task default).')
parser.add_argument("--request_max_retries", type=int, default=1,
                    help='Max retries per batch request before writing empty predictions for that batch.')
parser.add_argument("--request_timeout_seconds", type=float, default=600.0,
                    help='HTTP request timeout in seconds for API calls.')

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(',')))
if args.server_type == 'hf' or args.server_type == 'gemini':
    args.threads = 1


def get_llm(tokens_to_generate):
    if args.server_type == 'trtllm':
        from client_wrappers import TRTLLMClient
        llm = TRTLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            proxy_url=args.proxy_url if args.proxy_url else None,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
            max_attention_window_size=args.sliding_window_size,
        )

    elif args.server_type == 'vllm':
        from client_wrappers import VLLMClient
        llm = VLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            proxy_url=args.proxy_url if args.proxy_url else None,
            model_name=args.model_name_or_path,
            vllm_endpoint=args.vllm_endpoint,
            api_base_url=args.api_base_url if args.api_base_url else None,
            api_key=os.getenv(args.api_key_env, 'EMPTY'),
            disable_auth_header=args.disable_auth_header,
            reasoning_effort=args.reasoning_effort.strip() if args.reasoning_effort else None,
            truncate_prompt_tokens=args.truncate_prompt_tokens,
            request_timeout_seconds=args.request_timeout_seconds,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'sglang':
        from client_wrappers import SGLClient
        llm = SGLClient(
            server_host=args.server_host,
            server_port=args.server_port,
            proxy_url=args.proxy_url if args.proxy_url else None,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )
        
    elif args.server_type == 'openai':
        from client_wrappers import OpenAIClient
        llm = OpenAIClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'gemini':
        from client_wrappers import GeminiClient
        llm = GeminiClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )
        
    elif args.server_type == 'hf':
        from model_wrappers import HuggingFaceModel
        llm = HuggingFaceModel(
            name_or_path=args.model_name_or_path,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )
    
    elif args.server_type == 'mamba':
        from model_wrappers import MambaModel
        # mamba uses its own generation function, do not pass in do_sample
        # https://github.com/state-spaces/mamba/blob/009bec5ee37f586844a3fc89c040a9c1a9d8badf/mamba_ssm/utils/generation.py#L121
        llm = MambaModel(
            name_or_path=args.model_name_or_path,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )
        
    else:
        raise RuntimeError(f'Unsupported server type {args.server_type}')

    return llm


def main():
    start_time = time.time()
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_dir = args.data_dir / args.task
    task_file_plain = task_dir / f'{args.subset}.jsonl'
    task_file_gzip = task_dir / f'{args.subset}.jsonl.gz'
    task_file = task_file_plain if task_file_plain.exists() else task_file_gzip
    
    if args.chunk_amount > 1:
        pred_base = args.save_dir / f'{args.task}-{args.chunk_idx}'
    else:
        pred_base = args.save_dir / f'{args.task}'

    pred_file_plain = pred_base.with_suffix('.jsonl')
    pred_file_gzip = pred_base.with_suffix('.jsonl.gz')

    if args.overwrite:
        if pred_file_plain.exists():
            pred_file_plain.unlink()
        if pred_file_gzip.exists():
            pred_file_gzip.unlink()

    if pred_file_plain.exists():
        working_pred_file = pred_file_plain
    elif pred_file_gzip.exists():
        with gzip.open(pred_file_gzip, 'rt', encoding='utf-8') as fin, open(pred_file_plain, 'wt', encoding='utf-8') as fout:
            shutil.copyfileobj(fin, fout)
        working_pred_file = pred_file_plain
    else:
        working_pred_file = pred_file_plain if args.gzip_output else pred_file_plain

    final_pred_file = pred_file_gzip if args.gzip_output else pred_file_plain
        
    print(f'Predict {args.task} \nfrom {task_file}\nto {final_pred_file}')
    final_pred_file.parent.mkdir(parents=True, exist_ok=True)

    if not task_file.exists():
        print(f"[WARN] Task input file not found for task={args.task}: {task_file}. Skipping task.")
        return

    # Load data
    all_data = read_manifest(task_file)
    existing_pred_count = 0
    if os.path.exists(working_pred_file):
        pred_index = {sample['index'] for sample in read_manifest(working_pred_file)}
        existing_pred_count = len(pred_index)
        data = [sample for sample in all_data if sample['index'] not in pred_index]
    else:
        data = all_data

    if len(data) == 0:
        print('No remaining samples to process.')
        return

    tokens_to_generate = config['tokens_to_generate']
    if args.max_output_tokens > 0:
        tokens_to_generate = min(tokens_to_generate, args.max_output_tokens)
        print(
            f"Using max output tokens cap: min(task_default={config['tokens_to_generate']}, "
            f"client_cap={args.max_output_tokens}) = {tokens_to_generate}"
        )

    # Load api
    llm = get_llm(tokens_to_generate)
    realtime_request_progress = args.server_type == 'vllm' and args.vllm_endpoint in ('completions', 'chat')
    progress_marked_indices = set()
    completed_ok_indices = set()
    completed_failed_indices = set()
    written_output_indices = set()
    writer_state = {'fout': None}
    write_lock = threading.Lock()
    counter_state = {'written_count': 0}

    def update_progress_postfix(inflight_batches=None, elapsed=None):
        ok_count = len(completed_ok_indices)
        failed_count = len(completed_failed_indices)
        pending_count = max(0, len(data) - ok_count - failed_count)
        parts = [f"ok={ok_count}", f"failed={failed_count}", f"pending={pending_count}"]
        if inflight_batches is not None:
            parts.append(f"inflight_batches={inflight_batches}")
        if elapsed is not None:
            parts.append(f"elapsed={elapsed}s")
        progress.set_postfix_str(' '.join(parts))
        progress.refresh()

    def get_output(idx_list, index_list, input_list, outputs_list, others_list, truncation_list, length_list):
        nonlocal llm

        def sanitize_prompt(prompt):
            if prompt is None:
                return ''
            if not isinstance(prompt, str):
                prompt = str(prompt)
            return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', prompt)

        def compact_error(exc, max_chars=400):
            msg = str(exc).replace('\n', ' ')
            if len(msg) > max_chars:
                return msg[:max_chars] + ' ... [truncated]'
            return msg

        def on_request_done(prompt_pos, success, result):
            if not realtime_request_progress:
                return
            if prompt_pos < 0 or prompt_pos >= len(idx_list):
                return
            if success:
                local_idx = idx_list[prompt_pos]
                pred_text, pred_failed = normalize_pred_text(result)
                if pred_failed:
                    return

                output_entry = {
                    'index': index_list[prompt_pos],
                    'pred': pred_text,
                    'input': input_list[prompt_pos],
                    'outputs': outputs_list[prompt_pos],
                    'others': others_list[prompt_pos],
                    'truncation': truncation_list[prompt_pos],
                    'length': length_list[prompt_pos],
                    '_failed': False,
                }
                outputs_parallel[local_idx] = output_entry

                with write_lock:
                    fout = writer_state.get('fout')
                    if fout is not None and local_idx not in written_output_indices:
                        output_row = dict(output_entry)
                        output_row.pop('_failed', None)
                        fout.write(json.dumps(output_row) + '\n')
                        fout.flush()
                        written_output_indices.add(local_idx)
                        counter_state['written_count'] += 1

                mark_sample_result(local_idx, True)

        def mark_sample_result(local_idx, success):
            if not realtime_request_progress:
                return
            with progress_lock:
                if success:
                    completed_ok_indices.add(local_idx)
                    completed_failed_indices.discard(local_idx)
                else:
                    if local_idx not in completed_ok_indices:
                        completed_failed_indices.add(local_idx)
                if local_idx in progress_marked_indices:
                    update_progress_postfix()
                    return
                progress_marked_indices.add(local_idx)
                progress.update(1)
                update_progress_postfix()

        def should_skip_single_fallback(exc):
            msg = str(exc).lower()
            hard_down_signals = [
                "connection refused",
                "failed to establish a new connection",
                "max retries exceeded",
                "name or service not known",
                "temporary failure in name resolution",
            ]
            return any(signal in msg for signal in hard_down_signals)

        def is_failed_pred(pred):
            return isinstance(pred, dict) and pred.get('__failed__', False)

        def retry_positions_concurrently(preds, positions):
            pending = list(positions)
            for _round in range(1, args.request_max_retries + 1):
                if not pending:
                    break

                retry_prompts = [sanitized_input_list[pos] for pos in pending]

                def on_retry_done(retry_pos, success):
                    if success and retry_pos >= 0 and retry_pos < len(pending):
                        mark_sample_result(idx_list[pending[retry_pos]], True)

                retry_results = llm.process_batch(prompts=retry_prompts)
                if not isinstance(retry_results, list):
                    retry_results = [retry_results]

                next_pending = []
                for local_retry_idx, global_pos in enumerate(pending):
                    result = retry_results[local_retry_idx] if local_retry_idx < len(retry_results) else {'__failed__': True, 'text': ''}
                    preds[global_pos] = result
                    if is_failed_pred(result):
                        next_pending.append(global_pos)
                    else:
                        pred_text, pred_failed = normalize_pred_text(result)
                        if pred_failed:
                            next_pending.append(global_pos)
                            continue

                        output_entry = {
                            'index': index_list[global_pos],
                            'pred': pred_text,
                            'input': input_list[global_pos],
                            'outputs': outputs_list[global_pos],
                            'others': others_list[global_pos],
                            'truncation': truncation_list[global_pos],
                            'length': length_list[global_pos],
                            '_failed': False,
                        }
                        local_idx = idx_list[global_pos]
                        outputs_parallel[local_idx] = output_entry

                        with write_lock:
                            fout = writer_state.get('fout')
                            if fout is not None and local_idx not in written_output_indices:
                                output_row = dict(output_entry)
                                output_row.pop('_failed', None)
                                fout.write(json.dumps(output_row) + '\n')
                                fout.flush()
                                written_output_indices.add(local_idx)
                                counter_state['written_count'] += 1

                        mark_sample_result(local_idx, True)

                pending = next_pending
                if pending:
                    time.sleep(0.2)

            for global_pos in pending:
                sample_index = index_list[global_pos]
                sample_local_idx = idx_list[global_pos]
                err = preds[global_pos].get('error', 'retry failed') if isinstance(preds[global_pos], dict) else 'retry failed'
                print(
                    f"[WARN] Concurrent repair failed after {args.request_max_retries} retries "
                    f"for task={args.task}, sample_index={sample_index}, local_idx={sample_local_idx}: "
                    f"{compact_error(err)}"
                )
                preds[global_pos] = {'__failed__': True, 'text': ''}
                mark_sample_result(sample_local_idx, False)

        def normalize_pred_text(pred):
            if isinstance(pred, dict) and pred.get('__failed__', False):
                return '', True

            if pred is None:
                return '', True

            if isinstance(pred, str):
                return pred, False

            if isinstance(pred, dict):
                pred_text = pred.get('text', '')
            else:
                pred_text = getattr(pred, 'text', '')

            if pred_text is None:
                return '', True
            if isinstance(pred_text, str):
                return pred_text, False
            if isinstance(pred_text, list):
                if len(pred_text) == 0:
                    return '', True
                first_item = pred_text[0]
                return ('' if first_item is None else str(first_item)), (first_item is None)
            return str(pred_text), False

        sanitized_input_list = [sanitize_prompt(prompt) for prompt in input_list]

        pred_list = None
        last_exception = None
        for attempt in range(1, args.request_max_retries + 1):
            try:
                pred_list = llm.process_batch(prompts=sanitized_input_list, on_result=on_request_done)
                if not isinstance(pred_list, list):
                    pred_list = [pred_list]

                failed_positions = [i for i, pred in enumerate(pred_list) if is_failed_pred(pred)]
                if len(failed_positions) == len(sanitized_input_list) and len(sanitized_input_list) > 0:
                    first_error = pred_list[failed_positions[0]].get('error', 'all requests in batch failed')
                    raise RuntimeError(first_error)

                retry_positions_concurrently(pred_list, failed_positions)
                break
            except Exception as e:
                last_exception = e
                print(
                    f"[WARN] Batch request failed (attempt {attempt}/{args.request_max_retries}) "
                    f"for task={args.task}, sample_index_range={index_list[0]}..{index_list[-1]}: "
                    f"{compact_error(e)}"
                )
                time.sleep(min(attempt, 5))

        if pred_list is None:
            if last_exception is not None:
                print(
                    f"[ERROR] Batch failed after retries for task={args.task}, "
                    f"sample_index_range={index_list[0]}..{index_list[-1]}: {compact_error(last_exception)}"
                )

            if last_exception is not None and should_skip_single_fallback(last_exception):
                print(
                    f"[INFO] Endpoint appears unavailable; skipping per-sample fallback for "
                    f"task={args.task}, sample_index_range={index_list[0]}..{index_list[-1]}."
                )
                pred_list = [{'__failed__': True, 'text': ''} for _ in sanitized_input_list]
                for local_idx in idx_list:
                    mark_sample_result(local_idx, False)
            else:
                print(
                    f"[INFO] Falling back to per-sample retries for task={args.task}, "
                    f"sample_index_range={index_list[0]}..{index_list[-1]}."
                )
                pred_list = [{'__failed__': True, 'text': ''} for _ in sanitized_input_list]
                retry_positions_concurrently(pred_list, list(range(len(sanitized_input_list))))

        if not isinstance(pred_list, list):
            pred_list = [pred_list]

        if len(pred_list) != len(input_list):
            print(
                f"[WARN] Prediction count mismatch for batch: got {len(pred_list)} predictions "
                f"for {len(input_list)} inputs. Missing entries will be filled with empty predictions."
            )
            if len(pred_list) < len(input_list):
                pred_list = pred_list + ([{'__failed__': True, 'text': ''}] * (len(input_list) - len(pred_list)))
            else:
                pred_list = pred_list[:len(input_list)]

        zipped_iter = zip(
            pred_list,
            idx_list,
            index_list,
            input_list,
            outputs_list,
            others_list,
            truncation_list,
            length_list,
        )

        for pred, idx, index, input, outputs, others, truncation, length in zipped_iter:
            pred_text, pred_failed = normalize_pred_text(pred)

            outputs_parallel[idx] = {
                'index': index,
                'pred': pred_text,
                'input': input,
                'outputs': outputs,
                'others': others,
                'truncation': truncation,
                'length': length,
                '_failed': pred_failed,
            }

    threads = []
    outputs_parallel = [{} for _ in range(len(data))]
    progress_lock = threading.Lock()

    batched_data = []
    batch = []
    for idx, data_point in enumerate(data):
        if idx == 0:
            #print(data_point)
            pass
        data_point['idx'] = idx

        if len(batch) >= args.batch_size:
            batched_data.append(batch)
            batch = []

        batch.append(data_point)

    if len(batch):
        batched_data.append(batch)

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(working_pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        writer_state['fout'] = fout
        # the data is processed sequentially, so we can store the start and end of current processing window
        start_idx = 0  # window: [start_idx, end_idx]
        failed_skipped = 0
        written_count = counter_state['written_count']

        progress = tqdm(total=len(data), desc=args.task, unit='sample')
        if realtime_request_progress:
            with progress_lock:
                update_progress_postfix(inflight_batches=0, elapsed=0)
        for batch_idx, batch in enumerate(batched_data):
            idx_list = [data_point['idx'] for data_point in batch]
            end_idx = idx_list[-1]  # the data in a batch is ordered

            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    idx_list=idx_list,
                    index_list=[data_point['index'] for data_point in batch],
                    input_list=[data_point.get('input', data_point.get('question', '')) for data_point in batch],
                    outputs_list=[
                        data_point.get('outputs', data_point.get('expected_answer', [data_point.get('output', '')]))
                        for data_point in batch
                    ],
                    others_list=[data_point.get('others', {}) for data_point in batch],
                    truncation_list=[data_point.get('truncation', -1) for data_point in batch],
                    length_list=[data_point.get('length', -1) for data_point in batch],
                ),
            )
            thread.start()
            threads.append(thread)

            is_last_batch = (batch_idx == len(batched_data) - 1)

            if (len(threads) == args.threads) or is_last_batch:
                wait_start = time.time()
                while True:
                    alive_threads = [thread for thread in threads if thread.is_alive()]
                    if len(alive_threads) == 0:
                        break
                    elapsed = int(time.time() - wait_start)
                    if realtime_request_progress:
                        with progress_lock:
                            update_progress_postfix(inflight_batches=len(alive_threads), elapsed=elapsed)
                    else:
                        progress.set_postfix_str(
                            f"inflight_batches={len(alive_threads)} elapsed={elapsed}s"
                        )
                        progress.refresh()
                    time.sleep(0.5)

                for thread in threads:
                    thread.join()
                threads = []
                if realtime_request_progress:
                    with progress_lock:
                        update_progress_postfix(inflight_batches=0, elapsed=int(time.time() - wait_start))
                else:
                    progress.set_postfix_str("")

                # dump the results in current processing window on disk
                for idx in range(start_idx, end_idx + 1):
                    if len(outputs_parallel[idx]) > 0:
                        if outputs_parallel[idx].get('_failed', False):
                            failed_skipped += 1
                            continue
                        if idx in written_output_indices:
                            continue
                        output_row = dict(outputs_parallel[idx])
                        output_row.pop('_failed', None)
                        fout.write(json.dumps(output_row) + '\n')
                        written_count += 1
                        written_output_indices.add(idx)

                counter_state['written_count'] = written_count

                if not realtime_request_progress:
                    progress.update(end_idx - start_idx + 1)
                start_idx = end_idx + 1

        progress.close()
        writer_state['fout'] = None

    if args.gzip_output:
        with open(pred_file_plain, 'rb') as fin, gzip.open(pred_file_gzip, 'wb') as fout:
            shutil.copyfileobj(fin, fout)
        pred_file_plain.unlink()

    if failed_skipped > 0:
        print(f"[INFO] Skipped writing {failed_skipped} failed samples to {final_pred_file}. They will be retried on the next run.")

    attempted_count = len(data)
    total_samples = len(all_data)
    final_saved = existing_pred_count + written_count
    remaining_count = max(0, total_samples - final_saved)
    print(
        f"[SUMMARY] task={args.task} existing={existing_pred_count} attempted={attempted_count} "
        f"written={written_count} failed={failed_skipped} saved_total={final_saved}/{total_samples} "
        f"remaining_to_retry={remaining_count}"
    )

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == '__main__':
    main()
