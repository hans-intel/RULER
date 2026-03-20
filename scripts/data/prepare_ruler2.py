import argparse
import json
import subprocess
from pathlib import Path
import gzip


TASKS = [
    "mk_niah_basic",
    "mk_niah_easy",
    "mk_niah_medium",
    "mk_niah_hard",
    "mv_niah_basic",
    "mv_niah_easy",
    "mv_niah_medium",
    "mv_niah_hard",
    "qa_basic",
    "qa_easy",
    "qa_medium",
    "qa_hard",
]


def _module_cmd(task, output_folder, tokenizer_type, tokenizer_path, max_seq_length, num_samples):
    common = [
        "--output_folder", output_folder,
        "--tokenizer_type", tokenizer_type,
        "--tokenizer_path", tokenizer_path,
        "--max_seq_length", str(max_seq_length),
        "--num_samples", str(num_samples),
        "--random_seed", "42",
    ]

    if task == "mk_niah_basic":
        return ["python", "-m", "data.ruler2.prepare_niah", *common,
                "--num_needle_k", "1", "--num_needle_v", "1", "--num_needle_q", "1",
                "--type_haystack", "needle", "--type_needle_k", "words", "--type_needle_v", "numbers",
                "--num_digits_v", "10"]
    if task == "mk_niah_easy":
        return ["python", "-m", "data.ruler2.prepare_mmlu", *common,
                "--dataset", "mmlu", "--fewshot", "0", "--prompt_type", "instruct",
                "--num_order", "0", "--task_type", "retrieve", "--algo_type", "single"]
    if task == "mk_niah_medium":
        return ["python", "-m", "data.ruler2.prepare_mmlu", *common,
                "--dataset", "mmlu", "--fewshot", "5", "--prompt_type", "instruct",
                "--num_order", "0", "--task_type", "solve", "--algo_type", "2steps"]
    if task == "mk_niah_hard":
        return ["python", "-m", "data.ruler2.prepare_mmlu", *common,
                "--dataset", "mmlu", "--fewshot", "5", "--prompt_type", "instruct",
                "--num_order", "0", "--task_type", "solve", "--algo_type", "single"]
    if task == "mv_niah_basic":
        return ["python", "-m", "data.ruler2.prepare_niah", *common,
                "--num_needle_k", "1", "--num_needle_v", "4", "--num_needle_q", "1",
                "--type_haystack", "needle", "--type_needle_k", "words", "--type_needle_v", "numbers",
                "--num_digits_v", "10"]
    if task == "mv_niah_easy":
        return ["python", "-m", "data.ruler2.prepare_mmlu", *common,
                "--dataset", "mmlu", "--fewshot", "0", "--prompt_type", "instruct",
                "--num_order", "4", "--task_type", "niah", "--algo_type", "single"]
    if task == "mv_niah_medium":
        return ["python", "-m", "data.ruler2.prepare_mmlu", *common,
                "--dataset", "mmlu", "--fewshot", "0", "--prompt_type", "instruct",
                "--num_order", "4", "--task_type", "retrieve", "--algo_type", "2steps"]
    if task == "mv_niah_hard":
        return ["python", "-m", "data.ruler2.prepare_mmlu", *common,
                "--dataset", "mmlu", "--fewshot", "0", "--prompt_type", "instruct",
                "--num_order", "4", "--task_type", "retrieve", "--algo_type", "single"]
    if task == "qa_basic":
        return ["python", "-m", "data.ruler2.prepare_qa", *common,
                "--dataset", "hotpotqa", "--fewshot", "0", "--prompt_type", "instruct",
                "--task_type", "retrieve", "--query_type", "doc"]
    if task == "qa_easy":
        return ["python", "-m", "data.ruler2.prepare_qa", *common,
                "--dataset", "hotpotqa", "--fewshot", "0", "--prompt_type", "instruct",
                "--task_type", "retrieve", "--query_type", "question"]
    if task == "qa_medium":
        return ["python", "-m", "data.ruler2.prepare_qa", *common,
                "--dataset", "hotpotqa", "--fewshot", "0", "--prompt_type", "instruct",
                "--task_type", "solve", "--algo_type", "2steps"]
    if task == "qa_hard":
        return ["python", "-m", "data.ruler2.prepare_qa", *common,
                "--dataset", "hotpotqa", "--fewshot", "0", "--prompt_type", "instruct",
                "--task_type", "solve", "--algo_type", "single"]
    raise ValueError(f"Unsupported task: {task}")


def _convert_to_ruler_schema(task_dir: Path, subset: str, gzip_output: bool):
    src = task_dir / "test.jsonl"
    dst = task_dir / f"{subset}.jsonl.gz" if gzip_output else task_dir / f"{subset}.jsonl"
    writer = gzip.open if gzip_output else open
    with open(src, "rt", encoding="utf-8") as fin, writer(dst, "wt", encoding="utf-8") as fout:
        for line in fin:
            sample = json.loads(line)
            converted = {
                "index": sample["index"],
                "input": sample["question"],
                "outputs": sample.get("expected_answer", []),
                "length": sample.get("length", -1),
                "others": {},
            }
            fout.write(json.dumps(converted) + "\n")


def _gzip_file_in_place(src: Path):
    if not src.exists() or src.suffix == ".gz":
        return
    dst = src.with_suffix(src.suffix + ".gz")
    with open(src, "rb") as fin, gzip.open(dst, "wb") as fout:
        fout.writelines(fin)
    src.unlink()


def main():
    parser = argparse.ArgumentParser(description="Prepare local RULER2 data in standalone RULER repo.")
    parser.add_argument("--save_dir", type=Path, required=True, help="Output dataset root directory")
    parser.add_argument("--subset", type=str, default="validation", help="Subset filename")
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--tokenizer_type", type=str, default="hf", choices=["hf", "openai", "gemini"])
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument(
        "--gzip_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write prepared subset files as .jsonl.gz (default: true).",
    )
    parser.add_argument(
        "--gzip_test_jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gzip intermediate test.jsonl into test.jsonl.gz and remove test.jsonl (default: true).",
    )
    parser.add_argument(
        "--skip_unfit_tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tasks that cannot fit into max_seq_length instead of failing the whole run.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on first task error (overrides --skip_unfit_tasks behavior).",
    )
    args = parser.parse_args()

    scripts_root = Path(__file__).resolve().parent.parent
    args.save_dir.mkdir(parents=True, exist_ok=True)

    prepared_tasks = []
    skipped_tasks = []

    for task in args.tasks:
        task_dir = args.save_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        cmd = _module_cmd(
            task=task,
            output_folder=str(task_dir),
            tokenizer_type=args.tokenizer_type,
            tokenizer_path=args.tokenizer_path,
            max_seq_length=args.max_seq_length,
            num_samples=args.num_samples,
        )
        print("Running:", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=str(scripts_root),
                capture_output=True,
                text=True,
            )
            _convert_to_ruler_schema(task_dir, args.subset, args.gzip_output)
            if args.gzip_test_jsonl:
                _gzip_file_in_place(task_dir / "test.jsonl")
            prepared_tasks.append(task)
            prepared_name = f"{args.subset}.jsonl.gz" if args.gzip_output else f"{args.subset}.jsonl"
            print(f"Prepared {task} -> {task_dir / prepared_name}")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or ""
            stdout = exc.stdout or ""
            msg = stderr if stderr else stdout

            unfit_error = "Unable to fit sample within max_seq_length" in msg
            if args.strict:
                raise

            if args.skip_unfit_tasks and unfit_error:
                skipped_tasks.append(task)
                print(
                    f"[WARN] Skipping task '{task}' because it does not fit max_seq_length={args.max_seq_length}."
                )
                continue

            raise

    print("\nPreparation summary")
    print("  prepared:", prepared_tasks)
    if skipped_tasks:
        print("  skipped (unfit):", skipped_tasks)


if __name__ == "__main__":
    main()
