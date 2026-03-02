#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def length_to_tokens(length: str) -> int:
    value = length.strip().lower()
    if value.endswith("k"):
        return int(value[:-1]) * 1024
    if value.endswith("m"):
        return int(value[:-1]) * 1024 * 1024
    raise ValueError(f"Unsupported length suffix in '{length}'. Use forms like 4k, 100k, 1m.")


def parse_summary_macro_avg(summary_path: Path) -> float:
    df = pd.read_csv(summary_path)
    scores = pd.to_numeric(df.iloc[1, 1:], errors="coerce").dropna()
    if len(scores) == 0:
        raise ValueError(f"No score row found in {summary_path}")
    return float(scores.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate RULER2 summary.csv files into Avg/wAvg metrics.")
    parser.add_argument("--runs_root", type=Path, default=Path("local_runs"), help="Root directory containing local runs.")
    parser.add_argument("--run_prefix", type=str, required=True, help="Run prefix before '-<length>' (e.g., gpt-oss-20b).")
    parser.add_argument(
        "--lengths",
        type=str,
        default="4k,8k,16k,32k,64k,100k,128k",
        help="Comma-separated length suffixes to include (e.g., 4k,8k,16k,32k,64k,128k).",
    )
    parser.add_argument("--pred_subdir", type=str, default="pred", help="Prediction subfolder under each run directory.")
    parser.add_argument("--summary_name", type=str, default="summary.csv", help="Summary CSV filename.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested length run is missing. By default missing lengths are skipped.",
    )
    args = parser.parse_args()

    requested_lengths = [item.strip().lower() for item in args.lengths.split(",") if item.strip()]
    if not requested_lengths:
        raise ValueError("No lengths were provided.")

    rows = []
    missing = []

    for length in requested_lengths:
        run_name = f"{args.run_prefix}-{length}"
        summary_path = args.runs_root / run_name / args.pred_subdir / args.summary_name

        if not summary_path.exists():
            missing.append(str(summary_path))
            continue

        try:
            score = parse_summary_macro_avg(summary_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse {summary_path}: {exc}") from exc

        rows.append(
            {
                "length": length,
                "tokens": length_to_tokens(length),
                "run_name": run_name,
                "score": score,
            }
        )

    if args.strict and missing:
        missing_list = "\n".join(missing)
        raise FileNotFoundError(f"Missing required summary files:\n{missing_list}")

    if not rows:
        raise FileNotFoundError("No summary.csv files found for the requested run prefix/lengths.")

    result_df = pd.DataFrame(rows).sort_values("tokens").reset_index(drop=True)

    n = len(result_df)
    inc_weights = pd.Series(range(1, n + 1), dtype="float64")
    dec_weights = pd.Series(range(n, 0, -1), dtype="float64")

    avg = float(result_df["score"].mean())
    wavg_inc = float((result_df["score"] * inc_weights).sum() / inc_weights.sum())
    wavg_dec = float((result_df["score"] * dec_weights).sum() / dec_weights.sum())

    print("\nPer-length macro scores:")
    printable_df = result_df[["length", "score", "run_name"]].copy()
    printable_df["score"] = printable_df["score"].map(lambda x: round(x, 4))
    print(printable_df.to_string(index=False))

    print("\nConsolidated:")
    print(f"Avg        : {avg:.4f}")
    print(f"wAvg (inc) : {wavg_inc:.4f}")
    print(f"wAvg (dec) : {wavg_dec:.4f}")

    if missing:
        print("\nSkipped missing summaries:")
        for path in missing:
            print(f"- {path}")


if __name__ == "__main__":
    main()
