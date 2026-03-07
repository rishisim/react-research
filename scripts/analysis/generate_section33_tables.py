#!/usr/bin/env python3
"""Generate Section 3.3 tables from experiment artifacts.

Tables generated:
1) Main Results (per benchmark)
2) HotPotQA Relative Deltas vs ReAct
3) FEVER Relative Deltas vs ReAct
A1) Token distribution stats (optional appendix support)

Optional:
4) Budgeted coverage/accuracy table when --budgets is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


FRAMEWORKS: List[str] = [
    "react",
    "reflexion",
    "self_reflection",
    "majority_voting",
    "cot_sc",
]

BENCHMARKS: List[str] = ["hotpotqa", "fever"]

DISPLAY_NAMES: Dict[str, str] = {
    "react": "ReAct",
    "reflexion": "Reflexion",
    "self_reflection": "TCAR",
    "majority_voting": "Majority Voting",
    "cot_sc": "CoT-SC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results"),
        help="Root results directory (default: results)",
    )
    parser.add_argument(
        "--seed-dir",
        default="seed42_gemini-2.5-flash",
        help="Seed/model directory under each method (default: seed42_gemini-2.5-flash)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional markdown file output path",
    )
    parser.add_argument(
        "--out-csv-dir",
        type=Path,
        default=None,
        help="Optional directory to emit csv tables",
    )
    parser.add_argument(
        "--budgets",
        default="",
        help="Optional comma-separated token budgets for Table 4 (e.g. 100000,250000,500000)",
    )
    return parser.parse_args()


def _coerce_em(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes"}:
            return 1
        if v in {"0", "false", "no", ""}:
            return 0
        try:
            return int(float(v) != 0.0)
        except ValueError:
            return 0
    return 0


def _coerce_tokens(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return 0
        try:
            return int(float(s))
        except ValueError:
            return 0
    return 0


def _load_json_pairs(path: Path) -> List[Tuple[int, int]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")

    pairs: List[Tuple[int, int]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        em = _coerce_em(row.get("em"))
        tokens = _coerce_tokens(row.get("total_tokens"))
        pairs.append((em, tokens))
    return pairs


def _load_csv_pairs(path: Path) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            em = _coerce_em(row.get("em"))
            tokens = _coerce_tokens(row.get("total_tokens"))
            pairs.append((em, tokens))
    return pairs


def load_pairs(base_dir: Path, benchmark: str, framework: str, seed_dir: str) -> List[Tuple[int, int]]:
    base = base_dir / benchmark / framework / seed_dir

    json_name = f"{framework}.json"
    csv_candidates = ["react.csv", "pareto_table.csv"]

    json_path = base / json_name
    if json_path.exists():
        return _load_json_pairs(json_path)

    for csv_name in csv_candidates:
        csv_path = base / csv_name
        if csv_path.exists():
            return _load_csv_pairs(csv_path)

    raise FileNotFoundError(f"No result file found for {benchmark}/{framework} under {base}")


def percentile(values: Sequence[int], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return float(values[0])
    if p >= 100:
        return float(values[-1])
    n = len(values)
    idx = (p / 100.0) * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(values[lo])
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return num / den


def metric_dict(pairs: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    n_eval = len(pairs)
    n_correct = sum(em for em, _ in pairs)
    tokens_total = sum(tokens for _, tokens in pairs)
    acc = safe_div(n_correct, n_eval)
    tpt = safe_div(tokens_total, n_eval)
    # Spec-defined Tokens/Success uses TOTAL tokens / #correct.
    tps = safe_div(tokens_total, n_correct)

    tokens_sorted = sorted(tokens for _, tokens in pairs)
    return {
        "N_eval": float(n_eval),
        "N_correct": float(n_correct),
        "Accuracy": acc,
        "tokens_total": float(tokens_total),
        "Tokens/Task": tpt,
        "Tokens/Success": tps,
        "median_tokens": percentile(tokens_sorted, 50),
        "p90_tokens": percentile(tokens_sorted, 90),
        "p95_tokens": percentile(tokens_sorted, 95),
        "p99_tokens": percentile(tokens_sorted, 99),
        "max_tokens": float(tokens_sorted[-1]) if tokens_sorted else float("nan"),
    }


def fmt_float(x: float, ndigits: int = 2) -> str:
    if math.isnan(x):
        return "NA"
    return f"{x:.{ndigits}f}"


def fmt_int(x: float) -> str:
    if math.isnan(x):
        return "NA"
    return str(int(round(x)))


def fmt_int_commas(x: float) -> str:
    if math.isnan(x):
        return "NA"
    return f"{int(round(x)):,}"


def fmt_acc_with_fraction(acc: float, n_correct: float, n_eval: float) -> str:
    if math.isnan(acc):
        return "NA"
    return f"{acc * 100.0:.2f}% ({fmt_int(n_correct)}/{fmt_int(n_eval)})"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def table1_rows(metrics: Dict[str, Dict[str, float]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for fw in FRAMEWORKS:
        m = metrics[fw]
        rows.append(
            [
                DISPLAY_NAMES[fw],
                fmt_acc_with_fraction(m["Accuracy"], m["N_correct"], m["N_eval"]),
                fmt_int_commas(m["tokens_total"]),
                fmt_float(m["Tokens/Task"], 2),
                fmt_float(m["Tokens/Success"], 2),
            ]
        )
    return rows


def delta_rows(metrics: Dict[str, Dict[str, float]]) -> List[List[str]]:
    react = metrics["react"]
    rows: List[List[str]] = []
    for fw in FRAMEWORKS:
        if fw == "react":
            continue
        m = metrics[fw]
        gain_pp = (m["Accuracy"] - react["Accuracy"]) * 100.0
        dtpt = m["Tokens/Task"] - react["Tokens/Task"]
        dtps = m["Tokens/Success"] - react["Tokens/Success"]
        tpt_x = safe_div(m["Tokens/Task"], react["Tokens/Task"])

        rows.append(
            [
                DISPLAY_NAMES[fw],
                fmt_acc_with_fraction(m["Accuracy"], m["N_correct"], m["N_eval"]),
                fmt_float(gain_pp, 2),
                fmt_float(m["Tokens/Task"], 2),
                fmt_float(dtpt, 2),
                fmt_float(m["Tokens/Success"], 2),
                fmt_float(dtps, 2),
                fmt_float(tpt_x, 3),
            ]
        )
    return rows


def appendix_rows(metrics: Dict[str, Dict[str, float]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for fw in FRAMEWORKS:
        m = metrics[fw]
        rows.append(
            [
                DISPLAY_NAMES[fw],
                fmt_int(m["median_tokens"]),
                fmt_int(m["p90_tokens"]),
                fmt_int(m["p95_tokens"]),
                fmt_int(m["p99_tokens"]),
                fmt_int(m["max_tokens"]),
            ]
        )
    return rows


def table4_rows(
    pairs_by_framework: Dict[str, Sequence[Tuple[int, int]]],
    budgets: Sequence[int],
) -> List[List[str]]:
    rows: List[List[str]] = []
    for fw in FRAMEWORKS:
        pairs = list(pairs_by_framework[fw])
        pairs.sort(key=lambda x: x[1])
        for b in budgets:
            cum = 0
            prefix: List[Tuple[int, int]] = []
            for em, t in pairs:
                if cum + t > b:
                    break
                prefix.append((em, t))
                cum += t
            m = len(prefix)
            n_correct = sum(em for em, _ in prefix)
            acc = safe_div(n_correct, m) if m > 0 else float("nan")
            rows.append(
                [
                    DISPLAY_NAMES[fw],
                    str(b),
                    str(m),
                    f"{acc * 100.0:.2f}% ({n_correct}/{m})" if not math.isnan(acc) else "NA",
                ]
            )
    return rows


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    budgets: List[int] = []
    if args.budgets.strip():
        budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    benchmark_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    benchmark_pairs: Dict[str, Dict[str, Sequence[Tuple[int, int]]]] = {}

    for benchmark in BENCHMARKS:
        metrics_for_benchmark: Dict[str, Dict[str, float]] = {}
        pairs_for_benchmark: Dict[str, Sequence[Tuple[int, int]]] = {}
        for fw in FRAMEWORKS:
            pairs = load_pairs(args.base_dir, benchmark, fw, args.seed_dir)
            pairs_for_benchmark[fw] = pairs
            metrics_for_benchmark[fw] = metric_dict(pairs)
        benchmark_metrics[benchmark] = metrics_for_benchmark
        benchmark_pairs[benchmark] = pairs_for_benchmark

    lines: List[str] = []

    # Table 1
    headers_t1 = ["Method", "Accuracy (Correct/Eval)", "Total Tokens", "Tokens/Task", "Tokens/Success"]
    for benchmark in BENCHMARKS:
        lines.append(f"## Table 1 — Main Results ({benchmark.upper()})")
        rows = table1_rows(benchmark_metrics[benchmark])
        lines.append(markdown_table(headers_t1, rows))
        lines.append("")

    # Table 2 and Table 3
    headers_delta = [
        "Method",
        "Accuracy (Correct/Eval)",
        "Gain vs ReAct (pp)",
        "Tokens/Task",
        "Delta Tokens/Task",
        "Tokens/Success",
        "Delta Tokens/Success",
        "Tokens/Task (x ReAct)",
    ]

    lines.append("## Table 2 — HotPotQA Relative Deltas vs ReAct")
    rows_t2 = delta_rows(benchmark_metrics["hotpotqa"])
    lines.append(markdown_table(headers_delta, rows_t2))
    lines.append("")

    lines.append("## Table 3 — FEVER Relative Deltas vs ReAct")
    rows_t3 = delta_rows(benchmark_metrics["fever"])
    lines.append(markdown_table(headers_delta, rows_t3))
    lines.append("")

    # Appendix A1
    headers_a1 = ["Method", "median tokens", "p90", "p95", "p99", "max"]
    for benchmark in BENCHMARKS:
        lines.append(f"## Appendix Table A1 — Token Distribution Stats ({benchmark.upper()})")
        rows = appendix_rows(benchmark_metrics[benchmark])
        lines.append(markdown_table(headers_a1, rows))
        lines.append("")

    # Optional table 4
    if budgets:
        headers_t4 = ["Method", "Budget B", "coverage m(B)", "accuracy (correct/covered)"]
        for benchmark in BENCHMARKS:
            lines.append(f"## Table 4 — Budgeted Coverage/Accuracy ({benchmark.upper()})")
            rows = table4_rows(benchmark_pairs[benchmark], budgets)
            lines.append(markdown_table(headers_t4, rows))
            lines.append("")

    output = "\n".join(lines).strip() + "\n"
    print(output)

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(output, encoding="utf-8")

    if args.out_csv_dir is not None:
        args.out_csv_dir.mkdir(parents=True, exist_ok=True)

        t1_hotpot = table1_rows(benchmark_metrics["hotpotqa"])
        t1_fever = table1_rows(benchmark_metrics["fever"])
        write_csv(args.out_csv_dir / "table1_hotpotqa.csv", headers_t1, t1_hotpot)
        write_csv(args.out_csv_dir / "table1_fever.csv", headers_t1, t1_fever)

        write_csv(args.out_csv_dir / "table2_hotpotqa_deltas.csv", headers_delta, rows_t2)
        write_csv(args.out_csv_dir / "table3_fever_deltas.csv", headers_delta, rows_t3)

        a1_hotpot = appendix_rows(benchmark_metrics["hotpotqa"])
        a1_fever = appendix_rows(benchmark_metrics["fever"])
        write_csv(args.out_csv_dir / "appendix_a1_hotpotqa.csv", headers_a1, a1_hotpot)
        write_csv(args.out_csv_dir / "appendix_a1_fever.csv", headers_a1, a1_fever)

        if budgets:
            headers_t4 = ["Method", "Budget B", "coverage m(B)", "accuracy (correct/covered)"]
            for benchmark in BENCHMARKS:
                rows = table4_rows(benchmark_pairs[benchmark], budgets)
                write_csv(args.out_csv_dir / f"table4_{benchmark}_budget.csv", headers_t4, rows)


if __name__ == "__main__":
    main()
