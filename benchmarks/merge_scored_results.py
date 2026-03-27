"""
Merge answer + quality_score from *_results_scored.json into *_results.json.

This keeps canonical result files (fp16/8bit/4bit) as single source of truth
for dashboard + MLflow while preserving latest performance metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


DEFAULT_PAIRS = [
    ("fp16_results.json", "fp16_results_scored.json"),
    ("8bit_results.json", "8bit_results_scored.json"),
    ("4bit_results.json", "4bit_results_scored.json"),
]


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_lookup(scored_rows):
    by_qid = {}
    by_question = {}
    for row in scored_rows:
        qid = row.get("question_id")
        question = row.get("question")
        if qid is not None:
            by_qid[qid] = row
        if isinstance(question, str):
            by_question[question] = row
    return by_qid, by_question


def merge_pair(base_file: Path, scored_file: Path, backup: bool) -> tuple[int, int]:
    base_rows = load_json(base_file)
    original_base_rows = json.loads(json.dumps(base_rows))
    scored_rows = load_json(scored_file)

    if not isinstance(base_rows, list) or not isinstance(scored_rows, list):
        raise ValueError(f"Both files must contain JSON lists: {base_file.name}, {scored_file.name}")

    by_qid, by_question = build_lookup(scored_rows)

    merged = 0
    for row in base_rows:
        match = None
        qid = row.get("question_id")
        question = row.get("question")
        if qid is not None and qid in by_qid:
            match = by_qid[qid]
        elif isinstance(question, str):
            match = by_question.get(question)

        if not match:
            continue

        if "answer" in match:
            row["answer"] = match.get("answer")
        row["quality_score"] = match.get("quality_score")
        merged += 1

    if backup:
        backup_path = base_file.with_suffix(base_file.suffix + ".bak")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(original_base_rows, f, indent=2, ensure_ascii=False)

    with open(base_file, "w", encoding="utf-8") as f:
        json.dump(base_rows, f, indent=2, ensure_ascii=False)

    return merged, len(base_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge scored answers into canonical results files")
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS),
        help="Directory containing *_results.json and *_results_scored.json",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup file creation (.bak)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    backup = not args.no_backup
    for base_name, scored_name in DEFAULT_PAIRS:
        base_file = results_dir / base_name
        scored_file = results_dir / scored_name
        merged, total = merge_pair(base_file, scored_file, backup=backup)
        print(f"[merge] {base_name} <- {scored_name}: merged {merged}/{total}")

    print("[merge] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

