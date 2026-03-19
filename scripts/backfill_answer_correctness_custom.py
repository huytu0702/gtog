from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graphrag.eval.ragas_runner import DEFAULT_API_KEY
from graphrag.eval.ragas_runner import DEFAULT_BASE_URL
from graphrag.eval.ragas_runner import DEFAULT_MAX_RETRIES
from graphrag.eval.ragas_runner import DEFAULT_MODEL
from graphrag.eval.ragas_runner import DEFAULT_TIMEOUT
from graphrag.eval.ragas_runner import SimpleResultRow
from graphrag.eval.ragas_runner import build_ragas_scorers
from graphrag.eval.ragas_runner import score_row


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        msg = f"Expected JSON array in {path}"
        raise TypeError(msg)
    return data


def _build_row(payload: dict[str, Any]) -> SimpleResultRow:
    return SimpleResultRow(
        row_index=int(payload["row_index"]),
        question=str(payload["question"]),
        response=str(payload["response"]),
        ground_truth=str(payload["ground_truth"]),
        context_text=str(payload["context_text"]),
        method=str(payload.get("method") or "unknown"),
        context=str(payload.get("context") or ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill answer_correctness_custom into existing detailed Ragas outputs.",
    )
    parser.add_argument("paths", nargs="+", help="Detailed JSON files to update in place.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args()

    scorers = build_ragas_scorers(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        metric_names=["answer_correctness_custom"],
    )

    for raw_path in args.paths:
        path = Path(raw_path)
        rows = _load_rows(path)
        updated_count = 0

        for row in rows:
            if row.get("status") != "success":
                continue
            scores = row.get("scores")
            if not isinstance(scores, dict):
                continue
            if "answer_correctness_custom" in scores:
                continue

            simple_row = _build_row(row)
            metric_score = score_row(simple_row, scorers)["answer_correctness_custom"]
            scores["answer_correctness_custom"] = metric_score
            updated_count += 1

        with path.open("w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2, ensure_ascii=False)

        print(f"{path}: updated {updated_count} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
