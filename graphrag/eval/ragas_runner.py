# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Utilities for evaluating GraphRAG simple results with Ragas."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_BASE_URL = "http://127.0.0.1:8317/v1"
DEFAULT_API_KEY = "proxypal-local"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 5
REQUIRED_FIELDS = ("question", "response", "ground_truth", "context_text")

logger = logging.getLogger(__name__)


class MetricScorer(Protocol):
    """Protocol for row-level metric scorers."""

    name: str

    def score(self, **kwargs: Any) -> Any:
        """Score a single sample."""


@dataclass(slots=True)
class SimpleResultRow:
    """Normalized row from eval_results_simple.json."""

    row_index: int
    question: str
    response: str
    ground_truth: str
    context_text: str
    method: str
    context: str

    def to_sample_payload(self) -> dict[str, Any]:
        """Convert a simple result row into the Ragas single-turn payload."""
        return {
            "user_input": self.question,
            "response": self.response,
            "reference": self.ground_truth,
            "retrieved_contexts": [self.context_text],
        }


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def load_simple_results(input_path: str | Path) -> list[dict[str, Any]]:
    """Load the simple-results JSON file."""
    path = Path(input_path)
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        msg = f"Expected a JSON array in {path}."
        raise TypeError(msg)

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            msg = f"Row {index} is not a JSON object."
            raise TypeError(msg)
        normalized.append(row)
    return normalized


def parse_simple_results(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[SimpleResultRow], list[dict[str, Any]]]:
    """Validate and normalize raw simple-result rows."""
    valid_rows: list[SimpleResultRow] = []
    skipped_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        normalized: dict[str, str] = {}
        missing_fields: list[str] = []

        for field_name in REQUIRED_FIELDS:
            value = _normalize_text(row.get(field_name))
            if value is None:
                missing_fields.append(field_name)
                continue
            normalized[field_name] = value

        if missing_fields:
            skipped_rows.append(
                {
                    "row_index": index,
                    "status": "skipped",
                    "method": _normalize_text(row.get("method")) or "unknown",
                    "question": _normalize_text(row.get("question")) or "",
                    "response": _normalize_text(row.get("response")) or "",
                    "ground_truth": _normalize_text(row.get("ground_truth")) or "",
                    "context": _normalize_text(row.get("context")) or "",
                    "context_text": _normalize_text(row.get("context_text")) or "",
                    "error": (
                        "Missing required fields: " + ", ".join(sorted(missing_fields))
                    ),
                }
            )
            continue

        valid_rows.append(
            SimpleResultRow(
                row_index=index,
                question=normalized["question"],
                response=normalized["response"],
                ground_truth=normalized["ground_truth"],
                context_text=normalized["context_text"],
                method=_normalize_text(row.get("method")) or "unknown",
                context=_normalize_text(row.get("context")) or "",
            )
        )

    return valid_rows, skipped_rows


def build_single_turn_sample(row: SimpleResultRow) -> Any:
    """Create a Ragas SingleTurnSample from a normalized row."""
    from ragas.dataset_schema import SingleTurnSample

    return SingleTurnSample(**row.to_sample_payload())


def build_openai_client(
    *,
    api_key: str,
    base_url: str,
    timeout: float,
    max_retries: int,
) -> Any:
    """Create an OpenAI-compatible client."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )


def build_ragas_scorers(
    *,
    model: str,
    client: Any,
) -> dict[str, MetricScorer]:
    """Create the default set of Ragas scorers."""
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerCorrectness,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    evaluator_llm = llm_factory(model=model, provider="openai", client=client)

    return {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
        "answer_correctness": AnswerCorrectness(
            llm=evaluator_llm,
            weights=[1.0, 0.0],
        ),
    }


def score_row(
    row: SimpleResultRow,
    scorers: dict[str, MetricScorer],
) -> dict[str, float]:
    """Score a single normalized row across all configured metrics."""
    payload = row.to_sample_payload()
    scores: dict[str, float] = {}
    for metric_name, scorer in scorers.items():
        score_kwargs = _build_metric_kwargs(scorer, payload)
        metric_result = scorer.score(**score_kwargs)
        metric_value = getattr(metric_result, "value", metric_result)
        scores[metric_name] = float(metric_value)
    return scores


def _build_metric_kwargs(
    scorer: MetricScorer,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Filter the payload down to only the arguments accepted by a metric."""
    score_target = getattr(scorer, "ascore", scorer.score)
    signature = inspect.signature(score_target)

    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return payload

    accepted_parameters = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {
        field_name: field_value
        for field_name, field_value in payload.items()
        if field_name in accepted_parameters
    }


def aggregate_ragas_results(
    detailed_results: Sequence[dict[str, Any]],
    *,
    model: str,
    base_url: str,
    input_path: str | Path,
) -> dict[str, Any]:
    """Aggregate row-level Ragas outputs by method."""
    by_method: dict[str, dict[str, Any]] = {}
    success_count = 0
    fail_count = 0
    skip_count = 0

    for result in detailed_results:
        method = result.get("method") or "unknown"
        summary = by_method.setdefault(
            method,
            {
                "count": 0,
                "success_count": 0,
                "fail_count": 0,
                "skip_count": 0,
                "_score_sums": {},
            },
        )
        summary["count"] += 1

        status = result["status"]
        if status == "success":
            success_count += 1
            summary["success_count"] += 1
            for metric_name, metric_value in result["scores"].items():
                score_sums = summary["_score_sums"]
                score_sums[metric_name] = score_sums.get(metric_name, 0.0) + float(
                    metric_value
                )
        elif status == "failed":
            fail_count += 1
            summary["fail_count"] += 1
        else:
            skip_count += 1
            summary["skip_count"] += 1

    for summary in by_method.values():
        score_sums = summary.pop("_score_sums")
        success_total = summary["success_count"]
        if success_total:
            for metric_name, total in score_sums.items():
                summary[metric_name] = round(total / success_total, 6)

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_file": str(Path(input_path).resolve()),
            "model": model,
            "base_url": base_url,
        },
        "overall": {
            "count": len(detailed_results),
            "success_count": success_count,
            "fail_count": fail_count,
            "skip_count": skip_count,
        },
        "by_method": by_method,
    }


def run_ragas_evaluation(
    *,
    input_path: str | Path,
    output_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[Path, Path]:
    """Run Ragas evaluation over a simple-results JSON file."""
    input_path = Path(input_path)
    output_path = Path(output_dir) if output_dir is not None else input_path.parent
    output_path.mkdir(parents=True, exist_ok=True)

    raw_rows = load_simple_results(input_path)
    valid_rows, skipped_rows = parse_simple_results(raw_rows)

    logger.info("Loaded %s rows from %s", len(raw_rows), input_path)
    logger.info("Valid rows: %s", len(valid_rows))
    if skipped_rows:
        logger.info("Skipped rows before scoring: %s", len(skipped_rows))

    client = build_openai_client(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    scorers = build_ragas_scorers(model=model, client=client)
    logger.info("Metrics: %s", ", ".join(scorers.keys()))

    detailed_results = list(skipped_rows)
    for row in valid_rows:
        try:
            scores = score_row(row, scorers)
        except Exception as exc:  # noqa: BLE001
            detailed_results.append(
                {
                    "row_index": row.row_index,
                    "status": "failed",
                    "method": row.method,
                    "question": row.question,
                    "response": row.response,
                    "ground_truth": row.ground_truth,
                    "context": row.context,
                    "context_text": row.context_text,
                    "error": str(exc),
                }
            )
            logger.info("[row %s] failed: %s", row.row_index, exc)
            continue

        detailed_results.append(
            {
                "row_index": row.row_index,
                "status": "success",
                "method": row.method,
                "question": row.question,
                "response": row.response,
                "ground_truth": row.ground_truth,
                "context": row.context,
                "context_text": row.context_text,
                "scores": scores,
            }
        )
        logger.info("[row %s] success", row.row_index)

    summary = aggregate_ragas_results(
        detailed_results,
        model=model,
        base_url=base_url,
        input_path=input_path,
    )

    detailed_path = output_path / "eval_results_ragas_detailed.json"
    summary_path = output_path / "eval_results_ragas_summary.json"

    with detailed_path.open("w", encoding="utf-8") as file:
        json.dump(detailed_results, file, indent=2)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    logger.info("Detailed results: %s", detailed_path)
    logger.info("Summary results: %s", summary_path)

    return detailed_path, summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the internal Ragas eval script."""
    parser = argparse.ArgumentParser(
        description="Evaluate GraphRAG eval_results_simple.json with Ragas.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to eval_results_simple.json.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for eval_results_ragas_*.json outputs. Defaults to the input directory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI-compatible model name. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible base URL. Defaults to {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds. Defaults to {DEFAULT_TIMEOUT}.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Retry count for OpenAI-compatible client. Defaults to {DEFAULT_MAX_RETRIES}.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the internal Ragas eval script."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_ragas_evaluation(
        input_path=args.input,
        output_dir=args.output_dir,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
