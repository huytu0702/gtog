"""CLI implementation of the eval subcommand."""

import asyncio
import json
import sys
from pathlib import Path

from graphrag.config.load_config import load_config
from graphrag.config.models.eval_config import EvalConfig
from graphrag.eval import (
    LLMJudge,
    EvaluationRunner,
    aggregate_results,
)
from graphrag.language_model.manager import ModelManager

# ruff: noqa: T201


def eval_cli(
    root_dir: Path,
    eval_config: Path | None,
    methods: str | None,
    imdb_key: str | None,
    resume: bool,
    verbose: bool,
):
    """Run evaluation on GraphRAG indexes.

    Compares search methods (ToG, Local, Basic) using LLM-as-Judge metrics.
    """
    root = root_dir.resolve()

    # Load eval config
    if eval_config:
        config_path = root / eval_config if not eval_config.is_absolute() else eval_config
        eval_cfg = EvalConfig.from_yaml(str(config_path))
    else:
        # Look for default config
        default_config = root / "eval_config.yaml"
        if default_config.exists():
            eval_cfg = EvalConfig.from_yaml(str(default_config))
        else:
            print("Error: No eval_config.yaml found. Create one or specify --config.")
            sys.exit(1)

    # Override methods if specified
    if methods:
        eval_cfg.methods = methods.split(",")

    # Filter to single movie if specified
    if imdb_key:
        if imdb_key not in eval_cfg.indexes:
            print(f"Error: imdb_key '{imdb_key}' not found in config indexes.")
            sys.exit(1)
        eval_cfg.indexes = {imdb_key: eval_cfg.indexes[imdb_key]}

    # Run evaluation
    asyncio.run(run_evaluation(root, eval_cfg, resume, verbose))


async def run_evaluation(
    root: Path,
    eval_cfg: EvalConfig,
    resume: bool,
    verbose: bool,
):
    """Execute the evaluation run."""
    # Load main graphrag config for LLM settings
    graphrag_config = load_config(root)

    # Create LLM judge using the default chat model
    model_settings = graphrag_config.get_language_model_config("default_chat_model")
    judge_model = ModelManager().get_or_create_chat_model(
        name="eval_judge",
        model_type=model_settings.type,
        config=model_settings,
    )
    judge = LLMJudge(
        model=judge_model,
        temperature=eval_cfg.judge.temperature,
    )

    # Create runner
    runner = EvaluationRunner(
        config=graphrag_config,
        judge=judge,
        index_roots=eval_cfg.indexes,
    )

    # Load dataset
    dataset_path = root / eval_cfg.dataset.path
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Filter dataset to configured indexes
    dataset = [qa for qa in dataset if qa["imdb_key"] in eval_cfg.indexes]

    if not dataset:
        print("Error: No QA pairs found for configured indexes.")
        return

    print(f"Loaded {len(dataset)} QA pairs for {len(eval_cfg.indexes)} movies")
    print(f"Methods: {', '.join(eval_cfg.methods)}")
    print(f"Total evaluations: {len(dataset) * len(eval_cfg.methods)}")
    print()

    # Resume logic
    completed_results = []
    if resume:
        checkpoint_path = Path(eval_cfg.output.dir) / "checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            completed_results = checkpoint.get("results", [])
            completed_keys = {(r["imdb_key"], r["question"], r["method"]) for r in completed_results}
            print(f"Resuming from checkpoint: {len(completed_results)} completed")
        else:
            completed_keys = set()
    else:
        completed_keys = set()

    # Progress callback
    def progress(current: int, total: int, movie: str, method: str):
        pct = current / total * 100
        print(f"\r[{current}/{total}] ({pct:.1f}%) {movie} - {method}    ", end="")
        sys.stdout.flush()

    # Run evaluation
    try:
        results = await runner.run_evaluation(
            dataset=dataset,
            methods=eval_cfg.methods,
            progress_callback=progress if verbose else None,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving checkpoint...")
        results = []

    print()  # Newline after progress

    # Combine with resumed results
    all_results = completed_results + [r.to_dict() for r in results]

    # Save checkpoint
    if eval_cfg.output.save_intermediate:
        checkpoint_path = Path(eval_cfg.output.dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path / "checkpoint.json", "w") as f:
            json.dump({"results": all_results}, f)

    # Aggregate and save final results
    from graphrag.eval.runner import QueryResult, EfficiencyMetrics
    from graphrag.eval.metrics import MetricScores, JudgeResult

    # Convert dict results back to QueryResult objects for aggregation
    query_results = []
    for r in all_results:
        if isinstance(r, dict):
            scores = MetricScores(
                correctness=JudgeResult(r["scores"]["correctness"], ""),
                faithfulness=JudgeResult(r["scores"]["faithfulness"], ""),
                relevance=JudgeResult(r["scores"]["relevance"], ""),
                completeness=JudgeResult(r["scores"]["completeness"], ""),
            )
            efficiency = EfficiencyMetrics(**r["efficiency"])
            qr = QueryResult(
                imdb_key=r["imdb_key"],
                question=r["question"],
                method=r["method"],
                response=r["response"],
                ground_truth=r.get("ground_truth", ""),
                context_text=r.get("context_text", ""),
                scores=scores,
                efficiency=efficiency,
            )
            query_results.append(qr)
        else:
            query_results.append(r)

    aggregated = aggregate_results(query_results)
    aggregated.save(eval_cfg.output.dir)

    # Print summary
    print("\n=== Evaluation Summary ===\n")
    for method, scores in aggregated.by_method.items():
        print(f"{method.upper()}:")
        print(f"  Correctness:  {scores['correctness']:.2%}")
        print(f"  Faithfulness: {scores['faithfulness']:.2%}")
        print(f"  Relevance:    {scores['relevance']:.2%}")
        print(f"  Completeness: {scores['completeness']:.2%}")
        print()

    print(f"\nResults saved to {eval_cfg.output.dir}/")
