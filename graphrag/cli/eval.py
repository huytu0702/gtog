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
    resume: bool,
    skip_evaluation: bool,
    verbose: bool,
):
    """Run evaluation on GraphRAG indexes.

    Compares search methods (ToG, Local, Basic) using LLM-as-Judge metrics.
    """
    root = root_dir.resolve()

    # Load eval config
    if eval_config:
        config_path = (
            root / eval_config if not eval_config.is_absolute() else eval_config
        )
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

    # Run evaluation
    asyncio.run(run_evaluation(root, eval_cfg, resume, skip_evaluation, verbose))


async def run_evaluation(
    root: Path,
    eval_cfg: EvalConfig,
    resume: bool,
    skip_evaluation: bool,
    verbose: bool,
):
    """Execute the evaluation run."""
    # Load main graphrag config for LLM settings
    graphrag_config = load_config(root)

    # Create runner (judge only needed if not skipping evaluation)
    if skip_evaluation:
        runner = EvaluationRunner(
            config=graphrag_config,
            judge=None,  # type: ignore[arg-type]
            index_roots=eval_cfg.indexes,
        )
    else:
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
        runner = EvaluationRunner(
            config=graphrag_config,
            judge=judge,
            index_roots=eval_cfg.indexes,
        )

    # Load dataset
    dataset_path = root / eval_cfg.dataset.path
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Validate dataset format
    if not dataset:
        print("Error: Empty dataset.")
        return

    print(f"Loaded {len(dataset)} QA pairs")
    print(f"Methods: {', '.join(eval_cfg.methods)}")
    print(f"Total evaluations: {len(dataset) * len(eval_cfg.methods)}")
    print()

    # Load index data once for all queries
    from pathlib import Path as PathLib

    index_data = await runner._load_index(
        list(eval_cfg.indexes.keys())[0] if eval_cfg.indexes else "default"
    )

    # Resume logic
    completed_results = []
    if resume:
        checkpoint_path = Path(eval_cfg.output.dir) / "checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            completed_results = checkpoint.get("results", [])
            completed_keys = {(r["question"], r["method"]) for r in completed_results}
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
            index_data=index_data,
            progress_callback=progress if verbose else None,
            skip_evaluation=skip_evaluation,
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
            # Handle both simple results (no evaluation) and full results
            if "scores" in r and "efficiency" in r:
                scores = MetricScores(
                    correctness=JudgeResult(r["scores"]["correctness"], ""),
                    faithfulness=JudgeResult(r["scores"]["faithfulness"], ""),
                    relevance=JudgeResult(r["scores"]["relevance"], ""),
                    completeness=JudgeResult(r["scores"]["completeness"], ""),
                )
                efficiency = EfficiencyMetrics(**r["efficiency"])
            else:
                scores = None
                efficiency = None
            qr = QueryResult(
                question=r["question"],
                method=r["method"],
                response=r["response"],
                context=r.get("context", ""),
                context_text=r.get("context_text", ""),
                ground_truth=r.get("ground_truth", ""),
                scores=scores,
                efficiency=efficiency,
            )
            query_results.append(qr)
        else:
            query_results.append(r)

    aggregated = aggregate_results(query_results)

    # Save results (simple or full)
    if skip_evaluation:
        aggregated.save_simple(eval_cfg.output.dir)
        print(f"\n=== Simple Results Saved ===")
        print(f"Results saved to {eval_cfg.output.dir}/eval_results_simple.json")
    else:
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
