# Phase 3: CLI Integration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `graphrag eval` CLI command with configuration file support, progress display, and resume capability.

**Architecture:** Follow existing CLI patterns using typer. Create eval.py with command implementation, integrate into main.py.

**Tech Stack:** typer, pydantic, yaml

---

## Prerequisites

- Phase 2 must be complete (graphrag.eval module exists)

---

## Task 1: Create Eval Config Schema

**Files:**
- Create: `graphrag/config/models/eval_config.py`

**Step 1: Write the failing test**

```python
# tests/unit/config/test_eval_config.py
import pytest
from graphrag.config.models.eval_config import EvalConfig

def test_eval_config_from_dict():
    """EvalConfig should load from dictionary."""
    config_dict = {
        "dataset": {"path": "eval/qa_eval.json"},
        "indexes": {
            "tt0097576": "tt0097576",
            "tt0102798": "tt0102798",
        },
        "methods": ["tog", "local", "basic"],
        "output": {"dir": "eval/results", "save_intermediate": True},
        "judge": {"temperature": 0.0},
    }

    config = EvalConfig(**config_dict)

    assert config.dataset.path == "eval/qa_eval.json"
    assert len(config.indexes) == 2
    assert "tog" in config.methods

def test_eval_config_defaults():
    """EvalConfig should have sensible defaults."""
    config = EvalConfig(
        dataset={"path": "eval/qa_eval.json"},
        indexes={"tt1": "tt1"},
    )

    assert config.methods == ["tog", "local", "basic"]
    assert config.output.save_intermediate is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/config/test_eval_config.py -v`
Expected: FAIL

**Step 3: Create eval_config.py**

```python
# graphrag/config/models/eval_config.py
"""Configuration model for evaluation runs."""

from pydantic import BaseModel, Field
from typing import Any


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    path: str = Field(description="Path to QA dataset JSON file")


class OutputConfig(BaseModel):
    """Output configuration."""
    dir: str = Field(default="eval/results", description="Output directory for results")
    save_intermediate: bool = Field(default=True, description="Save intermediate results during run")


class JudgeConfig(BaseModel):
    """LLM judge configuration."""
    model: str | None = Field(default=None, description="Model to use for judging (None = use default)")
    temperature: float = Field(default=0.0, description="Temperature for judge LLM")


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""
    dataset: DatasetConfig
    indexes: dict[str, str] = Field(description="Mapping of imdb_key to index root directory")
    methods: list[str] = Field(default=["tog", "local", "basic"], description="Search methods to evaluate")
    output: OutputConfig = Field(default_factory=OutputConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        """Load configuration from YAML file."""
        import yaml
        from pathlib import Path

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/config/test_eval_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/config/models/eval_config.py tests/unit/config/test_eval_config.py
git commit -m "$(cat <<'EOF'
feat(config): add EvalConfig model for evaluation settings

Pydantic model for eval_config.yaml with:
- dataset.path: QA dataset location
- indexes: mapping of imdb_key to index roots
- methods: list of search methods to evaluate
- output: result directory and intermediate saving
- judge: LLM judge settings

EOF
)"
```

---

## Task 2: Create Eval CLI Module

**Files:**
- Create: `graphrag/cli/eval.py`

**Step 1: Write the failing test**

```python
# tests/unit/cli/test_eval_cli.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

def test_eval_cli_loads_config():
    """eval_cli should load eval_config.yaml."""
    from graphrag.cli.eval import eval_cli

    with patch('graphrag.cli.eval.EvalConfig.from_yaml') as mock_load:
        mock_config = MagicMock()
        mock_config.dataset.path = "eval/qa_eval.json"
        mock_config.indexes = {"tt1": "tt1"}
        mock_config.methods = ["tog"]
        mock_config.output.dir = "eval/results"
        mock_load.return_value = mock_config

        with patch('graphrag.cli.eval.run_evaluation') as mock_run:
            mock_run.return_value = None

            # Should not raise
            eval_cli(
                root_dir=Path("."),
                eval_config=Path("eval_config.yaml"),
                methods=None,
                imdb_key=None,
                resume=False,
                verbose=False,
            )

        mock_load.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/cli/test_eval_cli.py -v`
Expected: FAIL (module not found)

**Step 3: Create eval.py**

```python
# graphrag/cli/eval.py
"""CLI implementation of the eval subcommand."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from graphrag.config.load_config import load_config
from graphrag.config.models.eval_config import EvalConfig
from graphrag.eval import (
    LLMJudge,
    EvaluationRunner,
    aggregate_results,
)
from graphrag.language_model.factory import create_chat_model_from_config

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

    # Create LLM judge
    judge_model = create_chat_model_from_config(graphrag_config)
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/cli/test_eval_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/cli/eval.py tests/unit/cli/test_eval_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): add eval command implementation

eval_cli runs batch evaluation:
- Loads eval_config.yaml
- Supports --methods and --imdb-key filters
- Shows progress during run
- Saves checkpoint for resume
- Outputs summary and detailed JSON results

EOF
)"
```

---

## Task 3: Register Eval Command in Main CLI

**Files:**
- Modify: `graphrag/cli/main.py`

**Step 1: Write the failing test**

```python
# tests/unit/cli/test_main_eval.py
import pytest
from typer.testing import CliRunner
from graphrag.cli.main import app

runner = CliRunner()

def test_eval_command_exists():
    """CLI should have eval command."""
    result = runner.invoke(app, ["--help"])
    assert "eval" in result.stdout
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/cli/test_main_eval.py -v`
Expected: FAIL (eval not in output)

**Step 3: Add eval command to main.py**

Add after the query command in `graphrag/cli/main.py`:

```python
@app.command("eval")
def _eval_cli(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Evaluation configuration file (eval_config.yaml).",
        exists=True,
        file_okay=True,
        readable=True,
        autocompletion=CONFIG_AUTOCOMPLETE,
    ),
    root: Path = typer.Option(
        Path(),
        "--root",
        "-r",
        help="The project root directory.",
        exists=True,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        autocompletion=ROOT_AUTOCOMPLETE,
    ),
    methods: str | None = typer.Option(
        None,
        "--methods",
        "-m",
        help="Comma-separated list of methods to evaluate (e.g., tog,local,basic).",
    ),
    imdb_key: str | None = typer.Option(
        None,
        "--imdb-key",
        "-k",
        help="Evaluate only this movie (must be in config).",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from checkpoint if available.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show progress during evaluation.",
    ),
) -> None:
    """Evaluate GraphRAG search methods using LLM-as-Judge metrics."""
    from graphrag.cli.eval import eval_cli

    eval_cli(
        root_dir=root,
        eval_config=config,
        methods=methods,
        imdb_key=imdb_key,
        resume=resume,
        verbose=verbose,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/cli/test_main_eval.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/cli/main.py tests/unit/cli/test_main_eval.py
git commit -m "$(cat <<'EOF'
feat(cli): register eval command in main CLI

Added @app.command("eval") to main.py with options:
- --config: path to eval_config.yaml
- --root: project root directory
- --methods: filter to specific methods
- --imdb-key: filter to specific movie
- --resume: continue from checkpoint
- --verbose: show progress

EOF
)"
```

---

## Task 4: Create Example eval_config.yaml

**Files:**
- Create: `eval/eval_config.yaml` (example)

**Step 1: Create example config**

```yaml
# eval/eval_config.yaml
# Example evaluation configuration for GraphRAG

dataset:
  path: "eval/qa_eval.json"

indexes:
  tt0097576: "tt0097576"
  tt0102798: "tt0102798"
  tt2278388: "tt2278388"

methods:
  - tog
  - local
  - basic

output:
  dir: "eval/results"
  save_intermediate: true

judge:
  model: null  # uses default from settings.yaml
  temperature: 0.0
```

**Step 2: Validate config loads**

```bash
python -c "from graphrag.config.models.eval_config import EvalConfig; c = EvalConfig.from_yaml('eval/eval_config.yaml'); print(f'Loaded {len(c.indexes)} indexes')"
```
Expected: "Loaded 3 indexes"

**Step 3: Commit**

```bash
git add eval/eval_config.yaml
git commit -m "$(cat <<'EOF'
docs(eval): add example eval_config.yaml

Example configuration for running evaluation with all three movies
and all search methods.

EOF
)"
```

---

## Task 5: Add Config Export

**Files:**
- Modify: `graphrag/config/models/__init__.py`

**Step 1: Check current exports**

Read `graphrag/config/models/__init__.py` to see current structure.

**Step 2: Add EvalConfig export**

```python
# Add to graphrag/config/models/__init__.py
from graphrag.config.models.eval_config import EvalConfig

# Add to __all__ if it exists
```

**Step 3: Commit**

```bash
git add graphrag/config/models/__init__.py
git commit -m "$(cat <<'EOF'
feat(config): export EvalConfig from models package

EOF
)"
```

---

## Task 6: Test Full CLI Flow

**Step 1: Test help output**

```bash
graphrag eval --help
```

Expected output includes:
- `--config` option
- `--methods` option
- `--imdb-key` option
- `--resume` option
- `--verbose` option

**Step 2: Test dry run (no actual evaluation)**

Create a minimal test:

```python
# tests/integration/cli/test_eval_integration.py
import pytest
from pathlib import Path
from typer.testing import CliRunner
from graphrag.cli.main import app

runner = CliRunner()

def test_eval_requires_config():
    """Eval should fail gracefully without config."""
    result = runner.invoke(app, ["eval", "--root", "/nonexistent"])
    # Should exit with error about missing config
    assert result.exit_code != 0 or "Error" in result.stdout
```

Run: `pytest tests/integration/cli/test_eval_integration.py -v`

**Step 3: Commit**

```bash
git add tests/integration/cli/test_eval_integration.py
git commit -m "$(cat <<'EOF'
test(cli): add eval CLI integration test

Verifies eval command handles missing config gracefully.

EOF
)"
```

---

## Task 7: Final Phase 3 Commit

**Step 1: Run all CLI tests**

Run: `pytest tests/unit/cli/ tests/integration/cli/ -v`
Expected: All tests PASS

**Step 2: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(cli): complete Phase 3 - eval command

Complete CLI integration for evaluation:
- EvalConfig model for eval_config.yaml
- eval.py with full evaluation logic
- Registered in main.py
- Progress display and resume capability
- Example config file

Usage:
  graphrag eval --root ./project --verbose
  graphrag eval --config custom.yaml --methods tog,local
  graphrag eval --imdb-key tt0097576 --resume

EOF
)"
```

---

## Phase 3 Checklist

- [ ] Task 1: EvalConfig schema
- [ ] Task 2: eval.py implementation
- [ ] Task 3: Command registered in main.py
- [ ] Task 4: Example config file
- [ ] Task 5: Config exports
- [ ] Task 6: CLI flow tested
- [ ] Task 7: All tests pass
