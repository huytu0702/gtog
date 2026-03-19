from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graphrag.eval.ragas_runner import aggregate_ragas_results


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate a Ragas summary JSON from a detailed JSON file.",
    )
    parser.add_argument("--detailed", required=True, help="Path to detailed JSON file.")
    parser.add_argument("--summary", required=True, help="Path to summary JSON file.")
    args = parser.parse_args()

    detailed_path = Path(args.detailed)
    summary_path = Path(args.summary)

    detailed_results = _load_json(detailed_path)
    existing_summary = _load_json(summary_path)
    metadata = existing_summary["metadata"]

    updated_summary = aggregate_ragas_results(
        detailed_results,
        model=metadata["model"],
        base_url=metadata["base_url"],
        input_path=metadata["input_file"],
        settings_path=metadata["settings_path"],
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(updated_summary, file, indent=2, ensure_ascii=False)

    print(f"Updated {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
