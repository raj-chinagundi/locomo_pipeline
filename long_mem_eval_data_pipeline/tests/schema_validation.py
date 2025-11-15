"""Lightweight schema validation using the provided sample record."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import flatten_haystack_structure, load_dataset


def main() -> None:
    sample_path = PROJECT_ROOT / "sample_longmemeval_record.json"
    dataset = load_dataset(sample_path)
    records, stats = flatten_haystack_structure(dataset)
    assert records, "No records returned from sample dataset"
    assert stats.turns == len(records), "Turn count mismatch"
    print(
        "Schema validation succeeded:",
        f"questions={stats.questions}",
        f"sessions={stats.sessions}",
        f"turns={stats.turns}",
    )


if __name__ == "__main__":
    main()
