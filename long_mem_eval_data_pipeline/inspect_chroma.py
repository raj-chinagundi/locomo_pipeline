"""Inspect a Chroma collection to preview stored LongMemEval turns."""
from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from tabulate import tabulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview ChromaDB collection contents")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("./vector_store"),
        help="Directory where ChromaDB data is persisted",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="longmemeval_turns",
        help="Collection name to inspect",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of turn-level rows to display",
    )
    parser.add_argument(
        "--where",
        type=str,
        default=None,
        help="Optional metadata filter expressed as key=value",
    )
    return parser.parse_args()


def parse_where(where_clause: str | None) -> dict[str, str] | None:
    if not where_clause:
        return None
    key, value = where_clause.split("=", maxsplit=1)
    return {key.strip(): value.strip()}


def main() -> None:
    args = parse_args()
    client = chromadb.PersistentClient(path=str(args.persist_dir))
    collection = client.get_or_create_collection(args.collection)

    where = parse_where(args.where)
    results = collection.get(where=where, include=["documents", "metadatas"], limit=args.limit)

    rows = []
    for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
        rows.append(
            [
                meta.get("question_id"),
                meta.get("question_type"),
                meta.get("session_id"),
                meta.get("session_timestamp"),
                meta.get("turn_id"),
                meta.get("role"),
                meta.get("has_answer"),
                meta.get("is_answer_session"),
                meta.get("answer_session_ids"),
                doc[:80].replace("\n", " ") + ("..." if len(doc) > 80 else ""),
            ]
        )

    if not rows:
        print("No rows returned. Check the collection name, persist directory, or filters.")
        return

    headers = [
        "question_id",
        "question_type",
        "session_id",
        "session_timestamp",
        "turn_id",
        "role",
        "has_answer",
        "is_answer_session",
        "answer_session_ids",
        "content_preview",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
