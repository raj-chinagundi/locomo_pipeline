"""Utility to ingest only the first LongMemEval question for quick validation."""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import (
    create_collection,
    filter_new_questions,
    flatten_haystack_structure,
    init_embedding_model,
    load_dataset,
    run_retrieval_check,
    store_in_chromadb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-question LongMemEval ingest")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "LongMemEval"
        / "data"
        / "longmemeval_s_cleaned.json",
        help="Path to the LongMemEval_s dataset JSON file",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("./vector_store_single"),
        help="Directory where ChromaDB should persist data",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="longmemeval_turns_single",
        help="Collection name inside ChromaDB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of turns per batch",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=64,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Drop and recreate the collection before ingestion",
    )
    parser.add_argument(
        "--run-check",
        action="store_true",
        help="Run retrieval sanity check after ingesting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, limit=1)

    embedder = init_embedding_model(args.model)
    _, collection = create_collection(
        args.persist_dir,
        args.collection,
        reset_collection=args.reset_collection,
    )
    dataset = filter_new_questions(dataset, collection, skip_existing=True)
    if not dataset:
        print("The first question already exists in the collection. Exiting.")
        return

    records, stats = flatten_haystack_structure(dataset)
    store_in_chromadb(
        records,
        collection,
        embedder,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
    )

    if args.run_check and records:
        run_retrieval_check(records[0].content, collection, embedder)

    print(
        f"Single-question ingest complete: questions={stats.questions}, sessions={stats.sessions}, turns={stats.turns}"
    )


if __name__ == "__main__":
    main()
