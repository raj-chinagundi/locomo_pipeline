"""LongMemEval_s → ChromaDB ingestion pipeline."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models import Collection
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

REQUIRED_ROOT_FIELDS = [
    "question_id",
    "question_type",
    "question",
    "answer",
    "question_date",
    "haystack_session_ids",
    "haystack_dates",
    "haystack_sessions",
    "answer_session_ids",
]
REQUIRED_TURN_FIELDS = ["role", "content"]


@dataclass(frozen=True)
class TurnRecord:
    """Container for an individual dialogue turn ready for indexing."""

    record_id: str
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PipelineStats:
    """Simple summary of processed elements."""

    questions: int
    sessions: int
    turns: int


def load_dataset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load and validate the LongMemEval_s dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Expected dataset to be a list of question objects")

    if limit is not None and limit > 0:
        payload = payload[:limit]

    for idx, record in enumerate(payload):
        missing = [field for field in REQUIRED_ROOT_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Record {idx} missing required fields: {missing}")
        _validate_session_alignment(record, idx)
    return payload


def _validate_session_alignment(record: dict[str, Any], idx: int) -> None:
    """Ensure session IDs, dates, and transcripts align one-to-one."""

    session_ids = record["haystack_session_ids"]
    session_dates = record["haystack_dates"]
    sessions = record["haystack_sessions"]

    if not (isinstance(session_ids, list) and isinstance(session_dates, list) and isinstance(sessions, list)):
        raise ValueError(f"Record {idx} contains non-list session fields")

    length_set = {len(session_ids), len(session_dates), len(sessions)}
    if len(length_set) != 1:
        raise ValueError(
            f"Record {idx} session arrays must have identical lengths: "
            f"ids={len(session_ids)}, dates={len(session_dates)}, sessions={len(sessions)}"
        )


def flatten_haystack_structure(dataset: Sequence[dict[str, Any]]) -> Tuple[list[TurnRecord], PipelineStats]:
    """Convert nested question → session → turn hierarchy into turn-level records."""

    records: list[TurnRecord] = []
    session_count = 0
    for question in tqdm(dataset, desc="Flattening questions", unit="q"):
        qid = str(question["question_id"])
        question_type = question["question_type"]
        question_date = question["question_date"]
        answer_session_ids = [str(value) for value in question["answer_session_ids"]]
        answer_session_id_set = set(answer_session_ids)

        for session_idx, (raw_session_id, session_date, turns) in enumerate(
            zip(question["haystack_session_ids"], question["haystack_dates"], question["haystack_sessions"]),
            start=1,
        ):
            session_count += 1
            session_id = str(raw_session_id)
            session_date_str = str(session_date)
            if not isinstance(turns, list):
                raise ValueError(f"Session {session_id} in question {qid} must be a list of turns")
            for turn_idx, turn in enumerate(turns, start=1):
                for field in REQUIRED_TURN_FIELDS:
                    if field not in turn:
                        raise ValueError(
                            f"Turn {turn_idx} in session {session_id} missing required field '{field}'"
                        )
                content = str(turn["content"]).strip()
                if not content:
                    continue  # skip empty strings to avoid meaningless embeddings
                role = turn["role"]
                if role not in {"user", "assistant"}:
                    raise ValueError(
                        f"Turn {turn_idx} in session {session_id} has invalid role '{role}'"
                    )
                turn_id = f"{session_id}_t{turn_idx:03d}"
                record_id = f"{qid}__s{session_idx:03d}__{turn_id}"
                metadata = {
                    "question_id": qid,
                    "question_type": question_type,
                    "question_date": question_date,
                    "session_id": session_id,
                    "session_timestamp": session_date_str,
                    "turn_id": turn_id,
                    "role": role,
                    "has_answer": bool(turn.get("has_answer", False)),
                    "is_answer_session": session_id in answer_session_id_set,
                    "answer_session_ids": "|".join(answer_session_ids),
                }
                records.append(TurnRecord(record_id=record_id, content=content, metadata=metadata))
    stats = PipelineStats(questions=len(dataset), sessions=session_count, turns=len(records))
    return records, stats


def init_embedding_model(model_name: str) -> SentenceTransformer:
    """Instantiate the sentence-transformer model."""

    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = 512
    return model


def create_collection(
    persist_directory: Path,
    collection_name: str,
    reset_collection: bool = False,
) -> Tuple[ClientAPI, Collection.Collection]:
    """Create or reuse a ChromaDB collection."""

    persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_directory))
    if reset_collection:
        try:
            client.delete_collection(name=collection_name)
            print(f"Reset existing collection '{collection_name}'")
        except Exception:
            pass  # Ignore if the collection did not exist
    collection = client.get_or_create_collection(name=collection_name, metadata={"source": "LongMemEval_s"})
    return client, collection


def question_exists(collection: Collection.Collection, question_id: str) -> bool:
    """Check if any turn for the given question already exists in the collection."""

    result = collection.get(where={"question_id": question_id}, include=["metadatas"], limit=1)
    return bool(result.get("ids"))


def filter_new_questions(
    dataset: Sequence[dict[str, Any]],
    collection: Collection.Collection,
    skip_existing: bool,
) -> list[dict[str, Any]]:
    """Optionally drop questions already present in the collection."""

    if not skip_existing:
        return list(dataset)

    filtered: list[dict[str, Any]] = []
    skipped = 0
    for record in dataset:
        qid = str(record["question_id"])
        if question_exists(collection, qid):
            skipped += 1
            continue
        filtered.append(record)
    if skipped:
        print(f"Skipped {skipped} questions already present in the collection")
    return filtered


def store_in_chromadb(
    records: Sequence[TurnRecord],
    collection: Collection.Collection,
    embedder: SentenceTransformer,
    batch_size: int = 128,
    encode_batch_size: int = 256,
) -> None:
    """Embed records and store them inside ChromaDB in batches."""

    for start_idx in tqdm(range(0, len(records), batch_size), desc="Indexing", unit="batch"):
        batch = records[start_idx : start_idx + batch_size]
        texts = [item.content for item in batch]
        embeddings = embedder.encode(
            texts,
            convert_to_numpy=True,
            batch_size=min(max(1, encode_batch_size), len(texts)),
            show_progress_bar=False,
        )
        collection.upsert(
            ids=[item.record_id for item in batch],
            documents=texts,
            metadatas=[item.metadata for item in batch],
            embeddings=[vector.tolist() for vector in embeddings],
        )


def run_retrieval_check(
    sample_text: str,
    collection: Collection.Collection,
    embedder: SentenceTransformer,
    top_k: int = 3,
) -> None:
    """Perform a quick similarity search to verify storage."""

    query_vec = embedder.encode(sample_text, convert_to_numpy=True)
    results = collection.query(query_embeddings=[query_vec.tolist()], n_results=top_k)
    print("\nRetrieval sanity check:")
    for doc, meta in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]):
        print(f"- {meta.get('question_id')} | {meta.get('session_id')} | {meta.get('turn_id')}: {doc[:80]}...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LongMemEval_s to ChromaDB pipeline")
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
        default=Path("./vector_store"),
        help="Directory where ChromaDB should persist data",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="longmemeval_turns",
        help="Collection name inside ChromaDB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of turns to send to ChromaDB per batch",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="Number of texts to encode simultaneously (controls throughput)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N questions (0 = all)",
    )
    parser.add_argument(
        "--run-check",
        action="store_true",
        help="Run a retrieval sanity check using the first stored turn",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Drop and recreate the target collection before ingestion",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip questions already stored in the target collection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = args.limit if args.limit > 0 else None
    dataset = load_dataset(args.dataset, limit=limit)

    embedder = init_embedding_model(args.model)
    _, collection = create_collection(
        args.persist_dir,
        args.collection,
        reset_collection=args.reset_collection,
    )
    dataset = filter_new_questions(dataset, collection, skip_existing=args.skip_existing)
    if not dataset:
        print("No new questions to ingest. Exiting.")
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
        f"\nIngestion complete: questions={stats.questions}, sessions={stats.sessions}, turns={stats.turns}"
    )


if __name__ == "__main__":
    main()
