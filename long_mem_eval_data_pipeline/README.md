# LongMemEval → ChromaDB Pipeline

Production-ready utility that ingests the LongMemEval_s dataset, flattens each question → session → dialogue turn hierarchy, and stores per-turn embeddings with rich metadata in ChromaDB. Built for deterministic processing and schema validation.

## Features
- Validates every dataset record to ensure required LongMemEval fields are present.
- Flattens each haystack session into turn-level documents with `question_id`, `session_id`, `turn_id`, `role`, `has_answer`, timestamps, and answer-trace metadata.
- Uses `sentence-transformers/all-MiniLM-L6-v2` locally for deterministic embedding generation.
- Persists data into ChromaDB for retrieval by similarity or metadata filters.
- Includes a lightweight schema validation test using a sample record.

## Setup
```bash
cd /Users/ShahYash/Desktop/Projects/HyArg/long_mem_eval_data_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline
```bash
rm -rf ./vector_store
export CHROMA_TELEMETRY_ENABLED=FALSE
python pipeline.py \
  --dataset ../LongMemEval/data/longmemeval_s_cleaned.json \
  --persist-dir ./vector_store \
  --collection longmemeval_turns \
  --batch-size 512 \
  --encode-batch-size 128 \
  --limit 0 \
  --reset-collection \
  --skip-existing
```
- `--limit 0` ingests the full dataset; set `--limit 10` (for example) to process only the first 10 questions when testing.
- Tweak `--batch-size` (Chroma writes per batch) and `--encode-batch-size` (embedding throughput) to balance runtime vs. memory.
- Pass `--reset-collection` whenever you want a clean ingest without duplicate-ID conflicts.
- Use `--skip-existing` to skip any questions already present in the target collection when resuming a run.
- Add `--run-check` to trigger a quick retrieval sanity check after ingestion.

## Inspecting the Stored Turns
```bash
python inspect_chroma.py \
  --persist-dir ./vector_store \
  --collection longmemeval_turns \
  --limit 5 2>&1 | grep -v "capture()"
```
- Include `--where question_id=<id>` or `--where session_id=<id>` to filter the preview.
- Pipe through `grep -v "capture()"` (or set `CHROMA_TELEMETRY_ENABLED=FALSE`) to silence Chroma's telemetry warnings while inspecting.

Example output:
```
| question_id   | question_type       | session_id   | session_timestamp      | turn_id         | role      | has_answer   | is_answer_session   | answer_session_ids   | content_preview                                                                     |
|---------------|---------------------|--------------|------------------------|-----------------|-----------|--------------|---------------------|----------------------|-------------------------------------------------------------------------------------|
| 118b2229      | single-session-user | db73b7e4_4   | 2023/05/20 (Sat) 03:29 | db73b7e4_4_t001 | user      | False        | False               | answer_40a90d51      | I'm looking for some advice on how to take care of my leather boots. I've been w... |
| 118b2229      | single-session-user | db73b7e4_4   | 2023/05/20 (Sat) 03:29 | db73b7e4_4_t002 | assistant | False        | False               | answer_40a90d51      | Congratulations on your new boots! Taking good care of them will indeed help the... |
| 118b2229      | single-session-user | db73b7e4_4   | 2023/05/20 (Sat) 03:29 | db73b7e4_4_t003 | user      | False        | False               | answer_40a90d51      | I've been meaning to get some waterproofing spray for my boots, can you recommen... |
| 118b2229      | single-session-user | db73b7e4_4   | 2023/05/20 (Sat) 03:29 | db73b7e4_4_t004 | assistant | False        | False               | answer_40a90d51      | Waterproofing spray is an excellent addition to your boot care routine. There ar... |
| 118b2229      | single-session-user | db73b7e4_4   | 2023/05/20 (Sat) 03:29 | db73b7e4_4_t005 | user      | False        | False               | answer_40a90d51      | I've heard of Nikwax before, but I've never tried it. How often would I need to ... |
```

## Schema Validation Test (Sample Record)
```bash
python tests/schema_validation.py
```
This test exercises the loader and flattener on `sample_longmemeval_record.json`, ensuring the expected metadata schema before indexing the full dataset.

## Cleanup
The ChromaDB store is created under `--persist-dir`. Remove the directory to reset the index.
