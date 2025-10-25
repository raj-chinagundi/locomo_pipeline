# RAG Pipeline for Question Answering

Minimal RAG (Retrieval-Augmented Generation) pipeline with pluggable retrievers for question answering on conversation data.

## Features

- Category-based filtering of questions
- Multiple pluggable retrievers (TF-IDF, SVM, FAISS, NanoPQ, BM25)
- OpenAI-based answer generation with fallback
- Automatic answer evaluation
- Detailed JSON output with per-question results

## Installation

```bash
pip install -r requirements.txt
```

For optional retrievers:
```bash
# FAISS (faster similarity search)
pip install faiss-cpu

# NanoPQ (product quantization)
pip install nanopq
```

## Usage

Run on all questions:
```bash
python pipeline.py
```

Run on specific category only:
```bash
python pipeline.py --category 2
```

Specify output file:
```bash
python pipeline.py --category 2 --output results.json
```

## Configuration

Set Gemini API key for LLM-based answer generation:
```bash
export GEMINI_API_KEY="your-api-key-here"
# or
export GOOGLE_API_KEY="your-api-key-here"
```

If no API key is provided, the pipeline will use a simple fallback answer generator.

## Output Format

The pipeline generates a JSON file with the following structure:

```json
{
  "convid": 1,
  "qcount": 5,
  "retrievers": [
    {
      "name": "bm25",
      "total_answer_right": 5,
      "accuracy": 1.0,
      "q_lvl": [
        {
          "qid": 1,
          "category": 2,
          "source_q": "When was the product Alpha launched?",
          "source_answer": "2024-03-15",
          "source_evidence_ids": ["D1:9"],
          "generated_answer": "2024-03-15",
          "retrieved_evidence_ids": ["D1:2", "D1:9", "D4:7"],
          "status": "correct"
        }
      ]
    }
  ]
}
```

## Retrievers

The following retrievers are available:

1. **TF-IDF**: Term frequency-inverse document frequency with cosine similarity
2. **SVM**: Support Vector Machine-based retrieval (simplified with TF-IDF)
3. **FAISS**: Fast similarity search with sentence transformers
4. **NanoPQ**: Product quantization for compressed vector search
5. **BM25**: Best Matching 25 ranking function

All retrievers implement the `BaseRetriever` interface and can be easily extended.

## Project Structure

```
newcode/
├── pipeline.py          # Main pipeline script
├── data_loader.py       # Data loading and filtering
├── retrievers.py        # Retriever implementations
├── llm_helper.py        # LLM answer generation
├── evaluator.py         # Answer evaluation
├── requirements.txt     # Python dependencies
├── locomo10.json        # Input data file
└── README.md            # This file
```

## Adding New Retrievers

To add a new retriever:

1. Create a new class in `retrievers.py` that inherits from `BaseRetriever`
2. Implement the `retrieve()` method
3. Add initialization in `get_all_retrievers()` function

Example:

```python
class MyRetriever(BaseRetriever):
    def __init__(self, documents):
        super().__init__(documents)
        # Your initialization code
    
    def retrieve(self, query: str, top_k: int = 3):
        # Your retrieval logic
        return retrieved_docs
```
HOW TO RUN:
```

# Just SVM
python pipeline.py --category 2 --limit 1 --retriever svm

# Just BM25
python pipeline.py --category 2 --limit 1 --retriever bm25

# Just TF-IDF
python pipeline.py --retriever tfidf

# ALL CATEGORY 2 ques limit conversation to 1
python pipeline.py --category 2 --limit 1 (ALL RETRIEVERS DEFAULT)

```
