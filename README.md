# MEMAGENT - Retriever Comparison on Locomo Dataset

A clean, modular system for comparing retriever performance on conversational QA tasks using the Locomo dataset.

## 📋 Overview

This codebase provides a simple pipeline to:
1. Load conversational data from Locomo dataset
2. Test multiple retrieval algorithms (13 different retrievers)
3. Compare performance on question-answering tasks
4. Save results for analysis

**Key Features:**
- Clean separation of concerns (data loading, retrievers, execution)
- YAML-based configuration for easy customization
- Support for both basic (keyword/vector) and advanced (LLM-based) retrievers
- Consistent interface across all retrievers

---

## 🚀 Quick Start

### 1. Installation

```bash
cd /Users/raj/Desktop/MEMAGENT

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -r requirements.txt

# Verify installation
uv run python test_setup.py
```

### 2. Basic Configuration

The system uses `config.yaml` for all settings. Key configurations:

```yaml
# Which conversations to test
data:
  sample_ids: "all"  # or ["conv-26", "conv-27"]
  limit: null        # or 5 for quick testing

# Which retrievers to run
retrievers:
  to_run: "all"      # or ["bm25", "faiss", "ensemble"]
  top_k: 10

# Enable LLM for advanced retrievers (optional)
llm:
  enabled: true      # requires GOOGLE_API_KEY in .env
```

### 3. Run Pipeline

```bash
# Run with default config
uv run python run_pipeline.py

# Or with standard Python
python run_pipeline.py
```

### 4. View Results

Results are saved to `./results/` directory:
- `retriever_all_YYYYMMDD_HHMMSS.json` - Detailed results
- `summary_YYYYMMDD_HHMMSS.json` - Summary statistics

---

## 🔧 Configuration

Edit `config.yaml` to customize:

### Dataset Selection
```yaml
data:
  json_path: "locomo10.json"
  sample_ids: "all"           # Test all conversations
  # sample_ids: ["conv-26"]   # Test specific conversation
  limit: null                  # No limit
  # limit: 5                   # Test only 5 conversations
```

### Retriever Selection
```yaml
retrievers:
  to_run: "all"               # Run all available retrievers
  # to_run: ["bm25", "faiss"] # Run specific retrievers
  top_k: 10                    # Retrieve top 10 documents
```

### Evaluation Options
```yaml
evaluation:
  question_limit: null          # Test all questions
  # question_limit: 10          # Test only 10 questions per conversation
  filter_by_conversation: true  # Only retrieve from same conversation
```

---

## 🤖 LLM Setup (Optional)

Some retrievers use LLM for enhanced performance. To enable:

### 1. Get Gemini API Key
1. Visit: https://makersuite.google.com/app/apikey
2. Sign in and create an API key
3. Copy the key

### 2. Create `.env` File
```bash
# Create .env in project root
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### 3. Enable in Config
```yaml
llm:
  enabled: true
  temperature: 0.0
  model: "gemini-2.5-flash-lite"
```

### 4. Test LLM
```bash
uv run python llm_helper.py
```

**LLM-Enhanced Retrievers:**
- `multiquery` - Generates multiple query variations
- `contextual_compression` - Filters results with LLM
- `self_query` - Extracts filters from natural language
- `multivector` - Multiple representations per document

---

## 📊 Available Retrievers

### Basic Retrievers (No LLM Required)
| Retriever | Type | Description |
|-----------|------|-------------|
| `bm25` | Keyword | Probabilistic keyword ranking |
| `tfidf` | Keyword | Traditional term weighting |
| `faiss` | Vector | Semantic similarity search |
| `svm` | ML | Support Vector Machine ranking |
| `mmr` | Diversification | Max Marginal Relevance |
| `time_weighted` | Temporal | Recency-aware retrieval |

### Advanced Retrievers (No LLM Required)
| Retriever | Type | Description |
|-----------|------|-------------|
| `ensemble` | Hybrid | Combines FAISS + BM25 |
| `long_context_reorder` | Reordering | Optimizes document ordering |

### LLM-Enhanced Retrievers (Requires API Key)
| Retriever | Type | Description |
|-----------|------|-------------|
| `multiquery` | Query Expansion | Multiple query variations |
| `contextual_compression` | Reranking | LLM-based filtering |
| `self_query` | Metadata-Aware | Natural language filters |
| `multivector` | Multi-Representation | Multiple embeddings per doc |

### Specialized Retrievers
| Retriever | Type | Description |
|-----------|------|-------------|
| `parent_document` | Hierarchical | Returns full sessions |

---

## 📁 Code Structure

```
MEMAGENT/
├── config.yaml           # Configuration file
├── run_pipeline.py       # Main execution script
├── dataloader.py         # Data loading and filtering
├── retriever.py          # Unified retriever interface
├── llm_helper.py         # LLM initialization
├── locomo10.json         # Dataset
├── results/              # Output directory
│   ├── retriever_all_*.json
│   └── summary_*.json
└── README.md             # This file
```

### Key Components

**dataloader.py**
- `DataLoader` class for loading Locomo data
- Methods: `load_data()`, `get_session_documents()`
- Configurable filtering by sample_ids and limit

**retriever.py**
- `Retriever` class with unified interface
- Access via properties: `retriever.bm25`, `retriever.faiss`
- All retrievers use `.invoke(query)` method
- Consistent output: `List[Document]`

**run_pipeline.py**
- Main execution script
- Steps: Load config → Load data → Initialize retrievers → Run experiments → Save results
- Clean output with progress tracking

**llm_helper.py**
- Simple LLM initialization
- `get_llm()` function returns Gemini instance
- Gracefully handles missing API key

---

## 🎯 Usage Examples

### Test All Retrievers on One Conversation
```yaml
# config.yaml
data:
  sample_ids: ["conv-26"]
retrievers:
  to_run: "all"
```

### Quick Test with Basic Retrievers
```yaml
# config.yaml
data:
  limit: 2
  sample_ids: "all"
retrievers:
  to_run: ["bm25", "faiss", "ensemble"]
evaluation:
  question_limit: 5
```

### Compare Keyword vs Semantic
```yaml
# config.yaml
retrievers:
  to_run: ["bm25", "faiss"]
  top_k: 20
```

---

## 📈 Understanding Results

The pipeline outputs:

### Console Output
```
FINAL COMPARISON
--------------------------------------------------------------------------------
Retriever                 Questions       Evidence Found       Recall %
--------------------------------------------------------------------------------
bm25                      10/10           45/100                45.0%
faiss                     9/10            52/100                52.0%
ensemble                  10/10           58/100                58.0%
```

### JSON Results
```json
{
  "timestamp": "20251018_143022",
  "retrievers_tested": ["bm25", "faiss"],
  "results": [
    {
      "retriever": "bm25",
      "total_questions": 10,
      "questions_with_evidence": 8,
      "recall_pct": 45.0,
      "details": [...]
    }
  ]
}
```

**Key Metrics:**
- **Questions with evidence**: How many questions had at least 1 relevant document retrieved
- **Recall %**: Percentage of all ground truth evidence documents that were retrieved
- **Detailed results**: Per-question breakdown with retrieved documents

---

## 🛠️ Customization

### Adding New Retrievers

Edit `retriever.py` and add a new property:

```python
@property
def my_new_retriever(self):
    """My custom retriever."""
    if 'my_new_retriever' not in self._retrievers:
        print("[MyRetriever] Initializing...")
        # Your initialization code
        self._retrievers['my_new_retriever'] = your_retriever
    return self._retrievers['my_new_retriever']
```

### Modifying Evaluation

Edit `run_pipeline.py` → `evaluate_retriever()` function to change:
- Evaluation metrics
- Output format
- Progress display

### Custom Data Processing

Edit `dataloader.py` → `_process_conversations()` to:
- Change document granularity
- Add/modify metadata
- Filter specific utterances

---

## 🐛 Troubleshooting

### "No module named 'langchain_classic'"
Some advanced retrievers need this package:
```bash
uv add langchain-classic
```

### "GOOGLE_API_KEY not found"
LLM-based retrievers are skipped. To enable:
1. Get API key from https://makersuite.google.com/app/apikey
2. Create `.env` file with `GOOGLE_API_KEY=your_key`
3. Restart pipeline

### "No documents loaded"
Check `config.yaml`:
- Verify `json_path` points to valid file
- Check `sample_ids` match dataset
- Remove restrictive `limit` values

### Retriever returns empty results
- Some retrievers may not work on very small datasets
- Try increasing `limit` in config
- Check that questions are relevant to documents

---

## 📝 Dependencies

**Core:**
- `langchain` - Framework for retrievers
- `langchain-community` - Community retrievers
- `langchain-google-genai` - Gemini LLM (optional)
- `sentence-transformers` - Embeddings
- `faiss-cpu` - Vector search
- `PyYAML` - Config parsing
- `python-dotenv` - Environment variables

**Optional:**
- `langchain-classic` - Advanced retrievers (Ensemble, MultiQuery, etc.)

Install all with:
```bash
uv sync
```

---

## 📄 License

See `locomo/LICENSE.txt` for Locomo dataset license.

---

## 🎓 Citation

If you use this code or the Locomo dataset, please cite:
```
[Locomo Dataset Citation - Add from locomo/README.MD]
```

---

## 💡 Tips

1. **Start small**: Test on 1-2 conversations first (`limit: 2`)
2. **Basic first**: Test non-LLM retrievers before setting up API keys
3. **Compare incrementally**: Add retrievers one at a time to understand differences
4. **Monitor resources**: Vector-based retrievers use more memory
5. **Check timestamps**: Results include timestamps to avoid overwriting

---

## 📞 Support

For issues or questions:
1. Check this README
2. Review `config.yaml` comments
3. Check code comments in `dataloader.py`, `retriever.py`, `run_pipeline.py`

---

**Last Updated:** October 2025

