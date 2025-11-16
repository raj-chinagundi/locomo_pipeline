# Retrieval Evaluation Project

A comprehensive evaluation framework for comparing multiple retrieval methods (BM25, TF-IDF, SVM, FAISS, Time-Weighted) on two datasets: Locomo and LongMemEval.

## 📋 Project Overview

This project evaluates 5 different retrieval methods:
- **BM25**: Keyword-based retrieval using Okapi BM25
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **SVM**: Support Vector Machine-based retrieval
- **FAISS**: Dense vector search using HuggingFace embeddings
- **Time-Weighted**: LangChain's time-weighted vector store retriever

**Metrics Evaluated**: Recall@5 and NDCG@5

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd NLP-Project-Final
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup

Ensure your data files are in the `data/` directory:
- `locomo10.json` (or `locomo1.json` for smaller test)
- `longmemeval_s_cleaned.json` (or `longmem_small.json` for testing)

## 📁 Project Structure

```
NLP-Project-Final/
├── data/                          # Dataset files
│   ├── locomo10.json
│   ├── longmemeval_s_cleaned.json
│   └── ...
├── results/                       # Generated results
│   ├── locomo_results.json
│   ├── longmem_results.json
│   ├── locomo_analysis.csv
│   ├── longmem_analysis.csv
│   └── *.png (visualizations)
├── dataloader.py                  # Data loading utilities
├── orchestrator_approach.py       # Main evaluation for Locomo dataset
├── orchestrator_longmem.py        # Main evaluation for LongMemEval dataset
├── analyze_locomo.py              # Generate CSV analysis for Locomo
├── analyze_longmem.py             # Generate CSV analysis for LongMemEval
├── visualize_locomo.py            # Generate visualizations for Locomo
├── visualize_longmem.py           # Generate visualizations for LongMemEval
└── requirements.txt               # Python dependencies
```

## 🔧 Usage

### 1. Run Evaluation Pipelines

**Evaluate on Locomo dataset:**
```bash
python locomo_approach.py
```
This will:
- Load and process Locomo data
- Evaluate all 5 retrievers
- Calculate Recall@5 and NDCG@5 metrics
- Save results to `results/locomo_results.json`
- Display summary statistics

**Evaluate on LongMemEval dataset:**
```bash
python longmem_approach.py
```
This will:
- Load and process LongMemEval data
- Build indexes per question (isolated search space)
- Evaluate all 5 retrievers
- Calculate Recall@5 and NDCG@5 metrics
- Save results to `results/longmem_results.json`
- Display summary statistics

**Note**: For LongMemEval, you can limit the number of questions by editing `LIMIT_Q` variable in `orchestrator_longmem.py`:
```python
LIMIT_Q = 100  # Process only 100 questions
LIMIT_Q = None  # Process all questions (default)
```

### 2. Generate Analysis CSVs

**For Locomo:**
```bash
python analyze_locomo.py
```
Creates `locomo_analysis.csv` with:
- Each question as a row
- Binary indicators (1/0) for each retriever finding evidence
- `any_retriever_found` column

**For LongMemEval:**
```bash
python analyze_longmem.py
```
Creates `longmem_analysis.csv` with:
- Each question as a row
- Binary indicators (1/0) for each retriever finding answer_session_ids
- `any_retriever_found` column

### 3. Generate Visualizations

**For Locomo:**
```bash
python visualize_locomo.py
```
Generates 4 visualization files:
- `locomo_overall_performance.png` - Overall metrics comparison
- `locomo_category_heatmap.png` - Performance by category (heatmap)
- `locomo_category_comparison.png` - Performance by category (bar chart)
- `locomo_distribution.png` - Score distributions

**For LongMemEval:**
```bash
python visualize_longmem.py
```
Generates 4 visualization files:
- `longmem_overall_performance.png` - Overall metrics comparison
- `longmem_question_type_heatmap.png` - Performance by question type (heatmap)
- `longmem_question_type_comparison.png` - Performance by question type (bar chart)
- `longmem_distribution.png` - Score distributions

## 📊 Output Files

### JSON Results
- `results/locomo_results.json`: Complete evaluation results with Top5 retrieved IDs for each question
- `results/longmem_results.json`: Complete evaluation results with Top5 retrieved IDs for each question

### CSV Analysis
- `locomo_analysis.csv`: Binary matrix showing which retrievers found evidence for each question
- `longmem_analysis.csv`: Binary matrix showing which retrievers found answer_session_ids for each question

### Visualizations
- Performance comparison charts
- Category/Question type heatmaps
- Distribution plots

## 📈 Understanding the Results

### Metrics Explained

- **Recall@5**: Proportion of relevant documents found in top 5 retrieved results
- **NDCG@5**: Normalized Discounted Cumulative Gain at rank 5 (accounts for ranking quality)
- **Success Rate**: Percentage of questions where at least one relevant document was retrieved

### Dataset Details

**Locomo**:
- 1538 questions
- Categories: 1-5 (different question types)
- Evidence format: `D1:3` (dialogue IDs)

**LongMemEval**:
- 500 questions
- Question types: knowledge-update, multi-session, single-session-assistant, single-session-preference, single-session-user, temporal-reasoning
- Evidence format: `answer_session_ids` (session IDs)

## 🔍 Key Features

1. **Per-Question Indexing (LongMemEval)**: Each question has its own isolated search space, ensuring no cross-question retrieval
2. **User Message Only Indexing**: Only user messages are indexed as documents (not assistant responses)
3. **Comprehensive Evaluation**: Multiple metrics and visualizations for thorough analysis
4. **Category/Type Analysis**: Performance breakdown by question category or type

## 🛠️ Dependencies

See `requirements.txt` for complete list. Main packages:
- `langchain-core`, `langchain-community`, `langchain-classic`
- `rank-bm25`
- `scikit-learn`
- `matplotlib`, `seaborn`, `numpy`

## 📝 Notes

- The first run may take time to download embeddings model (sentence-transformers/all-MiniLM-L6-v2)
- LongMemEval evaluation builds indexes per question, which can be time-consuming for large datasets
- Results are saved incrementally, so you can analyze partial results if needed

## 🐛 Troubleshooting

**Import errors**: Make sure virtual environment is activated and all dependencies are installed
**Memory issues**: Use `LIMIT_Q` to process fewer questions at a time
**Encoding errors**: Ensure data files are UTF-8 encoded

