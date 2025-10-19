# RAG Answer Generation - Quick Guide

## What's New: RAG (Retrieval-Augmented Generation)

Your pipeline now supports **end-to-end question answering** using retrieved documents!

### Before (Retrieval Only):
```
Question: "What is Caroline's identity?"
Output: Retrieved documents [D1:5, D1:3, ...]
```

### After (RAG - Retrieval + Generation):
```
Question: "What is Caroline's identity?"
Retrieved: [D1:5: "I'm a transgender woman..."]
Generated Answer: "Caroline is a transgender woman."
```

---

## How to Enable

### Step 1: Set up LLM (if not done)
```bash
# Get API key from: https://makersuite.google.com/app/apikey
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### Step 2: Edit `config.yaml`
```yaml
llm:
  enabled: true              # Enable LLM
  generate_answers: true     # Enable RAG answer generation ← NEW!
  max_context_docs: 3        # How many retrieved docs to use as context
  temperature: 0.0           # 0.0 = deterministic answers
```

### Step 3: Run Pipeline
```bash
uv run python run_pipeline.py
```

---

## What You'll See

### Console Output (Enhanced):
```
Question #1
├─ Q: What is Caroline's identity?
├─ Ground Truth Answer: Transgender woman
├─ Generated Answer: Caroline is a transgender woman.  ← NEW!
├─ Ground Truth Evidence: ['D1:5']
├─ Retrieved: ['D1:5', 'D1:3', 'D2:8']... ✓
└─ Top result from: 1:56 pm on 8 May, 2023

SUMMARY for bm25:
├─ Questions tested: 15
├─ Answers generated: 15/15                           ← NEW!
├─ Questions with ≥1 evidence: 12 (80.0%)
└─ Evidence recall: 45/60 (75.0%)
```

### Results JSON (Enhanced):
```json
{
  "question": "What is Caroline's identity?",
  "ground_truth_answer": "Transgender woman",
  "generated_answer": "Caroline is a transgender woman.",  ← NEW!
  "retrieved_documents": [
    {
      "dia_id": "D1:5",
      "speaker": "Caroline",
      "session_datetime": "1:56 pm on 8 May, 2023",
      "text": "I'm a transgender woman...",
      "is_evidence": true
    }
  ]
}
```

---

## Configuration Options

```yaml
llm:
  # Enable/disable answer generation
  generate_answers: true
  
  # Number of retrieved docs to use as context (top-k)
  max_context_docs: 3
  # - 1-2: Fast, less context
  # - 3-5: Balanced (recommended)
  # - 5-10: More context, slower
  
  # Temperature for answer generation
  temperature: 0.0
  # - 0.0: Deterministic, factual (recommended for QA)
  # - 0.5: Some variation
  # - 1.0: Creative, diverse answers
```

---

## Use Cases

### 1. **Retriever Comparison (Original)**
```yaml
llm:
  generate_answers: false  # Disabled
```
**Focus**: Which retriever finds the right documents?

### 2. **End-to-End QA Comparison (NEW!)**
```yaml
llm:
  generate_answers: true   # Enabled
```
**Focus**: Which retriever produces the best final answers?

### 3. **Both!**
Run once with RAG enabled - you get:
- ✅ Retriever performance (what docs were found)
- ✅ Answer quality (what answers were generated)
- ✅ All in one run!

---

## Performance Impact

**Without RAG** (default):
- Fast: ~1-2 seconds per retriever
- No API costs
- No internet needed (except for LLM-based retrievers)

**With RAG** (enabled):
- Slower: +2-5 seconds per question (LLM generation time)
- API costs: ~$0.001 per question with Gemini 2.5 Flash
- Requires internet connection
- **Worth it**: You get complete QA evaluation!

---

## Tips

1. **Start small**: Test on 5-10 questions first
   ```yaml
   evaluation:
     question_limit: 10
   ```

2. **Adjust context docs**: 
   - More docs = more context but slower
   - Start with `max_context_docs: 3`

3. **Compare retrievers**: Run with RAG to see which retriever produces best answers

4. **Check generated answers**: Look in results JSON to verify quality

---

## Example Workflow

```bash
# 1. Test retrieval only (fast, no API)
# config.yaml: generate_answers: false
uv run python run_pipeline.py

# 2. Enable RAG for deeper analysis
# config.yaml: generate_answers: true
uv run python run_pipeline.py

# 3. Compare results
# - retrieval_only_results.json: Which retriever finds evidence?
# - rag_results.json: Which retriever produces best answers?
```

---

## Troubleshooting

**"Answers generated: 0/15"**
- Check that `llm.enabled: true`
- Check that `llm.generate_answers: true`
- Verify GOOGLE_API_KEY in .env

**"Warning: Answer generation failed"**
- Check API key is valid
- Check internet connection
- Check retrieved_docs has results

**Generated answers are wrong**
- Try increasing `max_context_docs`
- Check if right documents were retrieved
- Try different retriever

---

## What's Next?

**Future Enhancements** (not implemented yet, but easy to add):
- Answer quality metrics (ROUGE, F1, BERTScore)
- Multiple prompt strategies
- Chain-of-thought reasoning
- Answer confidence scores
- Citation tracking (which doc each answer part came from)

---

**Ready to try it?** Enable RAG in `config.yaml` and run the pipeline! 🚀

