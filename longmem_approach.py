from dataloader import CustomData
import sys
import os
import logging
import warnings
import json

# Limit number of questions to process (set to None to process all questions)
LIMIT_Q = None  # Set to a number like 100 to limit, or None for all questions

print("Loading data...", flush=True)
longmem_data = CustomData(
    path_locomo=None,
    path_longmem_eval="data/longmemeval_s_cleaned.json"
)

print("Processing longmem data...", flush=True)
longmem_processed = longmem_data.process_longmem_eval()

# Process the longmem data into documents similar to locomo format
# Split each session into individual messages (like locomo splits dialogues)
print("Creating documents from sessions...", flush=True)
documents_dict = []
for collection_key in longmem_processed:
    session_entries = longmem_processed[collection_key]
    for entry in session_entries:
        haystack_session = entry.get('haystack_session', [])
        haystack_date = entry.get('haystack_date', '')
        haystack_session_id = entry.get('haystack_session_id', '')
        
        # Create a separate document for each USER message in the session
        # Each user message is a document, indexed with its haystack_session_id
        # When retrieving, we get session_ids which are compared to evidence_ids
        if haystack_session and isinstance(haystack_session, list):
            for message in haystack_session:
                if isinstance(message, dict):
                    role = message.get('role', '')
                    content = message.get('content', '')
                    if content and role == 'user':  # Only index user messages
                        doc_meta = {
                            "text": content,
                            "timestamp": haystack_date,
                            "session_id": haystack_session_id,  # This session_id is compared to evidence_ids
                            "collection_key": collection_key,
                            "role": role
                        }
                        documents_dict.append(doc_meta)

print(f"Created {len(documents_dict)} documents", flush=True)

# Load QA data (limit if LIMIT_Q is set, otherwise process all)
if LIMIT_Q is not None:
    print(f"Loading QA data (limiting to first {LIMIT_Q} questions)...", flush=True)
    longmem_qa_full = longmem_data.load_source_qa_longmem_eval()
    longmem_qa = {}
    question_count = 0
    for question_id, questions in longmem_qa_full.items():
        if question_count >= LIMIT_Q:
            break
        for q in questions:
            if question_count >= LIMIT_Q:
                break
            if question_id not in longmem_qa:
                longmem_qa[question_id] = []
            longmem_qa[question_id].append(q)
            question_count += 1
        if question_count >= LIMIT_Q:
            break
    print(f"Limited to {question_count} questions", flush=True)
else:
    print("Loading QA data (processing all questions)...", flush=True)
    longmem_qa = longmem_data.load_source_qa_longmem_eval()
    total_q = sum(len(questions) for questions in longmem_qa.values())
    print(f"Loaded {total_q} questions", flush=True)

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import SVMRetriever
from langchain_classic.retrievers.time_weighted_retriever import TimeWeightedVectorStoreRetriever
from pprint import pprint
import re
import math

# Suppress LangChain warnings and verbose output
os.environ["LANGCHAIN_VERBOSE"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)
logging.getLogger("langchain_classic").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*HuggingFaceEmbeddings.*")

def preprocess(text):
    text = text.lower()
    # keep words and numbers, replace others with space
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = text.split()
    return tokens

# Create mapping from collection_key to documents for question-wise processing
print("Creating collection_key to documents mapping...", flush=True)
collection_key_to_docs = {}
for doc in documents_dict:
    collection_key = doc.get('collection_key', '')
    if collection_key not in collection_key_to_docs:
        collection_key_to_docs[collection_key] = []
    collection_key_to_docs[collection_key].append(doc)

print(f"Created {len(collection_key_to_docs)} collections", flush=True)

# Create mapping from question_id to question metadata (question_type, question_date)
print("Creating question metadata mapping...", flush=True)
question_metadata = {}
for item in longmem_data.longmem_eval_data:
    question_id = item.get('question_id')
    if question_id:
        question_metadata[question_id] = {
            'question_type': item.get('question_type', ''),
            'question_date': item.get('question_date', '')
        }

# Load embeddings model once (reused for each question)
print("Loading embeddings model (this may take a moment)...", flush=True)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def _parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object. Format: '2023/05/20 (Sat) 02:21'"""
    if not timestamp_str:
        return datetime(2000, 1, 1)
    try:
        parts = timestamp_str.split()
        if len(parts) >= 3:
            date_part = parts[0]  # "2023/05/20"
            time_part = parts[2]  # "02:21"
            date_obj = datetime.strptime(date_part, "%Y/%m/%d")
            time_parts = time_part.split(":")
            if len(time_parts) == 2:
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                return date_obj.replace(hour=hour, minute=minute)
        return datetime(2000, 1, 1)
    except:
        return datetime(2000, 1, 1)

def build_indexes_for_question(question_docs):
    """
    Build all indexes for a specific question's documents.
    
    Args:
        question_docs: List of document dictionaries for this question
    
    Returns:
        Dictionary with all indexes: bm25, tfidf_vectorizer, tfidf_matrix, svm_retriever, langchain_docs, vectorstore, time_weighted_retriever
    """
    if not question_docs:
        return None
    
    # Extract text
    question_texts = [doc['text'] for doc in question_docs]
    
    # Build BM25
    question_tokenized = [preprocess(text) for text in question_texts]
    bm25 = BM25Okapi(question_tokenized)
    
    # Build TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(question_texts)
    
    # Build LangChain documents (needed for SVM retriever)
    langchain_docs = []
    for i, doc in enumerate(question_docs):
        timestamp_str = doc.get('timestamp', '')
        last_accessed = _parse_timestamp(timestamp_str)
        langchain_doc = Document(
            page_content=doc['text'],
            metadata={
                'buffer_idx': i,
                'last_accessed_at': last_accessed,
                'session_id': doc.get('session_id', ''),
                'timestamp': timestamp_str,
                'original_dict': doc
            }
        )
        langchain_docs.append(langchain_doc)
    
    # Build SVM retriever
    if len(question_texts) > 0:
        svm_retriever = SVMRetriever.from_texts(question_texts, embeddings)
    else:
        svm_retriever = None
    
    # Build FAISS vectorstore
    if len(langchain_docs) > 0:
        vectorstore = FAISS.from_documents(langchain_docs, embeddings)
        time_weighted_retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore,
            memory_stream=langchain_docs,
            search_kwargs={"k": min(len(langchain_docs), 50)},
        )
    else:
        vectorstore = None
        time_weighted_retriever = None
    
    return {
        'bm25': bm25,
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'svm_retriever': svm_retriever,
        'question_docs': question_docs,
        'question_texts': question_texts,
        'langchain_docs': langchain_docs,
        'vectorstore': vectorstore,
        'time_weighted_retriever': time_weighted_retriever
    }

def retrieve_topk(query, k=3, indexes=None):
    """
    BM25 retrieval for question-specific documents.
    
    Args:
        query: Search query string
        k: Number of top results to return
        indexes: Dictionary with indexes built for this question
    
    Returns:
        top_n: List of full dictionary objects (with all metadata)
        topk_with_scores: List of tuples (index, score, full_dict)
    """
    if indexes is None or indexes['bm25'] is None:
        return [], []
    
    k = max(1, min(k, len(indexes['question_docs'])))
    tokenized_query = preprocess(query)
    scores = indexes['bm25'].get_scores(tokenized_query)
    topk_idx = (-scores).argsort()[:k]
    top_n = [indexes['question_docs'][i] for i in topk_idx]
    topk_with_scores = [(i, float(scores[i]), indexes['question_docs'][i]) for i in topk_idx]
    return top_n, topk_with_scores

def retrieve_topk_tfidf(query, k=3, indexes=None):
    """
    TF-IDF retrieval for question-specific documents.
    
    Args:
        query: Search query string
        k: Number of top results to return
        indexes: Dictionary with indexes built for this question
    
    Returns:
        top_n: List of full dictionary objects (with all metadata)
        topk_with_scores: List of tuples (index, score, full_dict)
    """
    if indexes is None or indexes['tfidf_vectorizer'] is None:
        return [], []
    
    k = max(1, min(k, len(indexes['question_docs'])))
    query_vec = indexes['tfidf_vectorizer'].transform([query])
    scores = cosine_similarity(query_vec, indexes['tfidf_matrix'])[0]
    topk_idx = scores.argsort()[::-1][:k]
    top_n = [indexes['question_docs'][i] for i in topk_idx]
    topk_with_scores = [(i, float(scores[i]), indexes['question_docs'][i]) for i in topk_idx]
    return top_n, topk_with_scores

def retrieve_topk_svm(query, k=3, indexes=None):
    """
    SVM retrieval for question-specific documents.
    
    Args:
        query: Search query string
        k: Number of top results to return
        indexes: Dictionary with indexes built for this question
    
    Returns:
        top_n: List of full dictionary objects (with all metadata)
        topk_with_scores: List of tuples (index, score, full_dict) - score is placeholder
    """
    if indexes is None or indexes['svm_retriever'] is None:
        return [], []
    
    k = max(1, min(k, len(indexes['question_docs'])))
    # SVM retriever returns LangChain Documents
    retrieved_docs = indexes['svm_retriever'].invoke(query)
    # Limit to k results
    retrieved_docs = retrieved_docs[:k]
    
    # Map retrieved documents back to original question_docs by matching text
    # Create a text-to-doc mapping for quick lookup
    text_to_doc = {doc['text']: doc for doc in indexes['question_docs']}
    
    top_n = []
    for doc in retrieved_docs:
        # Try to find matching original document by text content
        if doc.page_content in text_to_doc:
            top_n.append(text_to_doc[doc.page_content])
        else:
            # Fallback: create dict from document content
            top_n.append({
                'text': doc.page_content,
                'session_id': doc.metadata.get('session_id', '') if hasattr(doc, 'metadata') else '',
                'timestamp': doc.metadata.get('timestamp', '') if hasattr(doc, 'metadata') else ''
            })
    
    # Create placeholder scores (SVM doesn't provide explicit scores)
    topk_with_scores = [(i, 1.0, top_n[i]) for i in range(len(top_n))]
    return top_n, topk_with_scores

def retrieve_topk_faiss(query, k=3, indexes=None):
    """
    FAISS retrieval for question-specific documents.
    
    Args:
        query: Search query string
        k: Number of top results to return
        indexes: Dictionary with indexes built for this question
    
    Returns:
        top_n: List of full dictionary objects (with all metadata)
        topk_with_scores: List of tuples (index, score, full_dict)
    """
    if indexes is None or indexes['vectorstore'] is None:
        return [], []
    
    k = max(1, min(k, len(indexes['question_docs'])))
    results = indexes['vectorstore'].similarity_search_with_score(query, k=k)
    top_n = []
    topk_with_scores = []
    for doc, score in results:
        original_dict = doc.metadata.get('original_dict', {})
        buffer_idx = doc.metadata.get('buffer_idx', 0)
        top_n.append(original_dict)
        topk_with_scores.append((buffer_idx, float(score), original_dict))
    return top_n, topk_with_scores

def retrieve_topk_time_weighted(query, k=3, indexes=None):
    """
    Time-weighted retrieval for question-specific documents.
    
    Args:
        query: Search query string
        k: Number of top results to return
        indexes: Dictionary with indexes built for this question
    
    Returns:
        top_n: List of full dictionary objects (with all metadata)
    """
    if indexes is None or indexes['time_weighted_retriever'] is None:
        return []
    
    k = max(1, min(k, len(indexes['question_docs'])))
    retrieved_docs = indexes['time_weighted_retriever'].invoke(query)[:k]
    top_n = [doc.metadata.get('original_dict', {}) for doc in retrieved_docs]
    return top_n

def calculate_recall_at_k(retrieved_session_ids, relevant_session_ids, k=5):
    """
    Calculate Recall@k.
    
    Recall@k = (# of unique relevant documents in top k retrieved) / (Total # of relevant documents)
    
    Args:
        retrieved_session_ids: List of session_ids from top k retrieved documents
        relevant_session_ids: Set of relevant session_ids (ground truth)
        k: Number of top results to consider
    
    Returns:
        recall@k: Float value between 0 and 1
    """
    if not relevant_session_ids:
        return 0.0
    
    top_k_retrieved = retrieved_session_ids[:k]
    # Count UNIQUE relevant session_ids (not total occurrences)
    unique_retrieved = set(top_k_retrieved)
    relevant_in_top_k = len(unique_retrieved & relevant_session_ids)
    
    return relevant_in_top_k / len(relevant_session_ids)

def calculate_ndcg_at_k(retrieved_session_ids, relevant_session_ids, k=5):
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain).
    
    DCG@k = sum(i=1 to k) of (2^rel_i - 1) / log2(i + 1)
    where rel_i = 1 if document at rank i is relevant, else 0
    Note: Each unique relevant session_id is only counted once (at its first occurrence)
    
    IDCG@k = DCG of perfect ranking (all relevant docs in top positions)
    NDCG@k = DCG@k / IDCG@k
    
    Args:
        retrieved_session_ids: List of session_ids from top k retrieved documents
        relevant_session_ids: Set of relevant session_ids (ground truth)
        k: Number of top results to consider
    
    Returns:
        ndcg@k: Float value between 0 and 1
    """
    if not relevant_session_ids:
        return 0.0
    
    # Calculate DCG@k - only count each unique relevant session_id once (at first occurrence)
    dcg = 0.0
    top_k_retrieved = retrieved_session_ids[:k]
    seen_relevant = set()  # Track which relevant session_ids we've already counted
    for i, session_id in enumerate(top_k_retrieved, start=1):
        if session_id in relevant_session_ids and session_id not in seen_relevant:
            # First time seeing this relevant session_id - count it
            rel_i = 1
            seen_relevant.add(session_id)
        else:
            rel_i = 0
        dcg += (2**rel_i - 1) / math.log2(i + 1)
    
    # Calculate IDCG@k (perfect ranking: all relevant docs first)
    num_relevant = len(relevant_session_ids)
    idcg = 0.0
    for i in range(1, min(k, num_relevant) + 1):
        idcg += (2**1 - 1) / math.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

# QA data already loaded and limited above

# Initialize accumulators for each retriever
accumulated_metrics = {
    'BM25': {'recall@5': [], 'ndcg@5': []},
    'TF-IDF': {'recall@5': [], 'ndcg@5': []},
    'SVM': {'recall@5': [], 'ndcg@5': []},
    'FAISS': {'recall@5': [], 'ndcg@5': []},
    'Time-Weighted': {'recall@5': [], 'ndcg@5': []}
}

# Initialize results storage
results = []

k = 5
total_questions = 0
progress_interval = 10  # Print progress every N questions

# Count total questions first for progress display
total_questions_count = sum(len(questions) for questions in longmem_qa.values())
print(f"Starting evaluation of {total_questions_count} questions...")
print("Processing questions...\n")
sys.stdout.flush()

# Process questions - build indexes per question and aggregate results
# Each question gets its own search space built from its question_id's documents
for question_id, questions in longmem_qa.items():
    # Get collection_key and documents for this question_id
    collection_key = f"vectordb_collection_{question_id}"
    question_docs = collection_key_to_docs.get(collection_key, [])
    
    if not question_docs:
        continue
    
    # Process each question individually with its own search space
    for question_data in questions:
        query = question_data['question']
        evidence_ids = set(question_data['evidence_ids'])
        
        if not evidence_ids:
            continue
        
        total_questions += 1
        
        # Build indexes for THIS question's documents only (fresh search space)
        indexes = build_indexes_for_question(question_docs)
        if indexes is None:
            continue
        
        # Print progress after every question (with overwrite)
        print(f"\rProcessing question {total_questions}/{total_questions_count} (docs: {len(question_docs)})...", end='', flush=True)
        
        # BM25 retrieval
        top_texts_bm25, _ = retrieve_topk(query, k=k, indexes=indexes)
        retrieved_session_ids_bm25 = [doc.get('session_id', '') for doc in top_texts_bm25]
        accumulated_metrics['BM25']['recall@5'].append(calculate_recall_at_k(retrieved_session_ids_bm25, evidence_ids, k=k))
        accumulated_metrics['BM25']['ndcg@5'].append(calculate_ndcg_at_k(retrieved_session_ids_bm25, evidence_ids, k=k))
        
        # TF-IDF retrieval
        top_texts_tfidf, _ = retrieve_topk_tfidf(query, k=k, indexes=indexes)
        retrieved_session_ids_tfidf = [doc.get('session_id', '') for doc in top_texts_tfidf]
        accumulated_metrics['TF-IDF']['recall@5'].append(calculate_recall_at_k(retrieved_session_ids_tfidf, evidence_ids, k=k))
        accumulated_metrics['TF-IDF']['ndcg@5'].append(calculate_ndcg_at_k(retrieved_session_ids_tfidf, evidence_ids, k=k))
        
        # SVM retrieval
        top_texts_svm, _ = retrieve_topk_svm(query, k=k, indexes=indexes)
        retrieved_session_ids_svm = [doc.get('session_id', '') for doc in top_texts_svm]
        accumulated_metrics['SVM']['recall@5'].append(calculate_recall_at_k(retrieved_session_ids_svm, evidence_ids, k=k))
        accumulated_metrics['SVM']['ndcg@5'].append(calculate_ndcg_at_k(retrieved_session_ids_svm, evidence_ids, k=k))
        
        # FAISS retrieval
        top_texts_faiss, _ = retrieve_topk_faiss(query, k=k, indexes=indexes)
        retrieved_session_ids_faiss = [doc.get('session_id', '') for doc in top_texts_faiss]
        accumulated_metrics['FAISS']['recall@5'].append(calculate_recall_at_k(retrieved_session_ids_faiss, evidence_ids, k=k))
        accumulated_metrics['FAISS']['ndcg@5'].append(calculate_ndcg_at_k(retrieved_session_ids_faiss, evidence_ids, k=k))
        
        # Time-weighted retrieval
        top_texts_time = retrieve_topk_time_weighted(query, k=k, indexes=indexes)
        retrieved_session_ids_time = [doc.get('session_id', '') for doc in top_texts_time]
        accumulated_metrics['Time-Weighted']['recall@5'].append(calculate_recall_at_k(retrieved_session_ids_time, evidence_ids, k=k))
        accumulated_metrics['Time-Weighted']['ndcg@5'].append(calculate_ndcg_at_k(retrieved_session_ids_time, evidence_ids, k=k))
        
        # Store results on the fly
        metadata = question_metadata.get(question_id, {})
        result_entry = {
            "question_id": question_id,
            "question_type": metadata.get('question_type', ''),
            "question": query,
            "question_date": metadata.get('question_date', ''),
            "answer": question_data.get('answer', ''),
            "answer_session_ids": list(evidence_ids),
            "Retriever": {
                "BM25": {
                    "Top5_retrieved_ids": retrieved_session_ids_bm25[:k]
                },
                "TF-IDF": {
                    "Top5_retrieved_ids": retrieved_session_ids_tfidf[:k]
                },
                "SVM": {
                    "Top5_retrieved_ids": retrieved_session_ids_svm[:k]
                },
                "FAISS": {
                    "Top5_retrieved_ids": retrieved_session_ids_faiss[:k]
                },
                "Time-Weighted": {
                    "Top5_retrieved_ids": retrieved_session_ids_time[:k]
                }
            }
        }
        results.append(result_entry)
        
        # Print detailed progress periodically
        if total_questions % progress_interval == 0:
            # Calculate running averages
            running_avg = {}
            for retriever_name, metrics in accumulated_metrics.items():
                if metrics['recall@5']:
                    avg_recall = sum(metrics['recall@5']) / len(metrics['recall@5'])
                    avg_ndcg = sum(metrics['ndcg@5']) / len(metrics['ndcg@5'])
                    running_avg[retriever_name] = {'recall@5': avg_recall, 'ndcg@5': avg_ndcg}
            
            # Print progress update with metrics
            print()  # New line
            print(f"Progress: {total_questions}/{total_questions_count} questions processed")
            if total_questions % (progress_interval * 5) == 0:  # Full table every 50 questions
                print(f"Running Averages (after {total_questions} questions):")
                print("=" * 50)
                print(f"{'Retriever':<20} {'Recall@5':<15} {'NDCG@5':<15}")
                print("=" * 50)
                for retriever_name, metrics in running_avg.items():
                    print(f"{retriever_name:<20} {metrics['recall@5']:<15.4f} {metrics['ndcg@5']:<15.4f}")
                print("=" * 50)
            print()  # Blank line
            sys.stdout.flush()

# Print final progress line
print(f"\rProcessing question {total_questions}/{total_questions_count}... Complete!    ")
print()
sys.stdout.flush()

# Calculate average metrics for each retriever
average_results = {}
for retriever_name, metrics in accumulated_metrics.items():
    avg_recall = sum(metrics['recall@5']) / len(metrics['recall@5']) if metrics['recall@5'] else 0.0
    avg_ndcg = sum(metrics['ndcg@5']) / len(metrics['ndcg@5']) if metrics['ndcg@5'] else 0.0
    average_results[retriever_name] = {
        'recall@5': avg_recall,
        'ndcg@5': avg_ndcg
    }

# Display final results in tabular format
print("=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"Total Questions Evaluated: {total_questions}")
print("=" * 50)
print(f"{'Retriever':<20} {'Recall@5':<15} {'NDCG@5':<15}")
print("=" * 50)
for retriever_name, metrics in average_results.items():
    print(f"{retriever_name:<20} {metrics['recall@5']:<15.4f} {metrics['ndcg@5']:<15.4f}")
print("=" * 50)

# Save results to JSON file
output_file = "longmem_results.json"
print(f"\nSaving results to {output_file}...", flush=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {output_file} ({len(results)} questions)")

