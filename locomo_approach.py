from dataloader import CustomData

locomo_data = CustomData(
    path_locomo="data/locomo10.json",
    path_longmem_eval=None
)

locomo_processed = locomo_data.process_locomo()

# for d in locomo_processed:
#     print(locomo_processed[d])
#     print("-"*100)
#     break

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
import json

def preprocess(text):
    text = text.lower()
    # keep words and numbers, replace others with space
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = text.split()
    return tokens

# Step 1: Store full metadata dictionaries (we'll retrieve these later)
documents_dict = [docs for key in locomo_processed for docs in locomo_processed[key]]

# Step 2: Extract ONLY text field for keyword search
documents_text = [docs['text'] for docs in documents_dict]

# Step 3: Tokenize text for BM25 (keyword matching happens here)
documents_tokenized = [preprocess(text) for text in documents_text]

# Step 4: Initialize BM25 with tokenized text only (keyword search on text)
bm25 = BM25Okapi(documents_tokenized)

# Step 5: Initialize TF-IDF with raw text (TF-IDF handles tokenization internally)
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(documents_text)

def retrieve_topk(query, k=3):
    """
    Search query against text field only, but return full metadata.
    
    Args:
        query: Search query string
        k: Number of top results to return
    
    Returns:
        top_n: List of full dictionary objects (with all metadata)
        topk_with_scores: List of tuples (index, score, full_dict)
    """
    # Ensure k is valid
    k = max(1, min(k, len(documents_text)))
    
    # Tokenize query for BM25 matching
    tokenized_query = preprocess(query)
    
    # Get BM25 scores for all documents (matching happens on text only)
    scores = bm25.get_scores(tokenized_query)  # numpy array
    
    # Get indices of top k documents (sorted by score)
    topk_idx = (-scores).argsort()[:k]
    
    # Retrieve full metadata dictionaries using the indices
    # This is the key: we searched on text, but return full dicts
    top_n = [documents_dict[i] for i in topk_idx]
    
    # Also return with scores for detailed results
    topk_with_scores = [(i, float(scores[i]), documents_dict[i]) for i in topk_idx]
    
    return top_n, topk_with_scores

def retrieve_topk_tfidf(query, k=3):
    """
    TF-IDF retrieval: search query against text field only, but return full metadata.
    
    Args:
        query: Search query string
        k: Number of top results to return
    
    Returns:
        top_n: List of full dictionary objects (with all metadata)
        topk_with_scores: List of tuples (index, score, full_dict)
    """
    # Ensure k is valid
    k = max(1, min(k, len(documents_text)))
    
    # Transform query to TF-IDF vector
    query_vec = tfidf_vectorizer.transform([query])
    
    # Compute cosine similarity scores
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    
    # Get indices of top k documents (sorted by score, descending)
    topk_idx = scores.argsort()[::-1][:k]
    
    # Retrieve full metadata dictionaries using the indices
    top_n = [documents_dict[i] for i in topk_idx]
    
    # Return with scores for detailed results
    topk_with_scores = [(i, float(scores[i]), documents_dict[i]) for i in topk_idx]
    
    return top_n, topk_with_scores

def retrieve_topk_svm(query, k=3):
    """
    SVM retrieval: search query against text field, return full metadata.
    
    Args:
        query: Search query string
        k: Number of top results to return
    
    Returns:
        top_n: List of full dictionary objects (with all metadata)
        topk_with_scores: List of tuples (index, score, full_dict) - score is placeholder
    """
    # Ensure k is valid
    k = max(1, min(k, len(documents_text)))
    
    # SVM retriever returns LangChain Documents
    retrieved_docs = svm_retriever.invoke(query)
    # Limit to k results
    retrieved_docs = retrieved_docs[:k]
    
    # Map retrieved documents back to original documents_dict by matching text
    # Create a text-to-doc mapping for quick lookup
    text_to_doc = {doc['text']: doc for doc in documents_dict}
    
    top_n = []
    for doc in retrieved_docs:
        # Try to find matching original document by text content
        if doc.page_content in text_to_doc:
            top_n.append(text_to_doc[doc.page_content])
        else:
            # Fallback: create dict from document content
            top_n.append({
                'text': doc.page_content,
                'dia_id': doc.metadata.get('dia_id', '') if hasattr(doc, 'metadata') else '',
                'timestamp': doc.metadata.get('timestamp', '') if hasattr(doc, 'metadata') else ''
            })
    
    # Create placeholder scores (SVM doesn't provide explicit scores)
    topk_with_scores = [(i, 1.0, top_n[i]) for i in range(len(top_n))]
    return top_n, topk_with_scores

def _parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object. Format: '1:56 pm on 8 May, 2023'"""
    if not timestamp_str:
        # Use very old date if timestamp is missing (will have high time decay)
        return datetime(2000, 1, 1)
    try:
        return datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
    except:
        try:
            return datetime.strptime(timestamp_str, "%I:%M %p on %d %B %Y")
        except:
            # Use very old date if parsing fails (will have high time decay)
            return datetime(2000, 1, 1)

# Step 7: Create LangChain Documents with metadata for time-weighted retrieval
langchain_documents = []
for i, doc in enumerate(documents_dict):
    timestamp_str = doc.get('timestamp', '')
    last_accessed = _parse_timestamp(timestamp_str)
    
    langchain_doc = Document(
        page_content=doc['text'],
        metadata={
            'buffer_idx': i,
            'last_accessed_at': last_accessed,
            'speaker': doc.get('speaker', ''),
            'dia_id': doc.get('dia_id', ''),
            'timestamp': timestamp_str,
            'original_dict': doc  # Store original dict for retrieval
        }
    )
    langchain_documents.append(langchain_doc)

# Step 8: Initialize embeddings, vector store, and SVM retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(langchain_documents, embeddings)

# Step 9: Initialize SVM retriever
svm_retriever = SVMRetriever.from_texts(documents_text, embeddings)

def retrieve_topk_faiss(query, k=3):
    """
    FAISS retrieval: search query against text field using embeddings, return full metadata.
    
    Args:
        query: Search query string
        k: Number of top results to return
    
    Returns:
        top_n: List of full dictionary objects (with all metadata)
        topk_with_scores: List of tuples (index, score, full_dict)
        Note: score is L2 distance (smaller = more similar)
    """
    # Ensure k is valid
    k = max(1, min(k, len(langchain_documents)))
    
    # Search with scores using FAISS
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    # Extract original dictionaries and create results
    top_n = []
    topk_with_scores = []
    
    for doc, score in results:
        original_dict = doc.metadata.get('original_dict', {})
        buffer_idx = doc.metadata.get('buffer_idx', 0)
        top_n.append(original_dict)
        # Score is L2 distance (smaller = more similar)
        topk_with_scores.append((buffer_idx, float(score), original_dict))
    
    return top_n, topk_with_scores

# Step 9: Initialize TimeWeightedVectorStoreRetriever
# search_kwargs k should be large enough to support any requested k
time_weighted_retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    memory_stream=langchain_documents,
    search_kwargs={"k": min(len(langchain_documents), 50)},  # Retrieve enough docs for time weighting
)

def retrieve_topk_time_weighted(query, k=3):
    """
    Time-weighted retrieval using LangChain's TimeWeightedVectorStoreRetriever.
    Combines vector similarity with time decay.
    
    Args:
        query: Search query string
        k: Number of top results to return
    
    Returns:
        top_n: List of full dictionary objects (with all metadata)
    """
    # Ensure k is valid
    k = max(1, min(k, len(langchain_documents)))
    
    # Retrieve documents using LangChain's time-weighted retriever
    retrieved_docs = time_weighted_retriever.invoke(query)[:k]  # Limit to k results
    
    # Extract original dictionaries
    top_n = []
    for doc in retrieved_docs:
        original_dict = doc.metadata.get('original_dict', {})
        top_n.append(original_dict)
    
    return top_n

def calculate_recall_at_k(retrieved_dia_ids, relevant_dia_ids, k=5):
    """
    Calculate Recall@k.
    
    Recall@k = (# of unique relevant documents in top k retrieved) / (Total # of relevant documents)
    
    Args:
        retrieved_dia_ids: List of dia_ids from top k retrieved documents
        relevant_dia_ids: Set of relevant dia_ids (ground truth)
        k: Number of top results to consider
    
    Returns:
        recall@k: Float value between 0 and 1
    """
    if not relevant_dia_ids:
        return 0.0
    
    top_k_retrieved = retrieved_dia_ids[:k]
    # Count UNIQUE relevant dia_ids (not total occurrences)
    unique_retrieved = set(top_k_retrieved)
    relevant_in_top_k = len(unique_retrieved & relevant_dia_ids)
    
    return relevant_in_top_k / len(relevant_dia_ids)

def calculate_ndcg_at_k(retrieved_dia_ids, relevant_dia_ids, k=5):
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain).
    
    DCG@k = sum(i=1 to k) of (2^rel_i - 1) / log2(i + 1)
    where rel_i = 1 if document at rank i is relevant, else 0
    Note: Each unique relevant dia_id is only counted once (at its first occurrence)
    
    IDCG@k = DCG of perfect ranking (all relevant docs in top positions)
    NDCG@k = DCG@k / IDCG@k
    
    Args:
        retrieved_dia_ids: List of dia_ids from top k retrieved documents
        relevant_dia_ids: Set of relevant dia_ids (ground truth)
        k: Number of top results to consider
    
    Returns:
        ndcg@k: Float value between 0 and 1
    """
    if not relevant_dia_ids:
        return 0.0
    
    # Calculate DCG@k - only count each unique relevant dia_id once (at first occurrence)
    dcg = 0.0
    top_k_retrieved = retrieved_dia_ids[:k]
    seen_relevant = set()  # Track which relevant dia_ids we've already counted
    for i, dia_id in enumerate(top_k_retrieved, start=1):
        if dia_id in relevant_dia_ids and dia_id not in seen_relevant:
            # First time seeing this relevant dia_id - count it
            rel_i = 1
            seen_relevant.add(dia_id)
        else:
            rel_i = 0
        dcg += (2**rel_i - 1) / math.log2(i + 1)
    
    # Calculate IDCG@k (perfect ranking: all relevant docs first)
    num_relevant = len(relevant_dia_ids)
    idcg = 0.0
    for i in range(1, min(k, num_relevant) + 1):
        idcg += (2**1 - 1) / math.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

# Load QA data to get all questions with evidence_ids
locomo_qa = locomo_data.load_source_qa_locomo()

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

# Process all questions across all conversations
for conv_id, questions in locomo_qa.items():
    for question_data in questions:
        query = question_data['question']
        evidence_ids = set(question_data['evidence_ids'])
        
        if not evidence_ids:
            continue
        
        total_questions += 1
        
        # BM25 retrieval
        top_texts_bm25, _ = retrieve_topk(query, k=k)
        retrieved_dia_ids_bm25 = [doc.get('dia_id', '') for doc in top_texts_bm25]
        accumulated_metrics['BM25']['recall@5'].append(calculate_recall_at_k(retrieved_dia_ids_bm25, evidence_ids, k=k))
        accumulated_metrics['BM25']['ndcg@5'].append(calculate_ndcg_at_k(retrieved_dia_ids_bm25, evidence_ids, k=k))
        
        # TF-IDF retrieval
        top_texts_tfidf, _ = retrieve_topk_tfidf(query, k=k)
        retrieved_dia_ids_tfidf = [doc.get('dia_id', '') for doc in top_texts_tfidf]
        accumulated_metrics['TF-IDF']['recall@5'].append(calculate_recall_at_k(retrieved_dia_ids_tfidf, evidence_ids, k=k))
        accumulated_metrics['TF-IDF']['ndcg@5'].append(calculate_ndcg_at_k(retrieved_dia_ids_tfidf, evidence_ids, k=k))
        
        # SVM retrieval
        top_texts_svm, _ = retrieve_topk_svm(query, k=k)
        retrieved_dia_ids_svm = [doc.get('dia_id', '') for doc in top_texts_svm]
        accumulated_metrics['SVM']['recall@5'].append(calculate_recall_at_k(retrieved_dia_ids_svm, evidence_ids, k=k))
        accumulated_metrics['SVM']['ndcg@5'].append(calculate_ndcg_at_k(retrieved_dia_ids_svm, evidence_ids, k=k))
        
        # FAISS retrieval
        top_texts_faiss, _ = retrieve_topk_faiss(query, k=k)
        retrieved_dia_ids_faiss = [doc.get('dia_id', '') for doc in top_texts_faiss]
        accumulated_metrics['FAISS']['recall@5'].append(calculate_recall_at_k(retrieved_dia_ids_faiss, evidence_ids, k=k))
        accumulated_metrics['FAISS']['ndcg@5'].append(calculate_ndcg_at_k(retrieved_dia_ids_faiss, evidence_ids, k=k))
        
        # Time-weighted retrieval
        top_texts_time = retrieve_topk_time_weighted(query, k=k)
        retrieved_dia_ids_time = [doc.get('dia_id', '') for doc in top_texts_time]
        accumulated_metrics['Time-Weighted']['recall@5'].append(calculate_recall_at_k(retrieved_dia_ids_time, evidence_ids, k=k))
        accumulated_metrics['Time-Weighted']['ndcg@5'].append(calculate_ndcg_at_k(retrieved_dia_ids_time, evidence_ids, k=k))
        
        # Store results on the fly
        result_entry = {
            "question": query,
            "answer": question_data.get('answer', ''),
            "evidence": list(evidence_ids),
            "category": question_data.get('category', ''),
            "Retriever": {
                "BM25": {
                    "Top5_retrieved_ids": retrieved_dia_ids_bm25[:k]
                },
                "TF-IDF": {
                    "Top5_retrieved_ids": retrieved_dia_ids_tfidf[:k]
                },
                "SVM": {
                    "Top5_retrieved_ids": retrieved_dia_ids_svm[:k]
                },
                "FAISS": {
                    "Top5_retrieved_ids": retrieved_dia_ids_faiss[:k]
                },
                "Time-Weighted": {
                    "Top5_retrieved_ids": retrieved_dia_ids_time[:k]
                }
            }
        }
        results.append(result_entry)

# Calculate average metrics for each retriever
average_results = {}
for retriever_name, metrics in accumulated_metrics.items():
    avg_recall = sum(metrics['recall@5']) / len(metrics['recall@5']) if metrics['recall@5'] else 0.0
    avg_ndcg = sum(metrics['ndcg@5']) / len(metrics['ndcg@5']) if metrics['ndcg@5'] else 0.0
    average_results[retriever_name] = {
        'recall@5': avg_recall,
        'ndcg@5': avg_ndcg
    }

# Display results in tabular format
print(f"\nTotal Questions Evaluated: {total_questions}")
print("=" * 50)
print(f"{'Retriever':<20} {'Recall@5':<15} {'NDCG@5':<15}")
print("=" * 50)
for retriever_name, metrics in average_results.items():
    print(f"{retriever_name:<20} {metrics['recall@5']:<15.4f} {metrics['ndcg@5']:<15.4f}")
print("=" * 50)

# Save results to JSON file
output_file = "locomo_results.json"
print(f"\nSaving results to {output_file}...", flush=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {output_file} ({len(results)} questions)")