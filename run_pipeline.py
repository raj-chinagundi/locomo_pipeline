"""
Run Pipeline - Main execution script for retriever experiments
Clean and configurable pipeline for comparing retriever performance on Locomo dataset.
"""

import os
import json
import yaml
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from dataloader import DataLoader
from retriever import Retriever, RETRIEVER_INFO
from llm_helper import get_llm
from rate_limiter import RateLimiter
from answer_evaluator import evaluate_answer


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_timestamp(format_type: str = "datetime") -> str:
    """Generate timestamp for result files."""
    if format_type == "unix":
        return str(int(datetime.now().timestamp()))
    else:  # datetime format
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_answer(llm, question: str, retrieved_docs: List, max_context_docs: int = 3, rate_limiter: Optional[RateLimiter] = None) -> Optional[str]:
    """
    Generate natural language answer using retrieved documents (RAG).
    
    Args:
        llm: LLM instance for generation
        question: The question to answer
        retrieved_docs: List of retrieved Document objects
        max_context_docs: Maximum number of docs to use as context
        rate_limiter: Optional RateLimiter instance to prevent API rate limit errors
    
    Returns:
        Generated answer string or None if generation fails/disabled
    """
    if not llm or not retrieved_docs:
        return None
    
    try:
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:max_context_docs], 1):
            dia_id = doc.metadata.get('dia_id', '')
            speaker = doc.metadata.get('speaker', 'Unknown')
            timestamp = doc.metadata.get('session_datetime', '')
            text = doc.page_content
            
            # Concise format: dialog ID, speaker, timestamp, content
            context_parts.append(f"[{dia_id} - {speaker}, {timestamp}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for answer generation
        prompt = f"""Based on the following conversation excerpts, provide a direct and precise answer to the question.

Conversation Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the information provided in the context above
- Be SPECIFIC and PRECISE - include exact dates, numbers, names, and details
- DO NOT use vague terms like "yesterday", "recently", "a while ago", "some time", etc.
- If the context mentions a specific date, time, location, or detail - include it exactly
- If asking about a date, provide the exact date format given in the context
- Keep the answer concise but complete with all relevant specific details

Answer:"""
        
        # Wait for rate limit if limiter is provided
        if rate_limiter:
            rate_limiter.wait_if_needed()
        
        # Generate answer
        response = llm.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        print(f"   Warning: Answer generation failed: {e}")
        return None


def filter_by_sample(retriever, query: str, sample_id: str, k: int = 10) -> List:
    """
    Retrieve documents and filter by sample_id.
    
    Args:
        retriever: The retriever instance
        query: Query string
        sample_id: Conversation sample ID to filter by
        k: Number of results to return
    
    Returns:
        List of retrieved Document objects from the same conversation
    """
    try:
        docs = retriever.invoke(query)
        filtered = [d for d in docs if d.metadata.get('sample_id') == sample_id]
        return filtered[:k]
    except Exception as e:
        print(f"   Error during retrieval: {e}")
        return []


def evaluate_retriever(
    retriever,
    retriever_name: str,
    qa_pairs: List[Dict],
    config: dict,
    llm=None
) -> Dict:
    """
    Evaluate a single retriever on Q&A pairs.
    
    Args:
        retriever: The retriever instance
        retriever_name: Name of the retriever
        qa_pairs: List of Q&A dictionaries
        config: Configuration dict
        llm: Optional LLM instance for answer generation
    
    Returns:
        Dictionary containing evaluation results
    """
    print(f"\n{'='*80}")
    print(f"TESTING: {retriever_name}")
    print(f"{'='*80}\n")
    
    top_k = config['retrievers']['top_k']
    question_limit = config['evaluation'].get('question_limit')
    filter_by_conv = config['evaluation'].get('filter_by_conversation', True)
    
    # RAG settings
    generate_answers = config['llm'].get('generate_answers', False) and llm is not None
    max_context_docs = config['llm'].get('max_context_docs', 3)
    
    # Initialize rate limiter if enabled
    rate_limiter = None
    if generate_answers and config['llm'].get('rate_limit', {}).get('enabled', True):
        rate_limit_config = config['llm'].get('rate_limit', {})
        max_requests = rate_limit_config.get('max_requests', 30)
        time_window = rate_limit_config.get('time_window', 60)
        rate_limiter = RateLimiter(max_requests, time_window)
        print(f"  Rate limiting enabled: {max_requests} requests per {time_window}s")
    
    stats = {
        'retriever': retriever_name,
        'total_questions': 0,
        'questions_with_evidence': 0,
        'total_evidence_items': 0,
        'found_evidence_items': 0,
        'answers_generated': 0,
        'answers_correct': 0,
        'details': []
    }
    
    for idx, qa in enumerate(qa_pairs, start=1):
        if question_limit and idx > question_limit:
            break
        
        question = qa['question']
        answer = qa['answer']
        evidence = qa['evidence']
        sample_id = qa['sample_id']
        
        if not question or not evidence:
            continue
        
        stats['total_questions'] += 1
        stats['total_evidence_items'] += len(evidence)
        
        # Retrieve documents
        if filter_by_conv:
            retrieved_docs = filter_by_sample(retriever, question, sample_id, k=top_k)
        else:
            try:
                retrieved_docs = retriever.invoke(question)[:top_k]
            except:
                retrieved_docs = []
        
        retrieved_dia_ids = [doc.metadata.get('dia_id') for doc in retrieved_docs]
        
        # Generate answer using RAG if enabled
        generated_answer = None
        answer_eval = None
        if generate_answers:
            generated_answer = generate_answer(llm, question, retrieved_docs, max_context_docs, rate_limiter)
            if generated_answer:
                stats['answers_generated'] += 1
                # Evaluate generated answer against ground truth
                answer_eval = evaluate_answer(generated_answer, answer)
                if answer_eval['is_correct']:
                    stats['answers_correct'] += 1
        
        # Check which evidence was found
        found_items = [eid for eid in evidence if eid in retrieved_dia_ids]
        stats['found_evidence_items'] += len(found_items)
        
        if found_items:
            stats['questions_with_evidence'] += 1
        
        # Store detailed results
        detail = {
            'question_id': idx,
            'question': question,
            'ground_truth_answer': answer,
            'generated_answer': generated_answer,
            'answer_evaluation': answer_eval,
            'ground_truth_evidence': evidence,
            'retrieved_dia_ids': retrieved_dia_ids,
            'found_evidence': found_items,
            'retrieved_documents': [
                {
                    'dia_id': doc.metadata.get('dia_id'),
                    'sample_id': doc.metadata.get('sample_id'),
                    'speaker': doc.metadata.get('speaker'),
                    'session': doc.metadata.get('session'),
                    'session_datetime': doc.metadata.get('session_datetime'),
                    'text': doc.page_content[:200],  # Increased from 100 to 200 chars
                    'full_text': doc.page_content,   # Include full text
                    'is_evidence': doc.metadata.get('dia_id') in evidence,
                    'metadata': doc.metadata  # Include complete metadata
                }
                for doc in retrieved_docs
            ]
        }
        stats['details'].append(detail)
        
        # Print progress (with metadata preview)
        print(f"Question #{idx}")
        print(f"├─ Q: {question}")
        print(f"├─ Ground Truth Answer: {answer}")
        if generated_answer:
            # Truncate long answers for display
            display_answer = generated_answer if len(generated_answer) <= 100 else generated_answer[:97] + "..."
            print(f"├─ Generated Answer: {display_answer}")
        print(f"├─ Ground Truth Evidence: {evidence}")
        print(f"├─ Retrieved: {retrieved_dia_ids[:5]}... {'✓' if found_items else '✗'}")
        if retrieved_docs and len(retrieved_docs) > 0:
            first_doc = retrieved_docs[0]
            session_time = first_doc.metadata.get('session_datetime', 'N/A')
            print(f"└─ Top result from: {session_time}")
    
    # Calculate metrics
    if stats['total_questions'] > 0:
        stats['questions_with_evidence_pct'] = 100 * stats['questions_with_evidence'] / stats['total_questions']
    else:
        stats['questions_with_evidence_pct'] = 0.0
    
    if stats['total_evidence_items'] > 0:
        stats['recall_pct'] = 100 * stats['found_evidence_items'] / stats['total_evidence_items']
    else:
        stats['recall_pct'] = 0.0
    
    # Calculate answer accuracy
    if stats['answers_generated'] > 0:
        stats['answer_accuracy'] = 100 * stats['answers_correct'] / stats['answers_generated']
    else:
        stats['answer_accuracy'] = 0.0
    
    # Print summary
    print(f"\n{'-'*80}")
    print(f"SUMMARY for {retriever_name}:")
    print(f"├─ Questions tested: {stats['total_questions']}")
    if generate_answers:
        print(f"├─ Answers generated: {stats['answers_generated']}/{stats['total_questions']}")
        print(f"├─ Answer accuracy: {stats['answers_correct']}/{stats['answers_generated']} ({stats['answer_accuracy']:.1f}%)")
    print(f"├─ Questions with ≥1 evidence: {stats['questions_with_evidence']} ({stats['questions_with_evidence_pct']:.1f}%)")
    print(f"└─ Evidence recall: {stats['found_evidence_items']}/{stats['total_evidence_items']} ({stats['recall_pct']:.1f}%)")
    print(f"{'-'*80}\n")
    
    return stats


def save_results(results: List[Dict], retriever_names: List[str], config: dict):
    """
    Save evaluation results to JSON files.
    
    Args:
        results: List of evaluation result dictionaries
        retriever_names: List of retriever names tested
        config: Configuration dict
    """
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    timestamp = get_timestamp(config['output'].get('timestamp_format', 'datetime'))
    
    # Determine filename
    if len(retriever_names) == 1:
        filename = f"retriever_{retriever_names[0]}_{timestamp}.json"
    else:
        filename = f"retriever_all_{timestamp}.json"
    
    output_path = results_dir / filename
    
    # Prepare output data
    output_data = {
        'timestamp': timestamp,
        'config': config,
        'retrievers_tested': retriever_names,
        'results': results
    }
    
    # Save detailed results if enabled
    if config['output'].get('save_detailed', True):
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Detailed results saved to: {output_path}")
    
    # Save summary if enabled
    if config['output'].get('save_summary', True):
        summary_path = results_dir / f"summary_{timestamp}.json"
        summary_items = []
        for r in results:
            item = {
                'retriever': r['retriever'],
                'questions_tested': r['total_questions'],
                'questions_with_evidence': r['questions_with_evidence'],
                'questions_with_evidence_pct': r['questions_with_evidence_pct'],
                'evidence_found': r['found_evidence_items'],
                'evidence_total': r['total_evidence_items'],
                'recall_pct': r['recall_pct']
            }
            # Add answer generation stats if available
            if 'answers_generated' in r:
                item['answers_generated'] = r['answers_generated']
            summary_items.append(item)
        
        summary = {
            'timestamp': timestamp,
            'retrievers_tested': retriever_names,
            'rag_enabled': config['llm'].get('generate_answers', False),
            'summary': summary_items
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to: {summary_path}")


def main():
    """Main execution function."""
    
    print(f"\n{'='*80}")
    print("MEMAGENT - RETRIEVER COMPARISON PIPELINE")
    print(f"{'='*80}\n")
    
    # Load configuration
    print("Step 1: Loading configuration...")
    config = load_config()
    print(f"  ✓ Configuration loaded from config.yaml")
    
    # Load data
    print("\nStep 2: Loading dataset...")
    loader = DataLoader(config['data']['json_path'])
    
    sample_ids = config['data']['sample_ids']
    limit = config['data'].get('limit')
    
    documents, qa_pairs, conversations = loader.load_data(
        sample_ids=sample_ids,
        limit=limit
    )
    
    if len(documents) == 0:
        print("❌ No documents loaded. Check your configuration.")
        return
    
    print(f"  ✓ Loaded {len(documents)} utterances, {len(qa_pairs)} Q&A pairs")
    
    # Load session documents if needed
    session_docs = None
    retrievers_to_run = config['retrievers']['to_run']
    if retrievers_to_run == "all" or "parent_document" in retrievers_to_run:
        print("\nStep 3: Loading session documents (for Parent Document retriever)...")
        session_docs = loader.get_session_documents()
    
    # Load LLM if needed
    print("\nStep 4: Loading LLM (if enabled)...")
    llm = None
    if config['llm'].get('enabled', False):
        llm = get_llm(temperature=config['llm'].get('temperature', 0.0))
        if llm:
            print("  ✓ LLM loaded and ready")
        else:
            print("  ⚠️  LLM not available (check .env file)")
    else:
        print("  ⚠️  LLM disabled in config")
    
    # Initialize retriever
    print("\nStep 5: Initializing retrievers...")
    retriever_config = {
        'embedding_model': config['retrievers']['embedding_model'],
        'top_k': config['retrievers']['top_k'],
        'ensemble_weights': config['retrievers']['ensemble_weights'],
        'multiquery_variations': config['retrievers'].get('multiquery_variations', 3),
        'parent_chunk_size': config['retrievers'].get('parent_chunk_size', 400),
        'parent_chunk_overlap': config['retrievers'].get('parent_chunk_overlap', 50),
        'mmr_diversity': config['retrievers'].get('mmr_diversity', 0.5),
        'mmr_fetch_k': config['retrievers'].get('mmr_fetch_k', 20),
    }
    
    retriever = Retriever(
        documents=documents,
        llm=llm,
        session_docs=session_docs,
        config=retriever_config
    )
    
    # Determine which retrievers to run
    if retrievers_to_run == "all":
        retrievers_to_test = retriever.list_available()
        print(f"  ✓ Testing all available retrievers: {retrievers_to_test}")
    else:
        retrievers_to_test = retrievers_to_run
        print(f"  ✓ Testing selected retrievers: {retrievers_to_test}")
    
    # Run experiments
    print(f"\nStep 6: Running experiments...")
    all_results = []
    successful_retrievers = []
    
    for retriever_name in retrievers_to_test:
        try:
            # Get retriever instance
            retriever_instance = retriever.get(retriever_name)
            
            if retriever_instance is None:
                print(f"\n⚠️  Skipping {retriever_name} - not available")
                continue
            
            # Evaluate
            results = evaluate_retriever(
                retriever=retriever_instance,
                retriever_name=retriever_name,
                qa_pairs=qa_pairs,
                config=config,
                llm=llm
            )
            
            all_results.append(results)
            successful_retrievers.append(retriever_name)
            
        except Exception as e:
            print(f"\n❌ Error testing {retriever_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final comparison
    if all_results:
        print(f"\n{'='*80}")
        print("FINAL COMPARISON")
        print(f"{'='*80}\n")
        
        print(f"{'Retriever':<25} {'Questions':<15} {'Evidence Found':<20} {'Recall %':<10}")
        print(f"{'-'*80}")
        
        for result in all_results:
            name = result['retriever']
            total_q = result['total_questions']
            found_q = result['questions_with_evidence']
            found_e = result['found_evidence_items']
            total_e = result['total_evidence_items']
            recall = result['recall_pct']
            
            print(f"{name:<25} {found_q}/{total_q:<13} {found_e}/{total_e:<18} {recall:>6.1f}%")
        
        print(f"{'-'*80}\n")
        
        # Save results
        print("\nStep 7: Saving results...")
        save_results(all_results, successful_retrievers, config)
        
        print("\n✓ Pipeline completed successfully!\n")
    else:
        print("\n❌ No retrievers were successfully tested.\n")


if __name__ == "__main__":
    main()

