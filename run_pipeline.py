"""
Run Pipeline with Dynamic Orchestrator + Context Evaluation
FIXED VERSION - All evaluation features working correctly.
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
from query_orchestrator import QueryOrchestrator


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_timestamp(format_type: str = "datetime") -> str:
    """Generate timestamp for result files."""
    if format_type == "unix":
        return str(int(datetime.now().timestamp()))
    else:
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_answer(llm, question: str, retrieved_docs: List, max_context_docs: int = 3, rate_limiter: Optional[RateLimiter] = None) -> Optional[str]:
    """Generate natural language answer using retrieved documents (RAG)."""
    if not llm or not retrieved_docs:
        return None
    
    try:
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:max_context_docs], 1):
            dia_id = doc.metadata.get('dia_id', '')
            speaker = doc.metadata.get('speaker', 'Unknown')
            timestamp = doc.metadata.get('session_datetime', '')
            text = doc.page_content
            
            context_parts.append(f"[{dia_id} - {speaker}, {timestamp}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
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
        
        if rate_limiter:
            rate_limiter.wait_if_needed()
        
        response = llm.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        print(f"   Warning: Answer generation failed: {e}")
        return None


def filter_by_sample(retriever, query: str, sample_id: str, k: int = 10) -> List:
    """Retrieve documents and filter by sample_id."""
    try:
        docs = retriever.invoke(query)
        filtered = [d for d in docs if d.metadata.get('sample_id') == sample_id]
        return filtered[:k]
    except Exception as e:
        print(f"   Error during retrieval: {e}")
        return []


def calculate_retrieval_metrics(retrieved_dia_ids: List[str], ground_truth_evidence: List[str]) -> Dict:
    """
    Calculate comprehensive retrieval metrics.
    
    Args:
        retrieved_dia_ids: List of retrieved document IDs
        ground_truth_evidence: List of ground truth evidence IDs
        
    Returns:
        Dict with precision, recall, F1, MRR, etc.
    """
    if not ground_truth_evidence:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mrr": 0.0,
            "hit_rate": 0.0,
            "found_evidence": []
        }
    
    # Find which evidence items were retrieved
    found_evidence = [eid for eid in ground_truth_evidence if eid in retrieved_dia_ids]
    
    # Precision: How many retrieved docs are relevant?
    precision = len(found_evidence) / len(retrieved_dia_ids) if retrieved_dia_ids else 0.0
    
    # Recall: How many relevant docs were retrieved?
    recall = len(found_evidence) / len(ground_truth_evidence) if ground_truth_evidence else 0.0
    
    # F1 Score: Harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Mean Reciprocal Rank (MRR): Position of first relevant document
    mrr = 0.0
    for i, doc_id in enumerate(retrieved_dia_ids, start=1):
        if doc_id in ground_truth_evidence:
            mrr = 1.0 / i
            break
    
    # Hit Rate: Was at least one relevant document retrieved?
    hit_rate = 1.0 if found_evidence else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "mrr": round(mrr, 4),
        "hit_rate": hit_rate,
        "found_evidence": found_evidence,
        "total_retrieved": len(retrieved_dia_ids),
        "total_relevant": len(ground_truth_evidence),
        "true_positives": len(found_evidence)
    }


def evaluate_with_dynamic_selection(
    retriever_manager,
    orchestrator: QueryOrchestrator,
    qa_pairs: List[Dict],
    config: dict,
    llm=None
) -> Dict:
    """
    Evaluate queries using dynamic retriever selection WITH context quality evaluation.
    """
    print(f"\n{'='*80}")
    print("DYNAMIC ORCHESTRATOR EVALUATION WITH CONTEXT QUALITY ASSESSMENT")
    print(f"{'='*80}\n")
    
    top_k = config['retrievers']['top_k']
    question_limit = config['evaluation'].get('question_limit')
    filter_by_conv = config['evaluation'].get('filter_by_conversation', True)
    
    # RAG settings
    generate_answers = config['llm'].get('generate_answers', False) and llm is not None
    max_context_docs = config['llm'].get('max_context_docs', 3)
    
    # Context evaluation setting
    evaluate_context = config['evaluation'].get('evaluate_context_quality', True)
    
    # Rate limiter
    rate_limiter = None
    if config['llm'].get('rate_limit', {}).get('enabled', True):
        rate_limit_config = config['llm'].get('rate_limit', {})
        max_requests = rate_limit_config.get('max_requests', 30)
        time_window = rate_limit_config.get('time_window', 60)
        rate_limiter = RateLimiter(max_requests, time_window)
        print(f"  Rate limiting enabled: {max_requests} requests per {time_window}s")
    
    # Enhanced stats with retrieval metrics
    stats = {
        'evaluation_mode': 'dynamic_orchestrator_with_context_eval',
        'total_questions': 0,
        'questions_with_evidence': 0,
        'total_evidence_items': 0,
        'found_evidence_items': 0,
        'answers_generated': 0,
        'answers_correct': 0,
        'retriever_selection_stats': {},
        'avg_precision': 0.0,
        'avg_recall': 0.0,
        'avg_f1_score': 0.0,
        'avg_mrr': 0.0,
        'avg_context_quality': 0.0,
        'details': []
    }
    
    # Accumulators for averaging
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_mrr = 0.0
    total_context_quality = 0.0
    context_evals_count = 0
    
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
        
        # ============================================================
        # STEP 1: DYNAMIC RETRIEVER SELECTION
        # ============================================================
        print(f"\n{'─'*80}")
        print(f"Question #{idx}: {question}")
        print(f"{'─'*80}")
        
        selected_retriever_name, orchestrator_analysis = orchestrator.select_retriever(question)
        
        print(f"\n🤖 ORCHESTRATOR DECISION:")
        print(f"├─ Intent: {orchestrator_analysis.get('query_intent', 'N/A')}")
        print(f"├─ Selected: {selected_retriever_name}")
        print(f"├─ Confidence: {orchestrator_analysis.get('confidence', 'N/A')}")
        reasoning = orchestrator_analysis.get('reasoning', 'N/A')
        print(f"└─ Reasoning: {reasoning[:100]}...")
        
        # Track retriever usage
        if selected_retriever_name not in stats['retriever_selection_stats']:
            stats['retriever_selection_stats'][selected_retriever_name] = {
                'count': 0,
                'total_recall': 0.0,
                'total_precision': 0.0,
                'total_f1': 0.0,
                'total_context_quality': 0.0
            }
        stats['retriever_selection_stats'][selected_retriever_name]['count'] += 1
        
        # ============================================================
        # STEP 2: RETRIEVE DOCUMENTS
        # ============================================================
        retriever_instance = retriever_manager.get(selected_retriever_name)
        
        if retriever_instance is None:
            print(f"⚠️  Retriever '{selected_retriever_name}' not available, skipping...")
            continue
        
        if filter_by_conv:
            retrieved_docs = filter_by_sample(retriever_instance, question, sample_id, k=top_k)
        else:
            try:
                retrieved_docs = retriever_instance.invoke(question)[:top_k]
            except:
                retrieved_docs = []
        
        retrieved_dia_ids = [doc.metadata.get('dia_id') for doc in retrieved_docs]
        
        print(f"\n📄 RETRIEVAL RESULTS:")
        print(f"├─ Retrieved: {len(retrieved_docs)} documents")
        print(f"└─ Top IDs: {retrieved_dia_ids[:5]}")
        
        # ============================================================
        # STEP 3: CALCULATE RETRIEVAL METRICS (FIXED - NOW DISPLAYS!)
        # ============================================================
        retrieval_metrics = calculate_retrieval_metrics(retrieved_dia_ids, evidence)
        
        total_precision += retrieval_metrics['precision']
        total_recall += retrieval_metrics['recall']
        total_f1 += retrieval_metrics['f1_score']
        total_mrr += retrieval_metrics['mrr']
        
        # Update retriever-specific stats
        stats['retriever_selection_stats'][selected_retriever_name]['total_precision'] += retrieval_metrics['precision']
        stats['retriever_selection_stats'][selected_retriever_name]['total_recall'] += retrieval_metrics['recall']
        stats['retriever_selection_stats'][selected_retriever_name]['total_f1'] += retrieval_metrics['f1_score']
        
        print(f"\n📊 RETRIEVAL METRICS:")
        print(f"├─ Precision: {retrieval_metrics['precision']:.3f} ({retrieval_metrics['true_positives']}/{retrieval_metrics['total_retrieved']} relevant)")
        print(f"├─ Recall: {retrieval_metrics['recall']:.3f} ({retrieval_metrics['true_positives']}/{retrieval_metrics['total_relevant']} found)")
        print(f"├─ F1 Score: {retrieval_metrics['f1_score']:.3f}")
        print(f"└─ MRR: {retrieval_metrics['mrr']:.3f}")
        
        # ============================================================
        # STEP 4: EVALUATE CONTEXT QUALITY (FIXED - NOW RUNS!)
        # ============================================================
        context_evaluation = None
        if evaluate_context and orchestrator.llm and len(retrieved_docs) > 0:
            print(f"\n🔍 EVALUATING CONTEXT QUALITY...")
            
            if rate_limiter:
                rate_limiter.wait_if_needed()
            
            try:
                context_evaluation = orchestrator.evaluate_context_quality(
                    query=question,
                    retrieved_docs=retrieved_docs,
                    ground_truth_evidence=evidence
                )
                
                context_quality_score = context_evaluation.get('context_quality_score', 0.0)
                total_context_quality += context_quality_score
                context_evals_count += 1
                
                stats['retriever_selection_stats'][selected_retriever_name]['total_context_quality'] += context_quality_score
                
                print(f"├─ Quality Score: {context_quality_score:.1f}/10.0")
                print(f"├─ Relevance: {context_evaluation.get('relevance_assessment', 'N/A')[:60]}")
                print(f"├─ Completeness: {context_evaluation.get('completeness_assessment', 'N/A')[:60]}")
                overall = context_evaluation.get('overall_evaluation', 'N/A')
                print(f"└─ Overall: {overall[:80]}...")
            except Exception as e:
                print(f"   ⚠️ Context evaluation failed: {e}")
                context_evaluation = {
                    "context_quality_score": 0.0,
                    "relevance_assessment": f"Error: {e}",
                    "error": str(e)
                }
        
        # ============================================================
        # STEP 5: GENERATE ANSWER (RAG)
        # ============================================================
        generated_answer = None
        answer_eval = None
        if generate_answers:
            if rate_limiter:
                rate_limiter.wait_if_needed()
                
            generated_answer = generate_answer(llm, question, retrieved_docs, max_context_docs, rate_limiter)
            if generated_answer:
                stats['answers_generated'] += 1
                answer_eval = evaluate_answer(generated_answer, answer)
                if answer_eval['is_correct']:
                    stats['answers_correct'] += 1
                
                print(f"\n💬 GENERATED ANSWER:")
                display_answer = generated_answer if len(generated_answer) <= 150 else generated_answer[:147] + "..."
                print(f"   {display_answer}")
                print(f"   {'✓ Correct' if answer_eval['is_correct'] else '✗ Incorrect'} (Ground truth: {answer})")
        
        # ============================================================
        # STEP 6: CHECK EVIDENCE
        # ============================================================
        found_items = retrieval_metrics['found_evidence']
        stats['found_evidence_items'] += len(found_items)
        
        if found_items:
            stats['questions_with_evidence'] += 1
        
        print(f"\n🎯 EVIDENCE EVALUATION:")
        print(f"├─ Ground Truth: {evidence}")
        print(f"└─ Found: {found_items} {'✓' if found_items else '✗'}")
        
        # ============================================================
        # STEP 7: STORE DETAILED RESULTS
        # ============================================================
        detail = {
            'question_id': idx,
            'question': question,
            'ground_truth_answer': answer,
            'generated_answer': generated_answer,
            'answer_evaluation': answer_eval,
            'orchestrator_analysis': orchestrator_analysis,
            'selected_retriever': selected_retriever_name,
            'retriever_reasoning': orchestrator_analysis.get('reasoning', ''),
            'query_intent': orchestrator_analysis.get('query_intent', ''),
            'confidence': orchestrator_analysis.get('confidence', ''),
            'retrieval_metrics': retrieval_metrics,
            'context_evaluation': context_evaluation,
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
                    'text': doc.page_content[:200],
                    'full_text': doc.page_content,
                    'is_evidence': doc.metadata.get('dia_id') in evidence,
                    'rank': i + 1,
                    'metadata': doc.metadata
                }
                for i, doc in enumerate(retrieved_docs)
            ]
        }
        stats['details'].append(detail)
    
    # ============================================================
    # CALCULATE FINAL METRICS
    # ============================================================
    if stats['total_questions'] > 0:
        stats['questions_with_evidence_pct'] = 100 * stats['questions_with_evidence'] / stats['total_questions']
        stats['avg_precision'] = total_precision / stats['total_questions']
        stats['avg_recall'] = total_recall / stats['total_questions']
        stats['avg_f1_score'] = total_f1 / stats['total_questions']
        stats['avg_mrr'] = total_mrr / stats['total_questions']
    else:
        stats['questions_with_evidence_pct'] = 0.0
    
    if context_evals_count > 0:
        stats['avg_context_quality'] = total_context_quality / context_evals_count
    
    if stats['total_evidence_items'] > 0:
        stats['recall_pct'] = 100 * stats['found_evidence_items'] / stats['total_evidence_items']
    else:
        stats['recall_pct'] = 0.0
    
    if stats['answers_generated'] > 0:
        stats['answer_accuracy'] = 100 * stats['answers_correct'] / stats['answers_generated']
    else:
        stats['answer_accuracy'] = 0.0
    
    # Calculate per-retriever averages
    for ret_name, ret_stats in stats['retriever_selection_stats'].items():
        count = ret_stats['count']
        if count > 0:
            ret_stats['avg_precision'] = ret_stats['total_precision'] / count
            ret_stats['avg_recall'] = ret_stats['total_recall'] / count
            ret_stats['avg_f1'] = ret_stats['total_f1'] / count
            ret_stats['avg_context_quality'] = ret_stats['total_context_quality'] / count if ret_stats['total_context_quality'] > 0 else 0.0
    
    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    print(f"\n{'='*80}")
    print("DYNAMIC ORCHESTRATOR SUMMARY")
    print(f"{'='*80}")
    print(f"├─ Questions tested: {stats['total_questions']}")
    if generate_answers:
        print(f"├─ Answers generated: {stats['answers_generated']}/{stats['total_questions']}")
        print(f"├─ Answer accuracy: {stats['answers_correct']}/{stats['answers_generated']} ({stats['answer_accuracy']:.1f}%)")
    print(f"├─ Questions with ≥1 evidence: {stats['questions_with_evidence']} ({stats['questions_with_evidence_pct']:.1f}%)")
    print(f"└─ Evidence recall: {stats['found_evidence_items']}/{stats['total_evidence_items']} ({stats['recall_pct']:.1f}%)")
    
    print(f"\n📊 OVERALL RETRIEVAL METRICS:")
    print(f"├─ Avg Precision: {stats['avg_precision']:.3f}")
    print(f"├─ Avg Recall: {stats['avg_recall']:.3f}")
    print(f"├─ Avg F1 Score: {stats['avg_f1_score']:.3f}")
    print(f"├─ Avg MRR: {stats['avg_mrr']:.3f}")
    if context_evals_count > 0:
        print(f"└─ Avg Context Quality: {stats['avg_context_quality']:.2f}/10.0")
    
    print(f"\n🎯 RETRIEVER SELECTION DISTRIBUTION:")
    for ret_name, ret_stats in sorted(stats['retriever_selection_stats'].items(), 
                                      key=lambda x: x[1]['count'], reverse=True):
        count = ret_stats['count']
        pct = 100 * count / stats['total_questions']
        avg_recall = ret_stats.get('avg_recall', 0.0)
        avg_f1 = ret_stats.get('avg_f1', 0.0)
        avg_quality = ret_stats.get('avg_context_quality', 0.0)
        
        print(f"\n   {ret_name}:")
        print(f"   ├─ Used: {count} times ({pct:.1f}%)")
        print(f"   ├─ Avg Recall: {avg_recall:.3f}")
        print(f"   ├─ Avg F1: {avg_f1:.3f}")
        if avg_quality > 0:
            print(f"   └─ Avg Context Quality: {avg_quality:.2f}/10.0")
        else:
            print(f"   └─ Context Quality: Not evaluated")
    
    print(f"\n{'='*80}\n")
    
    return stats


def save_results(results: Dict, config: dict):
    """Save evaluation results with orchestrator decisions and context evaluations."""
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    timestamp = get_timestamp(config['output'].get('timestamp_format', 'datetime'))
    filename = f"dynamic_orchestrator_{timestamp}.json"
    output_path = results_dir / filename
    
    output_data = {
        'timestamp': timestamp,
        'evaluation_mode': 'dynamic_orchestrator_with_context_eval',
        'config': config,
        'results': results
    }
    
    # Save detailed results
    if config['output'].get('save_detailed', True):
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Detailed results saved to: {output_path}")
    
    # Save summary
    if config['output'].get('save_summary', True):
        summary_path = results_dir / f"summary_orchestrator_{timestamp}.json"
        summary = {
            'timestamp': timestamp,
            'evaluation_mode': 'dynamic_orchestrator_with_context_eval',
            'rag_enabled': config['llm'].get('generate_answers', False),
            'context_evaluation_enabled': config['evaluation'].get('evaluate_context_quality', True),
            'questions_tested': results['total_questions'],
            'questions_with_evidence': results['questions_with_evidence'],
            'questions_with_evidence_pct': results['questions_with_evidence_pct'],
            'evidence_recall_pct': results['recall_pct'],
            'answer_accuracy': results.get('answer_accuracy', 0.0),
            'avg_precision': results['avg_precision'],
            'avg_recall': results['avg_recall'],
            'avg_f1_score': results['avg_f1_score'],
            'avg_mrr': results['avg_mrr'],
            'avg_context_quality': results.get('avg_context_quality', 0.0),
            'retriever_selection_stats': results['retriever_selection_stats']
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to: {summary_path}")


def main():
    """Main execution with dynamic orchestrator and context evaluation."""
    
    print(f"\n{'='*80}")
    print("MEMAGENT - DYNAMIC ORCHESTRATOR PIPELINE")
    print("WITH CONTEXT QUALITY EVALUATION")
    print(f"{'='*80}\n")
    
    # Load configuration
    print("Step 1: Loading configuration...")
    config = load_config()
    print(f"  ✓ Configuration loaded")
    
    # Load data
    print("\nStep 2: Loading dataset...")
    loader = DataLoader(config['data']['json_path'])
    
    documents, qa_pairs, conversations = loader.load_data(
        sample_ids=config['data']['sample_ids'],
        limit=config['data'].get('limit')
    )
    
    if len(documents) == 0:
        print("✗ No documents loaded. Check your configuration.")
        return
    
    print(f"  ✓ Loaded {len(documents)} utterances, {len(qa_pairs)} Q&A pairs")
    
    # Load session documents if needed
    session_docs = None
    print("\nStep 3: Loading session documents...")
    session_docs = loader.get_session_documents()
    
    # Load LLM (REQUIRED for orchestrator)
    print("\nStep 4: Loading LLM...")
    llm = get_llm(temperature=config['llm'].get('temperature', 0.0))
    if not llm:
        print("  ✗ LLM required for dynamic orchestrator. Check .env file.")
        return
    print("  ✓ LLM loaded and ready")
    
    # Initialize Query Orchestrator
    print("\nStep 5: Initializing Query Orchestrator...")
    orchestrator = QueryOrchestrator(
        llm=llm,
        retrievers_catalog_path="retrievers.json"
    )
    print("  ✓ Orchestrator initialized with retriever catalog")
    
    # Initialize retriever manager
    print("\nStep 6: Initializing retriever manager...")
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
    
    retriever_manager = Retriever(
        documents=documents,
        llm=llm,
        session_docs=session_docs,
        config=retriever_config
    )
    
    available = retriever_manager.list_available()
    print(f"  ✓ Available retrievers: {available}")
    
    # Run dynamic evaluation
    print(f"\nStep 7: Running dynamic orchestrator evaluation...")
    try:
        results = evaluate_with_dynamic_selection(
            retriever_manager=retriever_manager,
            orchestrator=orchestrator,
            qa_pairs=qa_pairs,
            config=config,
            llm=llm
        )
        
        # Save results
        print("\nStep 8: Saving results...")
        save_results(results, config)
        
        print("\n✓ Pipeline completed successfully!\n")
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()