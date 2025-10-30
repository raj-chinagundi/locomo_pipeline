"""
Main RAG pipeline for question answering with category filtering and dynamic retriever orchestration.
"""
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any

from data_loader import load_data, filter_by_category
from retrievers import get_all_retrievers
from orchestrator import JuryOrchestrator, create_retriever_config_file
from llm_helper import generate_answer
from evaluator import evaluate_answer


def run_pipeline(category: int = None, limit: int = None, retriever: str = None, 
                 use_orchestrator: bool = False, config_path: str = None) -> Dict[str, Any]:
    """
    Run the RAG pipeline for all questions, optionally filtered by category.
    
    Args:
        category: Category to filter questions by (1-5), or None for all categories
        limit: Limit number of conversations to process, or None for all
        retriever: Specific retriever to run (tfidf, svm, faiss, nanopq, bm25, ensemble), 
                   or None for all, or "orchestrator" for dynamic selection
        use_orchestrator: Whether to use dynamic retriever orchestration
        config_path: Path to retriever configuration JSON file
        
    Returns:
        Dictionary with results for all retrievers
    """
    # Load data
    conversations = load_data("locomo10.json")
    
    # Apply conversation limit
    if limit is not None:
        conversations = conversations[:limit]
        print(f"Limited to first {limit} conversation(s)")
    
    results = []
    total_convs = len(conversations)
    
    for conv_idx, conversation in enumerate(conversations):
        print(f"\n{'='*60}")
        print(f"Processing conversation {conv_idx + 1}/{total_convs}")
        print(f"{'='*60}")
        
        # Get QA pairs and conversation text
        qa_pairs = conversation["qa"]
        conv_data = conversation["conversation"]
        
        # Filter by category if specified
        if category is not None:
            qa_pairs = filter_by_category(qa_pairs, category)
        
        if not qa_pairs:
            continue
            
        # Build document chunks from conversation
        documents = []
        doc_id_map = {}
        
        for session_key in sorted([k for k in conv_data.keys() if k.startswith("session_") and k.endswith(("1", "2", "3", "4", "5", "6", "7", "8", "9", "0"))]):
            session_data = conv_data[session_key]
            # Get session timestamp
            session_timestamp_key = session_key + "_date_time"
            session_timestamp = conv_data.get(session_timestamp_key, "")
            
            for dialogue in session_data:
                if "text" in dialogue and "dia_id" in dialogue:
                    doc_id = dialogue["dia_id"]
                    text = dialogue["text"]
                    documents.append({
                        "id": doc_id,
                        "text": text,
                        "speaker": dialogue.get("speaker", ""),
                        "session_timestamp": session_timestamp
                    })
                    doc_id_map[doc_id] = text
        
        # Initialize orchestrator if requested
        orchestrator = None
        if use_orchestrator or retriever == "orchestrator":
            try:
                orchestrator = JuryOrchestrator(documents, config_path)
                print(f"  Initialized Jury Orchestrator with dynamic retriever selection")
            except Exception as e:
                print(f"  Failed to initialize orchestrator: {e}. Falling back to individual retrievers.")
                orchestrator = None
        
        # Get all retrievers (for individual mode or fallback)
        all_retrievers = get_all_retrievers(documents)
        
        # Process based on retriever selection mode
        if orchestrator and (use_orchestrator or retriever == "orchestrator"):
            # Use orchestrator mode - dynamic retriever selection per query
            print(f"\n  Running with DYNAMIC RETRIEVER ORCHESTRATION")
            
            q_lvl_results = []
            correct_count = 0
            total_questions = len(qa_pairs)
            
            for qid, qa in enumerate(qa_pairs, 1):
                question = qa["question"]
                source_answer = str(qa["answer"])
                source_evidence_ids = qa.get("evidence", [])
                category_val = qa.get("category", 0)
                
                # Use orchestrator to select retriever and retrieve documents
                orchestrated_result = orchestrator.retrieve_with_orchestration(question, top_k=5)
                retrieved_docs = orchestrated_result["retrieved_documents"]
                selected_retriever = orchestrated_result["selected_retriever"]
                detected_intent = orchestrated_result["detected_intent"]
                
                retrieved_evidence_ids = [doc["id"] for doc in retrieved_docs]
                
                # Generate answer using LLM
                generated_answer = generate_answer(question, retrieved_docs)
                
                # Evaluate answer
                status = evaluate_answer(source_answer, generated_answer)
                
                if status == "correct":
                    correct_count += 1
                
                # Check if we retrieved the needed evidence
                has_needed_evidence = any(eid in retrieved_evidence_ids for eid in source_evidence_ids)
                
                # Clean output
                print(f"\n    Q{qid}/{total_questions}: {question}")
                print(f"    Intent:        {detected_intent}")
                print(f"    Retriever:     {selected_retriever}")
                print(f"    Retrieved:     {', '.join(retrieved_evidence_ids)} (need: {', '.join(source_evidence_ids)})")
                print(f"    Ground Truth:  {source_answer}")
                print(f"    Generated:     {generated_answer}")
                print(f"    Status:        {'✓ CORRECT' if status == 'correct' else '✗ INCORRECT'}")
                
                # If we had the right evidence but still got it wrong, show what LLM saw
                if has_needed_evidence and status == "incorrect":
                    print(f"    ⚠ Had correct evidence but LLM failed! Retrieved context:")
                    for doc in retrieved_docs:
                        timestamp = doc.get('session_timestamp', '')
                        ts_str = f"[{timestamp}] " if timestamp else ""
                        print(f"      {doc['id']}: {ts_str}{doc.get('speaker', '')}: {doc['text'][:50]}...")
                
                q_lvl_results.append({
                    "qid": qid,
                    "category": category_val,
                    "source_q": question,
                    "source_answer": source_answer,
                    "source_evidence_ids": source_evidence_ids,
                    "generated_answer": generated_answer,
                    "retrieved_evidence_ids": retrieved_evidence_ids,
                    "selected_retriever": selected_retriever,
                    "detected_intent": detected_intent,
                    "status": status
                })
            
            accuracy = correct_count / len(qa_pairs) if qa_pairs else 0.0
            print(f"    → Orchestrator: {correct_count}/{total_questions} correct ({accuracy:.1%})")
            
            results.append({
                "convid": conv_idx + 1,
                "qcount": len(qa_pairs),
                "mode": "orchestrator",
                "retrievers": [{
                    "name": "orchestrator",
                    "total_answer_right": correct_count,
                    "accuracy": accuracy,
                    "q_lvl": q_lvl_results
                }]
            })
            
        else:
            # Individual retriever mode (original logic)
            # Filter to specific retriever if requested
            if retriever and retriever != "orchestrator":
                if retriever in all_retrievers:
                    retrievers = {retriever: all_retrievers[retriever]}
                else:
                    print(f"Warning: Retriever '{retriever}' not found. Available: {list(all_retrievers.keys())}")
                    retrievers = all_retrievers
            else:
                retrievers = all_retrievers
            
            # Process each retriever
            retriever_results = []
            total_retrievers = len(retrievers)
            
            for ret_idx, (retriever_name, retriever) in enumerate(retrievers.items(), 1):
                print(f"\n  [{ret_idx}/{total_retrievers}] Running retriever: {retriever_name}")
                
                q_lvl_results = []
                correct_count = 0
                total_questions = len(qa_pairs)
                
                for qid, qa in enumerate(qa_pairs, 1):
                    question = qa["question"]
                    source_answer = str(qa["answer"])
                    source_evidence_ids = qa.get("evidence", [])
                    category_val = qa.get("category", 0)
                    
                    # Retrieve relevant documents
                    retrieved_docs = retriever.retrieve(question, top_k=5)
                    retrieved_evidence_ids = [doc["id"] for doc in retrieved_docs]
                    
                    # Generate answer using LLM
                    generated_answer = generate_answer(question, retrieved_docs)
                    
                    # Evaluate answer
                    status = evaluate_answer(source_answer, generated_answer)
                    
                    if status == "correct":
                        correct_count += 1
                    
                    # Check if we retrieved the needed evidence
                    has_needed_evidence = any(eid in retrieved_evidence_ids for eid in source_evidence_ids)
                    
                    # Clean output
                    print(f"\n    Q{qid}/{total_questions}: {question}")
                    print(f"    Retrieved:    {', '.join(retrieved_evidence_ids)} (need: {', '.join(source_evidence_ids)})")
                    print(f"    Ground Truth: {source_answer}")
                    print(f"    Generated:    {generated_answer}")
                    print(f"    Status:       {'✓ CORRECT' if status == 'correct' else '✗ INCORRECT'}")
                    
                    # If we had the right evidence but still got it wrong, show what LLM saw
                    if has_needed_evidence and status == "incorrect":
                        print(f"    ⚠ Had correct evidence but LLM failed! Retrieved context:")
                        for doc in retrieved_docs:
                            timestamp = doc.get('session_timestamp', '')
                            ts_str = f"[{timestamp}] " if timestamp else ""
                            print(f"      {doc['id']}: {ts_str}{doc.get('speaker', '')}: {doc['text'][:50]}...")
                    
                    q_lvl_results.append({
                        "qid": qid,
                        "category": category_val,
                        "source_q": question,
                        "source_answer": source_answer,
                        "source_evidence_ids": source_evidence_ids,
                        "generated_answer": generated_answer,
                        "retrieved_evidence_ids": retrieved_evidence_ids,
                        "status": status
                    })
                
                accuracy = correct_count / len(qa_pairs) if qa_pairs else 0.0
                print(f"    → {retriever_name}: {correct_count}/{total_questions} correct ({accuracy:.1%})")
                
                retriever_results.append({
                    "name": retriever_name,
                    "total_answer_right": correct_count,
                    "accuracy": accuracy,
                    "q_lvl": q_lvl_results
                })
            
            results.append({
                "convid": conv_idx + 1,
                "qcount": len(qa_pairs),
                "mode": "individual",
                "retrievers": retriever_results
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline with category filtering and dynamic retriever orchestration")
    parser.add_argument(
        "--category",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Filter questions by category (1-5). If not specified, runs on all questions."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. If not specified, uses timestamp."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of conversations to process (e.g., --limit 1 for first conversation only)"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["tfidf", "svm", "faiss", "nanopq", "bm25", "ensemble", "orchestrator"],
        default=None,
        help="Run specific retriever only, or 'orchestrator' for dynamic selection. If not specified, runs all retrievers."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to retriever configuration JSON file. If not specified, uses default configuration."
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default retriever configuration file and exit."
    )
    
    args = parser.parse_args()
    
    # Create config file if requested
    if args.create_config:
        config_path = create_retriever_config_file()
        print(f"Created default retriever configuration file: {config_path}")
        return
    
    print(f"Starting RAG pipeline...")
    if args.category:
        print(f"Filtering by category: {args.category}")
    else:
        print("Running on all categories")
    if args.limit:
        print(f"Limiting to first {args.limit} conversation(s)")
    
    if args.retriever == "orchestrator":
        print(f"Running with DYNAMIC RETRIEVER ORCHESTRATION")
        use_orchestrator = True
    elif args.retriever:
        print(f"Running retriever: {args.retriever}")
        use_orchestrator = False
    else:
        print("Running all retrievers individually")
        use_orchestrator = False
    
    if args.config:
        print(f"Using configuration from: {args.config}")
    
    # Run pipeline
    results = run_pipeline(
        category=args.category, 
        limit=args.limit, 
        retriever=args.retriever,
        use_orchestrator=use_orchestrator,
        config_path=args.config
    )
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "orchestrator" if use_orchestrator else "individual"
        output_file = f"rag_results_{mode}_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for conv_result in results:
        print(f"\nConversation {conv_result['convid']} ({conv_result['qcount']} questions):")
        
        if conv_result['mode'] == 'orchestrator':
            for ret_result in conv_result['retrievers']:
                print(f"  Orchestrator: {ret_result['total_answer_right']}/{conv_result['qcount']} correct ({ret_result['accuracy']:.1%})")
        else:
            for ret_result in conv_result['retrievers']:
                print(f"  {ret_result['name']}: {ret_result['total_answer_right']}/{conv_result['qcount']} correct ({ret_result['accuracy']:.1%})")


if __name__ == "__main__":
    main()