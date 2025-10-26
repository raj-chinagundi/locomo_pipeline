"""
Query Orchestrator - Intelligent Retriever Selection
IMPROVED VERSION: Better prompting for Locomo dataset
"""

import json
from typing import Dict, Optional, Tuple, List
from pathlib import Path


class QueryOrchestrator:
    """
    Intelligently selects the best retriever for a query using LLM analysis.
    Optimized for Locomo conversational memory dataset.
    """
    
    def __init__(self, llm, retrievers_catalog_path: str = "retrievers.json"):
        """
        Initialize orchestrator.
        
        Args:
            llm: LLM instance for query analysis
            retrievers_catalog_path: Path to retrievers.json catalog
        """
        self.llm = llm
        self.catalog = self._load_catalog(retrievers_catalog_path)
        
    def _load_catalog(self, path: str) -> dict:
        """Load retriever catalog from JSON."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return data['Retrieval_System_Blueprint']
        except Exception as e:
            print(f"Warning: Could not load retrievers catalog: {e}")
            return {}
    
    def _build_analysis_prompt(self, query: str) -> str:
        """Build improved prompt for Locomo-specific retriever selection."""
        
        # Extract retriever info (prioritize available ones)
        available_retrievers = ['bm25', 'tfidf', 'faiss', 'ensemble', 'mmr', 
                               'parent_document', 'multiquery', 'contextual_compression',
                               'time_weighted', 'svm', 'long_context_reorder']
        
        retrievers_info = []
        for name in available_retrievers:
            if name in self.catalog.get('retrievers', {}):
                info = self.catalog['retrievers'][name]
                retrievers_info.append(f"""
**{name}**
- Role: {info.get('role', 'N/A')}
- Description: {info.get('description', 'N/A')[:120]}...
- Strengths: {', '.join(info.get('strengths', [])[:2])}
- Locomo Success: {info.get('locomo_success', 'N/A')[:80]}...
- Locomo Failure: {info.get('locomo_failure', 'N/A')[:80]}...
""")
        
        retrievers_text = "\n".join(retrievers_info)
        
        # Get Locomo-specific patterns
        patterns = self.catalog.get('Locomo_Query_Analysis_Guide', {})
        patterns_text = "\n".join([
            f"- {key}: {value.get('best_retrievers', [])} - {value.get('reasoning', '')[:60]}..."
            for key, value in patterns.items()
        ])
        
        prompt = f"""You are selecting the SINGLE BEST retriever for a conversational memory query from the Locomo dataset.

DATASET CONTEXT:
- Locomo contains multi-session conversations with personal memory questions
- Queries often ask: "When did [person] [action]?", "What did [person] say?", "Who is [person]?"
- Critical metadata: speaker names (Caroline, Melanie), session timestamps, event details
- Success depends on: finding exact names, temporal info, and conversational context

QUERY TO ANALYZE: "{query}"

AVAILABLE RETRIEVERS:
{retrievers_text}

LOCOMO QUERY PATTERNS & BEST RETRIEVERS:
{patterns_text}

CRITICAL SELECTION RULES FOR LOCOMO:
1. **Temporal queries** ("When did X?") → Use 'ensemble' (best hybrid) or 'bm25' (keyword match)
2. **Speaker-specific** ("What did Caroline say?") → Use 'bm25' (exact name match) or 'ensemble'
3. **Conceptual** ("What fields would X pursue?") → Use 'faiss' (semantic) or 'multiquery' (expansion)
4. **Identity** ("Who is X?", "What is X's identity?") → Use 'faiss' (concept) or 'parent_document' (context)
5. **Ambiguous/short** → Use 'multiquery' (expand) or 'ensemble' (safe default)
6. **NOT AVAILABLE**: self_query (requires langchain_classic package)

ANALYSIS TASK:
1. Identify query type: temporal? speaker-specific? conceptual? identity?
2. Check for key signals:
   - Person names (Caroline, Melanie) → favor bm25/ensemble
   - Time words (when, recently) → favor ensemble/time_weighted  
   - Conceptual (fields, identity, likely) → favor faiss/multiquery
   - Activity keywords (paint, research, go to) → favor bm25/ensemble
3. Select the SINGLE BEST retriever (avoid self_query - not available!)
4. Provide specific reasoning tied to Locomo dataset characteristics

RESPONSE FORMAT (JSON ONLY):
{{
  "query_analysis": "What is the query asking? (temporal/speaker/conceptual/identity)",
  "query_intent": "Primary intent (temporal_event_queries, speaker_specific_queries, conceptual_questions, identity_relationships)",
  "key_signals": ["signal1: person name", "signal2: time reference", "signal3: activity"],
  "selected_retriever": "retriever_name (choose from available list, NOT self_query)",
  "reasoning": "Why this retriever is optimal for this Locomo query. Reference specific query characteristics and retriever strengths for Locomo.",
  "alternative_retrievers": ["alt1", "alt2"],
  "confidence": "high/medium/low",
  "locomo_specific_note": "Why this choice is good for conversational memory retrieval"
}}

IMPORTANT: Respond with ONLY valid JSON, no markdown, no extra text."""
        
        return prompt
    
    def select_retriever(self, query: str) -> Tuple[str, Dict]:
        """
        Select optimal retriever for query using improved LLM analysis.
        
        Args:
            query: User query to analyze
            
        Returns:
            Tuple of (retriever_name, analysis_dict)
        """
        if not self.llm:
            return "ensemble", {
                "query_analysis": "No LLM available",
                "selected_retriever": "ensemble",
                "reasoning": "Defaulting to ensemble (safe hybrid approach)",
                "confidence": "low"
            }
        
        # Hardcoded fallback for unavailable retrievers
        unavailable_retrievers = ['self_query']  # Add more if needed
        
        try:
            prompt = self._build_analysis_prompt(query)
            response = self.llm.invoke(prompt)
            
            response_text = response.content.strip()
            
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing non-JSON text
            if not response_text.startswith('{'):
                # Find first {
                start = response_text.find('{')
                if start != -1:
                    response_text = response_text[start:]
            
            analysis = json.loads(response_text)
            
            # Validate and fallback
            selected = analysis.get('selected_retriever', 'ensemble')
            
            # Check if unavailable
            if selected in unavailable_retrievers:
                print(f"  ⚠️  Selected retriever '{selected}' not available, using ensemble")
                selected = 'ensemble'
                analysis['selected_retriever'] = 'ensemble'
                analysis['reasoning'] += f" [Fallback: {unavailable_retrievers[0]} not available, using ensemble hybrid approach]"
            
            # Validate exists in catalog
            if selected not in self.catalog.get('retrievers', {}):
                print(f"  ⚠️  Selected retriever '{selected}' not in catalog, using ensemble")
                selected = 'ensemble'
                analysis['selected_retriever'] = 'ensemble'
            
            return selected, analysis
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse error: {e}")
            print(f"  Response preview: {response_text[:150]}...")
            return "ensemble", {
                "query_analysis": "JSON parse error",
                "selected_retriever": "ensemble",
                "reasoning": "Fallback due to parsing error - using ensemble hybrid",
                "confidence": "low",
                "error": str(e)
            }
        except Exception as e:
            print(f"  ⚠️  Selection error: {e}")
            return "ensemble", {
                "query_analysis": "Error during analysis",
                "selected_retriever": "ensemble",
                "reasoning": f"Fallback due to error: {e}",
                "confidence": "low"
            }
    
    def evaluate_context_quality(
        self, 
        query: str, 
        retrieved_docs: List, 
        ground_truth_evidence: List[str]
    ) -> Dict:
        """
        Evaluate the quality and relevance of retrieved context using LLM.
        
        Args:
            query: Original query
            retrieved_docs: List of retrieved Document objects
            ground_truth_evidence: List of ground truth evidence IDs
            
        Returns:
            Dict with context quality metrics and LLM evaluation
        """
        if not self.llm or not retrieved_docs:
            return {
                "context_quality_score": 0.0,
                "relevance_assessment": "No LLM or documents available",
                "completeness_assessment": "unknown",
                "context_strengths": [],
                "context_weaknesses": [],
                "overall_evaluation": "Cannot evaluate"
            }
        
        try:
            # Build context summary (top 5)
            context_summary = []
            for i, doc in enumerate(retrieved_docs[:5], 1):
                dia_id = doc.metadata.get('dia_id', 'N/A')
                speaker = doc.metadata.get('speaker', 'Unknown')
                timestamp = doc.metadata.get('session_datetime', 'N/A')
                text_preview = doc.page_content[:120]
                is_ground_truth = dia_id in ground_truth_evidence
                
                context_summary.append(
                    f"[Doc {i}] ID: {dia_id} | Speaker: {speaker} | Time: {timestamp} | Ground Truth: {'✓' if is_ground_truth else '✗'}\n"
                    f"Text: {text_preview}..."
                )
            
            context_text = "\n\n".join(context_summary)
            
            eval_prompt = f"""Evaluate the quality of retrieved conversational context for answering a memory question.

QUERY: "{query}"

RETRIEVED CONTEXT (Top 5 documents from conversations):
{context_text}

GROUND TRUTH EVIDENCE IDs: {ground_truth_evidence}
(These are the document IDs that SHOULD be retrieved for a perfect answer)

EVALUATION CRITERIA:
1. **Relevance** (0-10): Do documents contain information relevant to answering the query?
2. **Completeness** (0-10): Is there enough information to fully answer the query?
3. **Precision** (0-10): How much noise vs. signal? Are non-relevant docs present?
4. **Evidence Coverage** (0-10): How many ground truth documents were actually retrieved?

For Locomo conversational data, consider:
- Did it find the right speaker (Caroline/Melanie)?
- Did it find the right time period/session?
- Did it find the specific event or fact mentioned?

RESPONSE FORMAT (JSON ONLY):
{{
  "context_quality_score": 0.0-10.0,
  "relevance_score": 0.0-10.0,
  "completeness_score": 0.0-10.0,
  "precision_score": 0.0-10.0,
  "evidence_coverage_score": 0.0-10.0,
  "relevance_assessment": "high/medium/low - brief explanation (max 60 chars)",
  "completeness_assessment": "complete/partial/insufficient - brief explanation (max 60 chars)",
  "precision_assessment": "high/medium/low - brief explanation (max 60 chars)",
  "evidence_coverage": "X out of Y ground truth items found",
  "context_strengths": ["strength1", "strength2"],
  "context_weaknesses": ["weakness1", "weakness2"],
  "missing_information": "What key info is missing to answer the query?",
  "overall_evaluation": "Brief summary (max 80 chars)"
}}

Respond with ONLY valid JSON."""
            
            response = self.llm.invoke(eval_prompt)
            response_text = response.content.strip()
            
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Find JSON object
            if not response_text.startswith('{'):
                start = response_text.find('{')
                if start != -1:
                    response_text = response_text[start:]
            
            evaluation = json.loads(response_text)
            
            # Ensure context_quality_score exists
            if 'context_quality_score' not in evaluation:
                # Calculate average if individual scores present
                scores = [
                    evaluation.get('relevance_score', 0),
                    evaluation.get('completeness_score', 0),
                    evaluation.get('precision_score', 0),
                    evaluation.get('evidence_coverage_score', 0)
                ]
                evaluation['context_quality_score'] = sum(scores) / len(scores) if scores else 0.0
            
            return evaluation
            
        except Exception as e:
            print(f"  ⚠️  Context evaluation error: {e}")
            return {
                "context_quality_score": 0.0,
                "relevance_assessment": f"Error: {str(e)[:40]}",
                "completeness_assessment": "unknown",
                "precision_assessment": "unknown",
                "context_strengths": [],
                "context_weaknesses": [f"Evaluation failed: {str(e)[:40]}"],
                "overall_evaluation": "Could not evaluate"
            }
    
    def get_retriever_info(self, retriever_name: str) -> Optional[Dict]:
        """Get detailed info about a specific retriever."""
        return self.catalog.get('retrievers', {}).get(retriever_name)
    
    def explain_selection(self, query: str, analysis: Dict) -> str:
        """Generate human-readable explanation of retriever selection."""
        lines = [
            f"Query: {query}",
            f"",
            f"Analysis:",
            f"  Intent: {analysis.get('query_intent', 'N/A')}",
            f"  Key Signals: {', '.join(analysis.get('key_signals', []))}",
            f"",
            f"Selected Retriever: {analysis.get('selected_retriever', 'N/A')}",
            f"Confidence: {analysis.get('confidence', 'N/A')}",
            f"",
            f"Reasoning:",
            f"  {analysis.get('reasoning', 'N/A')}",
            f"",
            f"Alternatives: {', '.join(analysis.get('alternative_retrievers', []))}"
        ]
        return "\n".join(lines)