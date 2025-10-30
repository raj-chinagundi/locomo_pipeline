"""
Query Orchestrator - Intelligent Retriever Selection
LoCoMo-OPTIMIZED VERSION: Based on LoCoMo benchmark research (Maharana et al., 2024)
"""

import json
from typing import Dict, Optional, Tuple, List
from pathlib import Path


class QueryOrchestrator:
    """
    Intelligently selects the best retriever for a query using LLM analysis.
    Specifically optimized for LoCoMo (Long-term Conversational Memory) dataset.
    
    Key LoCoMo Characteristics:
    - Very long conversations (300 turns, 9K tokens avg, up to 35 sessions)
    - 5 reasoning types: single-hop, multi-hop, temporal, commonsense, adversarial
    - Temporal reasoning is HARDEST (73% below human performance)
    - Persona-driven with detailed backgrounds and event graphs
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
        """Build LoCoMo-optimized prompt for retriever selection."""
        
        # Extract retriever info with LoCoMo rankings
        available_retrievers = ['ensemble', 'parent_document', 'faiss', 'bm25', 'mmr',
                               'multiquery', 'contextual_compression', 'time_weighted', 
                               'tfidf', 'svm', 'long_context_reorder']
        
        retrievers_info = []
        for name in available_retrievers:
            if name in self.catalog.get('retrievers', {}):
                info = self.catalog['retrievers'][name]
                ranking = info.get('locomo_ranking', 'N/A')
                retrievers_info.append(f"""
**{name}** {ranking}
- Role: {info.get('role', 'N/A')}
- Best for: {', '.join(info.get('primary_use_cases', [])[:3])}
- LoCoMo Performance: {info.get('locomo_performance', 'N/A')[:100]}...
""")
        
        retrievers_text = "\n".join(retrievers_info)
        
        # Get LoCoMo decision tree rules
        decision_tree = self.catalog.get('LoCoMo_Orchestrator_Decision_Tree', {})
        rules = decision_tree.get('rules', [])
        rules_text = "\n".join([
            f"{i+1}. IF {rule['condition']} → USE '{rule['selected_retriever']}' ({rule['confidence']} confidence)\n   Reason: {rule['reasoning'][:120]}..."
            for i, rule in enumerate(rules[:8])  # Top 8 rules
        ])
        
        # Get query intent patterns
        patterns = self.catalog.get('LoCoMo_Query_Intent_Patterns', {})
        patterns_text = "\n".join([
            f"- **{key}**: Patterns: {', '.join(value.get('patterns', [])[:3])} → {value.get('best_retrievers', [])}"
            for key, value in list(patterns.items())[:6]  # Top 6 patterns
        ])
        
        # Get critical signals
        signals = self.catalog.get('Orchestrator_Prompt_Guidelines', {}).get('locomo_specific_signals', {})
        
        prompt = f"""You are an expert retriever selector for the LoCoMo (Long-term Conversational Memory) benchmark.

LOCOMO DATASET CONTEXT:
- Very long conversations: 300 turns, 9K tokens avg, up to 35 sessions
- Personas: Caroline, Melanie, Nate, Joanna (detailed backgrounds)
- 5 reasoning types: single-hop, multi-hop, temporal, commonsense, adversarial
- CRITICAL: Temporal reasoning is HARDEST (73% below human performance)
- Rich metadata: speaker, session_datetime, dia_id, event graphs

QUERY TO ANALYZE: "{query}"

AVAILABLE RETRIEVERS (LoCoMo-Ranked):
{retrievers_text}

LOCOMO DECISION RULES (Priority Order):
{rules_text}

LOCOMO QUERY INTENT PATTERNS:
{patterns_text}

CRITICAL SELECTION LOGIC FOR LOCOMO:

**1. TEMPORAL QUERIES** ("When did", "What date", "What time") - HARDEST CATEGORY
   ⚠️  73% performance gap - requires hybrid approach
   ✅ USE: 'ensemble' (BEST - combines keyword + semantic)
   ❌ NEVER: 'time_weighted' (only for "recent", not exact dates)
   Signals: {', '.join(signals.get('temporal_signals', []))}

**2. MULTI-HOP QUERIES** ("How did X progress", "What happened after")
   Requires information synthesis across sessions
   ✅ USE: 'parent_document' (returns entire sessions for context)
   Alternative: 'mmr' (diverse session coverage)
   Signals: {', '.join(signals.get('multi_hop_signals', []))}

**3. SPEAKER-SPECIFIC QUERIES** ("What did [person] say/do")
   Needs exact persona name matching
   ✅ USE: 'ensemble' (hybrid) or 'bm25' (keyword match)
   ❌ AVOID: 'self_query' (unavailable - requires langchain_classic)
   Signals: {', '.join(signals.get('speaker_signals', []))}

**4. COMMONSENSE/CONCEPTUAL** ("What fields would X pursue", "Why did")
   Requires inference and semantic understanding
   ✅ USE: 'faiss' (semantic) or 'multiquery' (expansion)
   Signals: {', '.join(signals.get('commonsense_signals', []))}

**5. ADVERSARIAL QUERIES** ("Did X mention Y" when they didn't)
   LLMs hallucinate on these - need exact verification
   ✅ USE: 'bm25' (exact keyword match to verify)
   Signals: {', '.join(signals.get('adversarial_signals', []))}

**6. EXPLORATORY QUERIES** ("What topics did X discuss")
   Need diverse results from multiple sessions
   ✅ USE: 'mmr' (diversity) or 'multiquery' (expansion)
   Signals: {', '.join(signals.get('exploratory_signals', []))}

**7. IDENTITY/RELATIONSHIP** ("Who is X", "What is X's identity")
   Often not explicitly stated - needs semantic understanding
   ✅ USE: 'faiss' (concepts) or 'parent_document' (full context)
   Signals: {', '.join(signals.get('identity_signals', []))}

**8. SINGLE-HOP FACTOID** (simple fact lookup with clear keywords)
   ✅ USE: 'bm25' or 'ensemble'
   Signals: {', '.join(signals.get('factoid_signals', []))}

**9. DEFAULT/UNCLEAR** (if query type ambiguous)
   ✅ USE: 'ensemble' (safest hybrid approach for LoCoMo)

CRITICAL CONSTRAINTS:
- NEVER select 'self_query' (unavailable - requires langchain_classic package)
- NEVER use 'time_weighted' for "When did X happen?" (use 'ensemble')
- AVOID 'tfidf' and 'svm' (outperformed by bm25/faiss)
- AVOID 'contextual_compression' (expensive, limited benefit per LoCoMo research)

ANALYSIS STEPS:
1. Identify query category (temporal/multi-hop/speaker-specific/commonsense/adversarial/exploratory/identity/factoid)
2. Check for key signals:
   - Temporal: when, date, time, year, month, ago
   - Multi-hop: progress, evolution, after, before, led to
   - Speaker: Caroline, Melanie, Nate, Joanna, said, mentioned
   - Commonsense: fields, pursue, likely, why, infer
   - Adversarial: Did [person] mention, ever, at any point
   - Exploratory: topics, discuss, activities, overview
   - Identity: who is, identity, background, describe
3. Apply decision rules (priority order)
4. Select SINGLE BEST retriever
5. Provide reasoning tied to LoCoMo characteristics

RESPONSE FORMAT (JSON ONLY - no markdown, no extra text):
{{
  "query_category": "temporal/multi_hop/speaker_specific/commonsense/adversarial/exploratory/identity/factoid/unclear",
  "key_signals": ["signal1: temporal word 'when'", "signal2: person name 'Caroline'", "signal3: activity 'paint'"],
  "query_intent": "specific intent from LoCoMo patterns (temporal_queries, multi_hop_queries, speaker_specific_queries, etc.)",
  "selected_retriever": "retriever_name (from available list, NOT self_query)",
  "reasoning": "Why this retriever is optimal for this LoCoMo query category. Reference: 1) query signals, 2) retriever strengths, 3) LoCoMo research findings (e.g., temporal hardest category)",
  "confidence": "high/medium/low",
  "alternative_retrievers": ["alt1", "alt2"],
  "locomo_specific_reasoning": "How this addresses LoCoMo challenges (long context, multi-session, persona consistency, etc.)",
  "warning_if_any": "Any warnings about potential failure modes or limitations"
}}

IMPORTANT: Return ONLY valid JSON. No markdown formatting, no extra text."""
        
        return prompt
    
    def select_retriever(self, query: str) -> Tuple[str, Dict]:
        """
        Select optimal retriever for query using LoCoMo-optimized LLM analysis.
        
        Args:
            query: User query to analyze
            
        Returns:
            Tuple of (retriever_name, analysis_dict)
        """
        if not self.llm:
            return "ensemble", {
                "query_category": "unclear",
                "query_analysis": "No LLM available",
                "selected_retriever": "ensemble",
                "reasoning": "Defaulting to ensemble (safest hybrid approach for LoCoMo)",
                "confidence": "low"
            }
        
        # Hardcoded fallback for unavailable retrievers
        unavailable_retrievers = ['self_query']  # Not available without langchain_classic
        
        try:
            prompt = self._build_analysis_prompt(query)
            response = self.llm.invoke(prompt)
            
            response_text = response.content.strip()
            
            # Extract JSON (handle markdown formatting)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing non-JSON text
            if not response_text.startswith('{'):
                start = response_text.find('{')
                if start != -1:
                    response_text = response_text[start:]
            
            # Remove trailing non-JSON
            if not response_text.endswith('}'):
                end = response_text.rfind('}')
                if end != -1:
                    response_text = response_text[:end+1]
            
            analysis = json.loads(response_text)
            
            # Validate and fallback
            selected = analysis.get('selected_retriever', 'ensemble')
            
            # Check if unavailable
            if selected in unavailable_retrievers:
                print(f"  ⚠️  Selected retriever '{selected}' not available, using ensemble")
                selected = 'ensemble'
                analysis['selected_retriever'] = 'ensemble'
                analysis['reasoning'] += f" [FALLBACK: self_query unavailable (requires langchain_classic), using ensemble hybrid]"
                analysis['warning_if_any'] = "self_query unavailable - fallback to ensemble"
            
            # Validate exists in catalog
            if selected not in self.catalog.get('retrievers', {}):
                print(f"  ⚠️  Selected retriever '{selected}' not in catalog, using ensemble")
                selected = 'ensemble'
                analysis['selected_retriever'] = 'ensemble'
                analysis['reasoning'] += f" [FALLBACK: {selected} not in catalog, using ensemble]"
            
            return selected, analysis
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse error: {e}")
            print(f"  Response preview: {response_text[:200]}...")
            return "ensemble", {
                "query_category": "unclear",
                "query_analysis": "JSON parse error",
                "selected_retriever": "ensemble",
                "reasoning": "Fallback due to parsing error - using ensemble hybrid (safest for LoCoMo)",
                "confidence": "low",
                "error": str(e)
            }
        except Exception as e:
            print(f"  ⚠️  Selection error: {e}")
            return "ensemble", {
                "query_category": "unclear",
                "query_analysis": "Error during analysis",
                "selected_retriever": "ensemble",
                "reasoning": f"Fallback due to error: {e}. Using ensemble (safest for LoCoMo)",
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
        LoCoMo-specific: considers multi-session reasoning, persona consistency, temporal accuracy.
        
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
                session = doc.metadata.get('session', 'N/A')
                text_preview = doc.page_content[:120]
                is_ground_truth = dia_id in ground_truth_evidence
                
                context_summary.append(
                    f"[Doc {i}] ID: {dia_id} | Session: {session} | Speaker: {speaker} | Time: {timestamp}\n"
                    f"Ground Truth: {'✓ YES' if is_ground_truth else '✗ NO'} | Text: {text_preview}..."
                )
            
            context_text = "\n\n".join(context_summary)
            
            eval_prompt = f"""Evaluate retrieved context quality for a LoCoMo conversational memory query.

LOCOMO CONTEXT:
- Very long conversations (300 turns, 9K tokens, up to 35 sessions)
- Personas: Caroline, Melanie, Nate, Joanna with detailed backgrounds
- Query types: single-hop, multi-hop, temporal (hardest), commonsense, adversarial

QUERY: "{query}"

RETRIEVED CONTEXT (Top 5 documents):
{context_text}

GROUND TRUTH EVIDENCE: {ground_truth_evidence}
(Document IDs that SHOULD be retrieved for perfect answer)

EVALUATION CRITERIA FOR LOCOMO:

1. **Relevance** (0-10): Do documents contain information relevant to the query?
   - Right speaker/persona? (Caroline, Melanie, etc.)
   - Right topic/event?
   - Right time period?

2. **Completeness** (0-10): Enough information to answer?
   - Single-hop: Is the specific fact present?
   - Multi-hop: Are all necessary sessions/connections present?
   - Temporal: Is the exact date/time present?
   - Commonsense: Is context sufficient for inference?

3. **Precision** (0-10): Signal vs. noise ratio
   - How many non-relevant documents retrieved?
   - Adversarial queries: Any false positives?

4. **Evidence Coverage** (0-10): Ground truth retrieval
   - How many ground truth documents were found?
   - Are they ranked highly?

5. **Multi-Session Coherence** (0-10): For multi-hop queries
   - Do retrieved docs span necessary sessions?
   - Is temporal/causal order preserved?

RESPONSE FORMAT (JSON ONLY):
{{
  "context_quality_score": 0.0-10.0,
  "relevance_score": 0.0-10.0,
  "completeness_score": 0.0-10.0,
  "precision_score": 0.0-10.0,
  "evidence_coverage_score": 0.0-10.0,
  "multi_session_coherence_score": 0.0-10.0,
  "relevance_assessment": "high/medium/low - explanation (max 80 chars)",
  "completeness_assessment": "complete/partial/insufficient - explanation (max 80 chars)",
  "precision_assessment": "high/medium/low - explanation (max 80 chars)",
  "evidence_coverage": "X out of Y ground truth items found",
  "context_strengths": ["strength1: e.g., found correct speaker", "strength2: e.g., right time period"],
  "context_weaknesses": ["weakness1: e.g., missing key session", "weakness2: e.g., wrong temporal order"],
  "missing_information": "What critical info is missing for a complete answer?",
  "locomo_specific_issues": "Any LoCoMo-specific problems? (e.g., multi-session gap, persona mismatch, temporal confusion)",
  "overall_evaluation": "Brief summary (max 100 chars)"
}}

Return ONLY valid JSON."""
            
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
            
            if not response_text.endswith('}'):
                end = response_text.rfind('}')
                if end != -1:
                    response_text = response_text[:end+1]
            
            evaluation = json.loads(response_text)
            
            # Ensure context_quality_score exists
            if 'context_quality_score' not in evaluation:
                # Calculate average of all scores
                scores = [
                    evaluation.get('relevance_score', 0),
                    evaluation.get('completeness_score', 0),
                    evaluation.get('precision_score', 0),
                    evaluation.get('evidence_coverage_score', 0),
                    evaluation.get('multi_session_coherence_score', 0)
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
                "overall_evaluation": "Could not evaluate",
                "locomo_specific_issues": f"Evaluation error: {str(e)[:50]}"
            }
    
    def get_retriever_info(self, retriever_name: str) -> Optional[Dict]:
        """Get detailed info about a specific retriever."""
        return self.catalog.get('retrievers', {}).get(retriever_name)
    
    def explain_selection(self, query: str, analysis: Dict) -> str:
        """Generate human-readable explanation of retriever selection."""
        lines = [
            f"Query: {query}",
            f"",
            f"LoCoMo Analysis:",
            f"  Category: {analysis.get('query_category', 'N/A')}",
            f"  Intent: {analysis.get('query_intent', 'N/A')}",
            f"  Key Signals: {', '.join(analysis.get('key_signals', []))}",
            f"",
            f"Selected Retriever: {analysis.get('selected_retriever', 'N/A')}",
            f"Confidence: {analysis.get('confidence', 'N/A')}",
            f"",
            f"Reasoning:",
            f"  {analysis.get('reasoning', 'N/A')}",
            f"",
            f"LoCoMo-Specific:",
            f"  {analysis.get('locomo_specific_reasoning', 'N/A')}",
            f"",
            f"Alternatives: {', '.join(analysis.get('alternative_retrievers', []))}",
        ]
        
        if analysis.get('warning_if_any'):
            lines.append(f"⚠️  Warning: {analysis['warning_if_any']}")
        
        return "\n".join(lines)