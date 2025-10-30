"""
Jury orchestrator for dynamically selecting retrievers based on query characteristics.
"""
import json
import re
from typing import Dict, List, Any, Tuple
from retrievers import get_all_retrievers


class JuryOrchestrator:
    """Orchestrator that dynamically selects the best retriever for each query."""
    
    def __init__(self, documents: List[Dict[str, Any]], retriever_config_path: str = None):
        """
        Initialize orchestrator with documents and retriever configuration.
        
        Args:
            documents: List of document dictionaries
            retriever_config_path: Path to retriever configuration JSON file
        """
        self.documents = documents
        self.retrievers = get_all_retrievers(documents)
        self.available_retrievers = list(self.retrievers.keys())
        
        print(f"Available retrievers for orchestration: {self.available_retrievers}")
        
        # Load retriever configuration
        if retriever_config_path:
            with open(retriever_config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Use default configuration
            self.config = {
                "query_intent_patterns": {
                    "temporal": {
                        "patterns": ["when did", "what date", "what time", "how long ago", "what year", "what month"],
                        "retrievers": ["ensemble", "bm25", "tfidf"]
                    },
                    "multi_hop": {
                        "patterns": ["how did.*progress", "describe evolution", "what happened after", "what led to", "trace the development"],
                        "retrievers": ["ensemble", "tfidf"]
                    },
                    "speaker_specific": {
                        "patterns": ["what did.*say", "what did.*do", "'.*s ", "did.*mention"],
                        "retrievers": ["ensemble", "bm25", "tfidf"]
                    },
                    "commonsense": {
                        "patterns": ["what fields would", "why did", "how likely", "what interests", "what career", "infer", "suggest"],
                        "retrievers": ["faiss", "ensemble", "tfidf"]
                    },
                    "adversarial": {
                        "patterns": ["did.*mention.*when.*didn", "ever mention", "at any point"],
                        "retrievers": ["bm25", "ensemble", "tfidf"]
                    },
                    "exploratory": {
                        "patterns": ["what topics did", "describe conversations", "what activities did", "broad overview"],
                        "retrievers": ["ensemble", "tfidf"]
                    },
                    "identity": {
                        "patterns": ["who is", "what is.*identity", "describe.*", "what is.*relationship"],
                        "retrievers": ["faiss", "ensemble", "tfidf"]
                    },
                    "single_hop": {
                        "patterns": ["what did.*say about", "what.*start", "what.*paint"],
                        "retrievers": ["bm25", "ensemble", "tfidf"]
                    },
                    "recent": {
                        "patterns": ["recently", "lately", "latest", "most recent", "newest"],
                        "retrievers": ["ensemble", "tfidf"]
                    },
                    "short_ambiguous": {
                        "patterns": [r"^\w+$", r"^\w+\s\w+$"],  # single word or two words
                        "retrievers": ["ensemble", "tfidf"]
                    }
                },
                "fallback_retriever": "tfidf",
                "retriever_priority": ["ensemble", "bm25", "faiss", "tfidf"]
            }
    
    def analyze_query_intent(self, query: str) -> str:
        """
        Analyze query to determine intent type.
        
        Args:
            query: The query string
            
        Returns:
            Intent type string
        """
        query_lower = query.lower().strip()
        
        # Check each intent pattern
        for intent_type, intent_config in self.config["query_intent_patterns"].items():
            for pattern in intent_config["patterns"]:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return intent_type
        
        return "default"
    
    def select_retriever(self, query: str) -> Tuple[str, Any]:
        """
        Select the best retriever for the given query.
        
        Args:
            query: The query string
            
        Returns:
            Tuple of (retriever_name, retriever_instance)
        """
        intent = self.analyze_query_intent(query)
        
        # Get recommended retrievers for this intent
        if intent in self.config["query_intent_patterns"]:
            recommended_retrievers = self.config["query_intent_patterns"][intent]["retrievers"]
        else:
            recommended_retrievers = [self.config["fallback_retriever"]]
        
        # Filter to only available retrievers
        available_recommendations = [r for r in recommended_retrievers if r in self.available_retrievers]
        
        # Find the first available retriever from recommendations
        for retriever_name in available_recommendations:
            if retriever_name in self.retrievers:
                return retriever_name, self.retrievers[retriever_name]
        
        # Fallback to priority order (only available ones)
        for retriever_name in self.config["retriever_priority"]:
            if retriever_name in self.available_retrievers:
                return retriever_name, self.retrievers[retriever_name]
        
        # Final fallback - use any available retriever
        if self.available_retrievers:
            retriever_name = self.available_retrievers[0]
            return retriever_name, self.retrievers[retriever_name]
        
        raise ValueError("No retrievers available")
    
    def retrieve_with_orchestration(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve documents using orchestrated retriever selection.
        
        Args:
            query: The query string
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with retrieval results and metadata
        """
        selected_retriever_name, selected_retriever = self.select_retriever(query)
        intent = self.analyze_query_intent(query)
        
        # Perform retrieval
        retrieved_docs = selected_retriever.retrieve(query, top_k)
        
        return {
            "retrieved_documents": retrieved_docs,
            "selected_retriever": selected_retriever_name,
            "detected_intent": intent,
            "query": query
        }


def create_retriever_config_file(config_path: str = "retrievers_config.json"):
    """
    Create a comprehensive retriever configuration file based on the provided JSON.
    """
    config = {
        "description": "Retriever catalog specifically optimized for LoCoMo dataset",
        "query_intent_patterns": {
            "temporal": {
                "patterns": ["when did", "what date", "what time", "how long ago", "what year", "what month"],
                "retrievers": ["ensemble", "bm25", "tfidf"],
                "confidence": "high"
            },
            "multi_hop": {
                "patterns": ["how did.*progress", "describe evolution", "what happened after", "what led to", "trace the development"],
                "retrievers": ["ensemble", "tfidf"],
                "confidence": "high"
            },
            "speaker_specific": {
                "patterns": ["what did.*say", "what did.*do", "'.*s ", "did.*mention"],
                "retrievers": ["ensemble", "bm25", "tfidf"],
                "confidence": "high"
            },
            "commonsense": {
                "patterns": ["what fields would", "why did", "how likely", "what interests", "what career", "infer", "suggest"],
                "retrievers": ["faiss", "ensemble", "tfidf"],
                "confidence": "high"
            },
            "adversarial": {
                "patterns": ["did.*mention.*when.*didn", "ever mention", "at any point"],
                "retrievers": ["bm25", "ensemble", "tfidf"],
                "confidence": "high"
            },
            "exploratory": {
                "patterns": ["what topics did", "describe conversations", "what activities did", "broad overview"],
                "retrievers": ["ensemble", "tfidf"],
                "confidence": "high"
            },
            "identity": {
                "patterns": ["who is", "what is.*identity", "describe.*", "what is.*relationship"],
                "retrievers": ["faiss", "ensemble", "tfidf"],
                "confidence": "high"
            },
            "single_hop": {
                "patterns": ["what did.*say about", "what.*start", "what.*paint"],
                "retrievers": ["bm25", "ensemble", "tfidf"],
                "confidence": "high"
            },
            "recent": {
                "patterns": ["recently", "lately", "latest", "most recent", "newest"],
                "retrievers": ["ensemble", "tfidf"],
                "confidence": "medium"
            }
        },
        "fallback_retriever": "tfidf",
        "retriever_priority": ["ensemble", "bm25", "faiss", "tfidf"],
        "best_practices": {
            "temporal_queries": "ALWAYS use ensemble for temporal queries (LoCoMo's hardest category)",
            "multi_hop_queries": "USE ensemble for multi-hop reasoning", 
            "adversarial_queries": "USE bm25 for exact keyword verification",
            "default_strategy": "DEFAULT to ensemble for unclear query types"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path