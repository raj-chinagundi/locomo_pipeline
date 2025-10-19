"""
Data Loader for Locomo Dataset
Handles loading and processing of conversation data with configurable filtering.
"""

import json
from typing import List, Dict, Tuple, Optional, Union
from langchain_core.documents import Document


class DataLoader:
    """Load and process Locomo dataset for retrieval experiments."""
    
    def __init__(self, json_path: str):
        """
        Initialize DataLoader with path to Locomo JSON file.
        
        Args:
            json_path: Path to locomo10.json file
        """
        self.json_path = json_path
        self.raw_data = []
        self.conversations = {}
        self.documents = []
        self.qa_pairs = []
        
    def load_data(
        self, 
        sample_ids: Optional[Union[List[str], str]] = None,
        limit: Optional[int] = None
    ) -> Tuple[List[Document], List[Dict], Dict]:
        """
        Load and process Locomo data with optional filtering.
        
        Args:
            sample_ids: Specific sample IDs to load. Options:
                       - None or "all": Load all conversations
                       - List of sample_ids: Load only specified conversations
                       - Single sample_id string: Load one conversation
            limit: Maximum number of conversations to load (None = no limit)
        
        Returns:
            Tuple of:
                - documents: List of utterances as Document objects
                - qa_pairs: List of Q&A dictionaries with evidence
                - conversations: Dict mapping sample_id to conversation data
        """
        print(f"Loading data from {self.json_path}...")
        
        # Read raw JSON
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        print(f"Found {len(self.raw_data)} total conversations")
        
        # Apply filtering
        filtered_data = self._filter_conversations(self.raw_data, sample_ids, limit)
        print(f"Processing {len(filtered_data)} conversations after filtering")
        
        # Process filtered conversations
        self._process_conversations(filtered_data)
        
        print(f"Loaded {len(self.documents)} utterances and {len(self.qa_pairs)} Q&A pairs")
        
        return self.documents, self.qa_pairs, self.conversations
    
    def _filter_conversations(
        self, 
        data: List[Dict], 
        sample_ids: Optional[Union[List[str], str]], 
        limit: Optional[int]
    ) -> List[Dict]:
        """Filter conversations based on sample_ids and limit."""
        filtered = data
        
        # Filter by sample_ids
        if sample_ids is not None and sample_ids != "all":
            if isinstance(sample_ids, str):
                sample_ids = [sample_ids]
            filtered = [conv for conv in filtered if conv.get('sample_id') in sample_ids]
            print(f"  Filtered to sample_ids: {sample_ids}")
        
        # Apply limit
        if limit is not None and limit > 0:
            filtered = filtered[:limit]
            print(f"  Limited to first {limit} conversations")
        
        return filtered
    
    def _process_conversations(self, data: List[Dict]):
        """Process conversations into documents and QA pairs."""
        self.documents = []
        self.qa_pairs = []
        self.conversations = {}
        
        for sample in data:
            sample_id = sample.get('sample_id', 'unknown')
            self.conversations[sample_id] = sample
            
            # Extract utterances from all sessions
            conversation = sample.get('conversation', {})
            for key, value in conversation.items():
                if key.startswith('session_') and isinstance(value, list):
                    # Get session timestamp (e.g., "session_1_date_time")
                    session_datetime_key = f"{key}_date_time"
                    session_datetime = conversation.get(session_datetime_key, None)
                    
                    for utterance in value:
                        text = utterance.get('text', '').strip()
                        dia_id = utterance.get('dia_id', '')
                        speaker = utterance.get('speaker', '')
                        
                        if text and dia_id:
                            doc = Document(
                                page_content=text,
                                metadata={
                                    'sample_id': sample_id,
                                    'dia_id': dia_id,
                                    'speaker': speaker,
                                    'session': key,
                                    'session_datetime': session_datetime
                                }
                            )
                            self.documents.append(doc)
            
            # Extract Q&A pairs with evidence
            qa_list = sample.get('qa', [])
            for qa in qa_list:
                self.qa_pairs.append({
                    'sample_id': sample_id,
                    'question': qa.get('question', ''),
                    'answer': qa.get('answer', ''),
                    'evidence': qa.get('evidence', []),
                    'category': qa.get('category', None)
                })
    
    def get_session_documents(self) -> List[Document]:
        """
        Get session-level documents (groups of utterances).
        Useful for Parent Document retriever that needs larger context.
        
        Returns:
            List of Document objects where each document is an entire session
        """
        if not self.conversations:
            raise ValueError("No data loaded. Call load_data() first.")
        
        session_docs = []
        
        for sample_id, sample in self.conversations.items():
            conversation = sample.get('conversation', {})
            
            for key, value in conversation.items():
                if key.startswith('session_') and isinstance(value, list):
                    # Get session timestamp
                    session_datetime_key = f"{key}_date_time"
                    session_datetime = conversation.get(session_datetime_key, None)
                    
                    # Combine all utterances in session
                    session_text = "\n".join([
                        f"{u.get('speaker', '')}: {u.get('text', '')}"
                        for u in value if u.get('text', '').strip()
                    ])
                    
                    # Get all dia_ids in this session
                    dia_ids = [u.get('dia_id', '') for u in value if u.get('dia_id')]
                    
                    if session_text.strip():
                        doc = Document(
                            page_content=session_text,
                            metadata={
                                'sample_id': sample_id,
                                'session': key,
                                'session_datetime': session_datetime,
                                'dia_ids': dia_ids,
                                'type': 'session'
                            }
                        )
                        session_docs.append(doc)
        
        print(f"Created {len(session_docs)} session-level documents")
        return session_docs
    
    def get_qa_pairs_for_sample(self, sample_id: str) -> List[Dict]:
        """
        Get QA pairs for a specific conversation.
        
        Args:
            sample_id: The conversation sample ID
            
        Returns:
            List of QA dictionaries for that conversation
        """
        return [qa for qa in self.qa_pairs if qa['sample_id'] == sample_id]

