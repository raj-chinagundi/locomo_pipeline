"""
Unified Retriever Class
Provides consistent interface for all retriever types with property-based access.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import TFIDFRetriever, BM25Retriever, SVMRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Advanced retrievers - with compatibility checks
try:
    from langchain_classic.retrievers import EnsembleRetriever
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False

try:
    from langchain_classic.retrievers import MultiQueryRetriever
    HAS_MULTIQUERY = True
except ImportError:
    HAS_MULTIQUERY = False

try:
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
    HAS_COMPRESSION = True
except ImportError:
    HAS_COMPRESSION = False

try:
    from langchain_classic.retrievers import ParentDocumentRetriever
    from langchain_core.stores import InMemoryStore
    HAS_PARENT = True
except ImportError:
    HAS_PARENT = False

try:
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
    from langchain.chains.query_constructor.base import AttributeInfo
    HAS_SELF_QUERY = True
except ImportError:
    HAS_SELF_QUERY = False

try:
    from langchain_classic.retrievers import MultiVectorRetriever
    from langchain.storage import InMemoryStore as MVInMemoryStore
    HAS_MULTIVECTOR = True
except ImportError:
    HAS_MULTIVECTOR = False

try:
    from langchain_community.document_transformers import LongContextReorder
    from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
    HAS_REORDER = True
except ImportError:
    HAS_REORDER = False


class Retriever:
    """
    Unified retriever class providing consistent access to all retriever types.
    
    Usage:
        retriever = Retriever(documents, llm=llm, config=config)
        results = retriever.bm25.invoke(query)
        results = retriever.faiss.invoke(query)
    
    All retrievers follow LangChain's standard interface:
        - invoke(query: str) -> List[Document]
        - get_relevant_documents(query: str) -> List[Document]
    """
    
    def __init__(
        self, 
        documents: List[Document],
        llm=None,
        session_docs: Optional[List[Document]] = None,
        config: Optional[dict] = None
    ):
        """
        Initialize Retriever with documents and optional components.
        
        Args:
            documents: List of Document objects (utterance-level)
            llm: Optional LLM instance for advanced retrievers
            session_docs: Optional session-level documents for Parent Document retriever
            config: Optional configuration dict (uses defaults if None)
        """
        self.documents = documents
        self.llm = llm
        self.session_docs = session_docs
        self.config = config or self._default_config()
        
        # Lazy-loaded components
        self._embeddings = None
        self._vectorstore = None
        
        # Cached retrievers
        self._retrievers = {}
    
    def _default_config(self) -> dict:
        """Default configuration for retrievers."""
        return {
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "top_k": 10,
            "ensemble_weights": [0.5, 0.5],
            "multiquery_variations": 3,
            "parent_chunk_size": 400,
            "parent_chunk_overlap": 50,
            "mmr_diversity": 0.5,
            "mmr_fetch_k": 20,
        }
    
    def _get_embeddings(self):
        """Lazy load embeddings (shared across retrievers)."""
        if self._embeddings is None:
            print(f"  Loading embeddings: {self.config['embedding_model']}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config['embedding_model']
            )
        return self._embeddings
    
    def _get_vectorstore(self):
        """Lazy build vector store (shared across retrievers)."""
        if self._vectorstore is None:
            print("  Building FAISS vector index...")
            self._vectorstore = FAISS.from_documents(
                self.documents,
                self._get_embeddings()
            )
        return self._vectorstore
    
    # ========================================================================
    # RETRIEVER PROPERTIES - Access via retriever.bm25, retriever.faiss, etc.
    # ========================================================================
    
    @property
    def bm25(self):
        """BM25 Retriever - Keyword-based probabilistic ranking."""
        if 'bm25' not in self._retrievers:
            print("[BM25] Initializing keyword-based retriever...")
            self._retrievers['bm25'] = BM25Retriever.from_documents(
                self.documents,
                k=self.config['top_k']
            )
        return self._retrievers['bm25']
    
    @property
    def tfidf(self):
        """TF-IDF Retriever - Traditional information retrieval."""
        if 'tfidf' not in self._retrievers:
            print("[TF-IDF] Initializing traditional IR retriever...")
            self._retrievers['tfidf'] = TFIDFRetriever.from_documents(
                self.documents,
                k=self.config['top_k']
            )
        return self._retrievers['tfidf']
    
    @property
    def faiss(self):
        """FAISS Retriever - Semantic vector similarity search."""
        if 'faiss' not in self._retrievers:
            print("[FAISS] Initializing semantic vector retriever...")
            vectorstore = self._get_vectorstore()
            self._retrievers['faiss'] = vectorstore.as_retriever(
                search_kwargs={"k": self.config['top_k']}
            )
        return self._retrievers['faiss']
    
    @property
    def ensemble(self):
        """Ensemble Retriever - Hybrid (FAISS + BM25) fusion."""
        if not HAS_ENSEMBLE:
            print("[Ensemble] Not available - missing langchain_classic package")
            return None
        
        if 'ensemble' not in self._retrievers:
            print("[Ensemble] Initializing hybrid retriever (FAISS + BM25)...")
            self._retrievers['ensemble'] = EnsembleRetriever(
                retrievers=[self.faiss, self.bm25],
                weights=self.config['ensemble_weights']
            )
        return self._retrievers['ensemble']
    
    @property
    def multiquery(self):
        """MultiQuery Retriever - Query expansion with LLM."""
        if not HAS_MULTIQUERY:
            print("[MultiQuery] Not available - missing langchain_classic package")
            return None
        
        if self.llm is None:
            print("[MultiQuery] Not available - requires LLM")
            return None
        
        if 'multiquery' not in self._retrievers:
            print("[MultiQuery] Initializing query expansion retriever...")
            self._retrievers['multiquery'] = MultiQueryRetriever.from_llm(
                retriever=self.faiss,
                llm=self.llm
            )
        return self._retrievers['multiquery']
    
    @property
    def contextual_compression(self):
        """Contextual Compression Retriever - LLM-based filtering."""
        if not HAS_COMPRESSION:
            print("[ContextualCompression] Not available - missing langchain_classic package")
            return None
        
        if self.llm is None:
            print("[ContextualCompression] Not available - requires LLM")
            return None
        
        if 'contextual_compression' not in self._retrievers:
            print("[ContextualCompression] Initializing reranking retriever...")
            compressor = LLMChainExtractor.from_llm(self.llm)
            self._retrievers['contextual_compression'] = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.faiss
            )
        return self._retrievers['contextual_compression']
    
    @property
    def parent_document(self):
        """Parent Document Retriever - Session-level context retrieval."""
        if not HAS_PARENT:
            print("[ParentDocument] Not available - missing langchain_classic package")
            return None
        
        if self.session_docs is None:
            print("[ParentDocument] Not available - requires session documents")
            return None
        
        if 'parent_document' not in self._retrievers:
            print("[ParentDocument] Initializing context-aware retriever...")
            
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['parent_chunk_size'],
                chunk_overlap=self.config['parent_chunk_overlap']
            )
            
            store = InMemoryStore()
            
            retriever = ParentDocumentRetriever(
                vectorstore=self._get_vectorstore(),
                docstore=store,
                child_splitter=child_splitter,
                search_kwargs={"k": self.config['top_k']}
            )
            
            retriever.add_documents(self.session_docs)
            self._retrievers['parent_document'] = retriever
        
        return self._retrievers['parent_document']
    
    @property
    def self_query(self):
        """Self-Query Retriever - Metadata-aware natural language filtering."""
        if not HAS_SELF_QUERY:
            print("[SelfQuery] Not available - missing langchain_classic package")
            return None
        
        if self.llm is None:
            print("[SelfQuery] Not available - requires LLM")
            return None
        
        if 'self_query' not in self._retrievers:
            print("[SelfQuery] Initializing metadata-aware retriever...")
            
            metadata_field_info = [
                AttributeInfo(
                    name="speaker",
                    description="The person speaking (e.g., Caroline, Melanie)",
                    type="string"
                ),
                AttributeInfo(
                    name="session",
                    description="The conversation session (e.g., session_1, session_2)",
                    type="string"
                ),
                AttributeInfo(
                    name="dia_id",
                    description="The dialogue ID (e.g., D1:3)",
                    type="string"
                ),
            ]
            
            document_content_description = "Utterances from conversations between speakers"
            
            self._retrievers['self_query'] = SelfQueryRetriever.from_llm(
                llm=self.llm,
                vectorstore=self._get_vectorstore(),
                document_contents=document_content_description,
                metadata_field_info=metadata_field_info,
                verbose=True
            )
        
        return self._retrievers['self_query']
    
    @property
    def svm(self):
        """SVM Retriever - Machine learning based ranking."""
        if 'svm' not in self._retrievers:
            print("[SVM] Initializing ML-based retriever...")
            self._retrievers['svm'] = SVMRetriever.from_documents(
                self.documents,
                self._get_embeddings(),
                k=self.config['top_k']
            )
        return self._retrievers['svm']
    
    @property
    def mmr(self):
        """MMR Retriever - Max Marginal Relevance (diversity-aware)."""
        if 'mmr' not in self._retrievers:
            print("[MMR] Initializing diversity-aware retriever...")
            vectorstore = self._get_vectorstore()
            self._retrievers['mmr'] = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.config['top_k'],
                    "fetch_k": self.config['mmr_fetch_k'],
                    "lambda_mult": self.config['mmr_diversity']
                }
            )
        return self._retrievers['mmr']
    
    @property
    def time_weighted(self):
        """Time-Weighted Retriever - Recency-aware retrieval."""
        if 'time_weighted' not in self._retrievers:
            print("[TimeWeighted] Initializing temporal-aware retriever...")
            print("  Note: Session timestamps now stored in metadata (session_datetime)")
            print("  Note: Current implementation uses basic FAISS (upgrade to TimeWeightedVectorStoreRetriever for full temporal scoring)")
            vectorstore = self._get_vectorstore()
            self._retrievers['time_weighted'] = vectorstore.as_retriever(
                search_kwargs={"k": self.config['top_k']}
            )
        return self._retrievers['time_weighted']
    
    @property
    def multivector(self):
        """Multi-Vector Retriever - Multiple representations per document."""
        if not HAS_MULTIVECTOR:
            print("[MultiVector] Not available - missing langchain_classic package")
            return None
        
        if self.llm is None:
            print("[MultiVector] Not available - requires LLM")
            return None
        
        if 'multivector' not in self._retrievers:
            print("[MultiVector] Initializing multi-representation retriever...")
            from uuid import uuid4
            
            store = MVInMemoryStore()
            id_key = "doc_id"
            
            vectorstore = self._get_vectorstore()
            
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                docstore=store,
                id_key=id_key,
            )
            
            doc_ids = [str(uuid4()) for _ in self.documents]
            retriever.vectorstore.add_documents(self.documents)
            retriever.docstore.mset(list(zip(doc_ids, self.documents)))
            
            print("  Note: Using original utterances")
            self._retrievers['multivector'] = retriever
        
        return self._retrievers['multivector']
    
    @property
    def long_context_reorder(self):
        """Long Context Reorder Retriever - Optimizes document ordering for LLMs."""
        if not HAS_REORDER:
            print("[LongContextReorder] Not available - missing required packages")
            return None
        
        if 'long_context_reorder' not in self._retrievers:
            print("[LongContextReorder] Initializing context-reordering retriever...")
            
            reordering = LongContextReorder()
            pipeline = DocumentCompressorPipeline(transformers=[reordering])
            
            self._retrievers['long_context_reorder'] = ContextualCompressionRetriever(
                base_compressor=pipeline,
                base_retriever=self.faiss
            )
        
        return self._retrievers['long_context_reorder']
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def list_available(self) -> List[str]:
        """List all available retriever names."""
        retrievers = [
            'bm25', 'tfidf', 'faiss', 'svm', 'mmr', 'time_weighted'
        ]
        
        if HAS_ENSEMBLE:
            retrievers.append('ensemble')
        if HAS_MULTIQUERY and self.llm:
            retrievers.append('multiquery')
        if HAS_COMPRESSION and self.llm:
            retrievers.append('contextual_compression')
        if HAS_PARENT and self.session_docs:
            retrievers.append('parent_document')
        if HAS_SELF_QUERY and self.llm:
            retrievers.append('self_query')
        if HAS_MULTIVECTOR and self.llm:
            retrievers.append('multivector')
        if HAS_REORDER:
            retrievers.append('long_context_reorder')
        
        return retrievers
    
    def get(self, name: str):
        """Get a retriever by name."""
        return getattr(self, name, None)


# Registry of retriever metadata
RETRIEVER_INFO = {
    "bm25": {"type": "keyword", "requires_llm": False, "requires_sessions": False},
    "tfidf": {"type": "keyword", "requires_llm": False, "requires_sessions": False},
    "faiss": {"type": "vector", "requires_llm": False, "requires_sessions": False},
    "ensemble": {"type": "hybrid", "requires_llm": False, "requires_sessions": False},
    "multiquery": {"type": "query_expansion", "requires_llm": True, "requires_sessions": False},
    "contextual_compression": {"type": "reranking", "requires_llm": True, "requires_sessions": False},
    "parent_document": {"type": "hierarchical", "requires_llm": False, "requires_sessions": True},
    "self_query": {"type": "metadata_aware", "requires_llm": True, "requires_sessions": False},
    "svm": {"type": "ml_lexical", "requires_llm": False, "requires_sessions": False},
    "mmr": {"type": "diversification", "requires_llm": False, "requires_sessions": False},
    "time_weighted": {"type": "temporal", "requires_llm": False, "requires_sessions": False},
    "multivector": {"type": "multi_representation", "requires_llm": True, "requires_sessions": False},
    "long_context_reorder": {"type": "reordering", "requires_llm": False, "requires_sessions": False}
}

