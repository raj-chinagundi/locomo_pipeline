"""
Retriever implementations for RAG pipeline.
Each retriever can be plugged in to retrieve relevant documents for a query.
"""
from typing import List, Dict, Any
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Base class for all retrievers."""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        """
        Initialize retriever with documents.
        
        Args:
            documents: List of document dictionaries with 'id', 'text', and 'speaker'
        """
        self.documents = documents
        self.texts = [doc["text"] for doc in documents]
        self.doc_ids = [doc["id"] for doc in documents]
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for the query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of document dictionaries with 'id', 'text', and 'speaker'
        """
        pass


class TFIDFRetriever(BaseRetriever):
    """TF-IDF based retriever using scikit-learn."""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        super().__init__(documents)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            self.vectorizer = TfidfVectorizer()
            self.doc_vectors = self.vectorizer.fit_transform(self.texts)
            self.cosine_similarity = cosine_similarity
            self.np = np
        except ImportError:
            raise ImportError("Please install scikit-learn: pip install scikit-learn")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.vectorizer.transform([query])
        similarities = self.cosine_similarity(query_vector, self.doc_vectors)[0]
        top_indices = self.np.argsort(similarities)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]


class SVMRetriever(BaseRetriever):
    """SVM based retriever using LangChain's SVMRetriever with HuggingFace embeddings."""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        super().__init__(documents)
        try:
            from langchain_community.retrievers import SVMRetriever as LangChainSVMRetriever
            from langchain_huggingface import HuggingFaceEmbeddings
            
            # Use HuggingFace embeddings with the same model
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Create retriever using from_texts
            self.retriever = LangChainSVMRetriever.from_texts(self.texts, embeddings)
            
        except ImportError as e:
            raise ImportError(f"Please install: pip install langchain-community langchain-huggingface. Error: {e}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Use LangChain's invoke method
        results = self.retriever.invoke(query)
        
        # Map back to our document format with IDs
        retrieved_docs = []
        for i, result in enumerate(results[:top_k]):
            # Find matching document by text
            for doc in self.documents:
                if doc["text"] == result.page_content:
                    retrieved_docs.append(doc)
                    break
        
        return retrieved_docs


class FAISSRetriever(BaseRetriever):
    """FAISS based retriever with HuggingFace embeddings."""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        super().__init__(documents)
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.doc_embeddings = self.model.encode(self.texts)
            self.np = np
            
            try:
                import faiss
                self.use_faiss = True
                dimension = self.doc_embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(self.doc_embeddings.astype('float32'))
            except ImportError:
                print("FAISS not installed, using numpy cosine similarity instead")
                self.use_faiss = False
                
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query])[0]
        
        if self.use_faiss:
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                top_k
            )
            return [self.documents[i] for i in indices[0]]
        else:
            # Fallback to cosine similarity
            similarities = self.np.dot(self.doc_embeddings, query_embedding)
            top_indices = self.np.argsort(similarities)[::-1][:top_k]
            return [self.documents[i] for i in top_indices]


class NanoPQRetriever(BaseRetriever):
    """Product Quantization retriever using nanopq."""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        super().__init__(documents)
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.doc_embeddings = self.model.encode(self.texts)
            self.np = np
            
            try:
                import nanopq
                self.use_pq = True
                # Train PQ with 2 subspaces and 2 clusters (M=2, Ks=2)
                self.pq = nanopq.PQ(M=2, Ks=256)
                self.pq.fit(self.doc_embeddings)
                self.encoded_docs = self.pq.encode(self.doc_embeddings)
            except ImportError:
                print("nanopq not installed, using numpy cosine similarity instead")
                self.use_pq = False
                
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query])[0]
        
        if self.use_pq:
            # Compute distances using PQ
            distances = self.pq.dtable(query_embedding).adist(self.encoded_docs)
            top_indices = self.np.argsort(distances)[:top_k]
            return [self.documents[i] for i in top_indices]
        else:
            # Fallback to cosine similarity
            similarities = self.np.dot(self.doc_embeddings, query_embedding)
            top_indices = self.np.argsort(similarities)[::-1][:top_k]
            return [self.documents[i] for i in top_indices]


class BM25Retriever(BaseRetriever):
    """BM25 retriever using rank-bm25."""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        super().__init__(documents)
        try:
            from rank_bm25 import BM25Okapi
            
            tokenized_docs = [doc.lower().split() for doc in self.texts]
            self.bm25 = BM25Okapi(tokenized_docs)
        except ImportError:
            raise ImportError("Please install rank-bm25: pip install rank-bm25")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]


def get_all_retrievers(documents: List[Dict[str, Any]]) -> Dict[str, BaseRetriever]:
    """
    Get all available retrievers initialized with documents.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Dictionary mapping retriever names to retriever instances
    """
    retrievers = {}
    
    # Try to initialize each retriever
    try:
        retrievers["tfidf"] = TFIDFRetriever(documents)
    except Exception as e:
        print(f"Failed to initialize TFIDFRetriever: {e}")
    
    try:
        retrievers["svm"] = SVMRetriever(documents)
    except Exception as e:
        print(f"Failed to initialize SVMRetriever: {e}")
    
    try:
        retrievers["faiss"] = FAISSRetriever(documents)
    except Exception as e:
        print(f"Failed to initialize FAISSRetriever: {e}")
    
    try:
        retrievers["nanopq"] = NanoPQRetriever(documents)
    except Exception as e:
        print(f"Failed to initialize NanoPQRetriever: {e}")
    
    try:
        retrievers["bm25"] = BM25Retriever(documents)
    except Exception as e:
        print(f"Failed to initialize BM25Retriever: {e}")
    
    return retrievers

