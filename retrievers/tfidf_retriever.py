import os
import logging
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

class TFIDFRetriever(BaseRetriever):
    """
    Simple TF-IDF retriever using scikit-learn's TfidfVectorizer.
    This retriever serves as a traditional baseline for comparison with
    more sophisticated embedding approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TF-IDF retriever.
        
        Args:
            config: Configuration dictionary with these keys:
                - max_features: Max number of features for TF-IDF (default: 5000)
                - ngram_range: n-gram range for TF-IDF (default: (1, 2))
                - chunk_size: Size of document chunks (default: 1000)
                - chunk_overlap: Overlap between chunks (default: 200)
        """
        super().__init__(config)
        self.max_features = config.get("max_features", 5000)
        self.ngram_range = config.get("ngram_range", (1, 2))
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        
        # Will be initialized in setup()
        self.vectorizer = None
        self.document_vectors = None
        self.documents = None
        self.chunks = None
    
    def setup(self, documents: List[Dict[str, Any]]) -> None:
        """Set up the retriever with documents.
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents
        
        # Create chunks
        if self.chunk_size > 0:
            self.chunks = self._create_chunks(documents)
            logger.info(f"Split {len(documents)} documents into {len(self.chunks)} chunks")
        else:
            self.chunks = documents
            logger.info(f"Using {len(documents)} documents without chunking")
        
        # Initialize TF-IDF vectorizer
        # Ensure ngram_range is a tuple as required by scikit-learn
        ngram_range_tuple = tuple(self.ngram_range) if isinstance(self.ngram_range, list) else self.ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=ngram_range_tuple,
            stop_words='english'
        )
        
        # Fit and transform documents
        chunk_texts = [c["content"] for c in self.chunks]
        self.document_vectors = self.vectorizer.fit_transform(chunk_texts)
        
        logger.info(f"TF-IDF vectorizer created with {self.document_vectors.shape[1]} features")
        self.is_initialized = True
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents using TF-IDF similarity.
        
        Args:
            query: The query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.is_initialized:
            raise ValueError("Retriever not initialized. Call setup() first.")
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "score": float(similarities[idx])
            })
        
        return results
    
    def _create_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks from documents with specified chunk size and overlap.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunks
        """
        chunks = []
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            # Simple character-based chunking
            doc_chunks = []
            for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunk_text = content[i:i + self.chunk_size]
                if not chunk_text.strip():
                    continue
                
                # Create a new chunk
                chunk = {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": len(doc_chunks),
                        "chunk_start": i,
                        "chunk_end": min(i + self.chunk_size, len(content))
                    }
                }
                doc_chunks.append(chunk)
            
            chunks.extend(doc_chunks)
        
        return chunks
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this retriever."""
        return {
            "type": "TFIDFRetriever",
            "config": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        }
