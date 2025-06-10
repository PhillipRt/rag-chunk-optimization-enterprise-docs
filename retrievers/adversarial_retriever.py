import os
import logging
import numpy as np
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

class AdversarialRetriever(BaseRetriever):
    """
    Retriever that deliberately returns the least semantically similar documents to the query.
    This serves as a "worst-case" baseline to evaluate how the LLM handles irrelevant context.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adversarial retriever.
        
        Args:
            config: Configuration dictionary with these keys:
                - max_features: Max number of features for TF-IDF (default: 5000)
                - chunk_size: Size of document chunks (default: 1000)
                - chunk_overlap: Overlap between chunks (default: 200)
        """
        super().__init__(config)
        self.max_features = config.get("max_features", 5000)
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
        
        # Initialize TF-IDF vectorizer for semantic dissimilarity calculation
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english'
        )
        
        # Fit and transform documents
        chunk_texts = [c["content"] for c in self.chunks]
        self.document_vectors = self.vectorizer.fit_transform(chunk_texts)
        
        logger.info(f"Adversarial vectorizer created with {self.document_vectors.shape[1]} features")
        self.is_initialized = True
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the least similar documents to the query.
        
        Args:
            query: The query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of the least semantically similar documents
        """
        if not self.is_initialized:
            raise ValueError("Retriever not initialized. Call setup() first.")
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Get bottom-k indices (least similar documents)
        # Exclude documents with zero similarity (no overlap at all)
        non_zero_indices = np.where(similarities > 0)[0]
        if len(non_zero_indices) >= top_k:
            # Sort by similarity and take the least similar documents
            bottom_indices = non_zero_indices[np.argsort(similarities[non_zero_indices])[:top_k]]
        else:
            # If not enough non-zero similarity documents, include some zero similarity ones
            zero_indices = np.where(similarities == 0)[0]
            needed_zeros = min(top_k - len(non_zero_indices), len(zero_indices))
            
            # Take all non-zero indices (sorted ascending) and some zero indices
            bottom_indices = np.concatenate([
                non_zero_indices[np.argsort(similarities[non_zero_indices])], 
                np.random.choice(zero_indices, size=needed_zeros, replace=False)
            ])[:top_k]
        
        # Format results
        results = []
        for idx in bottom_indices:
            chunk = self.chunks[idx]
            # Invert the similarity score to make it look like a regular score
            inverted_score = 1.0 - similarities[idx]
            results.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "score": float(inverted_score)
            })
        
        # Sort by inverted score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
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
            "type": "AdversarialRetriever",
            "config": {
                "max_features": self.max_features,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        }