import logging
import random
from typing import List, Dict, Any

from retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

class RandomRetriever(BaseRetriever):
    """
    Retriever that returns random documents regardless of the query.
    This serves as a lower-bound baseline to compare other retrievers against.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the random retriever.
        
        Args:
            config: Configuration dictionary with these keys:
                - seed: Random seed for reproducibility (default: 42)
                - chunk_size: Size of document chunks (default: 1000)
                - chunk_overlap: Overlap between chunks (default: 200)
        """
        super().__init__(config)
        self.seed = config.get("seed", 42)
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        
        # Set random seed
        random.seed(self.seed)
        
        # Will be initialized in setup()
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
        
        self.is_initialized = True
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve random documents regardless of the query.
        
        Args:
            query: The query string (ignored)
            top_k: Number of documents to retrieve
            
        Returns:
            List of randomly selected documents with random scores
        """
        if not self.is_initialized:
            raise ValueError("Retriever not initialized. Call setup() first.")
        
        # Randomly sample chunks
        sample_size = min(top_k, len(self.chunks))
        selected_chunks = random.sample(self.chunks, sample_size)
        
        # Generate random scores between 0.0 and 1.0
        results = []
        for chunk in selected_chunks:
            results.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "score": random.random()  # Random score between 0 and 1
            })
        
        # Sort by random score in descending order to simulate a ranked retrieval
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
            "type": "RandomRetriever",
            "config": {
                "seed": self.seed,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        }