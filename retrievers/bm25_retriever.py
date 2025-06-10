import os
import logging
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

class BM25Retriever(BaseRetriever):
    """Pure BM25 retriever using rank_bm25 library."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BM25 retriever.
        
        Args:
            config: Configuration dictionary containing:
                - chunk_size: Size of document chunks (default: 1000)
                - chunk_overlap: Overlap between chunks (default: 200)
                - min_score_threshold: Minimum score threshold (default: 0.0)
                - k1: BM25 k1 parameter (default: 1.5)
                - b: BM25 b parameter (default: 0.75)
                - epsilon: BM25 epsilon parameter (default: 0.25)
        """
        super().__init__(config)
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.min_score_threshold = config.get("min_score_threshold", 0.0)
        
        # BM25 specific parameters
        self.k1 = config.get("k1", 1.5)
        self.b = config.get("b", 0.75)
        self.epsilon = config.get("epsilon", 0.25)
        
        # Will be initialized in setup()
        self.bm25_retriever = None
        self.bm25_corpus = None
        self.bm25_docs = None
    
    def setup(self, documents: List[Dict[str, Any]]) -> None:
        """Set up the BM25 retriever with documents."""
        logger.info("Setting up BM25 retriever")
        
        # Convert to Langchain documents
        langchain_docs = [
            Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            ) for doc in documents
        ]
        
        # Apply chunking if needed
        if self.chunk_size > 0:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            langchain_docs = text_splitter.split_documents(langchain_docs)
            logger.info(f"Split documents into {len(langchain_docs)} chunks for BM25")
        
        # Extract text content and tokenize (simple whitespace tokenization)
        self.bm25_corpus = [doc.page_content for doc in langchain_docs]
        self.bm25_docs = langchain_docs  # Store for later reference
        tokenized_corpus = [text.split() for text in self.bm25_corpus]
        
        # Initialize BM25 retriever with the tokenized corpus and custom parameters
        self.bm25_retriever = BM25Okapi(
            tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )
        logger.info(f"BM25 retriever initialized with {len(tokenized_corpus)} chunks")
        logger.info(f"BM25 parameters: k1={self.k1}, b={self.b}, epsilon={self.epsilon}")
        
        self.is_initialized = True
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents using BM25."""
        if not self.is_initialized:
            raise ValueError("Retriever not initialized. Call setup() first.")
        
        # Tokenize query
        tokenized_query = query.split()
        
        # Get scores for all documents in the corpus
        bm25_scores = self.bm25_retriever.get_scores(tokenized_query)
        
        # Create document-score pairs and sort by score
        bm25_doc_scores = list(zip(self.bm25_docs, bm25_scores))
        bm25_doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k results
        top_bm25_doc_scores = bm25_doc_scores[:top_k]
        
        # Convert to our standard format with actual scores
        results = []
        for doc, score in top_bm25_doc_scores:
            # Only include if score is positive and above threshold
            if score > 0 and score >= self.min_score_threshold:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)  # Ensure score is a float
                })
        
        logger.info(f"BM25 retrieved {len(results)} documents")
        return results
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this retriever for experiment tracking."""
        return {
            "type": "BM25Retriever",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_score_threshold": self.min_score_threshold,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon
        }
