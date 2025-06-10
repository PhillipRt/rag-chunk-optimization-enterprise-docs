import os
import time
import random
import logging
from typing import List, Dict, Any
from tqdm import tqdm

from langchain.embeddings.base import Embeddings

from retrievers.embedding_retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)

class GeminiEmbeddings(Embeddings):
    """Embeddings implementation that uses Google's Gemini embedding model with retry logic."""

    def __init__(self, api_key: str = None, model_name: str = "text-embedding-004"):
        """Initialize the Gemini embeddings with API key."""
        # Get key directly from argument (passed from GeminiRetriever config)
        self.api_key = api_key

        # Ensure model name has the correct prefix
        if model_name.startswith("models/") or model_name.startswith("tunedModels/"):
            self.model_name = model_name
        else:
            self.model_name = f"models/{model_name}"
        
        logger.info(f"Using model: {self.model_name}")
        
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini embeddings")
        
        # Import and configure Google GenAI
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except ImportError:
            raise ImportError(
                "The google-generativeai package is required to use Gemini embeddings. "
                "Install it with `pip install google-generativeai`"
            )
        
        # Simple configuration for batch processing
        self.batch_size = 200  # Use large batch size for efficiency
        self.max_retries = 5  # Retry up to 5 times on errors
        self.base_wait_time = 1.0  # Base wait time for retries
        self.max_wait_time = 30.0  # Max wait time for retries
        
        logger.info(f"Gemini embeddings initialized with model: {self.model_name}")

    def _backoff_time(self, retry_count):
        """Calculate exponential backoff time with jitter for retries."""
        wait_time = min(
            self.max_wait_time,
            self.base_wait_time * (2 ** retry_count) * (0.5 + random.random())
        )
        return wait_time

    # Class variable to track overall progress
    _total_docs_processed = 0
    _total_docs_count = 0
    _pbar = None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents with batching and retry logic."""
        if not texts:
            return []
            
        # Update counters for more accurate progress tracking
        batch_num = getattr(self.__class__, '_total_docs_processed', 0) // self.batch_size + 1
        documents_to_process = len(texts)
        
        # Initialize progress bar if needed
        if GeminiEmbeddings._total_docs_count > 0 and GeminiEmbeddings._pbar is None:
            GeminiEmbeddings._pbar = tqdm(
                total=GeminiEmbeddings._total_docs_count, 
                desc="Embedding documents", 
                unit="docs"
            )
            GeminiEmbeddings._pbar.update(GeminiEmbeddings._total_docs_processed)
        
        # Only log for significant batches to avoid cluttering the logs
        if len(texts) > 1:
            logger.info(f"Starting to embed {len(texts)} documents (processed {GeminiEmbeddings._total_docs_processed} of total so far)")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_embeddings = []
            
            # Process each text in the batch
            for text in batch_texts:
                retry_count = 0
                while retry_count <= self.max_retries:
                    try:
                        # Make the API call without preemptive throttling
                        result = self.genai.embed_content(
                            model=self.model_name,
                            content=text,
                            task_type="retrieval_document"
                        )
                        batch_embeddings.append(result["embedding"])
                        # Increment the counter after successfully processing each document
                        GeminiEmbeddings._total_docs_processed += 1
                        
                        # Update progress bar
                        if GeminiEmbeddings._pbar:
                            GeminiEmbeddings._pbar.update(1)
                        
                        # Log progress periodically (every 500 docs or batch completion)
                        if GeminiEmbeddings._total_docs_processed % 500 == 0 or \
                           GeminiEmbeddings._total_docs_processed == GeminiEmbeddings._total_docs_count:
                            logger.info(f"Embedding progress: {GeminiEmbeddings._total_docs_processed}/{GeminiEmbeddings._total_docs_count}")
                            
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        retry_count += 1
                        error_message = str(e)
                        
                        # Handle rate limit errors with retries
                        if "429" in error_message or "Rate limit" in error_message or "quota" in error_message.lower():
                            if retry_count <= self.max_retries:
                                wait_time = self._backoff_time(retry_count)
                                logger.warning(f"Rate limit exceeded (429). Retrying in {wait_time:.2f}s (attempt {retry_count}/{self.max_retries})")
                                time.sleep(wait_time)
                            else:
                                logger.error(f"Failed after {self.max_retries} retries: {error_message}")
                                raise ValueError(f"Rate limit exceeded and maximum retries reached: {error_message}")
                        elif "payload size exceeds" in error_message.lower():
                            # Hard fail on payload size errors - this indicates chunks are too large
                            logger.error(f"Document too large for API limit: {len(text)} chars, error: {error_message}")
                            raise ValueError(f"Document exceeds Gemini API size limit ({len(text)} chars): {error_message}. "
                                             f"Ensure proper document chunking is configured (current chunk_size "
                                             f"may be too large - try a smaller value like 500-800 characters).")
                        else:
                            # Non-rate-limit error
                            logger.error(f"Error generating document embedding: {error_message}")
                            raise ValueError(f"Error generating document embedding: {error_message}")
            
            # Add batch results to the overall results
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Get embeddings for a query with retry logic for rate limits."""
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Make the API call without preemptive throttling
                result = self.genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_query"
                )
                return result["embedding"]
                
            except Exception as e:
                retry_count += 1
                
                # If we hit a rate limit (429 error), then wait and retry
                if "429" in str(e) or "Rate limit" in str(e) or "quota" in str(e).lower():
                    if retry_count <= self.max_retries:
                        wait_time = self._backoff_time(retry_count)
                        logger.warning(f"Rate limit exceeded (429). Retrying in {wait_time:.2f}s (attempt {retry_count}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {self.max_retries} retries: {str(e)}")
                        raise ValueError(f"Rate limit exceeded and maximum retries reached: {str(e)}")
                else:
                    # Non-rate-limit error
                    logger.error(f"Error generating query embedding: {str(e)}")
                    raise ValueError(f"Error generating query embedding: {str(e)}")
        
        # Should not reach here, but just in case
        raise ValueError("Failed to generate query embedding after retries")


class GeminiRetriever(EmbeddingRetriever):
    """
    Retriever using Google's Gemini embedding model.
    This class extends EmbeddingRetriever to ensure identical document processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Gemini retriever.
        
        Args:
            config: Configuration dictionary with these keys:
                - api_key: Google API key (optional, can use environment variable)
                - model_name: Name of the embedding model (default: text-embedding-004)
                - chunk_size: Size of document chunks (default: 1000)
                - chunk_overlap: Overlap between chunks (default: 200)
                - cache_dir: Directory for caching the vector store
        """
        # Initialize parent class to inherit all processing
        super().__init__(config)

        # Override embedding-specific attributes using keys injected by ExperimentManager
        self.api_key = config.get("GOOGLE_API_KEY") # Use the key name from environment
        self.model_name = config.get("model_name", "text-embedding-004")

    def _setup_embedding_model(self) -> None:
        """Override to set up the Gemini embedding model."""
        logger.info(f"Setting up Gemini embedding model: {self.model_name}")
        
        self.embedding_model = GeminiEmbeddings(
            api_key=self.api_key,
            model_name=self.model_name
        )
