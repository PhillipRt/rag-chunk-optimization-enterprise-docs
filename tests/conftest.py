"""
Pytest configuration and fixtures.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use relative import for fixtures
from .fixtures.sample_documents import SAMPLE_DOCUMENTS, SAMPLE_QUERIES


@pytest.fixture
def sample_documents():
    """Return a list of sample documents for testing retrievers."""
    return SAMPLE_DOCUMENTS


@pytest.fixture
def sample_queries():
    """Return a list of sample queries for testing retrievers."""
    return SAMPLE_QUERIES


@pytest.fixture
def mock_embedding_api():
    """Mock the embedding API to avoid making real API calls during tests."""
    with patch('requests.post') as mock_post, patch('requests.get') as mock_get:
        # Mock the health check
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "status": "ok",
            "models": ["e5", "bge", "nvidia"]
        }
        
        # Mock embedding generation
        mock_post.return_value.status_code = 200
        
        # Return a mock 1024-dimensional embedding for each text
        def mock_embeddings(*args, **kwargs):
            texts = kwargs.get('json', {}).get('texts', [])
            if not texts and args and isinstance(args[1], dict):
                texts = args[1].get('texts', [])
            
            # Generate deterministic but unique mock embeddings based on text content
            mock_embedding_results = []
            for text in texts:
                # Create a deterministic embedding based on the text hash
                import hashlib
                text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
                mock_embedding = [(text_hash + i) % 100 / 100.0 for i in range(1024)]
                mock_embedding_results.append(mock_embedding)
            
            return {"embeddings": mock_embedding_results}
        
        mock_post.return_value.json.side_effect = mock_embeddings
        
        yield {
            "mock_post": mock_post,
            "mock_get": mock_get
        }


@pytest.fixture
def mock_llm():
    """Mock an LLM to avoid making real API calls during tests."""
    mock = MagicMock()
    
    # Set up the invoke method to return predetermined responses
    def mock_invoke(prompt, **kwargs):
        if "reformulate" in prompt.lower() or "generate" in prompt.lower():
            return """1. What are the key features of Python?
2. Describe the syntax and design principles of Python programming language
3. How does Python handle dynamic typing?"""
        
        if "answer" in prompt.lower():
            return "Python is a high-level programming language known for readability with its use of significant indentation."
        
        return "Mock LLM response"
    
    mock.invoke.side_effect = mock_invoke
    
    with patch('langchain_google_genai.ChatGoogleGenerativeAI', return_value=mock):
        yield mock 