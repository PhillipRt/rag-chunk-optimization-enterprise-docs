"""
Tests for the BaseRetriever class.
"""
import pytest
from unittest.mock import patch, MagicMock

from retrievers.base_retriever import BaseRetriever


class ConcreteRetriever(BaseRetriever):
    """Concrete implementation of BaseRetriever for testing purposes."""
    
    def setup(self, documents):
        self.documents = documents
        self.is_initialized = True
    
    def retrieve(self, query, top_k=5):
        # Simple implementation that returns documents containing any word from the query
        results = []
        for doc in self.documents:
            for word in query.lower().split():
                if word in doc["content"].lower():
                    results.append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": 0.8  # Mock score
                    })
                    break
        return results[:top_k]


def test_base_retriever_initialization():
    """Test that a BaseRetriever subclass can be initialized with a config."""
    config = {"test_param": "test_value"}
    retriever = ConcreteRetriever(config)
    
    assert retriever.config == config
    assert retriever.is_initialized is False


def test_base_retriever_setup(sample_documents):
    """Test that a BaseRetriever subclass can be set up with documents."""
    retriever = ConcreteRetriever({})
    retriever.setup(sample_documents)
    
    assert retriever.is_initialized is True
    assert hasattr(retriever, "documents")
    assert len(retriever.documents) == len(sample_documents)


def test_base_retriever_retrieve(sample_documents):
    """Test that a BaseRetriever subclass can retrieve documents."""
    retriever = ConcreteRetriever({})
    retriever.setup(sample_documents)
    
    results = retriever.retrieve("Python programming")
    
    assert len(results) > 0
    assert all(isinstance(r, dict) for r in results)
    assert all("content" in r for r in results)
    assert all("metadata" in r for r in results)
    assert all("score" in r for r in results)


def test_base_retriever_get_metadata():
    """Test that a BaseRetriever subclass can return metadata."""
    config = {"param1": "value1", "param2": "value2"}
    retriever = ConcreteRetriever(config)
    
    metadata = retriever.get_metadata()
    
    assert metadata["type"] == "ConcreteRetriever"
    assert metadata["config"] == config


def test_base_retriever_cleanup():
    """Test that a BaseRetriever subclass can clean up resources."""
    retriever = ConcreteRetriever({})
    retriever.is_initialized = True
    
    retriever.cleanup()
    
    assert retriever.is_initialized is False 