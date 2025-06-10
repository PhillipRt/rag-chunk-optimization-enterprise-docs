import pytest
from unittest.mock import MagicMock, patch
from retrievers.bm25_retriever import BM25Retriever

@pytest.fixture
def bm25_config():
    """Basic BM25 retriever configuration."""
    return {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "min_score_threshold": 0.1,
        "k1": 1.2,
        "b": 0.8,
        "epsilon": 0.3
    }

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "content": "This is a document about artificial intelligence and machine learning.",
            "metadata": {"source": "doc1.txt"}
        },
        {
            "content": "Natural language processing is a subfield of artificial intelligence.",
            "metadata": {"source": "doc2.txt"}
        },
        {
            "content": "Deep learning is a type of machine learning based on neural networks.",
            "metadata": {"source": "doc3.txt"}
        }
    ]

@patch('retrievers.bm25_retriever.BM25Okapi')
def test_bm25_retriever_setup(mock_bm25, bm25_config, sample_documents):
    """Test that BM25Retriever initializes correctly."""
    # Set up mock
    mock_bm25_instance = MagicMock()
    mock_bm25.return_value = mock_bm25_instance
    
    # Create and set up retriever
    retriever = BM25Retriever(bm25_config)
    retriever.setup(sample_documents)
    
    # Check that BM25Okapi was called with the right parameters
    mock_bm25.assert_called_once()
    # Check that the parameters were passed correctly
    args, kwargs = mock_bm25.call_args
    assert kwargs["k1"] == 1.2
    assert kwargs["b"] == 0.8
    assert kwargs["epsilon"] == 0.3
    
    # Check that the retriever is initialized
    assert retriever.is_initialized
    assert retriever.bm25_retriever == mock_bm25_instance

@patch('retrievers.bm25_retriever.BM25Okapi')
def test_bm25_retriever_retrieve(mock_bm25, bm25_config, sample_documents):
    """Test that BM25Retriever can retrieve documents."""
    # Set up mock
    mock_bm25_instance = MagicMock()
    mock_bm25.return_value = mock_bm25_instance
    
    # Mock the get_scores method to return some scores
    mock_bm25_instance.get_scores.return_value = [0.8, 0.6, 0.2]
    
    # Create and set up retriever
    retriever = BM25Retriever(bm25_config)
    retriever.setup(sample_documents)
    
    # Test retrieve method
    results = retriever.retrieve("artificial intelligence", top_k=2)
    
    # Check that get_scores was called with the right query
    mock_bm25_instance.get_scores.assert_called_once()
    
    # Check that the correct number of results were returned
    assert len(results) == 2
    
    # Check that results are sorted by score
    assert results[0]["score"] > results[1]["score"]
    
    # Check that the min_score_threshold is applied
    for result in results:
        assert result["score"] >= 0.1

def test_bm25_retriever_metadata(bm25_config):
    """Test that BM25Retriever returns correct metadata."""
    retriever = BM25Retriever(bm25_config)
    metadata = retriever.get_metadata()
    
    assert metadata["type"] == "BM25Retriever"
    assert metadata["chunk_size"] == 500
    assert metadata["chunk_overlap"] == 100
    assert metadata["min_score_threshold"] == 0.1
    assert metadata["k1"] == 1.2
    assert metadata["b"] == 0.8
    assert metadata["epsilon"] == 0.3
