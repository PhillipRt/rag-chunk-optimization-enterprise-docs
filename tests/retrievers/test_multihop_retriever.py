"""
Tests for the MultiHopRetriever class.
"""
import pytest
from unittest.mock import patch, MagicMock

from retrievers.multihop_retriever import MultiHopRetriever


@pytest.fixture
def multihop_config():
    """Return a configuration for MultiHopRetriever."""
    return {
        "base_retriever": {
            "model_name": "e5",
            "api_url": "https://fake-api.com",
            "auth_token": "fake-token",
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        "num_queries": 3,
        "llm_model": "gemini-1.5-flash",
        "temperature": 0.5,
        "combination_strategy": "union"
    }


def test_multihop_retriever_initialization(multihop_config):
    """Test that MultiHopRetriever can be initialized with a config."""
    retriever = MultiHopRetriever(multihop_config)
    
    assert retriever.config == multihop_config
    assert retriever.is_initialized is False
    assert retriever.base_retriever_config == multihop_config["base_retriever"]
    assert retriever.num_queries == 3
    assert retriever.llm_model == "gemini-1.5-flash"
    assert retriever.temperature == 0.5
    assert retriever.combination_strategy == "union"


@patch('retrievers.multihop_retriever.ChatGoogleGenerativeAI')
@patch('retrievers.multihop_retriever.EmbeddingRetriever')
def test_multihop_retriever_setup(mock_embedding_retriever, mock_chat_google, 
                                 multihop_config, sample_documents, mock_llm):
    """Test that MultiHopRetriever can be set up with documents."""
    # Mock the embedding retriever
    mock_retriever_instance = MagicMock()
    mock_embedding_retriever.return_value = mock_retriever_instance
    
    retriever = MultiHopRetriever(multihop_config)
    retriever.setup(sample_documents)
    
    assert retriever.is_initialized is True
    assert retriever.base_retriever == mock_retriever_instance
    assert retriever.query_generator is not None
    
    # Check that the base retriever was set up correctly
    mock_embedding_retriever.assert_called_once_with(multihop_config["base_retriever"])
    mock_retriever_instance.setup.assert_called_once_with(sample_documents)


@patch('retrievers.multihop_retriever.ChatGoogleGenerativeAI')
@patch('retrievers.multihop_retriever.EmbeddingRetriever')
def test_multihop_retriever_generate_queries(mock_embedding_retriever, mock_chat_google, 
                                           multihop_config, sample_documents, mock_llm):
    """Test that MultiHopRetriever can generate additional queries."""
    # Mock the embedding retriever
    mock_retriever_instance = MagicMock()
    mock_embedding_retriever.return_value = mock_retriever_instance
    
    retriever = MultiHopRetriever(multihop_config)
    retriever.setup(sample_documents)
    
    # Test generating queries
    original_query = "What is Python?"
    additional_queries = retriever._generate_queries(original_query)
    
    # Verify that the correct number of queries were generated
    assert len(additional_queries) == retriever.num_queries
    assert all(isinstance(q, str) for q in additional_queries)
    
    # Check that the LLM was invoked correctly
    assert mock_llm.invoke.called


@patch('retrievers.multihop_retriever.ChatGoogleGenerativeAI')
@patch('retrievers.multihop_retriever.EmbeddingRetriever')
def test_multihop_retriever_retrieve_union(mock_embedding_retriever, mock_chat_google, 
                                        multihop_config, sample_documents, mock_llm):
    """Test that MultiHopRetriever can retrieve using the union strategy."""
    # Use the union strategy
    multihop_config["combination_strategy"] = "union"
    
    # Mock the embedding retriever
    mock_retriever_instance = MagicMock()
    mock_embedding_retriever.return_value = mock_retriever_instance
    
    # Set up mock returns for base retriever's retrieve method
    def mock_retrieve(query, top_k):
        if query == "What is Python?":
            return [
                {"content": "Python info 1", "metadata": {"source": "doc1"}, "score": 0.9},
                {"content": "Python info 2", "metadata": {"source": "doc2"}, "score": 0.8}
            ]
        elif "features" in query:
            return [
                {"content": "Python features", "metadata": {"source": "doc3"}, "score": 0.7},
                {"content": "Python info 2", "metadata": {"source": "doc2"}, "score": 0.6}  # Duplicate
            ]
        else:
            return [
                {"content": "Python syntax", "metadata": {"source": "doc4"}, "score": 0.5}
            ]
    
    mock_retriever_instance.retrieve.side_effect = mock_retrieve
    
    # Mock the query generation
    def mock_generate_queries(query):
        return [
            "What are the key features of Python?",
            "How does Python handle dynamic typing?",
            "Describe the syntax of Python"
        ]
    
    retriever = MultiHopRetriever(multihop_config)
    retriever.setup(sample_documents)
    retriever._generate_queries = mock_generate_queries
    
    # Test retrieving
    results = retriever.retrieve("What is Python?", top_k=4)
    
    # Verify that the correct number of results were returned (should be 4 unique docs)
    assert len(results) == 4
    
    # Check that the base retriever was called for all queries
    assert mock_retriever_instance.retrieve.call_count == 4  # Original + 3 generated


@patch('retrievers.multihop_retriever.ChatGoogleGenerativeAI')
@patch('retrievers.multihop_retriever.EmbeddingRetriever')
def test_multihop_retriever_retrieve_reciprocal_rank_fusion(mock_embedding_retriever, mock_chat_google, 
                                                        multihop_config, sample_documents, mock_llm):
    """Test that MultiHopRetriever can retrieve using the reciprocal rank fusion strategy."""
    # Use the reciprocal rank fusion strategy
    multihop_config["combination_strategy"] = "reciprocal_rank_fusion"
    
    # Mock the embedding retriever
    mock_retriever_instance = MagicMock()
    mock_embedding_retriever.return_value = mock_retriever_instance
    
    # Set up mock returns for base retriever's retrieve method
    def mock_retrieve(query, top_k):
        if query == "What is Python?":
            return [
                {"content": "Python info 1", "metadata": {"source": "doc1"}, "score": 0.9},
                {"content": "Python info 2", "metadata": {"source": "doc2"}, "score": 0.8}
            ]
        elif "features" in query:
            return [
                {"content": "Python features", "metadata": {"source": "doc3"}, "score": 0.7},
                {"content": "Python info 1", "metadata": {"source": "doc1"}, "score": 0.6}  # Duplicate but lower score
            ]
        else:
            return [
                {"content": "Python syntax", "metadata": {"source": "doc4"}, "score": 0.5}
            ]
    
    mock_retriever_instance.retrieve.side_effect = mock_retrieve
    
    # Mock the query generation
    def mock_generate_queries(query):
        return [
            "What are the key features of Python?",
            "How does Python handle dynamic typing?",
            "Describe the syntax of Python"
        ]
    
    retriever = MultiHopRetriever(multihop_config)
    retriever.setup(sample_documents)
    retriever._generate_queries = mock_generate_queries
    
    # Test retrieving
    results = retriever.retrieve("What is Python?", top_k=4)
    
    # Verify that the correct number of results were returned (should be 4 unique docs)
    assert len(results) == 4
    
    # Check that results are in the correct order (based on RRF score)
    sources = [r["metadata"]["source"] for r in results]
    
    # Assuming doc1 appears in multiple result sets, it should be ranked higher
    assert "doc1" in sources
    
    # Check that the base retriever was called for all queries
    assert mock_retriever_instance.retrieve.call_count == 4  # Original + 3 generated 