"""
Tests for the HybridRetriever class.
"""
import pytest
from unittest.mock import patch, MagicMock

from retrievers.hybrid_retriever import HybridRetriever


@pytest.fixture
def hybrid_config():
    """Return a configuration for HybridRetriever."""
    return {
        "bm25_weight": 0.3,
        "vector_weight": 0.7,
        "use_reranker": True,
        "reranker_model": "flax-ui/bert-base-uncased-reranker",
        "reranker_threshold": 0.3,
        "dense_retriever": {
            "model_name": "e5",
            "api_url": "https://fake-api.com",
            "auth_token": "fake-token",
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    }


def test_hybrid_retriever_initialization(hybrid_config):
    """Test that HybridRetriever can be initialized with a config."""
    retriever = HybridRetriever(hybrid_config)
    
    assert retriever.config == hybrid_config
    assert retriever.is_initialized is False
    assert retriever.bm25_weight == 0.3
    assert retriever.vector_weight == 0.7
    assert retriever.use_reranker is True
    assert retriever.reranker_model == "flax-ui/bert-base-uncased-reranker"
    assert retriever.reranker_threshold == 0.3


@patch('retrievers.hybrid_retriever.FAISS')
@patch('retrievers.hybrid_retriever.BM25Retriever')
@patch('retrievers.hybrid_retriever.RemoteEmbeddings')
def test_hybrid_retriever_setup(mock_remote_embeddings, mock_bm25, mock_faiss, 
                             hybrid_config, sample_documents, mock_embedding_api):
    """Test that HybridRetriever can be set up with documents."""
    # Mock the BM25 retriever
    mock_bm25_instance = MagicMock()
    mock_bm25.return_value = mock_bm25_instance
    
    # Mock the FAISS vector store
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store
    
    # Create HybridRetriever
    retriever = HybridRetriever(hybrid_config)
    
    # Mock the reranker creation
    with patch.object(retriever, '_setup_reranker'):
        retriever.setup(sample_documents)
    
    assert retriever.is_initialized is True
    assert retriever.documents == sample_documents
    assert retriever.bm25_retriever == mock_bm25_instance
    assert retriever.vector_store == mock_vector_store
    
    # Check that BM25Retriever was called correctly
    mock_bm25.assert_called_once()
    mock_bm25_instance.init_from_documents.assert_called_once_with(sample_documents)
    
    # Check that RemoteEmbeddings was called correctly
    mock_remote_embeddings.assert_called_once()


@patch('retrievers.hybrid_retriever.CrossEncoder')
def test_setup_reranker(mock_cross_encoder, hybrid_config):
    """Test that HybridRetriever can set up a reranker."""
    # Mock the CrossEncoder
    mock_reranker = MagicMock()
    mock_cross_encoder.return_value = mock_reranker
    
    # Create HybridRetriever
    retriever = HybridRetriever(hybrid_config)
    
    # Set up the reranker
    retriever._setup_reranker()
    
    assert retriever.reranker == mock_reranker
    mock_cross_encoder.assert_called_once_with(
        "flax-ui/bert-base-uncased-reranker",
        max_length=512
    )


@patch('retrievers.hybrid_retriever.RemoteEmbeddings')
@patch('retrievers.hybrid_retriever.BM25Retriever')
@patch('retrievers.hybrid_retriever.FAISS')
def test_hybrid_retrieve_no_reranker(mock_faiss, mock_bm25, mock_embeddings, 
                                  hybrid_config, sample_documents, mock_embedding_api):
    """Test that HybridRetriever can retrieve documents without reranking."""
    # Disable reranker
    hybrid_config["use_reranker"] = False
    
    # Set up mocks
    mock_bm25_instance = MagicMock()
    mock_bm25.return_value = mock_bm25_instance
    
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store
    
    # Mock BM25 results
    bm25_results = [
        {"content": "BM25 doc 1", "metadata": {"source": "bm25_1"}, "score": 0.9},
        {"content": "BM25 doc 2", "metadata": {"source": "bm25_2"}, "score": 0.8}
    ]
    mock_bm25_instance.search.return_value = bm25_results
    
    # Mock vector results
    vector_results_docs = []
    for i in range(2):
        mock_doc = MagicMock()
        mock_doc.page_content = f"Vector doc {i+1}"
        mock_doc.metadata = {"source": f"vector_{i+1}"}
        vector_results_docs.append((mock_doc, 0.7 - i * 0.1))
    
    mock_vector_store.similarity_search_with_score.return_value = vector_results_docs
    
    # Create and set up retriever
    retriever = HybridRetriever(hybrid_config)
    retriever.setup(sample_documents)
    
    # Test retrieve method
    results = retriever.retrieve("test query", top_k=3)
    
    # Check that the correct number of results were returned
    assert len(results) == 3
    
    # Check that both retrievers were called
    mock_bm25_instance.search.assert_called_once_with("test query", k=3)
    mock_vector_store.similarity_search_with_score.assert_called_once_with("test query", k=3)


@patch('retrievers.hybrid_retriever.CrossEncoder')
@patch('retrievers.hybrid_retriever.RemoteEmbeddings')
@patch('retrievers.hybrid_retriever.BM25Retriever')
@patch('retrievers.hybrid_retriever.FAISS')
def test_hybrid_retrieve_with_reranker(mock_faiss, mock_bm25, mock_embeddings, 
                                    mock_cross_encoder, hybrid_config, 
                                    sample_documents, mock_embedding_api):
    """Test that HybridRetriever can retrieve documents with reranking."""
    # Set up mocks
    mock_bm25_instance = MagicMock()
    mock_bm25.return_value = mock_bm25_instance
    
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store
    
    mock_reranker = MagicMock()
    mock_cross_encoder.return_value = mock_reranker
    
    # Mock BM25 results
    bm25_results = [
        {"content": "BM25 doc 1", "metadata": {"source": "bm25_1"}, "score": 0.9},
        {"content": "BM25 doc 2", "metadata": {"source": "bm25_2"}, "score": 0.8}
    ]
    mock_bm25_instance.search.return_value = bm25_results
    
    # Mock vector results
    vector_results_docs = []
    for i in range(2):
        mock_doc = MagicMock()
        mock_doc.page_content = f"Vector doc {i+1}"
        mock_doc.metadata = {"source": f"vector_{i+1}"}
        vector_results_docs.append((mock_doc, 0.7 - i * 0.1))
    
    mock_vector_store.similarity_search_with_score.return_value = vector_results_docs
    
    # Mock reranker scores
    mock_reranker.predict.return_value = [0.95, 0.85, 0.55, 0.35]
    
    # Create and set up retriever
    retriever = HybridRetriever(hybrid_config)
    retriever.setup(sample_documents)
    
    # Test retrieve method
    results = retriever.retrieve("test query", top_k=3)
    
    # Check that the correct number of results were returned
    assert len(results) == 3
    
    # Check that both retrievers were called
    mock_bm25_instance.search.assert_called_once_with("test query", k=3)
    mock_vector_store.similarity_search_with_score.assert_called_once_with("test query", k=3)
    
    # Check that the reranker was called
    mock_reranker.predict.assert_called_once()
    
    # Check that results are in the correct order (from highest reranker score to lowest)
    assert results[0]["score"] > results[1]["score"] > results[2]["score"]


@patch('retrievers.hybrid_retriever.RemoteEmbeddings')
@patch('retrievers.hybrid_retriever.BM25Retriever')
@patch('retrievers.hybrid_retriever.FAISS')
def test_hybrid_retrieve_with_threshold(mock_faiss, mock_bm25, mock_embeddings, 
                                     hybrid_config, sample_documents, mock_embedding_api):
    """Test that HybridRetriever can filter results based on a threshold."""
    # Use a higher threshold to filter out low-scoring documents
    hybrid_config["use_reranker"] = False
    hybrid_config["min_score_threshold"] = 0.75
    
    # Set up mocks
    mock_bm25_instance = MagicMock()
    mock_bm25.return_value = mock_bm25_instance
    
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store
    
    # Mock BM25 results with varying scores
    bm25_results = [
        {"content": "BM25 doc 1", "metadata": {"source": "bm25_1"}, "score": 0.9},  # Above threshold
        {"content": "BM25 doc 2", "metadata": {"source": "bm25_2"}, "score": 0.7}   # Below threshold
    ]
    mock_bm25_instance.search.return_value = bm25_results
    
    # Mock vector results with varying scores
    vector_results_docs = []
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "Vector doc 1"
    mock_doc1.metadata = {"source": "vector_1"}
    vector_results_docs.append((mock_doc1, 0.8))  # Above threshold
    
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Vector doc 2"
    mock_doc2.metadata = {"source": "vector_2"}
    vector_results_docs.append((mock_doc2, 0.6))  # Below threshold
    
    mock_vector_store.similarity_search_with_score.return_value = vector_results_docs
    
    # Create and set up retriever
    retriever = HybridRetriever(hybrid_config)
    retriever.setup(sample_documents)
    
    # Test retrieve method with threshold filtering
    results = retriever.retrieve("test query", top_k=4)
    
    # Only documents above the threshold should be returned (2 of the 4)
    assert len(results) == 2
    
    # Check that all returned results are above the threshold
    assert all(r["score"] >= 0.75 for r in results) 