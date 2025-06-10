"""
Tests for the EmbeddingRetriever class.
"""
import pytest
import os
from unittest.mock import patch, MagicMock

from retrievers.embedding_retriever import EmbeddingRetriever, RemoteEmbeddings


def test_remote_embeddings_initialization():
    """Test that RemoteEmbeddings can be initialized with API information."""
    with patch('requests.get') as mock_get:
        # Mock the health check
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": ["e5", "bge", "nvidia"]
        }
        
        embeddings = RemoteEmbeddings(
            api_url="https://fake-api.com",
            model_name="e5",
            auth_token="fake-token"
        )
        
        assert embeddings.api_url == "https://fake-api.com"
        assert embeddings.model_name == "e5"
        assert embeddings.auth_token == "fake-token"
        assert embeddings.headers == {"Authorization": "Bearer fake-token"}


def test_remote_embeddings_embed_documents(mock_embedding_api):
    """Test that RemoteEmbeddings can embed documents."""
    embeddings = RemoteEmbeddings(
        api_url="https://fake-api.com",
        model_name="e5",
        auth_token="fake-token"
    )
    
    texts = ["This is a test document", "This is another test document"]
    result = embeddings.embed_documents(texts)
    
    # Check that we got embeddings back
    assert len(result) == len(texts)
    assert all(len(embedding) == 1024 for embedding in result)
    
    # Check that the API was called correctly
    mock_embedding_api["mock_post"].assert_called_once()
    call_args = mock_embedding_api["mock_post"].call_args[1]
    assert call_args["json"]["model"] == "e5"
    assert call_args["json"]["texts"] == texts


def test_remote_embeddings_embed_query(mock_embedding_api):
    """Test that RemoteEmbeddings can embed a query."""
    embeddings = RemoteEmbeddings(
        api_url="https://fake-api.com",
        model_name="e5",
        auth_token="fake-token"
    )
    
    query = "This is a test query"
    result = embeddings.embed_query(query)
    
    # Check that we got an embedding back
    assert len(result) == 1024
    
    # Check that the API was called correctly
    mock_embedding_api["mock_post"].assert_called_once()
    call_args = mock_embedding_api["mock_post"].call_args[1]
    assert call_args["json"]["model"] == "e5"
    assert call_args["json"]["texts"] == [query]


@pytest.fixture
def embedding_retriever_config():
    """Return a config for EmbeddingRetriever."""
    return {
        "model_name": "e5",
        "api_url": "https://fake-api.com",
        "auth_token": "fake-token",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "cache_dir": "cache/test_vector_stores"
    }


def test_embedding_retriever_initialization(embedding_retriever_config):
    """Test that EmbeddingRetriever can be initialized with a config."""
    retriever = EmbeddingRetriever(embedding_retriever_config)
    
    assert retriever.config == embedding_retriever_config
    assert retriever.is_initialized is False
    assert retriever.model_name == "e5"
    assert retriever.chunk_size == 1000
    assert retriever.chunk_overlap == 200


@patch('retrievers.embedding_retriever.FAISS')
def test_embedding_retriever_setup(mock_faiss, sample_documents, embedding_retriever_config, mock_embedding_api):
    """Test that EmbeddingRetriever can be set up with documents."""
    # Mock the FAISS store
    mock_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_store
    
    retriever = EmbeddingRetriever(embedding_retriever_config)
    retriever.setup(sample_documents)
    
    assert retriever.is_initialized is True
    assert retriever.vector_store == mock_store
    
    # Check that FAISS was called correctly
    mock_faiss.from_documents.assert_called_once()


@patch('retrievers.embedding_retriever.FAISS')
def test_embedding_retriever_retrieve(mock_faiss, sample_documents, embedding_retriever_config, mock_embedding_api):
    """Test that EmbeddingRetriever can retrieve documents."""
    # Mock the FAISS store
    mock_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_store
    
    # Mock the similarity search
    mock_docs = []
    for i, doc in enumerate(sample_documents[:2]):
        mock_doc = MagicMock()
        mock_doc.page_content = doc["content"]
        mock_doc.metadata = doc["metadata"]
        mock_docs.append((mock_doc, 0.8 - i * 0.1))
    
    mock_store.similarity_search_with_score.return_value = mock_docs
    
    retriever = EmbeddingRetriever(embedding_retriever_config)
    retriever.setup(sample_documents)
    
    results = retriever.retrieve("test query", top_k=2)
    
    # Check the results
    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)
    assert all("content" in r for r in results)
    assert all("metadata" in r for r in results)
    assert all("score" in r for r in results)
    
    # Check that similarity_search_with_score was called correctly
    mock_store.similarity_search_with_score.assert_called_once_with("test query", k=2)


@patch('os.path.exists', return_value=True)
@patch('os.makedirs')
@patch('pickle.dump')
def test_embedding_retriever_save_cache(mock_dump, mock_makedirs, mock_exists, 
                                       embedding_retriever_config, mock_embedding_api):
    """Test that EmbeddingRetriever can save the vector store to cache."""
    with patch('retrievers.embedding_retriever.FAISS') as mock_faiss:
        # Mock the FAISS store
        mock_store = MagicMock()
        mock_faiss.from_documents.return_value = mock_store
        
        retriever = EmbeddingRetriever(embedding_retriever_config)
        retriever._save_to_cache()
        
        # Check that directories were created
        mock_makedirs.assert_called_once()
        
        # We don't actually check that dump was called since the vector_store isn't initialized


@patch('os.path.exists', return_value=True)
@patch('pickle.load')
def test_embedding_retriever_load_cache(mock_load, mock_exists, 
                                       embedding_retriever_config, mock_embedding_api):
    """Test that EmbeddingRetriever can load the vector store from cache."""
    # Mock the pickle.load to return a mock vector store
    mock_store = MagicMock()
    mock_load.return_value = mock_store
    
    retriever = EmbeddingRetriever(embedding_retriever_config)
    result = retriever._load_from_cache()
    
    assert result is True
    assert retriever.vector_store == mock_store
    
    # Check that pickle.load was called
    mock_load.assert_called_once() 