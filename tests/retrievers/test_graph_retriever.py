"""
Tests for the GraphRetriever class.
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Set up mocks for external dependencies
mock_kg_index = MagicMock()
mock_service_context = MagicMock()
mock_storage_context = MagicMock()
mock_llama_simple_graph_store = MagicMock()
mock_chat_message = MagicMock()
mock_message_role = MagicMock()
mock_llama_document = MagicMock()
mock_metadata_mode = MagicMock()
mock_kg_extract_prompt = MagicMock()
mock_chat_google = MagicMock()

# Mock the llama-index imports
sys.modules['llama_index'] = MagicMock()
sys.modules['llama_index'].KnowledgeGraphIndex = mock_kg_index
sys.modules['llama_index'].ServiceContext = mock_service_context
sys.modules['llama_index.storage.storage_context'] = MagicMock()
sys.modules['llama_index.storage.storage_context'].StorageContext = mock_storage_context
sys.modules['llama_index.graph_stores'] = MagicMock()
sys.modules['llama_index.graph_stores'].SimpleGraphStore = mock_llama_simple_graph_store
sys.modules['llama_index.llms'] = MagicMock()
sys.modules['llama_index.llms'].ChatMessage = mock_chat_message
sys.modules['llama_index.llms'].MessageRole = mock_message_role
sys.modules['llama_index.schema'] = MagicMock()
sys.modules['llama_index.schema'].Document = mock_llama_document
sys.modules['llama_index.schema'].MetadataMode = mock_metadata_mode
sys.modules['llama_index.prompts.prompts'] = MagicMock()
sys.modules['llama_index.prompts.prompts'].KnowledgeGraphExtractPrompt = mock_kg_extract_prompt
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['langchain_google_genai'].ChatGoogleGenerativeAI = mock_chat_google

# Now import the module under test
from retrievers.graph_retriever import GraphRetriever, SimpleGraphStore


@pytest.fixture
def graph_config():
    """Return a configuration for GraphRetriever."""
    return {
        "llm_model": "gemini-pro",
        "cache_dir": "cache/test_graph_indices",
        "max_triplets_per_chunk": 10,
        "include_embeddings": True
    }


def test_graph_retriever_initialization(graph_config):
    """Test that GraphRetriever can be initialized with a config."""
    retriever = GraphRetriever(graph_config)
    
    assert retriever.config == graph_config
    assert retriever.is_initialized is False
    assert retriever.llm_model == "gemini-pro"
    assert retriever.cache_dir == "cache/test_graph_indices"
    assert retriever.max_triplets == 10
    assert retriever.include_embeddings is True


def test_graph_retriever_setup(graph_config, sample_documents):
    """Test that GraphRetriever can be set up with documents."""
    # Mock the service context
    mock_service_context_instance = MagicMock()
    mock_service_context.from_defaults.return_value = mock_service_context_instance
    
    # Mock the storage context
    mock_storage_context_instance = MagicMock()
    mock_storage_context.from_defaults.return_value = mock_storage_context_instance
    
    # Mock the knowledge graph index
    mock_kg_index_instance = MagicMock()
    mock_kg_index.from_documents.return_value = mock_kg_index_instance
    
    # Create and set up the retriever
    with patch.object(GraphRetriever, '_load_from_cache', return_value=False), \
         patch.object(GraphRetriever, '_save_to_cache'):
        
        retriever = GraphRetriever(graph_config)
        retriever.setup(sample_documents)
    
    assert retriever.is_initialized is True
    assert retriever.knowledge_graph is not None
    
    # Check that the necessary components were set up correctly
    mock_service_context.from_defaults.assert_called_once()
    mock_storage_context.from_defaults.assert_called_once()
    
    # Check that the KG index was created
    mock_kg_index.from_documents.assert_called_once()


def test_graph_retriever_load_from_cache(graph_config):
    """Test that GraphRetriever can load from cache."""
    # Mock file operations
    mock_kg = MagicMock()
    mock_open = MagicMock()
    mock_pickle = MagicMock()
    mock_pickle.load.return_value = mock_kg
    
    # Create the retriever
    retriever = GraphRetriever(graph_config)
    retriever.index_id = "test_index_id"
    
    # Mock the file operations
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open), \
         patch('pickle.load', mock_pickle.load):
        
        # Load from cache
        result = retriever._load_from_cache()
    
    assert result is True
    assert retriever.knowledge_graph is not None


def test_graph_retriever_save_to_cache(graph_config):
    """Test that GraphRetriever can save to cache."""
    # Create the retriever
    retriever = GraphRetriever(graph_config)
    retriever.index_id = "test_index_id"
    retriever.knowledge_graph = MagicMock()
    
    # Mock file operations
    mock_makedirs = MagicMock()
    mock_open = MagicMock()
    mock_pickle = MagicMock()
    
    # Save to cache with mocked file operations
    with patch('os.makedirs', mock_makedirs), \
         patch('builtins.open', mock_open), \
         patch('pickle.dump', mock_pickle):
        
        retriever._save_to_cache()
    
    # Check that directories were created
    mock_makedirs.assert_called_once()


def test_setup_llm(graph_config):
    """Test that GraphRetriever can set up an LLM."""
    # Reset the mocks
    mock_chat_google.reset_mock()
    mock_service_context.reset_mock()
    mock_service_context.from_defaults.reset_mock()
    
    # Mock the LLM
    mock_llm_instance = MagicMock()
    mock_chat_google.return_value = mock_llm_instance
    
    # Mock the service context
    mock_service_ctx = MagicMock()
    mock_service_context.from_defaults.return_value = mock_service_ctx
    
    # Create the retriever
    retriever = GraphRetriever(graph_config)
    
    # Set up the LLM
    retriever._setup_llm()
    
    # Check that the LLM was created correctly
    mock_chat_google.assert_called_with(
        model="gemini-pro",
        temperature=0.2
    )
    
    # Check that service context was created
    assert mock_service_context.from_defaults.call_count >= 1


def test_graph_retriever_retrieve(graph_config, sample_documents):
    """Test that GraphRetriever can retrieve documents."""
    # Mock the knowledge graph and query engine
    mock_query_engine = MagicMock()
    mock_kg_instance = MagicMock()
    mock_kg_instance.as_query_engine.return_value = mock_query_engine
    
    # Mock the query results
    mock_node = MagicMock()
    mock_node.node.get_content.return_value = "Result content"
    mock_node.score = 0.85
    mock_node.node.metadata = {"source": "test_source"}
    mock_node.node.node_type = "text"
    
    mock_result = MagicMock()
    mock_result.source_nodes = [mock_node]
    mock_query_engine.query.return_value = mock_result
    
    # Create and set up the retriever
    retriever = GraphRetriever(graph_config)
    retriever.is_initialized = True
    retriever.knowledge_graph = mock_kg_instance
    retriever.query_engine = mock_query_engine
    
    # Test retrieving
    results = retriever.retrieve("What is Python?", top_k=3)
    
    # Check that query was called correctly
    mock_query_engine.query.assert_called_once_with("What is Python?")
    
    # Check the results
    assert len(results) > 0
    assert results[0]["content"] == "Result content"
    assert "metadata" in results[0]
    assert results[0]["score"] == 0.85
    assert results[0]["node_type"] == "text" 