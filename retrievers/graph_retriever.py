import os
import logging
import pickle
from typing import List, Dict, Any, Optional, Callable, Tuple, Set

from retrievers.base_retriever import BaseRetriever
from retrievers.gemini_retriever import GeminiEmbeddings

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import (
    VectorStoreIndex,
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage,  # Import for loading
)

from llama_index.core import Document as LlamaDocument
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts import PromptTemplate as LlamaPromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.graph_stores.simple import SimpleGraphStore
# Use native Gemini LLM from llama-index instead of LangChain wrapper
from llama_index.llms.gemini import Gemini as GeminiLLM
from llama_index.embeddings.gemini import GeminiEmbedding



logger = logging.getLogger(__name__)


# Renamed from GraphRetriever to keep the old implementation
class KnowledgeGraphRetriever(BaseRetriever):
    """Knowledge Graph based retriever using LlamaIndex (deprecated KnowledgeGraphIndex)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the graph-based retriever.
        
        Args:
            config: Configuration dictionary containing:
                - llm_model: LLM model for entity extraction and querying (default: gemini-pro)
                - cache_dir: Directory for caching the knowledge graph (default: cache/graph_indices)
                - entity_extract_template: Custom prompt for entity extraction (optional)
                - max_triplets_per_chunk: Max number of triplets to extract per chunk (default: 10)
                - chunk_size: Size of document chunks (default: 1000)
                - chunk_overlap: Overlap between chunks (default: 200)
        """
        super().__init__(config)
        
        self.llm_model = config.get("llm_model", "gemini-pro")
        self.cache_dir = config.get("cache_dir", "cache/graph_indices")
        self.entity_extract_template = config.get("entity_extract_template", None)
        self.max_triplets = config.get("max_triplets_per_chunk", 10)
        self.include_embeddings = config.get("include_embeddings", True)
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        
        # Will be initialized in setup()
        self.knowledge_graph = None
        self.query_engine = None
        self.index_id = None
        self.llm = None
        self.embed_model = None
    
    def setup(self, documents: List[Dict[str, Any]]) -> None:
        """Set up the retriever with documents."""
        logger.info("Setting up graph-based retriever")
        
        # Create a unique ID for this index based on documents and config
        import hashlib
        doc_hash = hashlib.md5("".join(doc["content"][:100] for doc in documents[:5]).encode()).hexdigest()[:10]
        self.index_id = f"graph_{doc_hash}_{self.llm_model.replace('-', '_')}"
        
        # Set up LLM
        self._setup_llm()
        
        # Try to load existing index
        if self._load_from_cache():
            logger.info(f"Loaded knowledge graph from cache ({self.index_id})")
        else:
            # Build knowledge graph
            self._build_knowledge_graph(documents)
            
            # Save to cache
            self._save_to_cache()
        
        # Create query engine
        self._setup_query_engine()
        
        self.is_initialized = True
    
    def _validate_components(self):
        """Validate that the LLM and embedding models are properly initialized."""
        if not self.llm:
            logger.error("LLM not initialized")
            raise ValueError("LLM not initialized")
            
        if not self.embed_model:
            logger.error("Embedding model not initialized")
            raise ValueError("Embedding model not initialized")
            
        logger.info(f"Components validated - Using LLM: {self.llm.__class__.__name__}, Embeddings: {self.embed_model.__class__.__name__}")
    
    def _setup_llm(self) -> None:
        """Set up the LLM and embeddings for the knowledge graph."""
        from llama_index.core import Settings
        
        # Create native Gemini LLM directly
        # Ensure model name has the required 'models/' prefix
        model_name = self.llm_model
        if not model_name.startswith("models/") and not model_name.startswith("tunedModels/"):
            model_name = f"models/{model_name}"
            
        self.llm = GeminiLLM(
            model_name=model_name,
            temperature=0.2
        )
        logger.info(f"Created native GeminiLLM with model: {model_name}")
        
        # Set up Gemini embeddings using the native implementation
        embedding_model = self.config.get("embedding_model", "text-embedding-004")
        # Ensure embedding model name has the required 'models/' prefix
        if not embedding_model.startswith("models/") and not embedding_model.startswith("tunedModels/"):
            embedding_model = f"models/{embedding_model}"
            
        self.embed_model = GeminiEmbedding(
            model_name=embedding_model
        )
        logger.info(f"Created native GeminiEmbedding with model: {embedding_model}")
        # Configure global LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 1024
        
        # Validate the components
        self._validate_components()
        
        logger.info(f"LLM and embeddings setup complete")

    def _build_knowledge_graph(self, documents: List[Dict[str, Any]]) -> None:
        """Build knowledge graph from documents."""
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        
        # Validate components before building
        self._validate_components()
        
        # Convert to LlamaIndex documents
        logger.info("Converting documents to LlamaIndex format...")
        llama_docs = [
            LlamaDocument(
                text=doc["content"],
                metadata=doc["metadata"]
            ) for doc in documents
        ]
        logger.info(f"Converted {len(llama_docs)} documents")
        
        # Apply chunking if configured
        if self.chunk_size > 0:
            logger.info(f"Chunking documents with size={self.chunk_size}, overlap={self.chunk_overlap}...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            # Convert LlamaIndex docs to LangChain docs for splitting
            langchain_docs = [
                Document(
                    page_content=doc.text,
                    metadata=doc.metadata
                ) for doc in llama_docs
            ]
            
            # Apply chunking
            chunked_docs = text_splitter.split_documents(langchain_docs)
            num_chunks = len(chunked_docs)
            logger.info(f"Split documents into {num_chunks} chunks for knowledge graph")
            
            # Convert back to LlamaIndex format
            llama_docs = [
                LlamaDocument(
                    text=doc.page_content,
                    metadata=doc.metadata
                ) for doc in chunked_docs
            ]
        else:
            num_chunks = len(llama_docs)  # Use original count if no chunking
            logger.info(f"Not chunking, using {num_chunks} documents directly")
        
        # Set up the graph store
        logger.info("Setting up graph store...")
        graph_store = SimpleGraphStore()  # Use our implementation
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        
        # Log starting triplet extraction
        logger.info(f"Starting knowledge graph construction with {num_chunks} documents/chunks...")
        logger.info(f"This may take a while for large document collections. Built-in progress bar enabled.")
        
        try:
            # Prepare the extraction prompt
            extract_prompt_tmpl = None
            if self.entity_extract_template:
                logger.info("Using custom extraction prompt template")
                extract_prompt_tmpl = LlamaPromptTemplate(template=self.entity_extract_template)
            else:
                logger.info("Using default extraction prompt template")
                extract_prompt_tmpl = DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
            
            # Create the index with explicit components and built-in progress tracking
            self.knowledge_graph = KnowledgeGraphIndex.from_documents(
                llama_docs,
                storage_context=storage_context,
                max_triplets_per_chunk=self.max_triplets,
                extract_prompt=extract_prompt_tmpl,
                include_embeddings=self.include_embeddings,
                llm=self.llm,                # Explicitly pass LLM
                embed_model=self.embed_model, # Explicitly pass embedding model
                show_progress=True,          # Use built-in progress indicator
            )
            
            # Log statistics
            triplets = graph_store.get_all_relationships()
            logger.info(f"Knowledge graph built successfully with {len(triplets)} triplets")
            
            # Log some sample triplets for verification
            sample_size = min(5, len(triplets))
            if sample_size > 0:
                logger.info(f"Sample triplets (showing {sample_size} of {len(triplets)}):")
                for i, (subj, rel, obj) in enumerate(triplets[:sample_size]):
                    logger.info(f"  {i+1}. {subj} -- {rel} --> {obj}")
        
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            raise
    
    def _save_to_cache(self) -> None:
        """Save the knowledge graph to cache."""
        if not self.knowledge_graph or not hasattr(self.knowledge_graph, 'storage_context'):
            logger.warning("Knowledge graph or storage context not available for saving.")
            return

        # Use directory path for LlamaIndex persistence
        cache_path = os.path.join(self.cache_dir, self.index_id) # Use index_id as directory name

        try:
            # Make the directory if it doesn't exist
            os.makedirs(cache_path, exist_ok=True)

            # Use LlamaIndex's persistence method
            self.knowledge_graph.storage_context.persist(persist_dir=cache_path)
            logger.info(f"Saved knowledge graph index components to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving knowledge graph index components: {str(e)}")

    def _load_from_cache(self) -> bool:
        """Load the knowledge graph index from cache using LlamaIndex methods."""
        # Use directory path based on index_id
        cache_path = os.path.join(self.cache_dir, self.index_id)

        if not os.path.exists(cache_path):
            logger.info(f"Cache directory not found at {cache_path}, will build index.")
            return False

        try:
            logger.info(f"Attempting to load knowledge graph index from {cache_path}")
            # Ensure LLM and embeddings are set up before loading
            if not self.llm or not self.embed_model:
                self._setup_llm()

            # Create ServiceContext with the configured LLM and embed_model
            # Use LlamaIndex Settings which should be configured in _setup_llm
            from llama_index.core import Settings
            service_context = ServiceContext.from_defaults(
                llm=Settings.llm,
                embed_model=Settings.embed_model
            )

            # Load the storage context from the directory
            storage_context = StorageContext.from_defaults(persist_dir=cache_path)

            # Load the index using the storage and service contexts
            self.knowledge_graph = load_index_from_storage(
                storage_context,
                service_context=service_context
            )

            # Check if loading actually returned an index
            if not isinstance(self.knowledge_graph, KnowledgeGraphIndex):
                 logger.error(f"Loaded object from {cache_path} is not a KnowledgeGraphIndex.")
                 return False # Treat as cache miss

            logger.info(f"Successfully loaded knowledge graph index from {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge graph index from cache: {str(e)}")
            # If loading fails, return False so it rebuilds
            return False
    
    def _setup_query_engine(self) -> None:
        """Set up the query engine for the knowledge graph."""
        if not self.knowledge_graph:
            raise ValueError("Knowledge graph not initialized")
        
        # Validate components before creating query engine
        self._validate_components()
        
        logger.info(f"Setting up query engine with explicitly provided components")
        logger.info(f"Using LLM: {self.llm.__class__.__name__} with model: {self.llm_model}")
        
        # Create query engine with explicit parameter passing to avoid global settings issues
        try:
            self.query_engine = self.knowledge_graph.as_query_engine(
                llm=self.llm,                # Explicitly pass the LLM
                embed_model=self.embed_model, # Explicitly pass the embedding model
                include_text=True,           # Include source text in results
                response_mode="compact",     # Concise responses
                similarity_top_k=10          # Number of relevant items to retrieve
            )
            logger.info("Query engine created successfully with explicit component configuration")
        except Exception as e:
            logger.error(f"Failed to create query engine: {str(e)}")
            # Re-raise the exception to ensure we're not hiding errors
            raise
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents based on the query using the knowledge graph."""
        if not self.is_initialized:
            raise ValueError("Retriever not initialized. Call setup() first.")
        
        logger.info(f"Retrieving with knowledge graph: {query}")
        
        # Validate components before retrieval
        self._validate_components()
        
        try:
            # Use retrieve instead of query to avoid synthesis (which requires LLM)
            source_nodes = self.query_engine.retrieve(query)
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise ValueError(f"Retrieval failed: {e}") from e
        
        # Validate that we got real results
        if not source_nodes or len(source_nodes) == 0:
            logger.error("No results returned from knowledge graph retrieval")
            raise ValueError("Knowledge graph retrieval returned no results")
            
        # Format results
        formatted_results = []
        seen_content = set()
        
        for node in source_nodes[:top_k]:
            # Validate that each node has the expected structure
            if not hasattr(node, 'node') or not hasattr(node.node, 'get_content'):
                logger.error(f"Retrieved node missing expected attributes: {node}")
                raise ValueError("Retrieved nodes have invalid structure")
                
            content = node.node.get_content(metadata_mode=MetadataMode.NONE)
            
            # Validate content
            if not content or not isinstance(content, str) or len(content.strip()) == 0:
                logger.warning(f"Skipping node with empty or invalid content: {node}")
                continue
                
            # Skip duplicates
            if content in seen_content:
                continue
                
            seen_content.add(content)
            
            # Extract metadata
            metadata = {}
            if hasattr(node.node, 'metadata'):
                metadata = dict(node.node.metadata)
            
            # Safely get node_type with error handling
            node_type = "unknown"
            if hasattr(node.node, 'node_type'):
                node_type = node.node.node_type
            
            formatted_results.append({
                "content": content,
                "metadata": metadata,
                "score": node.score or 1.0,
                "node_type": node_type
            })
        
        # Ensure we have at least some results after filtering
        if len(formatted_results) == 0:
            logger.error("All retrieved results were filtered out (duplicates/empty)")
            raise ValueError("Knowledge graph retrieval returned no valid results after filtering")
            
        return formatted_results[:top_k]
