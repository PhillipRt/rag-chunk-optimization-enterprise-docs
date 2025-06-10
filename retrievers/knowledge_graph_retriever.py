import os
import logging
import pickle
from typing import List, Dict, Any, Optional, Callable, Tuple, Set

from retrievers.base_retriever import BaseRetriever
from retrievers.gemini_retriever import GeminiEmbeddings # Keep for now, might be needed by unpickling

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
# Use newer Google GenAI integrations instead of deprecated ones
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding



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
        self.cache_dir = config.get("cache_dir", "cache/graph_indices") # Keep original cache dir for this version
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
        logger.info("Setting up KnowledgeGraphRetriever (using deprecated KnowledgeGraphIndex)")

        # Create a unique ID for this index based on documents and config
        import hashlib
        doc_hash = hashlib.md5("".join(doc["content"][:100] for doc in documents[:5]).encode()).hexdigest()[:10]
        # Keep original naming convention for this version
        self.index_id = f"graph_{doc_hash}_{self.llm_model.replace('-', '_')}"

        # Set up LLM
        self._setup_llm()

        # Try to load existing index using the corrected directory-based loading
        if self._load_from_cache():
            logger.info(f"Loaded knowledge graph index from cache ({self.index_id})")
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

        # Use newer GoogleGenAI LLM
        # Model name format might not require 'models/' prefix with newer classes, but check docs if issues arise
        self.llm = GoogleGenAI(
            model_name=self.llm_model,
            api_key=self.config.get("GOOGLE_API_KEY"), # Pass API key
            temperature=0.2
        )
        logger.info(f"Using GoogleGenAI LLM: {self.llm_model}")

        # Use newer GoogleGenAIEmbedding
        embedding_model_name = self.config.get("embedding_model", "models/text-embedding-004") # Use specific key
        self.embed_model = GoogleGenAIEmbedding(
            model_name=embedding_model_name,
            api_key=self.config.get("GOOGLE_API_KEY") # Pass API key
        )
        logger.info(f"Using GoogleGenAIEmbedding: {embedding_model_name}")

        # Configure global LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 1024 # Or use self.chunk_size if needed globally

        # Validate the components
        self._validate_components()

        logger.info(f"LLM and embeddings setup complete for KnowledgeGraphRetriever")

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
        graph_store = SimpleGraphStore()
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
        """Save the knowledge graph index components to cache."""
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
                # Ensure LLM/Embeddings are set in global Settings if not already
                self._setup_llm()

            # Load only the graph store from the directory
            # Assuming the conversion script saved it as 'graph_store.json'
            graph_store_path = os.path.join(cache_path, "graph_store.json")
            if not os.path.exists(graph_store_path):
                 logger.error(f"Converted graph_store.json not found at {graph_store_path}")
                 return False

            from llama_index.core.graph_stores.simple import SimpleGraphStore as CurrentSimpleGraphStore
            graph_store = CurrentSimpleGraphStore.from_persist_path(graph_store_path)
            logger.info("Successfully loaded graph store data.")

            # Reconstruct the index using from_existing (relies on global Settings)
            self.knowledge_graph = KnowledgeGraphIndex.from_existing(
                graph_store=graph_store,
                # LLM and embed_model are taken from Settings implicitly
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
            # Return empty list instead of raising error, consistent with other retrievers
            return []

        # Format results
        formatted_results = []
        seen_content = set()

        for node in source_nodes[:top_k]:
            # Validate that each node has the expected structure
            if not hasattr(node, 'node') or not hasattr(node.node, 'get_content'):
                logger.error(f"Retrieved node missing expected attributes: {node}")
                # Continue to next node instead of raising error
                continue

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

        # No need to raise error if filtering results in empty list
        if len(formatted_results) == 0:
             logger.warning("All retrieved results were filtered out (duplicates/empty).")

        return formatted_results[:top_k]
