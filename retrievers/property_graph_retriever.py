import os
import logging
import time
import asyncio
import functools  # For wraps decorator
# Removed duplicate os import
import re # Import regex for parsing
# Removed json import as it's only needed within fsspec patch scope now
from typing import List, Dict, Any, Optional, Tuple
# Import common Google API core exceptions for more specific retry handling
from google.api_core.exceptions import ServiceUnavailable, InternalServerError, GatewayTimeout, TooManyRequests, GoogleAPIError

from retrievers.base_retriever import BaseRetriever

# Updated LlamaIndex imports
from llama_index.core import (
    StorageContext,
    ServiceContext,
    Settings,
    Document as LlamaDocument,
    load_index_from_storage, # Import load_index_from_storage
)
from llama_index.core.schema import MetadataMode
# SimpleGraphStore might also be in core now, adjust if needed later
from llama_index.core.graph_stores import SimplePropertyGraphStore # Corrected import
# Import the specific store class for patching
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
# Assuming SimplePropertyGraphStore moved to core based on common patterns
# from llama_index.core.graph_stores import SimplePropertyGraphStore # Duplicate removed
from llama_index.core.indices.property_graph import (
    PropertyGraphIndex,
    SimpleLLMPathExtractor,
    ImplicitPathExtractor,
)
# Import PromptTemplate for custom prompt
from llama_index.core import PromptTemplate
# Import KnowledgeGraphTriplet for custom parsing - Removed as parse_fn returns tuples


# Use newer Google GenAI integrations
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr # Import PrivateAttr
# Import CallbackManager and LlamaDebugHandler
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType


# Langchain imports for splitting (optional, could use LlamaIndex splitter too)
from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
logger = logging.getLogger(__name__)

# Define retryable exceptions
import fsspec # Import fsspec for the patch
RETRYABLE_EXCEPTIONS = (
    ServiceUnavailable,    # Typically 503
    InternalServerError,   # Typically 500
    GatewayTimeout,        # Typically 504
    TooManyRequests,       # Typically 429 (Rate limiting)
)

# --- Custom Prompt Template for KG Extraction (Original - kept for reference) ---
KG_TRIPLET_EXTRACT_PROMPT_TMPL_ORIG = """
Some text is provided below. Given the text, extract up to {max_knowledge_triplets}
knowledge triplets in the form of (subject, predicate, object).
Avoid stopwords. Output ONLY the triplets, one per line. Do not include any other text,
markdown, or explanations.

Example:
Text: Alice is Bob's mother.
Triplets:
(Alice, is mother of, Bob)

Text: Philz is a coffee shop founded in Berkeley in 1982.
Triplets:
(Philz, is, coffee shop)
(Philz, founded in, Berkeley)
(Philz, founded in, 1982)
---------------------
Text: {text}
Triplets:
"""
KG_TRIPLET_EXTRACT_PROMPT_ORIG = PromptTemplate(KG_TRIPLET_EXTRACT_PROMPT_TMPL_ORIG)
# --- End Custom Prompt Template ---

# --- Simpler Custom Prompt Template for KG Extraction ---
KG_TRIPLET_EXTRACT_PROMPT_SIMPLE_TMPL = """
Your task is to extract knowledge triplets from the **Input Text** below.
Your output MUST strictly adhere to the **Output Format Requirements**.

**Input Text:**
{text}

**Output Format Requirements:**
1.  Output ONLY knowledge triplets.
2.  Each triplet MUST be on its own new line.
3.  Each line MUST follow the exact format: `(subject, predicate, object)` including the opening parenthesis `(`, the comma-space `, ` separators, and the closing parenthesis `)`.
4.  DO NOT include ANY text before the first triplet or after the last triplet.
5.  DO NOT include introductions (e.g., "Here are the triplets:").
6.  DO NOT include explanations, comments, or apologies.
7.  DO NOT include markdown formatting (like `*` or `-` bullet points).
8.  DO NOT output lines that are not valid triplets in the specified format (e.g., avoid lines like `(subject predicate object)` or `subject, relation, object` or `(item1, item2)`).

**Output Triplets:**
"""
# Update the PromptTemplate object
KG_TRIPLET_EXTRACT_PROMPT_SIMPLE = PromptTemplate(KG_TRIPLET_EXTRACT_PROMPT_SIMPLE_TMPL)
# --- End Simpler Custom Prompt Template ---

# --- Custom Parsing Function ---
def custom_parse_triplets_fn(output: str) -> List[Tuple[str, str, str]]: # Changed return type hint
    """Parse LLM output into triplets."""
    triplets = []
    logger.debug(f"Attempting to parse LLM output for triplets:\n{output}")
    lines = output.strip().split("\n")
    for line in lines:
        # Attempt to parse lines like "(Subject, Predicate, Object)"
        # Use regex to be more robust against potential variations
        match = re.match(r"\(\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*\)", line.strip())
        if match:
            subj, pred, obj = match.groups()
            # Basic cleaning
            subj = subj.strip().strip("'\"")
            pred = pred.strip().strip("'\"")
            obj = obj.strip().strip("'\"")
            if subj and pred and obj:
                triplets.append((subj, pred, obj)) # Changed to append tuple
                logger.debug(f"Successfully parsed triplet: ({subj}, {pred}, {obj})")
            else:
                logger.warning(f"Skipped potentially incomplete triplet after parsing: ({subj}, {pred}, {obj}) from line: {line}")
        else:
            logger.warning(f"Could not parse line into triplet: {line}")
    logger.info(f"Parsed {len(triplets)} triplets from output.")
    return triplets
# --- End Custom Parsing Function ---


# --- Async Rate Limited Embedding Class (Corrected Init v3 + Embedding Log) ---
class AsyncRateLimitedEmbedding(GoogleGenAIEmbedding):
    """
    Async-native rate limited embedding model for Google Gemini.
    Designed to work properly with PropertyGraphIndex's asyncio operations.
    """
    # Configuration - Declare all fields used by the class
    requests_per_minute: int = Field(default=150, description="Maximum requests per minute (RPM)")
    max_retries: int = Field(default=5, description="Maximum retries for rate limit errors")
    initial_backoff: float = Field(default=1.0, description="Initial backoff time in seconds.")
    backoff_factor: float = Field(default=2.0, description="Multiplier for each subsequent backoff.")

    # Internal rate limiting state - Use PrivateAttr for non-config state
    _request_timestamps: List[float] = PrivateAttr(default_factory=list)
    _lock: asyncio.Lock = PrivateAttr(default=None) # Initialize in __init__
    _count: int = PrivateAttr(default=0)
    _last_request_time: float = PrivateAttr(default=0.0)


    def __init__(self, *args, **kwargs):
        # Pop custom args *before* super().__init__
        requests_per_minute = kwargs.pop("requests_per_minute", 150)
        max_retries = kwargs.pop("max_retries", 5)
        initial_backoff = kwargs.pop("initial_backoff", 1.0)
        backoff_factor = kwargs.pop("backoff_factor", 2.0)
        # Keep embed_batch_size in kwargs for the parent

        # Initialize the parent class FIRST
        super().__init__(*args, **kwargs)

        # Now set our custom attributes AFTER parent init
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor

        # Initialize private attributes
        self._lock = asyncio.Lock()
        self._request_timestamps = []
        self._count = 0
        self._last_request_time = 0.0

        logger.info(f"Initialized AsyncRateLimitedEmbedding with RPM={self.requests_per_minute}, "
                   f"embed_batch_size={self.embed_batch_size}") # Log parent's batch size

    def _is_retryable(self, error: Exception) -> bool:
        """Check if the error is a known retryable type."""
        # (Implementation remains the same)
        if isinstance(error, RETRYABLE_EXCEPTIONS): return True
        if hasattr(error, '__cause__') and isinstance(error.__cause__, RETRYABLE_EXCEPTIONS): return True
        if '429' in str(error) or 'rate limit' in str(error).lower() or 'quota' in str(error).lower(): return True
        try:
            import httpcore
            if isinstance(error, httpcore.ReadError):
                logger.debug("Identified httpcore.ReadError as potentially retryable.")
                return True
        except ImportError: pass
        try:
            import google.genai.errors
            if isinstance(error, google.genai.errors.ServerError) and ('502' in str(error) or '503' in str(error) or '504' in str(error)):
                logger.debug(f"Identified {type(error).__name__} as retryable.")
                return True
        except ImportError: pass
        except AttributeError: pass
        return False

    async def _async_adaptive_wait(self):
        """Asynchronously adaptively wait."""
        async with self._lock: # Use async lock
            now = time.time()
            minute_ago = now - 60
            self._request_timestamps = [ts for ts in self._request_timestamps if ts > minute_ago]

            if len(self._request_timestamps) >= self.requests_per_minute:
                oldest = min(self._request_timestamps) if self._request_timestamps else now
                wait_time = (oldest + 60) - now
                if wait_time > 0:
                    logger.warning(f"Rate limit ({self.requests_per_minute} RPM) reached, "
                                  f"waiting {wait_time:.2f}s before next request")
                    await asyncio.sleep(wait_time) # Use async sleep

            self._request_timestamps.append(time.time())
            self._count += 1

            if self._count % 10 == 0:
                 logger.info(f"Embedding progress: {self._count} requests made, "
                            f"{len(self._request_timestamps)}/{self.requests_per_minute} requests in last minute")


    async def _retryable_request(self, func, *args, **kwargs):
        """Execute a request with retries for rate limit errors."""
        retry_count = 0
        backoff = self.initial_backoff

        while True:
            try:
                await self._async_adaptive_wait() # Use async wait
                return await func(*args, **kwargs)
            except Exception as e:
                retry_count += 1
                is_err_retryable = self._is_retryable(e)
                if is_err_retryable and retry_count <= self.max_retries:
                    logger.warning(f"Retryable error: {type(e).__name__}, retrying in {backoff:.2f}s "
                                  f"(attempt {retry_count}/{self.max_retries})")
                    await asyncio.sleep(backoff)
                    backoff *= self.backoff_factor
                else:
                    if retry_count > self.max_retries: logger.error(f"Max retries ({self.max_retries}) exceeded for {type(e).__name__}: {str(e)}")
                    else: logger.error(f"Non-retryable error during embedding: {type(e).__name__} - {str(e)}")
                    raise

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text with rate limiting."""
        async def _embed():
            return await super(AsyncRateLimitedEmbedding, self)._aget_text_embedding(text)
        return await self._retryable_request(_embed)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts, respecting rate limits via individual calls."""
        logger.debug(f"Starting _aget_text_embeddings for {len(texts)} texts (processing individually)")
        all_embeddings = []
        self._count = 0 # Reset count for this overall call

        # Process texts one by one using the rate-limited single embedding method
        tasks = [self._aget_text_embedding(text) for text in texts]
        all_embeddings = await asyncio.gather(*tasks)

        # Log sample embedding
        if all_embeddings and len(all_embeddings) > 0 and all_embeddings[0]:
             sample_embedding_snippet = str(all_embeddings[0][:5]) + "..."
             logger.debug(f"Sample embedding generated (first 5 dims): {sample_embedding_snippet}")
        elif all_embeddings and len(all_embeddings) > 0:
             logger.warning("First embedding generated was empty or invalid.")
        else:
             logger.warning("No embeddings were generated.")


        logger.debug(f"Completed _aget_text_embeddings for {len(texts)} texts")
        return all_embeddings

    # Override sync methods - Raise error as async is expected by PropertyGraphIndex
    def _get_text_embedding(self, text: str) -> List[float]:
        raise NotImplementedError("Sync embedding not supported by AsyncRateLimitedEmbedding")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Sync embedding not supported by AsyncRateLimitedEmbedding")

# --- End Custom Rate-Limited Embedding Class ---


class PropertyGraphRetriever(BaseRetriever):
    """Property Graph based retriever using LlamaIndex."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the property graph-based retriever.

        Args:
            config: Configuration dictionary containing:
                - llm_model: LLM model name (e.g., "gemini-2.0-flash")
                - embedding_model: Embedding model name (e.g., "models/text-embedding-004")
                - cache_dir: Directory for caching the property graph store.
                - chunk_size: Size of document chunks (default: 1000)
                - chunk_overlap: Overlap between chunks (default: 200)
                - GOOGLE_API_KEY: Google API Key (injected by ExperimentManager)
        """
        super().__init__(config)

        self.llm_model_name = config.get("llm_model", "gemini-2.0-flash")
        self.embedding_model_name = config.get("embedding_model", "models/text-embedding-004")
        self.cache_dir = config.get("cache_dir", "cache/property_graph_indices") # Specific cache dir
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.google_api_key = config.get("GOOGLE_API_KEY") # Get injected key
        # Ensure num_workers for LLM extractor is 1 as per previous step
        self.num_workers = config.get("num_workers", 1) # Default to 1 worker

        # Will be initialized in setup()
        self.property_graph_index: Optional[PropertyGraphIndex] = None # Type hint added
        self.retriever = None
        self.index_id = None
        self.llm = None
        self.embed_model = None
        self.storage_context: Optional[StorageContext] = None # Explicitly manage storage context
        self.llama_debug_handler = None # Add placeholder for debug handler

    def setup(self, documents: List[Dict[str, Any]]) -> None:
        """Set up the retriever with documents."""
        logger.debug("Setting up PropertyGraphRetriever")

        if not self.google_api_key:
             logger.warning("GOOGLE_API_KEY not found in config. Gemini models might fail.")
             # Depending on auth setup (e.g., gcloud login), it might still work

        # Create a unique ID for this index
        import hashlib
        sample_size = min(len(documents), 20) # Use up to 20 docs for hash
        content_to_hash = "".join(doc["content"][:100] for doc in documents[:sample_size])
        doc_hash = hashlib.md5(content_to_hash.encode()).hexdigest()[:10]

        safe_llm_name = self.llm_model_name.split('/')[-1].replace('-', '_')
        safe_embed_name = self.embedding_model_name.split('/')[-1].replace('-', '_')
        self.index_id = f"prop_graph_{doc_hash}_{safe_llm_name}_{safe_embed_name}"
        logger.info(f"Generated Index ID: {self.index_id}") # Log the generated ID

        # Set up LLM and Embeddings via LlamaIndex Settings, including CallbackManager
        self._setup_llama_settings()

        # Try to load existing index from cache
        if self._load_from_cache():
            logger.info(f"Loaded property graph index from cache ({self.index_id})")
            # Index is already loaded in _load_from_cache
        else:
            # Build property graph index
            self._build_property_graph(documents)
            # Save to cache
            #self._save_to_cache() # Persist happens inside _build_property_graph now

        # --- Print LLM Debug Info ---
        if self.llama_debug_handler:
            try:
                logger.info("--- LLAMA DEBUG HANDLER: LLM Calls ---")
                # Use get_event_pairs to get (start_event, end_event) tuples
                event_pairs = self.llama_debug_handler.get_event_pairs(CBEventType.LLM)
                if event_pairs:
                    for i, pair in enumerate(event_pairs):
                        start_event, end_event = pair
                        logger.info(f"--- LLM Call {i+1} ---")
                        # Access payload from the start event for input, end event for output
                        logger.info(f"Input: {start_event.payload.get('messages') or start_event.payload.get('prompt')}")
                        logger.info(f"Output: {end_event.payload.get('response')}")
                        logger.info("-" * 20)
                else:
                    logger.info("No LLM events captured by LlamaDebugHandler.")
                logger.info("--- END LLAMA DEBUG HANDLER ---")
            except Exception as debug_err:
                logger.error(f"Error printing LlamaDebugHandler info: {debug_err}")
        # --- End Print LLM Debug Info ---


        # Create retriever
        self._setup_retriever()

        self.is_initialized = True

    def _validate_components(self):
        """Validate that the LLM and embedding models are properly initialized."""
        if not Settings.llm: # Check global settings
            logger.error("LLM not initialized in LlamaIndex Settings")
            raise ValueError("LLM not initialized in LlamaIndex Settings")

        if not Settings.embed_model: # Check global settings
            logger.error("Embedding model not initialized in LlamaIndex Settings")
            raise ValueError("Embedding model not initialized in LlamaIndex Settings")

        logger.debug(f"Components validated - Using LLM: {Settings.llm.__class__.__name__}, Embeddings: {Settings.embed_model.__class__.__name__}")

    def _setup_llama_settings(self) -> None:
        """Set up the LLM, embeddings, and callback manager using LlamaIndex Settings."""
        # Use newer Google GenAI integrations
        self.llm = GoogleGenAI(model_name=self.llm_model_name, api_key=self.google_api_key, temperature=0.2)
        logger.debug(f"Using GoogleGenAI LLM: {self.llm_model_name}")

        # Use our async-native rate-limited embeddings implementation
        self.embed_model = AsyncRateLimitedEmbedding(
            model_name=self.embedding_model_name,
            api_key=self.google_api_key,
            embed_batch_size=5,        # Pass standard batch size
            requests_per_minute=150    # Respect Google's rate limit
        )
        logger.info(f"Using AsyncRateLimitedEmbedding: {self.embedding_model_name}")

        # Setup LlamaDebugHandler
        self.llama_debug_handler = LlamaDebugHandler(print_trace_on_end=False) # Don't print full trace automatically
        callback_manager = CallbackManager([self.llama_debug_handler])

        # Configure global LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.callback_manager = callback_manager # Set globally
        # Set chunk size globally
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap

        # Validate the components
        self._validate_components()

        logger.debug(f"LlamaIndex Settings configured with LlamaDebugHandler.")

    def _build_property_graph(self, documents: List[Dict[str, Any]]) -> None:
        """Build property graph index from documents."""
        logger.info(f"Building property graph index from {len(documents)} documents.")

        # Validate components are set in Settings
        self._validate_components()

        # Define KG extractors explicitly to control num_workers and use custom parsing function
        logger.debug(f"Setting up KG extractors with num_workers=1 and custom parsing function")
        kg_extractors = [
            SimpleLLMPathExtractor(
                llm=Settings.llm, # Use LLM from global settings
                num_workers=1, # Keep workers=1
                parse_fn=custom_parse_triplets_fn, # Use custom parsing function
                extract_prompt=KG_TRIPLET_EXTRACT_PROMPT_SIMPLE, # Use simpler prompt

            ),
            ImplicitPathExtractor(),
        ]
        logger.debug(f"KG extractors defined: {[e.__class__.__name__ for e in kg_extractors]}")

        # --- DEBUG: Limit documents for faster testing ---
        doc_limit = 5
        logger.warning(f'DEBUG: Limiting graph building to first {doc_limit} documents for testing!')
        documents_to_process = documents[:doc_limit]
        logger.info(f"Processing {len(documents_to_process)} documents due to limit.")
        # --- END DEBUG ---

        # Convert to LlamaIndex documents
        logger.debug("Converting documents to LlamaIndex format...")
        llama_docs = [
            LlamaDocument(text=doc["content"], metadata=doc["metadata"])
            for doc in documents_to_process # Use limited list
        ]
        logger.debug(f"Converted {len(llama_docs)} documents")

        # --- Optional Chunking ---
        docs_or_nodes = llama_docs # Use original documents
        num_input_items = len(docs_or_nodes)
        logger.debug(f"Using {num_input_items} documents/nodes for graph construction")
        # --- End Optional Chunking ---

        # --- Explicitly create StorageContext ---
        # --- Monkey Patching Setup for Persist ---
        # Store original persist method
        original_persist = SimplePropertyGraphStore.persist

        # Define patched persist method
        def patched_persist(store_self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None) -> None:
            """Patched persist method to enforce UTF-8 encoding."""
            fs = fs or fsspec.filesystem("file")
            dirpath = os.path.dirname(persist_path)
            if not fs.exists(dirpath):
                fs.makedirs(dirpath)
            # Explicitly open with UTF-8 encoding
            logger.debug(f"Patched persist: Opening {persist_path} with UTF-8 encoding.")
            with fs.open(persist_path, "w", encoding="utf-8") as f:
                f.write(store_self.graph.model_dump_json())
            logger.debug(f"Patched persist: Finished writing to {persist_path}.")

        # Apply the patch
        SimplePropertyGraphStore.persist = patched_persist
        logger.debug("Applied monkey patch to SimplePropertyGraphStore.persist for UTF-8 encoding.")
        # --- End Monkey Patching Setup ---

        logger.info("Creating new SimplePropertyGraphStore and StorageContext.")
        property_graph_store = SimplePropertyGraphStore()
        self.storage_context = StorageContext.from_defaults(property_graph_store=property_graph_store)
        # --- End explicit creation ---

        # Log starting graph construction
        logger.info(f"Starting PropertyGraphIndex.from_documents with {num_input_items} documents/nodes...")
        logger.info("This may take a while. Built-in progress bar enabled.")
        start_time = time.time()

        try:
 # This try block now includes the patching logic scope
            # Build the index using the configured extractors and explicit storage context
            self.property_graph_index = PropertyGraphIndex.from_documents(
                docs_or_nodes, # Use limited list
                kg_extractors=kg_extractors, # Pass configured extractors
                storage_context=self.storage_context, # Pass the created context
                show_progress=True, # Show progress bar
                include_embeddings=True # Explicitly set to include embeddings
            )
            end_time = time.time()
            logger.info(f"PropertyGraphIndex.from_documents finished successfully in {end_time - start_time:.2f} seconds.")
            # Persist using the potentially patched method
            persist_target_dir = self.cache_dir # Use the base cache dir
            logger.info(f"Attempting to persist storage context to directory: {persist_target_dir}")
            self.property_graph_index.storage_context.persist(persist_dir=persist_target_dir)
            logger.info(f"Successfully persisted property graph index to {persist_target_dir}")

        except Exception as e:
            end_time = time.time()
            logger.error(f"PropertyGraphIndex.from_documents or persist failed after {end_time - start_time:.2f} seconds: {e}", exc_info=True)
            raise
        finally:
            # --- Monkey Patching Teardown for Persist ---
            # Restore original method (important!)
            SimplePropertyGraphStore.persist = original_persist
            logger.debug("Restored original SimplePropertyGraphStore.persist method.")
            # --- End Monkey Patching Teardown ---

    # Removed _log_in_memory_graph_state as it was potentially misleading

    def _save_to_cache(self) -> None:
        """Save the property graph index's storage context to cache. (Deprecated - persist happens in build)"""
        logger.warning("_save_to_cache is deprecated, persistence now happens within _build_property_graph.")
        # logger.info(f"Cache directory configured as: {self.cache_dir}")
        # # Use the explicitly managed storage_context
        # if not self.storage_context:
        #     logger.warning("Storage context not available for saving.")
        #     return
        #
        # # Persist the entire storage context managed by the retriever
        # cache_path = os.path.join(self.cache_dir, self.index_id) # This path might be wrong if persist uses base dir
        # try:
        #     logger.info(f"Attempting to save storage context to {cache_path}")
        #     self.property_graph_index.storage_context.persist(persist_dir=cache_path)
        #     logger.info(f"Persist call completed for {cache_path}")
        #     # Verify directory exists after persist
        #     if os.path.exists(cache_path):
        #         logger.info(f"Verified cache directory exists at {cache_path}")
        #         # Optionally list contents for deeper verification
        #         try:
        #             contents = os.listdir(cache_path)
        #             logger.info(f"Cache directory contents: {contents}")
        #         except Exception as list_err:
        #             logger.warning(f"Could not list contents of cache directory {cache_path}: {list_err}")
        #     else:
        #         logger.error(f"Cache directory NOT found at {cache_path} immediately after persist call!")
        #     logger.info(f"Saved storage context to {cache_path}")
        # except Exception as e:
        #     logger.error(f"Error saving storage context: {str(e)}", exc_info=True)

    def _load_from_cache(self) -> bool:
        """Load the property graph index from cache."""
        # Corrected Path: Use the base cache_dir directly, as persist saves files there.
        cache_path = self.cache_dir
        logger.debug(f"Attempting to load cache from directory: {cache_path}") # Added log

        # Check if the directory exists (StorageContext saves multiple files)
        # Specifically check for the graph store file as a better indicator
        graph_store_file_path = os.path.join(cache_path, "property_graph_store.json")
        if not os.path.isfile(graph_store_file_path):
            logger.info(f"Index cache file ({graph_store_file_path}) not found, will build index.")
            return False
        if not os.path.isdir(cache_path): # Keep directory check as well
             logger.info(f"Index cache directory not found at {cache_path}, will build index.")
             return False

        # --- Monkey Patching Setup for fsspec open ---
        original_file_open = fsspec.filesystem("file").open
        patch_applied = False # Flag to track if patch was applied

        def patched_file_open(*args, **kwargs):
            """Patched fsspec open to force UTF-8 for property_graph_store.json reading."""
            nonlocal patch_applied # Allow modification of outer scope variable
            # args[0] is the file path
            file_path_arg = args[0] if args else ""
            mode_arg = kwargs.get("mode", "r") # Default mode is 'r'

            # Check if it's the target file being opened for reading
            target_file = "property_graph_store.json"
            # Use os.path.basename for robust path comparison
            is_target_file = os.path.basename(str(file_path_arg)) == target_file

            if is_target_file and mode_arg == "r":
                if kwargs.get("encoding") is None:
                    logger.debug(f"Patched fsspec open: Forcing UTF-8 for reading {file_path_arg}")
                    kwargs["encoding"] = "utf-8"
                    patch_applied = True # Mark that the patch logic was triggered
                else:
                    logger.debug(f"Patched fsspec open: Encoding already specified for {file_path_arg}, using '{kwargs.get('encoding')}'")
            # Call the original open method
            return original_file_open(*args, **kwargs)

        # Apply the patch
        fsspec.filesystem("file").open = patched_file_open
        logger.debug("Applied monkey patch to fsspec file open for UTF-8 encoding.")
        # --- End Monkey Patching Setup ---

        try:
            logger.debug(f"Attempting to load index from storage context at {cache_path}")
            # Ensure LLM and embeddings are set up in Settings (needed for loading)
            if not Settings.llm or not Settings.embed_model:
                 self._setup_llama_settings()

            # Load the storage context first
            # This call will now use the patched fsspec open method internally
            self.storage_context = StorageContext.from_defaults(persist_dir=cache_path)
            logger.debug(f"StorageContext.from_defaults call completed using persist_dir: {cache_path}")

            # Load the index using the loaded storage context
            self.property_graph_index = load_index_from_storage(self.storage_context)

            # Verify it's the correct type (optional but good practice)
            if not isinstance(self.property_graph_index, PropertyGraphIndex):
                logger.error(f"Loaded index from {cache_path} is not a PropertyGraphIndex.")
                self.property_graph_index = None # Reset if wrong type
                self.storage_context = None # Reset context too
                return False

            logger.info(f"Successfully loaded property graph index from {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index from cache at {cache_path} (fsspec patch active: {patch_applied}): {str(e)}", exc_info=True) # Log if patch was active
            self.property_graph_index = None # Ensure index is None on failure
            self.storage_context = None # Reset context too
            return False
        finally:
            # --- Monkey Patching Teardown for fsspec open ---
            # Restore original method (important!)
            fsspec.filesystem("file").open = original_file_open
            logger.debug("Restored original fsspec file open method.")
            # --- End Monkey Patching Teardown ---

    def _setup_retriever(self) -> None:
        """Set up the retriever for the property graph."""
        if not self.property_graph_index:
            raise ValueError("Property graph index not initialized")

        # Validate components are set in Settings
        self._validate_components()

        logger.debug("Setting up retriever for PropertyGraphIndex")
        try:
            # PropertyGraphIndex typically uses .as_retriever()
            self.retriever = self.property_graph_index.as_retriever(
                 # similarity_top_k=10 # Configure retriever parameters if needed
            )
            logger.debug("Retriever created successfully.")
        except Exception as e:
            logger.error(f"Failed to create retriever: {str(e)}")
            raise

    # Removed retry decorator to allow hard fail on persistent errors
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents based on the query using the property graph."""
        if not self.is_initialized or not self.retriever:
            raise ValueError("Retriever not initialized. Call setup() first.")

        logger.info(f"Retrieving with property graph: {query}")

        # Validate components are set in Settings
        self._validate_components()

        try:
            # Use the retriever
            source_nodes = self.retriever.retrieve(query)
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise ValueError(f"Retrieval failed: {e}") from e

        if not source_nodes:
            logger.warning("Property graph retrieval returned no results.")
            return [] # Return empty list if no nodes found

        # Format results
        formatted_results = []
        seen_ids = set() # Use node_id for deduplication

        for node_with_score in source_nodes[:top_k]:
            node = node_with_score.node # Access the actual node object

            # Use node_id for deduplication if available
            node_id = getattr(node, 'node_id', None)
            if node_id and node_id in seen_ids:
                continue
            if node_id:
                seen_ids.add(node_id)

            if not hasattr(node, 'get_content'):
                logger.error(f"Retrieved node missing get_content method: {node}")
                continue # Skip malformed nodes

            # Get content, potentially including properties based on MetadataMode
            content = node.get_content(metadata_mode=MetadataMode.LLM)

            if not content or not isinstance(content, str) or len(content.strip()) == 0:
                logger.warning(f"Skipping node with empty or invalid content: {node_id}")
                continue

            # If we didn't use node_id for deduplication, use content
            if not node_id:
                if content in seen_ids:
                    continue
                seen_ids.add(content)

            metadata = dict(node.metadata) if hasattr(node, 'metadata') else {}

            formatted_results.append({
                "content": content,
                "metadata": metadata,
                "score": node_with_score.score or 1.0, # Use score from NodeWithScore
            })

        if len(formatted_results) == 0:
             logger.warning("All retrieved results were filtered out (duplicates/empty).")

        return formatted_results[:top_k] # Ensure we respect top_k even after filtering

    def export_graph_html(self, output_path: str = "property_graph.html") -> None:
        """
        Exports the current property graph to an HTML file using networkx visualization.

        Args:
            output_path (str): The path where the HTML file should be saved.
                               Defaults to "property_graph.html".
        """
        if not self.is_initialized or not self.property_graph_index or not self.property_graph_index.property_graph_store:
            logger.error("Retriever or graph store not initialized. Cannot export graph.")
            raise ValueError("Retriever or graph store not initialized. Call setup() first.")

        try:
            # The save_networkx_graph method is part of the graph store instance
            logger.info(f"Attempting export using graph store: {type(self.property_graph_index.property_graph_store)}")
            self.property_graph_index.property_graph_store.save_networkx_graph(name=output_path)
            logger.info(f"Successfully exported property graph to HTML: {output_path}")
        except AttributeError:
            logger.error("The current graph store does not support HTML export via save_networkx_graph.")
            logger.error("Please ensure you are using a compatible graph store (like SimplePropertyGraphStore).")
            # Re-raise or handle as appropriate for your application
            raise NotImplementedError("Graph store does not support HTML export.")
        except Exception as e:
            logger.error(f"Failed to export property graph to HTML: {e}", exc_info=True)
            raise
