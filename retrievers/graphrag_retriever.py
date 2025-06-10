# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
GraphRAG Retriever wrapper implementation.
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import pandas as pd
import numpy as np # Import numpy

from .base_retriever import BaseRetriever

# --- GraphRAG Specific Imports ---
# Corrected import path for GraphRagConfig and other models
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.models.input_config import InputConfig
from graphrag.config.models.output_config import OutputConfig
from graphrag.config.models.cache_config import CacheConfig
from graphrag.config.models.vector_store_config import VectorStoreConfig
from graphrag.config.models.text_embedding_config import TextEmbeddingConfig
from graphrag.config.models.extract_graph_config import ExtractGraphConfig
# Add other workflow configs as needed, e.g.:
# from graphrag.config.models.summarize_descriptions_config import SummarizeDescriptionsConfig
# from graphrag.config.models.community_reports_config import CommunityReportsConfig

from graphrag.config.enums import (
    ModelType,
    AuthType,
    InputType, # This enum is for storage type (file/blob)
    InputFileType, # This enum is for file format (csv/text/json)
    OutputType,
    # VectorStoreType, # Moved import
    TextEmbeddingTarget,
)
from graphrag.api.index import build_index # Corrected import path
from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks # Import console callbacks
from graphrag.query.factory import (
    get_local_search_engine,
    get_global_search_engine,
    get_drift_search_engine,
    # get_basic_search_engine, # If needed later
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
# from graphrag.query.structured_search.basic_search.search import BasicSearch
from graphrag.vector_stores.factory import VectorStoreFactory, VectorStoreType # Corrected import location
# Corrected data model imports (again)
from graphrag.data_model.community_report import CommunityReport
from graphrag.data_model.text_unit import TextUnit
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.data_model.community import Community
from graphrag.data_model.covariate import Covariate


log = logging.getLogger(__name__)

# Load environment variables for API keys
load_dotenv()

class GraphRAGRetriever(BaseRetriever):
    """
    Wrapper class for Microsoft GraphRAG retrieval.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GraphRAGRetriever.

        Args:
            config: Configuration dictionary for the retriever. Expected keys:
                    - llm_config: Dict containing LLM related settings (e.g., api_keys)
                    - retriever_config: Dict containing GraphRAG specific settings
                                        (e.g., cache_path_root, query_mode)
        """
        super().__init__(config)
        self.llm_config = config.get("llm_config", {})
        self.retriever_config = config.get("retriever_config", {})
        self.graphrag_cache_path: Optional[Path] = None
        self.graphrag_config: Optional[Any] = None # Placeholder for GraphRagConfig object
        self.query_engine: Optional[Any] = None # Placeholder for GraphRAG query engine/runner
        # Store the last generated answer from GraphRAG
        self.last_generated_answer: Optional[str] = None
        self.generates_answer = True  # Indicate this retriever generates answers

        # --- Get API Keys ---
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY") # Use GOOGLE_API_KEY for Gemini
        if not self.deepseek_api_key:
            log.warning("DEEPSEEK_API_KEY not found in environment variables.")
        if not self.google_api_key:
            log.warning("GOOGLE_API_KEY not found in environment variables.") # Updated warning

    def setup(self, documents: List[Dict[str, Any]]) -> None:
        """
        Set up the GraphRAG index. This involves configuring and running the
        GraphRAG indexing pipeline.

        Args:
            documents: A list of documents to index. Each document is a dictionary
                        expected to have 'id' and 'page_content' keys, plus optional metadata.
        """
        log.info(f"Setting up GraphRAGRetriever for {len(documents)} documents...")
        # --- Debug: Log keys of the first document ---
        if documents:
            log.info(f"Keys in first document received: {list(documents[0].keys())}")
        else:
            log.warning("Received empty document list in setup.")
        # --- End Debug ---

        if not self.deepseek_api_key or not self.google_api_key: # Check for google_api_key
            msg = "Missing API keys (DEEPSEEK_API_KEY or GOOGLE_API_KEY) required for GraphRAG setup." # Updated error message
            log.error(msg)
            raise ValueError(msg)

        # --- Limit documents for indexing if configured ---
        # Limit removed - always using all documents for indexing
        documents_to_index = documents
        limit_str = "all" # Keep limit_str for cache path consistency, always "all"
        log.info(f"GraphRAG will index {len(documents_to_index)} documents.")
        # --- End limit ---

        # --- Define Cache Path (incorporating limit) ---
        cache_root = Path(self.retriever_config.get("cache_path_root", "cache"))
        # Simple hash based on query mode and limit for differentiation
        query_mode = self.retriever_config.get("query_mode", "local")
        config_hash = f"{query_mode}_{limit_str}"
        self.graphrag_cache_path = cache_root / "graphrag" / config_hash
        self.graphrag_cache_path.mkdir(parents=True, exist_ok=True)
        log.info(f"GraphRAG cache path: {self.graphrag_cache_path}")
        # --- End Cache Path ---

        # Construct GraphRagConfig programmatically (defines settings, doesn't create files yet)
        try:
            # Pass the potentially limited list
            self.graphrag_config = self._build_graphrag_config(documents_to_index)
        except Exception as e:
            log.error(f"Error building GraphRagConfig: {e}", exc_info=True)
            raise

        # Check if index already exists in the specific cache path
        expected_output_file = self.graphrag_cache_path / "output" / "entities.parquet"
        index_exists = expected_output_file.exists()

        if not index_exists:
            log.info("GraphRAG index not found in cache. Building index...")

            # --- Create temporary input file ONLY if indexing ---
            log.info("Creating temporary input CSV for GraphRAG indexing...")
            docs_df = pd.DataFrame(documents_to_index)
            if 'content' not in docs_df.columns:
                 # This check should ideally be done earlier, but double-check here
                 raise ValueError("Input documents must contain a 'content' key.")
            temp_input_dir = self.graphrag_cache_path / "input"
            temp_input_dir.mkdir(exist_ok=True)
            temp_input_file = temp_input_dir / "input_docs.csv"
            docs_df.to_csv(temp_input_file, index=False)
            log.info(f"Temporary input CSV created at {temp_input_file} with {len(docs_df)} rows.")
            # --- End CSV Creation ---

            try:
                # Need to run the async function in a sync context
                log.info("Starting GraphRAG build_index...")
                # Instantiate and pass console callbacks for detailed logging
                callbacks = [ConsoleWorkflowCallbacks()]
                # Pass the config object which points to the temp input CSV
                results = asyncio.run(build_index(config=self.graphrag_config, callbacks=callbacks))

                # Check results for errors
                has_errors = False
                for result in results:
                    if result.errors and len(result.errors) > 0:
                        log.error(f"Errors in workflow '{result.workflow}': {result.errors}")
                        has_errors = True
                if has_errors:
                     raise RuntimeError("GraphRAG indexing failed with errors.")
                else:
                    log.info("GraphRAG indexing completed successfully.")

            except Exception as e:
                log.error(f"Error during GraphRAG indexing: {e}", exc_info=True)
                raise
        else:
            log.info("GraphRAG index found in cache. Skipping build.")

        # Set up the query engine/runner
        try:
            self.query_engine = self._setup_query_engine()
            log.info("GraphRAG query engine setup complete.")
        except Exception as e:
            log.error(f"Error setting up GraphRAG query engine: {e}", exc_info=True)
            raise

        log.info("GraphRAGRetriever setup complete.")


    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query using GraphRAG.

        Args:
            query: The query string.
            top_k: The number of documents to retrieve (used for formatting output).

        Returns:
            A list of retrieved document chunks, each as a dictionary.
        """
        log.info(f"Retrieving top {top_k} documents for query: '{query}' using GraphRAG")

        if not self.query_engine:
             msg = "GraphRAG query engine not set up. Run setup() first."
             log.error(msg)
             msg = "GraphRAG query engine not set up. Run setup() first."
             log.error(msg)
             raise RuntimeError(msg)

        try:
            # Execute the query using GraphRAG's query API (async)
            # Note: GraphRAG search methods might not directly support 'top_k'.
            # The context builder likely handles relevance ranking internally.
            # We retrieve the context used for the response.
            log.info(f"Running GraphRAG {type(self.query_engine).__name__}...")
            results: Any = asyncio.run(self.query_engine.search(query=query)) # Assuming SearchResult type
            log.info("GraphRAG search completed.")

            # Store the generated answer if it exists in the results
            if hasattr(results, 'answer'):
                self.last_generated_answer = results.answer
                log.info(f"Stored GraphRAG generated answer: {self.last_generated_answer[:100]}...")
            elif hasattr(results, 'response'):
                self.last_generated_answer = results.response
                log.info(f"Stored GraphRAG generated answer: {self.last_generated_answer[:100]}...")
            elif hasattr(results, 'generated_response'):
                self.last_generated_answer = results.generated_response
                log.info(f"Stored GraphRAG generated answer: {self.last_generated_answer[:100]}...")
            else:
                self.last_generated_answer = None
                log.warning("No generated answer found in GraphRAG results")

            # Format GraphRAG results into the expected list of dicts
            # The primary result is the generated response string, but we want the context.
            # Pass top_k to the formatter
            formatted_results = self._format_results(results, top_k)

        except Exception as e:
            log.error(f"Error during GraphRAG retrieval: {e}", exc_info=True)
            formatted_results = [] # Return empty list on error
            self.last_generated_answer = None

        log.info(f"Retrieved {len(formatted_results)} context documents using GraphRAG.")
        return formatted_results

    def _build_graphrag_config(self, documents: List[Dict[str, Any]]) -> Any: # Replace Any with GraphRagConfig
        """Helper to construct the GraphRagConfig object."""
        log.debug("Building GraphRagConfig...")

        # --- Define Model Configs ---
        deepseek_lm_config = LanguageModelConfig(
            api_key=self.deepseek_api_key,
            type=ModelType.OpenAIChat, # Treat as OpenAI compatible
            model="deepseek-chat", # Or "deepseek-reasoner"
            api_base="https://api.deepseek.com/v1",
            encoding_model="cl100k_base", # Explicitly set encoding for tiktoken
            # Add other params like temperature, max_tokens if needed from self.llm_config
            # Ensure model_supports_json is set if needed by GraphRAG workflows
            model_supports_json=True, # Assuming DeepSeek supports JSON mode
        )

        gemini_emb_config = LanguageModelConfig(
            api_key=self.google_api_key, # Use google_api_key
            type=ModelType.OpenAIEmbedding, # Treat as OpenAI compatible
            model="text-embedding-004", # Or newer Gemini embedding model
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            encoding_model="cl100k_base", # Explicitly set encoding for tiktoken
        )

        # --- Define Input Config ---
        # GraphRAG often works with files. We might need to temporarily write
        # the input documents to a CSV/JSON file within the cache dir.
        # For simplicity now, assume we can adapt later or GraphRAG accepts DataFrames.
        # Let's prepare a DataFrame first.
        docs_df = pd.DataFrame(documents)
        # Use 'content' as the text key, as DataManager standardizes it
        if 'content' not in docs_df.columns:
             raise ValueError("Input documents must contain a 'content' key.")
        # TODO: Decide on input strategy (temp file vs direct DataFrame if supported)
        # Using a temporary file approach for now:
        temp_input_dir = self.graphrag_cache_path / "input"
        temp_input_dir.mkdir(exist_ok=True)
        temp_input_file = temp_input_dir / "input_docs.csv"
        docs_df.to_csv(temp_input_file, index=False)

        input_config = InputConfig(
            file_type=InputFileType.csv, # Use InputFileType enum
            base_dir="input", # Correct base_dir relative to root_dir
            file_pattern=".*\\.csv$",
            text_column="content", # Use 'content'
            # Add other columns like 'id' if needed by GraphRAG
            # title_column="id", # Assuming 'id' exists in input documents
            # metadata=list(docs_df.columns.drop('content', errors='ignore')) # Pass other columns as metadata
        )


        # --- Define Storage Configs ---
        # Pointing outputs, cache, and vector store to subdirs within self.graphrag_cache_path
        output_config = OutputConfig(
            type=OutputType.file, # Corrected back to lowercase
            base_dir="output", # Relative to root_dir
        )
        cache_config = CacheConfig(
            type=OutputType.file, # Corrected back to lowercase
            base_dir="cache", # Relative to root_dir
        )
        # Using LanceDB as the default vector store type
        vector_store_config = VectorStoreConfig(
            type=VectorStoreType.LanceDB,
            db_uri="vector_store/lancedb", # Relative to root_dir
        )

        # --- Define Workflow Configs ---
        # Configure text embedding to use Gemini
        text_embedding_config = TextEmbeddingConfig(
            model_id="gemini_embedding_model",
            vector_store_id="default", # Use the key defined in vector_store dict below
            target=TextEmbeddingTarget.required, # Embed necessary fields
            # batch_size=... # Configure if needed
        )

        # Configure graph extraction to use DeepSeek
        extract_graph_config = ExtractGraphConfig(
            model_id="deepseek_chat_model",
            # prompt=... # Use default or specify custom prompt path
            # entity_types=... # Define entity types if needed
            # max_gleanings=...
        )

        # TODO: Configure other workflows as needed (summarization, clustering, etc.)
        # based on self.retriever_config settings (e.g., query_mode might influence this)

        # --- Instantiate GraphRagConfig ---
        config = GraphRagConfig(
            root_dir=str(self.graphrag_cache_path), # Set root for relative paths
            models={
                "deepseek_chat_model": deepseek_lm_config,
                "gemini_embedding_model": gemini_emb_config,
                # GraphRAG requires default models to be defined, let's map ours
                "default_chat_model": deepseek_lm_config,
                "default_embedding_model": gemini_emb_config,
            },
            input=input_config,
            output=output_config,
            cache=cache_config,
            vector_store={"default": vector_store_config}, # Key must match vector_store_id above
            embed_text=text_embedding_config,
            extract_graph=extract_graph_config,
            # Add other workflow configs here if customizing beyond defaults
            # e.g., summarize_descriptions=SummarizeDescriptionsConfig(model_id="deepseek_chat_model", ...)
            # community_reports=CommunityReportsConfig(model_id="deepseek_chat_model", ...)
            # cluster_graph=ClusterGraphConfig(...)
            # ...
        )
        log.debug(f"Constructed GraphRagConfig: {config}")
        return config

    def _setup_query_engine(self) -> Any:
        """Helper to set up the GraphRAG query engine."""
        log.debug("Setting up GraphRAG query engine...")
        if not self.graphrag_config or not self.graphrag_cache_path:
            raise RuntimeError("GraphRAG config or cache path not initialized. Run setup() first.")

        output_dir = self.graphrag_cache_path / "output"
        vector_store_dir = self.graphrag_cache_path / "vector_store" / "lancedb" # Assuming LanceDB default path

        # --- Load Data Artifacts ---
        try:
            log.info(f"Loading GraphRAG artifacts from: {output_dir}")
            entities_df = pd.read_parquet(output_dir / "entities.parquet")
            reports_df = pd.read_parquet(output_dir / "community_reports.parquet")
            text_units_df = pd.read_parquet(output_dir / "text_units.parquet")
            relationships_df = pd.read_parquet(output_dir / "relationships.parquet")
            # communities_df = pd.read_parquet(output_dir / "communities.parquet") # Needed for Global
            # covariates_df = pd.read_parquet(output_dir / "covariates.parquet") # Needed for Local? Check factory args

            # Convert DataFrames to lists of data model objects using their from_dict methods
            # Ensure list-like columns are converted to Python lists to avoid NumPy boolean errors
            def ensure_python_list(item):
                """Converts item to a Python list if possible, otherwise returns None."""
                if item is None:
                    return None
                # Handle numpy arrays explicitly
                if isinstance(item, np.ndarray):
                    # Check for NaN within the numpy array if necessary
                    # This check might depend on the expected dtype and how NaNs are represented
                    # For simplicity, we convert to list first, then check for pd.isna on elements if needed
                    item_list = item.tolist()
                    # Filter out potential NaN values if they cause issues downstream
                    # return [x for x in item_list if not pd.isna(x)]
                    return item_list
                # Handle standard lists/tuples
                if isinstance(item, (list, tuple)):
                    return list(item)
                # Attempt to handle string representations of lists if necessary
                if isinstance(item, str) and item.startswith('[') and item.endswith(']'):
                    try:
                        # Use json.loads for safer parsing than eval
                        import json
                        # Handle potential single quotes if needed, be cautious
                        parsed = json.loads(item.replace("'", '"'))
                        if isinstance(parsed, list):
                            return parsed
                        else:
                            log.warning(f"Parsed string list resulted in non-list type: {type(parsed)}")
                            return None
                    except Exception as e:
                        log.warning(f"Failed to parse string list '{item}': {e}")
                        return None
                # Handle scalar non-list types (check for NaN specifically)
                if pd.isna(item):
                     return None

                log.warning(f"Unexpected type for ID list: {type(item)}. Returning None.")
                return None

            entity_records = entities_df.to_dict(orient="records")
            for record in entity_records:
                record['text_unit_ids'] = ensure_python_list(record.get('text_unit_ids'))
                record['community_ids'] = ensure_python_list(record.get('community_ids'))
            entities = [Entity.from_dict(record) for record in entity_records]

            report_records = reports_df.to_dict(orient="records")
            # Add similar list conversion if CommunityReport has list attributes causing issues
            reports = [CommunityReport.from_dict(record) for record in report_records]

            text_unit_records = text_units_df.to_dict(orient="records")
            for record in text_unit_records:
                record['relationship_ids'] = ensure_python_list(record.get('relationship_ids'))
                record['entity_ids'] = ensure_python_list(record.get('entity_ids'))
                record['document_ids'] = ensure_python_list(record.get('document_ids'))
                # Covariate IDs might be dicts, handle if necessary
            text_units = [TextUnit.from_dict(record) for record in text_unit_records]

            relationship_records = relationships_df.to_dict(orient="records")
            # Add similar list conversion if Relationship has list attributes causing issues
            relationships = [Relationship.from_dict(record) for record in relationship_records]

            # Load communities and covariates if needed for other modes (or if LocalSearch needs them)
            communities_path = output_dir / "communities.parquet"
            covariates_path = output_dir / "covariates.parquet" # Assuming this path, adjust if needed
            communities = []
            if communities_path.exists():
                 communities_df = pd.read_parquet(communities_path)
                 community_records = communities_df.to_dict(orient="records")
                 # Add similar list conversion if Community has list attributes causing issues
                 communities = [Community.from_dict(record) for record in community_records]

            covariates = {} # LocalSearch factory expects dict[str, list[Covariate]]
            if covariates_path.exists():
                covariates_df = pd.read_parquet(covariates_path)
                # Assuming the Parquet file has columns like 'covariate_type', 'id', 'subject_id', etc.
                # We need to group by covariate type.
                for cov_type, group in covariates_df.groupby('covariate_type'): # Adjust column name if different
                    covariate_records = group.to_dict(orient="records")
                    # Add similar list conversion if Covariate has list attributes causing issues
                    covariates[cov_type] = [Covariate.from_dict(record) for record in covariate_records]
            else:
                 log.warning(f"Covariates file not found at {covariates_path}. Passing empty dict to search engine.")

        except FileNotFoundError as e:
            log.error(f"Required GraphRAG output file not found: {e}. Ensure indexing ran successfully.")
            raise
        except Exception as e:
            log.error(f"Error loading GraphRAG artifacts: {e}", exc_info=True)
            raise

        # --- Load Vector Store ---
        try:
            log.info(f"Loading vector store from: {vector_store_dir}")
            # Assuming LanceDB and default collection names used by GraphRAG
            # Need to know the collection name for entity description embeddings
            # Check GraphRAG defaults or config output if needed.
            vector_store_config = self.graphrag_config.vector_store["default"]
            # The collection name is constructed from container_name and embedding target name
            container_name = vector_store_config.container_name or "default"
            embedding_target_name = "entity.description" # Target for description embeddings
            collection_name = f"{container_name}-{embedding_target_name.replace('.', '-')}" # e.g., "default-entity-description"
            log.info(f"Attempting to load LanceDB collection '{collection_name}' from {vector_store_dir}")
            # Use create_vector_store instead of get_vector_store and pass collection_name
            description_embedding_store = VectorStoreFactory.create_vector_store(
                vector_store_config.type,
                kwargs={"uri": str(vector_store_dir), "collection_name": collection_name}
            )
            # Explicitly connect to load the collection table, passing the path as db_uri
            description_embedding_store.connect(db_uri=str(vector_store_dir), collection_name=collection_name)

            # Verify the collection was loaded AFTER connect attempt
            if description_embedding_store.document_collection is None:
                raise RuntimeError(f"Failed to load LanceDB collection '{collection_name}' from {vector_store_dir}. Check if indexing created it or if the name is correct.")

            log.info(f"Vector store loaded and connected to collection '{collection_name}'.")
        except Exception as e:
            log.error(f"Error loading or connecting to vector store: {e}", exc_info=True)
            raise

        # --- Initialize Query Engine based on config ---
        query_mode = self.retriever_config.get("query_mode", "local").lower()
        response_type = self.retriever_config.get("response_type", "Multiple Paragraphs") # Example default

        log.info(f"Initializing GraphRAG query engine with mode: {query_mode}")
        if query_mode == "local":
            engine = get_local_search_engine(
                config=self.graphrag_config,
                reports=reports, # Pass loaded data
                text_units=text_units,
                entities=entities,
                relationships=relationships,
                covariates=covariates, # Pass loaded data (adjust loading if needed)
                description_embedding_store=description_embedding_store,
                response_type=response_type,
                # system_prompt=... # Optional: load from config/file
                # callbacks=... # Optional
            )
        elif query_mode == "global":
             # Load communities data if not already loaded
             # communities_df = pd.read_parquet(output_dir / "communities.parquet")
             # communities = [Community(**record) for record in communities_df.to_dict(orient="records")]
             engine = get_global_search_engine(
                 config=self.graphrag_config,
                 reports=reports,
                 entities=entities,
                 communities=communities, # Pass loaded data
                 response_type=response_type,
                 # dynamic_community_selection=... # Optional: from config
                 # map_system_prompt=... # Optional
                 # reduce_system_prompt=... # Optional
                 # callbacks=... # Optional
             )
        elif query_mode == "drift":
             engine = get_drift_search_engine(
                 config=self.graphrag_config,
                 reports=reports,
                 text_units=text_units,
                 entities=entities,
                 relationships=relationships,
                 description_embedding_store=description_embedding_store,
                 response_type=response_type,
                 # local_system_prompt=... # Optional
                 # reduce_system_prompt=... # Optional
                 # callbacks=... # Optional
             )
        # elif query_mode == "basic":
        #     # Load text unit embeddings vector store
        #     # text_unit_embedding_store = ...
        #     engine = get_basic_search_engine(...)
        else:
            raise ValueError(f"Unsupported GraphRAG query_mode: {query_mode}")

        log.info(f"GraphRAG {query_mode} search engine initialized.")
        return engine


    def _format_results(self, graphrag_results: Any, top_k: int) -> List[Dict[str, Any]]:
        """
        Helper to format GraphRAG query results (SearchResult) into the
        standard List[Dict[str, Any]] format.
        """
        log.debug("Formatting GraphRAG results...")
        formatted = []
        # Use a default top_k if not provided or invalid
        # top_k = top_k if isinstance(top_k, int) and top_k > 0 else 5 # Removed as top_k is now passed correctly

        if not hasattr(graphrag_results, 'context_data') or not graphrag_results.context_data:
            log.warning("No context data found in GraphRAG search result.")
            return formatted

        context_records_to_process = []
        if isinstance(graphrag_results.context_data, dict):
            # Handle dict of DataFrames (e.g., from local search)
            for key, df in graphrag_results.context_data.items():
                if isinstance(df, pd.DataFrame):
                    # Add a 'type' field to distinguish context source
                    df_records = df.to_dict(orient='records')
                    for record in df_records:
                        record['context_type'] = key
                    context_records_to_process.extend(df_records)
                else:
                    log.warning(f"Unexpected item type in context_data dict: {type(df)} for key {key}")
        elif isinstance(graphrag_results.context_data, list):
             # Handle list (might contain DataFrames or other types)
             for item in graphrag_results.context_data:
                 if isinstance(item, pd.DataFrame):
                     context_records_to_process.extend(item.to_dict(orient='records'))
                 elif isinstance(item, dict): # Assume it's already a record
                     context_records_to_process.append(item)
                 elif hasattr(item, '__dict__'): # Handle potential data model objects
                     context_records_to_process.append(item.__dict__)
                 else:
                     log.warning(f"Unrecognized item type in context_data list: {type(item)}")
        else:
            log.warning(f"Unrecognized context_data type: {type(graphrag_results.context_data)}")
            return formatted

        # Now format the collected records
        for i, record in enumerate(context_records_to_process):
            if i >= top_k: # Apply top_k limit
                break

            doc = {}
            # Prioritize 'text' or 'content', fall back to description or title
            doc_text = record.get('text', record.get('content', record.get('description', record.get('title', ''))))
            doc['id'] = record.get('id', f"graphrag_context_{i}")
            doc['content'] = doc_text if isinstance(doc_text, str) else str(doc_text) # Ensure text is string
            # Assign score based on rank or position if available
            doc['score'] = record.get('rank', record.get('score', 1.0 - (i / len(context_records_to_process))))
            doc['metadata'] = record # Store the whole record as metadata

            formatted.append(doc)

        return formatted

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the retriever setup.
        """
        # TODO: Return relevant metadata (e.g., cache path, config used)
        return {
            "retriever_type": "graphrag",
            "cache_path": str(self.graphrag_cache_path) if self.graphrag_cache_path else None,
            # Add other relevant config details
        }

    def cleanup(self) -> None:
        """
        Clean up any resources used by the retriever.
        """
        log.info("Cleaning up GraphRAGRetriever resources...")
        # No specific cleanup needed for file-based cache unless we want to delete it.
        # If using in-memory stores or connections, close them here.
        pass

    def get_generated_answer(self) -> Optional[str]:
        """
        Get the last answer generated by GraphRAG during retrieval.
        
        Returns:
            The generated answer string or None if no answer was generated
        """
        return self.last_generated_answer
