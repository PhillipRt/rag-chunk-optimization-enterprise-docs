# core/evaluation.py
import os
# import json # Removed - unused
import logging
import pandas as pd
import numpy as np # Added numpy import
from typing import List, Dict, Any # Removed Optional as it wasn't directly used
from dotenv import load_dotenv
from pydantic.types import SecretStr # Added import

# Import Ragas components carefully to avoid circular dependencies if any
from ragas.metrics.base import Metric, MetricWithEmbeddings, MetricWithLLM, MetricType # Added MetricType
# It's generally better to import specific metrics rather than the whole module
# if you need direct access, but __init__ handles dynamic loading well.
from ragas.metrics import (
    AnswerRelevancy, FactualCorrectness, AnswerCorrectness, AnswerAccuracy,
    Faithfulness, ContextPrecision, ContextUtilization, ContextRecall,
    ContextEntityRecall
)
from ragas.llms import LangchainLLMWrapper # Import necessary wrappers
from ragas.embeddings import LangchainEmbeddingsWrapper # Import necessary wrappers
from ragas.cache import DiskCacheBackend
from ragas import evaluate, EvaluationDataset, SingleTurnSample
# from ragas.utils import set_logging_level # Removed - unused
from ragas.run_config import RunConfig

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class RagEvaluator:
    """Evaluates RAG system using RAGAS metrics."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.

        Args:
            config: Configuration dictionary containing settings potentially across root and 'evaluation' keys.
                    It should include:
                    - evaluation.llm_model: LLM for evaluation (default: deepseek-chat)
                    - evaluation.cache_dir: Directory for RAGAS cache (default: cache/ragas)
                    - evaluation.metrics: List of metrics to use (default: see code below)
                    - GOOGLE_CLOUD_PROJECT_ID, GOOGLE_CLOUD_LOCATION (for Vertex AI Embeddings)
                    - DEEPSEEK_API_KEY (if using deepseek-chat)
        """
        self.config = config # Store the full merged config

        # --- FIX APPLIED HERE: Read evaluation-specific config from 'evaluation' section ---
        evaluation_section = config.get("evaluation", {}) # Get the evaluation section, default to empty dict if missing

        # Read metric names from evaluation_section['metrics'] or use default
        self.metrics_config = evaluation_section.get("metrics", [
            "answer_relevancy",
            "answer_correctness",
            "factual_correctness"
        ])
        # Read LLM model from evaluation_section['llm_model'] or use default
        self.llm_model = evaluation_section.get("llm_model", "deepseek-chat")
        # Read cache directory from evaluation_section['cache_dir'] or use default
        self.cache_dir = evaluation_section.get("cache_dir", "cache/ragas")
        # --- END OF FIX ---

        logger.info(f"RagEvaluator initialized with metrics: {self.metrics_config}")
        logger.info(f"Evaluation LLM model: {self.llm_model}")
        logger.info(f"RAGAS cache directory: {self.cache_dir}")

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Enable RAGAS cache logging to verify cache hits (optional)
        # set_logging_level("ragas.cache", logging.DEBUG) # Removed unused import

        # Will be initialized in setup()
        self.eval_llm = None
        self.embeddings = None
        self.metrics: List[Metric] = [] # Explicitly type hint

    def setup(self) -> None:
        """Set up the evaluator with LLM and metrics."""
        logger.info(f"Setting up evaluator with {self.llm_model} model")

        # Set up LLM for evaluation
        self._setup_llm()

        # Set up embeddings for evaluation (needed for some metrics)
        self._setup_embeddings()

        # Set up metrics
        self._setup_metrics()

    def _setup_llm(self) -> None:
        """Set up the LLM for evaluation."""
        # Import necessary modules only when needed
        logger.info(f"Configuring evaluation LLM: {self.llm_model}")
        # Note: API keys are accessed directly from self.config or os.environ inside the conditions

        if self.llm_model == "deepseek-chat":
            try:
                from langchain_openai.chat_models.base import BaseChatOpenAI
                api_key = self.config.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                     logger.error("DEEPSEEK_API_KEY not found for evaluation LLM.")
                     # Decide how to handle this: raise error or set self.eval_llm to None?
                     # Setting to None might allow some metrics to run, but better to fail early.
                     raise ValueError("DEEPSEEK_API_KEY is required for 'deepseek-chat' evaluation LLM.")
                self.eval_llm = BaseChatOpenAI(
                    model="deepseek-chat",
                    temperature=0.3, # Keep temperature low for eval
                    max_tokens=self.config.get("evaluation", {}).get("max_output_tokens", 8192), # Read from config if exists
                    base_url="https://api.deepseek.com/beta",
                    api_key=SecretStr(api_key) if api_key else None # Pass the key explicitly, wrapped
                )
                logger.info("DeepSeek LLM initialized for evaluation.")
            except ImportError:
                 logger.error("langchain-openai not installed. Cannot use DeepSeek LLM.")
                 raise
            except Exception as e:
                 logger.error(f"Failed to initialize DeepSeek LLM: {e}", exc_info=True)
                 raise

        elif self.llm_model.startswith("gemini"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                # Gemini uses GOOGLE_API_KEY implicitly via genai.configure or GOOGLE_APPLICATION_CREDENTIALS
                # No need to pass api_key explicitly unless specifically required by older versions/setups
                # Ensure ExperimentManager loads GOOGLE_API_KEY/GOOGLE_APPLICATION_CREDENTIALS if needed.
                self.eval_llm = ChatGoogleGenerativeAI(
                    model=self.llm_model,
                    temperature=0.2, # Keep temperature low for eval
                    max_tokens=self.config.get("evaluation", {}).get("max_output_tokens", 8192) # Read from config if exists
                    # safety_settings can be added here if needed
                )
                logger.info(f"Gemini LLM ({self.llm_model}) initialized for evaluation.")
            except ImportError:
                logger.error("langchain-google-genai not installed. Cannot use Gemini LLM.")
                raise
            except Exception as e:
                 logger.error(f"Failed to initialize Gemini LLM ({self.llm_model}): {e}", exc_info=True)
                 raise # Re-raise the exception
        else:
            # If self.llm_model is not recognized, raise an error
            raise ValueError(f"LLM model '{self.llm_model}' not supported for evaluation. Supported models: 'deepseek-chat', 'gemini-*'.")

    def _setup_embeddings(self) -> None:
        """Set up embeddings for evaluation metrics."""
        # Use Google's text-embedding-005 model via Vertex AI
        evaluation_section = self.config.get("evaluation", {})
        embedding_model_name = evaluation_section.get("embedding_model", "text-embedding-005")
        logger.info(f"Attempting to set up embeddings using model: {embedding_model_name}")

        # Embedding setup logic requires GOOGLE_CLOUD_PROJECT_ID and credentials
        project_id = self.config.get("GOOGLE_CLOUD_PROJECT_ID")
        # Prefer GOOGLE_APPLICATION_CREDENTIALS if set, otherwise rely on gcloud auth
        # credentials_path = self.config.get("GOOGLE_APPLICATION_CREDENTIALS") # Removed - unused variable

        if not project_id:
             logger.warning("GOOGLE_CLOUD_PROJECT_ID not found in config. Cannot initialize Vertex AI Embeddings.")
             self.embeddings = None
             return
        # if not credentials_path and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        #      logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set in config or environment. Vertex AI initialization might fail if gcloud auth is not configured.")
             # Continue attempt, vertexai.init() might still work via gcloud auth

        try:
            import vertexai
            from langchain_google_vertexai import VertexAIEmbeddings

            location = evaluation_section.get("GOOGLE_CLOUD_LOCATION", "europe-west1") # Use location from config or default
            logger.info(f"Initializing Vertex AI with Project ID: {project_id}, Location: {location}")
            try:
                # Initialize Vertex AI SDK. It automatically handles credentials from env var or gcloud login.
                vertexai.init(project=project_id, location=location)
                logger.info("Vertex AI initialized successfully.")
            except Exception as e:
                 logger.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)
                 logger.error("Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly OR you are logged in via 'gcloud auth application-default login'.")
                 self.embeddings = None
                 return # Cannot proceed

            # Use the embedding model specified in the config
            self.embeddings = VertexAIEmbeddings(
                model_name=embedding_model_name,
                project=project_id, # Explicitly pass project_id
                location=location, # Explicitly pass location
                # request_parallelism=1 # Adjust parallelism as needed, default is usually fine
            )
            logger.info(f"Using VertexAIEmbeddings model: {embedding_model_name}")

            # # Optional: Test embeddings (can be slow, consider removing for production)
            # try:
            #     test_embed = self.embeddings.embed_query("Test embedding dimensions")
            #     logger.info(f"Embedding dimensions: {len(test_embed)}")
            # except Exception as e:
            #     logger.warning(f"Failed to verify embedding dimensions: {str(e)}")

        except ImportError:
            logger.warning("Google AI vertex libraries (google-cloud-aiplatform) not available. Cannot set up Vertex AI embeddings.")
            self.embeddings = None
        except Exception as e:
             logger.error(f"Unexpected error setting up Vertex AI Embeddings: {e}", exc_info=True)
             self.embeddings = None


    def _setup_metrics(self) -> None:
        """Set up metrics for evaluation."""
        if not self.eval_llm:
            logger.error("Evaluation LLM is not initialized. Cannot set up LLM-dependent metrics.")
            # Depending on requirements, either return here or allow setup of non-LLM metrics.
            # For now, we return as most metrics likely depend on the LLM.
            return

        logger.info("Setting up metrics for evaluation...")

        # Set up RAGAS cache
        # cacher = DiskCacheBackend(cache_dir=self.cache_dir) # Removed - unused variable
        # Note: Consider making caching optional via config

        # Wrap the LLM with caching and run config
        # Use timeout/retries from the root config if available, else defaults
        llm_run_config = RunConfig(
            timeout=self.config.get("timeout", 1800),
            max_retries=self.config.get("max_retries", 10),
            max_workers=self.config.get("max_workers", 48) # Default workers
        )
        cached_llm = LangchainLLMWrapper(
            self.eval_llm,
            # cache=cacher, # Uncomment to enable caching
            run_config=llm_run_config
        )

        # Wrap embeddings with caching and run config if available
        embeddings_wrapper = None
        if self.embeddings:
             # Use potentially different run_config for embeddings if specified
             emb_run_config = RunConfig(
                 timeout=self.config.get("embeddings_timeout", 2400),
                 max_retries=self.config.get("embeddings_max_retries", 10),
                 max_workers=self.config.get("embeddings_max_workers", 8) # Default workers
             )
             embeddings_wrapper = LangchainEmbeddingsWrapper(
                 self.embeddings,
                 # cache=cacher, # Uncomment to enable caching
                 run_config=emb_run_config
             )
             logger.info("Embeddings wrapper created.")
        else:
             logger.warning("Embeddings are not available. Metrics requiring embeddings will be skipped.")


        # Initialize metrics based on self.metrics_config (read in __init__)
        self.metrics = []

        # Define available metrics (initialize instances here for modification)
        # Use actual Ragas class names for clarity
        metric_instances = {
            "answer_relevancy": AnswerRelevancy(),
            "answer_correctness": AnswerCorrectness(),
            "answer_accuracy": AnswerAccuracy(), # This is the Nvidia metric
            "nv_accuracy": AnswerAccuracy(),  # Explicit alias for clarity
            "factual_correctness": FactualCorrectness(),
            "faithfulness": Faithfulness(),
            "context_precision": ContextPrecision(),
            "context_utilization": ContextUtilization(), # Requires embeddings
            "context_recall": ContextRecall(),
            "context_entity_recall": ContextEntityRecall() # Requires LLM
        }

        # Add configured metrics from self.metrics_config
        for metric_name in self.metrics_config:
            logger.info(f"Attempting to initialize metric: {metric_name}")
            if metric_name in metric_instances:
                metric = metric_instances[metric_name] # Get the pre-initialized instance

                # Check requirements (LLM/Embeddings)
                requires_llm = isinstance(metric, MetricWithLLM)
                requires_embeddings = isinstance(metric, MetricWithEmbeddings)

                # Skip if LLM required but not available
                if requires_llm and not self.eval_llm:
                    logger.warning(f"  Metric {metric_name} requires an LLM, but it's not available. Skipping.")
                    continue

                 # Skip if Embeddings required but not available
                if requires_embeddings and not embeddings_wrapper:
                    logger.warning(f"  Metric {metric_name} requires embeddings, but they are not available. Skipping.")
                    continue

                # Assign LLM if required
                if requires_llm:
                    metric.llm = cached_llm
                    logger.info(f"  Assigned LLM to {metric_name}.")
                    # Optional: Prompt modification for Gemini (commented out as reverted in original)
                    # try:
                    #     prompts = metric.get_prompts()
                    #     # ... (modification logic if needed) ...
                    #     # metric.set_prompts(**modified_prompts)
                    #     logger.info(f"    Successfully modified prompts for {metric_name}.")
                    # except AttributeError:
                    #      logger.info(f"    Metric {metric_name} does not support prompt modification.")
                    # except Exception as e_prompt:
                    #      logger.error(f"    Error modifying prompts for {metric_name}: {e_prompt}", exc_info=True)

                # Assign Embeddings if required
                if requires_embeddings:
                     metric.embeddings = embeddings_wrapper
                     logger.info(f"  Assigned embeddings to {metric_name}.")


                # Call the metric's init method explicitly AFTER assigning LLM/Embeddings
                # This ensures dependencies are ready before metric-specific setup
                try:
                    if hasattr(metric, 'init'):
                        logger.info(f"    Calling init() method for {metric_name}...")
                        # Pass the appropriate run config (LLM or Embedding)
                        current_run_config = emb_run_config if requires_embeddings else llm_run_config
                        metric.init(run_config=current_run_config)
                        logger.info(f"    Called init() method for {metric_name}.")
                    else:
                         logger.debug(f"    Metric {metric_name} does not have an init() method.")
                except Exception as e_init:
                     logger.error(f"    Failed during init() method for {metric_name}: {e_init}", exc_info=True)
                     continue # Skip if metric's init fails

                # Add the fully initialized metric to the list
                self.metrics.append(metric)
                # Use getattr for safety in case 'name' attribute isn't present
                metric_display_name = getattr(metric, 'name', metric_name)
                logger.info(f"  Successfully initialized and added metric: {metric.__class__.__name__} (using name: '{metric_display_name}')")

            else:
                logger.warning(f"Metric '{metric_name}' not found in available instances, skipping.")

        if not self.metrics:
             logger.warning("No metrics were successfully initialized for evaluation.")


    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the dataset using RAGAS metrics.

        Args:
            dataset: List of dictionaries, typically the 'augmented_data' part from QA phase.
                     Each dict should have keys needed by the configured metrics
                     (e.g., 'user_input', 'response', 'retrieved_docs', 'reference').

        Returns:
            Dictionary with evaluation results, including average scores and raw results.
            Returns {"error": "No metrics initialized"} if setup failed.
            Returns {"error": "Evaluation failed: [exception message]"} on Ragas evaluate() error.
        """
        if not self.metrics:
             logger.error("No metrics initialized. Cannot run evaluation.")
             # Return a dictionary indicating the error, consistent with other error returns
             return {"error": "No metrics initialized", "average_scores": {}, "results": pd.DataFrame()}


        logger.info(f"Starting evaluation with {len(self.metrics)} metrics on {len(dataset)} samples...")

        # Convert dataset format if needed (assuming input is list of dicts)
        # Ragas evaluate expects a Dataset object or specific dictionary format
        samples = []
        required_keys = set()
        for m in self.metrics:
             # Assuming single turn for now based on original code structure
             required_keys.update(m.get_required_columns(with_optional=True).get(MetricType.SINGLE_TURN.name, set())) # Added MetricType import

        logger.debug(f"Required keys for metrics: {required_keys}")

        for i, item in enumerate(dataset):
            # Map standard keys, handle missing keys gracefully if metrics allow optional
            sample_data = {
                'user_input': item.get('user_input', item.get('question')), # Allow both keys
                'response': item.get('response', item.get('answer')), # Allow both keys
                'retrieved_contexts': item.get('retrieved_docs_content'), # Use pre-extracted content
                'reference': item.get('reference', item.get('ground_truth')) # Allow both keys
            }
            # Filter out None values only if the key is truly optional for all used metrics
            # For simplicity, we pass all keys and let Ragas handle validation per metric
            # This might be slightly less efficient but avoids complex logic here.
            filtered_sample_data = {k: v for k, v in sample_data.items() if v is not None}

            # Warn if a required key (non-optional) is missing
            # missing_required = required_keys - set(filtered_sample_data.keys()) - {'retrieved_contexts'} # Removed - unused variable

            if 'retrieved_docs_content' in filtered_sample_data: # If we have content, satisfy the need for contexts
                filtered_sample_data['retrieved_contexts'] = filtered_sample_data['retrieved_docs_content']
            elif 'retrieved_contexts' not in required_keys: # if contexts not required, don't worry
                 pass
            elif 'retrieved_contexts' in required_keys and 'retrieved_contexts' not in filtered_sample_data :
                 logger.warning(f"Sample {i+1} is missing 'retrieved_contexts'/'retrieved_docs_content', which might be required by some metrics.")


            # Check required keys specifically (stricter check)
            for key in required_keys:
                 if key not in filtered_sample_data and not key.endswith(':optional'):
                      # Check if an alternative key exists (e.g., question for user_input)
                      alternative_exists = False
                      # --- Refactored if blocks for clarity ---
                      if key == 'user_input' and 'question' in item:
                            alternative_exists = True
                      if key == 'response' and 'answer' in item:
                            alternative_exists = True
                      if key == 'reference' and 'ground_truth' in item:
                            alternative_exists = True
                      if key == 'retrieved_contexts' and 'retrieved_docs_content' in item:
                           alternative_exists = True
                      # --- End Refactor ---

                      if not alternative_exists:
                           logger.error(f"Sample {i+1} is missing required key '{key}' for evaluation.")
                           # Option: Skip sample or raise error? Returning error for now.
                           return {"error": f"Sample {i+1} missing required key '{key}'", "average_scores": {}, "results": pd.DataFrame()}

            samples.append(SingleTurnSample(**filtered_sample_data))


        # Create RAGAS evaluation dataset from samples
        try:
             evaluation_dataset = EvaluationDataset(samples)
             logger.info(f"Created RAGAS evaluation dataset with {len(evaluation_dataset)} sample(s)")
        except Exception as e:
             logger.error(f"Failed to create RAGAS EvaluationDataset: {e}", exc_info=True)
             return {"error": f"Failed to create RAGAS dataset: {e}", "average_scores": {}, "results": pd.DataFrame()}


        # Get the wrapped LLM and Embeddings for the evaluate function
        # These should already be configured with caching/run_config in _setup_metrics
        eval_llm_for_ragas = self.metrics[0].llm if self.metrics and hasattr(self.metrics[0], 'llm') else None
        eval_embeddings_for_ragas = self.metrics[0].embeddings if self.metrics and hasattr(self.metrics[0], 'embeddings') else None


        # Use the same run_config for the overall evaluate call
        evaluate_run_config = RunConfig(
            timeout=self.config.get("timeout", 1800),
            max_retries=self.config.get("max_retries", 10),
            max_workers=self.config.get("max_workers", 48) # Use max_workers from root config for overall coordination
        )

        try:
            self.evaluation_result = evaluate(
                dataset=evaluation_dataset,
                metrics=self.metrics,
                llm=eval_llm_for_ragas, # Pass the wrapped LLM
                embeddings=eval_embeddings_for_ragas, # Pass the wrapped embeddings
                run_config=evaluate_run_config,
                raise_exceptions=False # Prevent evaluate from stopping on single sample errors
            )
            logger.info("RAGAS evaluation completed.")
            # logger.debug(f"Raw RAGAS result object: {self.evaluation_result}")

            # Automatically upload results to app.ragas.io if API token is provided
            dashboard_url = None
            ragas_app_token = self.config.get("RAGAS_APP_TOKEN") or os.environ.get("RAGAS_APP_TOKEN")
            if ragas_app_token:
                logger.info("RAGAS_APP_TOKEN found, attempting to upload results...")
                try:
                    # Simple upload attempt
                    dashboard_url = self.evaluation_result.upload(verbose=False)
                    logger.info(f"📊 Results uploaded to RAGAS dashboard: {dashboard_url}")
                except Exception as e:
                    logger.warning(f"Could not upload to RAGAS dashboard (continuing anyway): {str(e)}")
                    # logger.debug("Ragas upload error details:", exc_info=True) # Optional detailed logging
            else:
                logger.info("RAGAS_APP_TOKEN not found, skipping dashboard upload.")

            # Convert to DataFrame for processing
            if hasattr(self.evaluation_result, 'to_pandas'):
                results_df = self.evaluation_result.to_pandas()
            else:
                 logger.error("Ragas evaluation result does not have 'to_pandas' method.")
                 # Handle cases where the result might be unexpected (e.g., None or dict)
                 if isinstance(self.evaluation_result, pd.DataFrame):
                     results_df = self.evaluation_result
                 elif isinstance(self.evaluation_result, dict):
                      # Attempt to create DataFrame from dict if possible, might need specific keys
                      try:
                           results_df = pd.DataFrame(self.evaluation_result)
                      except Exception:
                           results_df = pd.DataFrame() # Fallback to empty DF
                 else:
                     results_df = pd.DataFrame() # Fallback to empty DF


            logger.info(f"Results DataFrame shape: {results_df.shape}")
            # logger.debug(f"Results DataFrame head:\n{results_df.head()}")

            # Process results
            processed_results = self._process_results(results_df, dataset) # Pass original list of dicts

            # Add dashboard URL to results if available
            if dashboard_url:
                processed_results["dashboard_url"] = dashboard_url

            return processed_results

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            return {"error": f"Evaluation failed: {e}", "average_scores": {}, "results": pd.DataFrame()}


    def _process_results(self,
                        results_df: pd.DataFrame,
                        original_data: List[Dict[str, Any]] # Keep original data for reference if needed
                        ) -> Dict[str, Any]:
        """
        Process the results DataFrame into a structured output.

        Args:
            results_df: The raw DataFrame with evaluation results from Ragas evaluate().
            original_data: The original list of dataset dictionaries that were evaluated.

        Returns:
            Dictionary with structured evaluation results including raw df, average scores, and detailed per-sample scores.
        """
        processed_results: Dict[str, Any] = {
            "results": results_df,          # Include the raw results DataFrame
            "metrics": {},                  # Store average metric scores here too
            "average_scores": {},           # Main place for average scores
            "detailed_results": []          # Store per-sample results
        }

        # --- Calculate metric averages ---
        # Identify numeric columns that are likely metric scores
        metric_columns = [
            col for col in results_df.columns
            if pd.api.types.is_numeric_dtype(results_df[col]) and
            col not in ['user_input', 'response', 'retrieved_contexts', 'reference'] # Exclude standard input/output columns
        ]
        logger.info(f"Identified metric columns for averaging: {metric_columns}")

        for col in metric_columns:
            try:
                # Use numpy's nanmean to handle potential NaN scores gracefully
                average_score = np.nanmean(results_df[col]) # Used np
                if not np.isnan(average_score): # Used np
                     processed_results["average_scores"][col] = float(average_score)
                     processed_results["metrics"][col] = {
                         "name": col,
                         "score": float(average_score)
                     }
                else:
                    logger.warning(f"Could not calculate a valid mean for column '{col}' (all values might be NaN).")
                    processed_results["average_scores"][col] = None # Indicate inability to calculate
                    processed_results["metrics"][col] = {
                        "name": col,
                        "score": None
                     }
            except Exception as e:
                logger.warning(f"Could not calculate mean for column '{col}': {str(e)}")
                processed_results["average_scores"][col] = None # Indicate error
                processed_results["metrics"][col] = {
                    "name": col,
                    "score": None
                 }

        # --- Process individual sample results ---
        # Iterate through the DataFrame rows to create detailed results list
        for i, row in results_df.iterrows():
            # Get corresponding original data if needed (e.g., for richer context)
            # original_sample = original_data[i] if i < len(original_data) else {}

            sample_result = {
                # Include essential inputs for reference
                "input": row.get("user_input", ""), # Get from DF column if exists
                # Include original keys if needed:
                # "original_question": original_sample.get('question', original_sample.get('user_input')),
                # "original_answer": original_sample.get('answer', original_sample.get('response')),
                # Include scores for this sample
                "scores": {}
            }

            # Add scores for each identified metric column
            for col in metric_columns:
                 score_value = row.get(col)
                 # Convert numpy types (like float64) to standard Python float/int
                 # Handle potential NaNs gracefully
                 if pd.isna(score_value):
                      sample_result["scores"][col] = None
                 elif isinstance(score_value, (np.integer, np.int64)): # Used np
                      sample_result["scores"][col] = int(score_value)
                 elif isinstance(score_value, (np.floating, np.float64)): # Used np
                      sample_result["scores"][col] = float(score_value)
                 else:
                      # Try converting to float, handle errors
                      try:
                           sample_result["scores"][col] = float(score_value)
                      except (ValueError, TypeError):
                           logger.warning(f"Could not convert score '{score_value}' for metric '{col}' in sample {i} to float. Storing as is.")
                           sample_result["scores"][col] = score_value # Store original if conversion fails

            processed_results["detailed_results"].append(sample_result)

        return processed_results

    @staticmethod
    def get_available_metrics() -> Dict[str, str]:
        """
        Get information about all available Ragas metrics.

        Returns:
            Dictionary mapping metric names to descriptions
        """
        # Descriptions adapted from Ragas documentation/source code
        return {
            "answer_relevancy": "Measures how relevant and to-the-point the answer is to the given question by generating questions from the answer and comparing them to the original.",
            "answer_correctness": "Evaluates the accuracy of the answer compared to the reference using a weighted combination of factual similarity (TP/FP/FN) and semantic similarity.",
            "answer_accuracy": "(Nvidia) Evaluates the accuracy of the answer compared to the reference using LLM judgments on a scale (e.g., 0-Not accurate, 1-Partially accurate, 2-Accurate). Uses dual LLM judges.", # Changed judgements to judgments
            "nv_accuracy": "(Nvidia) Alias for answer_accuracy.",
            "factual_correctness": "Checks for factual errors by decomposing the answer into claims and validating them against the reference, calculating precision, recall, or F1 score based on verified claims.",
            "faithfulness": "Measures how factually consistent the answer is with the provided context by verifying if claims in the answer can be inferred from the context.",
            "context_precision": "Evaluates whether the most relevant items in the context appear at the top ranks using average precision scoring.",
            "context_utilization": "Measures how well the answer utilizes the retrieved context by assessing which context chunks contributed to the answer.",
            "context_recall": "Assesses if the retrieved context contains the key information needed to answer the question compared to the reference.",
            "context_entity_recall": "Evaluates if the context contains the specific entities needed for the answer by comparing entities in the context with those in the reference.",
            "nv_context_relevance": "(Nvidia) Evaluates whether the retrieved contexts are pertinent to the user input using dual LLM judges.",
            "nv_response_groundedness": "(Nvidia) Measures how well the response is supported by the retrieved contexts using dual LLM judges.",
            # Add other metrics as needed
        }