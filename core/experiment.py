import os
import json
import yaml
import logging
import datetime
from typing import List, Dict, Any, Optional, Union # Added Union
import copy
import re # Added for env var substitution
import glob # Added for compare_experiments globbing
import traceback # <-- ADDED IMPORT
import pandas as pd # <-- ADDED IMPORT

from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Use relative imports for modules within the same package structure
from retrievers.base_retriever import BaseRetriever
from retrievers.embedding_retriever import EmbeddingRetriever
from retrievers.multihop_retriever import MultiHopRetriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.knowledge_graph_retriever import KnowledgeGraphRetriever # Old implementation
from retrievers.property_graph_retriever import PropertyGraphRetriever # New implementation
from retrievers.gemini_retriever import GeminiRetriever
from retrievers.tfidf_retriever import TFIDFRetriever
from retrievers.random_retriever import RandomRetriever
from retrievers.adversarial_retriever import AdversarialRetriever
from retrievers.graphrag_retriever import GraphRAGRetriever
from retrievers.bm25_retriever import BM25Retriever
from .data_manager import DataManager
from .evaluation import RagEvaluator

logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    Manages RAG experiments from configuration to evaluation.
    """

    def __init__(self, base_config_path: str = "config/base_config.yaml"):
        """
        Initialize the experiment manager. Raises ValueError on critical init failures.
        """
        # API keys are loaded first, as they might be needed for config substitution
        self.api_keys = self._load_and_validate_api_keys()

        self.base_config = self._load_config(base_config_path)
        if not self.base_config:
            logger.error("Base configuration could not be loaded. Exiting.")
            raise ValueError("Failed to load base configuration.")

        data_config = self.base_config.get("data", {})
        if not data_config:
             logger.warning("No 'data' section found in base config, DataManager might use defaults.")
             data_config = {} # Provide empty dict to prevent potential errors
        try:
            # Make sure DataManager is initialized correctly
            self.data_manager = DataManager(data_config)
        except ValueError as e:
            logger.error(f"Failed to initialize DataManager: {e}", exc_info=True)
            raise # Propagate critical init error
        except KeyError as e:
             logger.error(f"DataManager config missing required key: {e}. Check base_config.yaml.", exc_info=True)
             raise ValueError(f"DataManager config missing required key: {e}")


    def _load_and_validate_api_keys(self) -> Dict[str, Optional[str]]:
        """Load API keys from environment variables. Raises ValueError if a required key is missing."""
        # --- No changes needed here, keeping original logic ---
        keys_to_load_spec = {
            "GOOGLE_API_KEY": False,
            "DEEPSEEK_API_KEY": False,
            "COHERE_API_KEY": False,
            "EMBEDDING_API_URL": False,
            "EMBEDDING_API_TOKEN": False,
            "GOOGLE_CLOUD_PROJECT_ID": False,
            "GOOGLE_APPLICATION_CREDENTIALS": False,
            "RAGAS_APP_TOKEN": False
        }
        loaded_keys = {}
        logger.info("Loading API keys from environment variables...")
        missing_required_keys = []

        for key, required in keys_to_load_spec.items():
            value = os.environ.get(key)
            if value:
                loaded_keys[key] = value
            else:
                loaded_keys[key] = None

        google_creds_set = bool(loaded_keys.get("GOOGLE_APPLICATION_CREDENTIALS"))

        for key, required in keys_to_load_spec.items():
            value = loaded_keys[key]
            is_missing = value is None

            if is_missing and required:
                if key == "GOOGLE_API_KEY" and google_creds_set:
                    logger.info(f"  Optional {key} not set, but GOOGLE_APPLICATION_CREDENTIALS is. Assuming Vertex AI auth.")
                else:
                    logger.error(f"  Missing REQUIRED environment variable: {key}")
                    missing_required_keys.append(key)
            elif not is_missing:
                masked_value = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
                logger.info(f"  Found {key}: {masked_value} {'(Required)' if required else '(Optional)'}")
            else:
                 logger.warning(f"  Optional environment variable not set: {key}")

        if missing_required_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_required_keys)}. Please set them to proceed.")

        return loaded_keys


    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file, substituting ${ENV_VAR}."""
        # --- No changes needed here, keeping original logic ---
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()

            def env_var_substituter(match):
                var_name = match.group(1)
                var_value = self.api_keys.get(var_name, os.environ.get(var_name))
                if var_value is None:
                    logger.warning(f"Environment variable '{var_name}' not found for substitution in {config_path}. Replacing with empty string.")
                    return ""
                return var_value

            processed_config_content = re.sub(r'\$\{([^}]+)\}', env_var_substituter, config_content)
            config = yaml.safe_load(processed_config_content)

            if config is None:
                logger.warning(f"Config file {config_path} is empty or invalid YAML. Returning empty dict.")
                return {}
            if not isinstance(config, dict):
                logger.error(f"Config file {config_path} did not parse into a dictionary. Content: {config_content[:200]}...")
                return {}
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML from {config_path}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred loading config from {config_path}: {str(e)}", exc_info=True)
            return {}


    def _create_retriever(self, retriever_sub_config: Dict[str, Any]) -> BaseRetriever:
        """Create a retriever instance based on configuration."""
        # --- No changes needed here, keeping original logic ---
        retriever_type = retriever_sub_config.get("type", "embedding")
        logger.info(f"Creating retriever of type: {retriever_type}")

        for key, value in self.api_keys.items():
             if value is not None and key not in retriever_sub_config:
                  retriever_sub_config[key] = value

        try:
            if retriever_type == "embedding":
                return EmbeddingRetriever(retriever_sub_config)
            elif retriever_type == "multihop":
                return MultiHopRetriever(retriever_sub_config)
            elif retriever_type == "hybrid":
                return HybridRetriever(retriever_sub_config)
            elif retriever_type == "graph" or retriever_type == "knowledge_graph":
                logger.warning("Using deprecated 'graph'/'knowledge_graph' (KnowledgeGraphRetriever). Consider 'property_graph'.")
                return KnowledgeGraphRetriever(retriever_sub_config)
            elif retriever_type == "property_graph":
                logger.info("Using PropertyGraphRetriever.")
                return PropertyGraphRetriever(retriever_sub_config)
            elif retriever_type == "gemini":
                return GeminiRetriever(retriever_sub_config)
            elif retriever_type == "tfidf":
                return TFIDFRetriever(retriever_sub_config)
            elif retriever_type == "bm25":
                logger.info("Using pure BM25 retriever.")
                return BM25Retriever(retriever_sub_config)
            elif retriever_type == "random":
                return RandomRetriever(retriever_sub_config)
            elif retriever_type == "adversarial":
                return AdversarialRetriever(retriever_sub_config)
            elif retriever_type == "graphrag":
                logger.info("Using GraphRAG retriever.")
                return GraphRAGRetriever(retriever_sub_config)
            else:
                raise ValueError(f"Unsupported retriever type: {retriever_type}")
        except Exception as e:
             logger.error(f"Failed to create retriever of type '{retriever_type}': {e}", exc_info=True)
             raise


    def run_qa_only(self,
                   config_path: str,
                   test_data_path: Optional[str] = None,
                   limit: int = 0,
                   cached_docs_path: Optional[str] = None
                   ) -> Optional[Dict[str, Any]]:
        """
        Run only the QA part. Returns None if a critical error occurs.
        """
        # --- No changes needed here, keeping original logic ---
        experiment_config = self._load_config(config_path)
        if not experiment_config: # Check if config loading failed
            logger.error(f"Failed to load experiment config from {config_path}. QA phase aborted.")
            return None

        config = copy.deepcopy(self.base_config) # Start with a copy of base
        for key, value in experiment_config.items():
             if key in config and isinstance(config.get(key), dict) and isinstance(value, dict):
                  config[key].update(value) # Use update for merging sub-dictionaries
             else:
                  config[key] = value # Override or add new keys

        for key, value in self.api_keys.items():
            if value is not None and key not in config:
                config[key] = value

        experiment_name = config.get("name", os.path.basename(config_path).split('.')[0])
        experiment_id = f"{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting QA phase for experiment: {experiment_name} (ID: {experiment_id})")


        try:
            documents = self.data_manager.load_documents(cached_path=cached_docs_path)
            if not documents:
                logger.error(f"No documents loaded for experiment {experiment_name}. QA phase cannot proceed.")
                return None
            logger.info(f"Loaded {len(documents)} documents")

            test_data = self.data_manager.load_test_data(data_path=test_data_path, limit=limit)
            if not test_data:
                logger.error(f"No test data loaded for experiment {experiment_name}. QA phase cannot proceed.")
                return None
            logger.info(f"Loaded {len(test_data)} test samples")

            retriever_config = config.get("retriever", {})
            retriever = self._create_retriever(retriever_config)
            retriever.setup(documents) # Pass the loaded documents

            # --- Optional: HTML export for PropertyGraphRetriever ---
            if isinstance(retriever, PropertyGraphRetriever):
                try:
                    export_dir = os.path.join(self.data_manager.evaluation_dir, "graph_exports")
                    os.makedirs(export_dir, exist_ok=True)
                    export_filename = os.path.join(export_dir, f"property_graph_export_{experiment_name}_{experiment_id}.html")
                    logger.info(f"Exporting property graph to {export_filename}...")
                    retriever.export_graph_html(output_path=export_filename)
                except Exception as export_err:
                    logger.error(f"Failed to export property graph HTML: {export_err}", exc_info=True)
            # --- End HTML export ---

            generation_specific_config = config.get("generation", {})
            augmented_data = self._generate_responses(
                retriever=retriever,
                dataset=test_data,
                generation_config=generation_specific_config
            )

            qa_result_output = {
                "augmented_data": augmented_data,
                "experiment": {
                    "id": experiment_id,
                    "name": experiment_name,
                    "config_path": config_path,
                    "merged_config": config,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            }

            self.data_manager.save_results(
                results=qa_result_output,
                filename="augmented_dataset",
                approach_name=experiment_name
            )

            return qa_result_output

        except Exception as e:
            logger.error(f"Critical error during QA phase for {experiment_name}: {e}", exc_info=True)
            return None


    def run_evaluation_only(self,
                          qa_result: Dict[str, Any],
                          config_path: str
                          ) -> Optional[Dict[str, Any]]:
        """
        Run only the evaluation part. Handles potential NaNs and errors from Ragas evaluate.

        Args:
            qa_result: The dictionary containing QA results (augmented_data, experiment metadata).
                       Expected structure: {"augmented_data": List[Dict], "experiment": Dict}
            config_path: Path to the experiment config (used for evaluation settings and fallback name).

        Returns:
            A dictionary with evaluation results, or None if evaluation fails critically.
        """
        # --- **MODIFIED TO USE THE CORRECT QA RESULT STRUCTURE** ---
        augmented_data = qa_result.get("augmented_data", [])
        experiment_metadata = qa_result.get("experiment", {})
        experiment_name = experiment_metadata.get("name", os.path.basename(config_path).split('.')[0])
        experiment_id = experiment_metadata.get("id", f"{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        # --- **END MODIFICATION** ---

        if not augmented_data:
            logger.error(f"No augmented data found for experiment {experiment_name} (ID: {experiment_id}). Evaluation aborted.")
            # Optionally save an error result here if needed
            return None

        logger.info(f"Starting evaluation phase for experiment: {experiment_name} (ID: {experiment_id})")

        # --- Get the merged config (using logic from original code) ---
        merged_config = experiment_metadata.get("merged_config")
        if not merged_config or not isinstance(merged_config, dict):
            logger.warning(f"Merged config not found or invalid in QA result for {experiment_name}. Re-loading from config path.")
            experiment_specific_config = self._load_config(config_path)
            if not experiment_specific_config:
                logger.error(f"Failed to load experiment config from {config_path} for evaluation. Aborting.")
                # Save error marker result
                error_results = {"error": f"Failed to load config {config_path}", "experiment": experiment_metadata}
                self.data_manager.save_results(results=error_results, filename="evaluation_results_ERROR", approach_name=experiment_name)
                return None
            merged_config = copy.deepcopy(self.base_config)
            for key, value in experiment_specific_config.items():
                if key in merged_config and isinstance(merged_config.get(key), dict) and isinstance(value, dict):
                    # Safely merge dictionaries
                    merged_config[key].update(value) # Use update for merging sub-dictionaries
                else:
                    merged_config[key] = value
            # Re-inject API keys if not substituted
            for key, value in self.api_keys.items():
                if value is not None and key not in merged_config:
                    merged_config[key] = value
            # Store the re-merged config back into metadata for consistency
            experiment_metadata["merged_config"] = merged_config
        # --- End Get the merged config ---

        # Add evaluation timestamp to metadata NOW, before potential errors
        experiment_metadata["evaluation_timestamp"] = datetime.datetime.now().isoformat()

        # --- **RECOMMENDED ERROR HANDLING LOGIC** ---
        results_df = None # Initialize results_df to handle potential early exit
        try:
            # 1. Initialize and Setup Evaluator
            evaluator = RagEvaluator(merged_config)
            evaluator.setup() # Can raise exceptions (e.g., model auth)

            # 2. Run Evaluation
            # This call returns the PROCESSED dictionary from RagEvaluator.evaluate (which calls _process_results)
            evaluation_output_dict = evaluator.evaluate(augmented_data)

            # 3. Check for Evaluation Failures / Issues
            evaluation_failed = False
            critical_failure = False # Differentiate between partial and total failure
            error_message = "Evaluation completed with potential issues." # Default

            if not evaluation_output_dict: # Case 1: Evaluator returned None or empty
                evaluation_failed = True
                critical_failure = True
                error_message = "Evaluator returned no result object."
            elif isinstance(evaluation_output_dict, dict) and "error" in evaluation_output_dict: # Case 2: Evaluator returned a dict with an "error" key
                evaluation_failed = True
                critical_failure = True # Assume errors from evaluator are critical
                error_message = f"Evaluation failed: {evaluation_output_dict['error']}"
            elif isinstance(evaluation_output_dict, dict) and "results" in evaluation_output_dict and "average_scores" in evaluation_output_dict:
                # Case 3: Successfully processed dictionary from evaluator
                evaluation_failed = False # Assume success for now, further checks below
                error_message = "Evaluation completed successfully." # Default for this path

                results_list_or_df = evaluation_output_dict.get("results") # Get the results, could be DF or list
                avg_scores = evaluation_output_dict.get("average_scores", {}) # Should exist

                # Check if the results (which should be a DataFrame from _process_results) is empty
                if isinstance(results_list_or_df, pd.DataFrame) and results_list_or_df.empty:
                    if evaluator.metrics_config: # Only a critical failure if metrics were expected
                        evaluation_failed = True
                        critical_failure = True
                        error_message = "Evaluation ran but produced no detailed scores (empty 'results' DataFrame)."
                    else:
                        logger.info(f"No metrics were configured for {experiment_name}, so empty 'results' DataFrame is expected.")
                        error_message = "Evaluation completed (no metrics configured)."
                elif not isinstance(results_list_or_df, pd.DataFrame) and not results_list_or_df: # If it was a list and empty
                    if evaluator.metrics_config:
                        evaluation_failed = True
                        critical_failure = True
                        error_message = "Evaluation ran but produced no detailed scores (empty 'results' list)."
                    else:
                        logger.info(f"No metrics were configured for {experiment_name}, so empty 'results' list is expected.")
                        error_message = "Evaluation completed (no metrics configured)."
                else:
                    # If results_list_or_df is a non-empty DataFrame or a non-empty list, proceed to NaN checks
                    # Check for widespread NaNs using average_scores, if they exist.
                    # This logic attempts to replicate the spirit of the original NaN check on results_df.
                    if not avg_scores:
                        if evaluator.metrics_config: # If metrics were run, avg_scores should ideally exist
                            logger.warning(f"Average scores for {experiment_name} are empty, though detailed results exist. Check processing.")
                            # Not necessarily a critical failure if detailed results are there.
                            error_message = "Evaluation completed, but average_scores dictionary is unexpectedly empty."
                    else:
                        # Check if all configured metrics that appear in average_scores are NaN
                        # Get base names of configured metrics
                        configured_base_metrics = {m.split("(")[0] for m in evaluator.metrics_config}
                        
                        avg_scores_for_configured_metrics = {
                            k: v for k, v in avg_scores.items() 
                            if k.split("(")[0] in configured_base_metrics
                        }

                        if configured_base_metrics and not avg_scores_for_configured_metrics:
                             # Configured metrics, but none of them are in average_scores
                             evaluation_failed = True
                             critical_failure = True
                             error_message = "Evaluation completed, but no configured metrics found in average_scores."
                        elif avg_scores_for_configured_metrics and all(pd.isna(score) for score in avg_scores_for_configured_metrics.values()):
                            evaluation_failed = True
                            critical_failure = True # Widespread failure
                            error_message = "Evaluation completed, but all configured metric average scores are NaN."
                        elif any(pd.isna(score) for score in avg_scores.values()): # Check all avg_scores for any NaN
                            logger.warning(f"Evaluation for {experiment_name} completed but average_scores contains some NaN values.")
                            error_message = "Evaluation completed with some NaN average scores."
                            # evaluation_failed remains False

            else: # Case 4: Unexpected structure from evaluator's output
                evaluation_failed = True
                critical_failure = True
                error_message = f"Evaluation returned an unexpected object type or structure: {type(evaluation_output_dict)}. Content: {str(evaluation_output_dict)[:200]}"
            # --- **END MODIFIED LOGIC FOR CHECKING evaluation_output_dict** ---


            # 4. Handle Failure or Success
            if evaluation_failed:
                logger.error(f"Evaluation phase failed for {experiment_name}: {error_message}")
                # Prepare results structure for saving, even on failure
                output_results = {
                    "error": error_message,
                    "experiment": experiment_metadata,
                    # Include partial scores if available from the dict
                    "results": evaluation_output_dict.get("results", []) if isinstance(evaluation_output_dict, dict) else [],
                    "average_scores": evaluation_output_dict.get("average_scores", {}) if isinstance(evaluation_output_dict, dict) else {}
                }
                # Use a specific filename for clear failures
                filename = "evaluation_results_FAILED" if critical_failure else "evaluation_results_PARTIAL_FAIL"
                self.data_manager.save_results(
                    results=output_results,
                    filename=filename,
                    approach_name=experiment_name
                )
                return None # Indicate failure

            else:
                # Evaluation succeeded (potentially with warnings about NaNs)
                logger.info(f"Evaluation phase completed for {experiment_name}. Status: {error_message}")
                
                # evaluation_output_dict IS the processed_results dictionary
                # Ensure 'experiment' metadata is included
                processed_results_with_meta = {**evaluation_output_dict, "experiment": experiment_metadata}

                self.data_manager.save_results(
                    results=processed_results_with_meta, 
                    filename="evaluation_results", # Standard success filename
                    approach_name=experiment_name
                )
                return processed_results_with_meta # Return successful, processed results

        except Exception as e:
            # Catch setup errors or other critical exceptions
            logger.error(f"Critical error during evaluation phase setup or execution for {experiment_name}: {e}", exc_info=True)
            error_results = {"error": str(e), "traceback": traceback.format_exc(), "experiment": experiment_metadata}
            self.data_manager.save_results(
                results=error_results,
                filename="evaluation_results_CRITICAL_ERROR",
                approach_name=experiment_name
            )
            return None # Indicate critical failure
        # --- **END RECOMMENDED ERROR HANDLING LOGIC** ---


    def run_experiment(self,
                     config_path: str,
                     test_data_path: Optional[str] = None,
                     limit: int = 0,
                     cached_docs_path: Optional[str] = None
                     ) -> Optional[Dict[str, Any]]:
        """
        Run a single experiment (QA and Evaluation). Returns None if a critical phase fails.
        """
        # --- No changes needed here, keeping original logic ---
        qa_result = self.run_qa_only(
            config_path=config_path,
            test_data_path=test_data_path,
            limit=limit,
            cached_docs_path=cached_docs_path
        )

        if qa_result is None:
            logger.error(f"QA phase failed for config {config_path}. Experiment aborted.")
            return None

        return self.run_evaluation_only(
            qa_result=qa_result,
            config_path=config_path
        )

    def _generate_responses(self,
                          retriever: BaseRetriever,
                          dataset: List[Dict[str, Any]],
                          generation_config: Dict[str, Any] # This is already resolved config['generation']
                          ) -> List[Dict[str, Any]]:
        """
        Generate responses for each sample using the retriever and an LLM.
        """
        # --- No changes needed here, keeping original logic ---
        dataset_copy = copy.deepcopy(dataset)

        model_name = generation_config.get("model", "gemini-1.5-flash") # Default if not in config
        temperature = generation_config.get("temperature", 0.7)
        google_api_key_for_gen = generation_config.get("GOOGLE_API_KEY")

        llm = None # Initialize llm as None
        if not google_api_key_for_gen and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.error("GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS not found for generation LLM. Cannot generate responses.")
            for sample in dataset_copy:
                sample['response'] = "Error: Generation LLM not configured due to missing API key/credentials."
                sample['retrieved_docs'] = []
                sample['retrieved_docs_content'] = []
            return dataset_copy

        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_tokens=generation_config.get("max_output_tokens", 8192),
            )
            logger.info(f"Generation LLM ({model_name}) initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize generation LLM ({model_name}): {e}", exc_info=True)
            for sample in dataset_copy:
                sample['response'] = f"Error: Could not initialize generation LLM: {e}"
                sample['retrieved_docs'] = []
                sample['retrieved_docs_content'] = []
            return dataset_copy

        prompt_template_str = generation_config.get("prompt_template",
            "Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        logger.info(f"Generating responses for {len(dataset_copy)} samples using LLM: {model_name}...")

        for i, sample in enumerate(dataset_copy):
            question = sample.get('user_input', sample.get('question', ''))
            if not question:
                logger.warning(f"Sample {i+1} has no 'user_input' or 'question'. Skipping response generation.")
                sample['response'] = "Error: No question provided in sample."
                sample['retrieved_docs'] = []
                sample['retrieved_docs_content'] = []
                continue

            logger.info(f"Processing sample {i+1}/{len(dataset_copy)}: {question[:50]}...")

            try:
                retrieved_docs = retriever.retrieve(question, top_k=generation_config.get("retrieval_top_k", 5))
                sample['retrieved_docs'] = retrieved_docs
                sample['retrieved_docs_content'] = [doc.get('content', '') for doc in retrieved_docs]

                generated_answer_by_retriever = None
                if hasattr(retriever, 'generates_answer') and retriever.generates_answer:
                    generated_answer_by_retriever = retriever.get_generated_answer()

                if generated_answer_by_retriever:
                    logger.info(f"Using answer generated directly by retriever {retriever.__class__.__name__}")
                    response_text = generated_answer_by_retriever
                else:
                    if hasattr(retriever, 'generates_answer') and retriever.generates_answer:
                         logger.info(f"Retriever {retriever.__class__.__name__} indicated answer generation but returned None. Falling back to LLM.")

                    context_str = self._format_context_from_docs(retrieved_docs)
                    prompt_filled = prompt_template_str.format(context=context_str, question=question)

                    llm_response = llm.invoke([HumanMessage(content=prompt_filled)])
                    response_text = llm_response.content

                sample['response'] = response_text

            except Exception as e:
                logger.error(f"Error processing sample {i+1} ('{question[:50]}...'): {str(e)}", exc_info=True)
                sample['response'] = f"Error generating response: {str(e)}"
                if 'retrieved_docs' not in sample: sample['retrieved_docs'] = []
                if 'retrieved_docs_content' not in sample: sample['retrieved_docs_content'] = []


        logger.info("Finished generating responses for all samples.")
        return dataset_copy

    def _format_context_from_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Formats context string from retrieved documents."""
        # --- No changes needed here, keeping original logic ---
        if not docs:
            return "No context provided."
        formatted_docs = []
        for j, doc in enumerate(docs):
            content = doc.get('content', doc.get('text', ''))
            metadata_str = f" (Source: {doc.get('metadata', {}).get('source', 'N/A')})" if doc.get('metadata') else ""
            formatted_docs.append(f"Document {j+1}{metadata_str}:\n{content}")
        return "\n\n".join(formatted_docs)


    def compare_experiments(self, experiment_ids_or_names: List[str]) -> Optional[Dict[str, Any]]:
        """Compare results from multiple experiments, finding files by ID or name."""
        # --- No changes needed here, keeping original logic ---
        experiment_results_data = []
        output_base_dir = self.data_manager.evaluation_dir

        logger.info(f"Attempting to compare experiments for: {experiment_ids_or_names}")

        for identifier in experiment_ids_or_names:
            found_file_for_identifier = None
            search_dirs = [output_base_dir, os.path.join(output_base_dir, "reruns")]
            for s_dir in search_dirs:
                if not os.path.isdir(s_dir): continue

                potential_files = []
                id_pattern = os.path.join(s_dir, f"evaluation_results*{identifier}.json")
                potential_files.extend(glob.glob(id_pattern))

                name_pattern = os.path.join(s_dir, f"evaluation_results_{identifier}_*.json")
                potential_files.extend(glob.glob(name_pattern))

                rerun_pattern = os.path.join(s_dir, f"evaluation_results_{identifier}*_rerun_*.json")
                potential_files.extend(glob.glob(rerun_pattern))

                if potential_files:
                    unique_files = sorted(list(set(potential_files)), key=os.path.getmtime, reverse=True)
                    found_file_for_identifier = unique_files[0]
                    logger.info(f"Found result file for identifier '{identifier}': {found_file_for_identifier}")
                    break

            if not found_file_for_identifier:
                logger.warning(f"No evaluation result file found for identifier: '{identifier}'. Skipping.")
                continue

            try:
                with open(found_file_for_identifier, 'r', encoding='utf-8') as f:
                    results = json.load(f)

                if not isinstance(results, dict):
                     logger.warning(f"File {found_file_for_identifier} for '{identifier}' does not contain a dictionary. Skipping.")
                     continue
                if "error" in results:
                    logger.warning(f"File {found_file_for_identifier} for '{identifier}' contains an error marker. Skipping.")
                    continue
                if "experiment" not in results or "average_scores" not in results:
                    logger.warning(f"File {found_file_for_identifier} for '{identifier}' is missing 'experiment' or 'average_scores' data. Skipping.")
                    continue
                if not isinstance(results.get("average_scores"), dict): # Use .get() for safety
                     logger.warning(f"'average_scores' in {found_file_for_identifier} for '{identifier}' is not a dictionary. Skipping.")
                     continue

                experiment_results_data.append(results)
            except json.JSONDecodeError as e:
                 logger.error(f"Error decoding JSON from {found_file_for_identifier} for '{identifier}': {e}")
            except Exception as e:
                logger.error(f"Error loading or processing results from {found_file_for_identifier} for '{identifier}': {e}", exc_info=True)


        if len(experiment_results_data) < 2:
            logger.warning(f"Found {len(experiment_results_data)} valid experiment result(s). Need at least 2 to generate a comparison.")
            return None

        comparison = {
            "comparison_timestamp": datetime.datetime.now().isoformat(),
            "compared_experiments_metadata": [],
            "metrics_comparison_table": {}
        }

        for res_data in experiment_results_data:
            exp_meta = res_data.get("experiment", {})
            exp_display_name = exp_meta.get("name", exp_meta.get("id", "UnknownExperiment"))

            comparison["compared_experiments_metadata"].append({
                "name": exp_display_name,
                "id": exp_meta.get("id", ""),
                "timestamp": exp_meta.get("timestamp", ""),
                "config_path": exp_meta.get("config_path", "N/A")
            })

            avg_scores = res_data.get("average_scores", {})
            for metric_name, score in avg_scores.items():
                if score is not None:
                    try:
                        score_float = float(score)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert score '{score}' for metric '{metric_name}' in experiment '{exp_display_name}' to float. Storing as None.")
                        score_float = None
                else:
                     score_float = None

                if metric_name not in comparison["metrics_comparison_table"]:
                    comparison["metrics_comparison_table"][metric_name] = {}
                comparison["metrics_comparison_table"][metric_name][exp_display_name] = score_float

        comp_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_filename = f"comparison_report_{comp_ts}.json"
        comparison_path = os.path.join(output_base_dir, comparison_filename)
        try:
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"Saved detailed comparison report to {comparison_path}")
        except Exception as e:
            logger.error(f"Failed to save comparison JSON: {e}", exc_info=True)

        try:
            import pandas as pd
            metrics_table = comparison["metrics_comparison_table"]
            if metrics_table:
                comparison_df_pivot = pd.DataFrame(metrics_table)
                comparison_df_pivot = comparison_df_pivot.fillna(float('nan'))

                csv_pivot_path = os.path.join(output_base_dir, f"comparison_pivot_{comp_ts}.csv")
                comparison_df_pivot.to_csv(csv_pivot_path, index=True)
                logger.info(f"Saved pivoted comparison CSV to {csv_pivot_path}")

                comparison_df_flat = comparison_df_pivot.reset_index().melt(id_vars='index', var_name='approach', value_name='score')
                comparison_df_flat.rename(columns={'index': 'metric'}, inplace=True)
                csv_flat_path = os.path.join(output_base_dir, f"comparison_flat_{comp_ts}.csv")
                comparison_df_flat[['metric', 'approach', 'score']].to_csv(csv_flat_path, index=False)
                logger.info(f"Saved flat comparison CSV to {csv_flat_path}")
            else:
                logger.info("Metrics comparison table is empty, skipping CSV generation.")

        except ImportError:
            logger.warning("Pandas not installed. Skipping CSV comparison report generation.")
        except Exception as e:
            logger.error(f"Failed to save comparison CSVs: {e}", exc_info=True)

        return comparison