import os
import json
import pickle
import logging
import numpy as np
import pandas as pd # <--- MOVED IMPORT HERE
from typing import List, Dict, Any, Optional, Union
import datetime # Import datetime

logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Add handling for datetime objects
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

class DataManager:
    """Handles document loading, caching, and preprocessing."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data manager.

        Args:
            config: Configuration dictionary with:
                - cache_dir: Directory for cached documents (default: cache/documents)
                - documents_dir: Directory containing documents (default: data/documents)
                - synthetic_data_dir: Directory for synthetic test data (default: data/synthetic_data)
                - evaluation_dir: Directory for evaluation results
        """
        self.config = config
        required_keys = ["cache_dir", "documents_dir", "synthetic_data_dir", "evaluation_dir"]
        # Add 'approaches_dir' check if it's truly required by DataManager itself,
        # otherwise ExperimentManager can handle its absence.
        # required_keys.append("approaches_dir")

        missing = [key for key in required_keys if key not in config]
        if missing:
            # Allow defaults if only approaches_dir is missing and handled elsewhere
            if missing == ["approaches_dir"] and len(missing) == 1: # Check if only approaches_dir is missing
                 logger.warning("DataManager config missing 'approaches_dir', ensure ExperimentManager handles this.")
            else:
                 # Raise error if other required keys are missing, or if more than just approaches_dir is missing
                 raise ValueError(f"DataManager config missing required keys: {missing}")


        self.cache_dir = config["cache_dir"]
        self.documents_dir = config["documents_dir"]
        self.synthetic_data_dir = config["synthetic_data_dir"]
        self.evaluation_dir = config["evaluation_dir"]
        # Store approaches_dir if present, but don't strictly require it here
        self.approaches_dir = config.get("approaches_dir", "config/approaches")


        logger.info(f"DataManager configured with paths:")
        logger.info(f"  Cache Dir: {self.cache_dir}")
        logger.info(f"  Documents Dir: {self.documents_dir}")
        logger.info(f"  Synthetic Data Dir: {self.synthetic_data_dir}")
        logger.info(f"  Evaluation Dir: {self.evaluation_dir}")
        logger.info(f"  Approaches Dir: {self.approaches_dir}")


        # Create directories if they don't exist
        # Ensure approaches_dir exists only if it's provided and needed by DataManager directly
        dirs_to_create = [self.cache_dir, self.documents_dir, self.synthetic_data_dir, self.evaluation_dir]
        if self.approaches_dir: # Only create approaches dir if it's configured
             dirs_to_create.append(self.approaches_dir)

        for dir_path in dirs_to_create:
            if dir_path: # Check if path is not None or empty
                 os.makedirs(dir_path, exist_ok=True)

    def load_documents(self,
                    cached_path: Optional[str] = None,
                    force_reload: bool = False
                    ) -> List[Dict[str, Any]]:
        """
        Load documents from cache or process raw documents.

        Args:
            cached_path: Path to cached documents file (optional)
            force_reload: Force reload from raw documents (default: False)

        Returns:
            List of document dictionaries with 'content' and 'metadata' keys
        """
        # Determine cache path
        if cached_path is None:
            cached_path = os.path.join(self.cache_dir, "cached_documents.pkl")

        # Try to load from cache first (unless force_reload)
        if not force_reload and os.path.exists(cached_path):
            # Pass cached_path to _load_from_cache
            loaded_docs = self._load_from_cache(cached_path)
            # Check if loading from cache was successful (returned non-empty list)
            if loaded_docs:
                return loaded_docs
            else:
                # If cache loading failed (e.g., invalid format), log and proceed to reprocess
                logger.warning(f"Loading from cache at {cached_path} failed or returned empty. Reprocessing raw documents.")


        # If no cache, force_reload, or cache loading failed, process raw documents
        logger.info(f"Processing raw documents from {self.documents_dir}")
        documents = self._process_raw_documents()

        # Cache the processed documents
        self._cache_documents(documents, cached_path)

        return documents

    def _load_from_cache(self, cached_path: str) -> List[Dict[str, Any]]:
        """Load documents from the cached pickle file."""
        logger.info(f"Loading cached documents from {cached_path}")

        try:
            with open(cached_path, 'rb') as f:
                cached_data = pickle.load(f)

            # --- Enhanced Cache Format Handling ---
            documents = []
            if isinstance(cached_data, list):
                # Iterate through items and check structure
                for doc in cached_data:
                    if isinstance(doc, dict):
                        if "content" in doc and "metadata" in doc:
                            # Expected format
                            documents.append(doc)
                        elif "page_content" in doc and "metadata" in doc:
                            # Convert LangChain Document format
                            logger.debug("Converting LangChain document format from cache.")
                            documents.append({
                                "content": doc["page_content"],
                                "metadata": doc["metadata"]
                            })
                        else:
                            logger.warning(f"Skipping cached item due to unexpected dictionary structure: {list(doc.keys())}")
                    # Add handling for other potential old formats if necessary
                    # elif isinstance(doc, SomeOldClass):
                    #     # convert SomeOldClass to dict
                    #     pass
                    else:
                         logger.warning(f"Skipping cached item of unexpected type: {type(doc)}")

                if documents:
                    logger.info(f"Successfully loaded {len(documents)} documents from cache.")
                    return documents
                else:
                    # Cache file exists but contained no valid documents
                    logger.error("Cache file exists but contained no valid documents.")
                    return [] # Return empty list to indicate failure

            else:
                 # Cache file exists but has completely unexpected format (not a list)
                 logger.error("Cached data has unexpected format (expected list).")
                 return [] # Return empty list to indicate failure

        except (pickle.UnpicklingError, EOFError, FileNotFoundError, Exception) as e:
            logger.error(f"Error loading cached documents from {cached_path}: {str(e)}", exc_info=True)
            logger.info("Falling back to processing raw documents")
            return [] # Return empty list to indicate failure


    def _process_raw_documents(self) -> List[Dict[str, Any]]:
        """Process raw documents from the documents directory."""
        logger.info(f"Processing raw documents from {self.documents_dir}")

        # Check if directory exists
        if not os.path.exists(self.documents_dir) or not os.path.isdir(self.documents_dir):
            logger.error(f"Documents directory '{self.documents_dir}' does not exist or is not a directory.")
            return []

        documents = []
        supported_extensions = ('.txt', '.md', '.html', '.jsonl', '.json') # Add json/jsonl
        logger.info(f"Looking for files with extensions: {supported_extensions}")

        for root, _, files in os.walk(self.documents_dir):
            logger.debug(f"Scanning directory: {root}")
            for file in files:
                file_path = os.path.join(root, file)
                _, file_ext = os.path.splitext(file)

                if file_ext.lower() in supported_extensions:
                    logger.debug(f"Processing file: {file_path}")
                    try:
                        if file_ext.lower() in ('.txt', '.md', '.html'):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            documents.append({
                                "content": content,
                                "metadata": {
                                    "source": os.path.relpath(file_path, self.documents_dir), # Use relative path
                                    "filename": file
                                }
                            })
                            # logger.info(f"Processed text file: {file_path}") # Reduce verbosity

                        elif file_ext.lower() == '.jsonl':
                             with open(file_path, 'r', encoding='utf-8') as f:
                                 for i, line in enumerate(f):
                                     try:
                                         data = json.loads(line)
                                         # Expecting dicts with 'content' and optionally 'metadata'
                                         if isinstance(data, dict) and 'content' in data:
                                              doc_metadata = data.get('metadata', {})
                                              # Add source info if not present
                                              if 'source' not in doc_metadata:
                                                   doc_metadata['source'] = f"{os.path.relpath(file_path, self.documents_dir)}#L{i+1}"
                                              if 'filename' not in doc_metadata:
                                                  doc_metadata['filename'] = file
                                              documents.append({
                                                 "content": data['content'],
                                                 "metadata": doc_metadata
                                              })
                                         else:
                                             logger.warning(f"Skipping invalid line {i+1} in {file_path}: Expected dict with 'content'. Found: {type(data)}")
                                     except json.JSONDecodeError:
                                         logger.warning(f"Skipping invalid JSON line {i+1} in {file_path}")
                             # logger.info(f"Processed JSONL file: {file_path}") # Reduce verbosity

                        elif file_ext.lower() == '.json':
                             with open(file_path, 'r', encoding='utf-8') as f:
                                 try:
                                      data = json.load(f)
                                      # Expecting a list of dicts or a single dict
                                      docs_in_file = []
                                      if isinstance(data, list):
                                           docs_in_file = data
                                      elif isinstance(data, dict):
                                           docs_in_file = [data] # Treat single dict as list with one item
                                      else:
                                           logger.warning(f"Skipping JSON file {file_path}: Expected list or dict, found {type(data)}.")

                                      for i, doc_data in enumerate(docs_in_file):
                                           if isinstance(doc_data, dict) and 'content' in doc_data:
                                                doc_metadata = doc_data.get('metadata', {})
                                                # Add source info if not present
                                                if 'source' not in doc_metadata:
                                                     doc_metadata['source'] = f"{os.path.relpath(file_path, self.documents_dir)}#Doc{i+1}"
                                                if 'filename' not in doc_metadata:
                                                    doc_metadata['filename'] = file
                                                documents.append({
                                                   "content": doc_data['content'],
                                                   "metadata": doc_metadata
                                                })
                                           else:
                                               logger.warning(f"Skipping invalid document entry {i+1} in {file_path}: Expected dict with 'content'.")

                                 except json.JSONDecodeError:
                                     logger.warning(f"Skipping invalid JSON file {file_path}")
                             # logger.info(f"Processed JSON file: {file_path}") # Reduce verbosity

                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
                else:
                    logger.debug(f"Skipping file with unsupported extension: {file_path}")

        if not documents:
             logger.warning(f"No documents were successfully processed from {self.documents_dir}. Please check the directory and file formats.")
        else:
            logger.info(f"Successfully processed {len(documents)} documents from {self.documents_dir}")
        return documents

    def _cache_documents(self, documents: List[Dict[str, Any]], cached_path: str) -> None:
        """Cache the processed documents to disk."""
        if not documents:
            logger.warning("No documents to cache.")
            return
        try:
            os.makedirs(os.path.dirname(cached_path), exist_ok=True)
            with open(cached_path, 'wb') as f:
                pickle.dump(documents, f)
            logger.info(f"Cached {len(documents)} documents to {cached_path}")
        except Exception as e:
            logger.error(f"Error caching documents to {cached_path}: {str(e)}", exc_info=True)

    def load_test_data(self, data_path: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Load test data for evaluation.

        Args:
            data_path: Path to the test data file (optional). Expects JSONL format.
            limit: Maximum number of samples to load (default: 10, 0 or negative means no limit)

        Returns:
            List of test samples (dictionaries)
        """
        # Find the latest test data file if not specified
        if data_path is None:
            try:
                logger.info(f"No specific test data path provided. Searching in {self.synthetic_data_dir}...")
                # Look for files starting with 'test_data_' and ending with '.jsonl'
                data_files = [f for f in os.listdir(self.synthetic_data_dir)
                            if f.startswith("test_data_") and f.endswith(".jsonl")]

                if not data_files:
                    logger.error(f"No test data files found in {self.synthetic_data_dir} matching 'test_data_*.jsonl'.")
                    return []

                # Sort by name (assuming timestamp in name makes latest last lexicographically, reverse=True for latest first)
                data_files.sort(reverse=True)
                data_path = os.path.join(self.synthetic_data_dir, data_files[0])
                logger.info(f"Found latest test data file: {data_path}")
            except FileNotFoundError:
                 logger.error(f"Synthetic data directory not found: {self.synthetic_data_dir}")
                 return []
            except Exception as e:
                 logger.error(f"Error searching for test data files: {e}", exc_info=True)
                 return []


        # Load the data from the specified or found path
        if not os.path.exists(data_path):
             logger.error(f"Test data file not found: {data_path}")
             return []

        logger.info(f"Loading test data from {data_path}")
        data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        sample = json.loads(line)
                        if isinstance(sample, dict): # Ensure it's a dictionary
                             data.append(sample)
                        else:
                             logger.warning(f"Skipping line {i+1} in {data_path}: Expected JSON object (dict), found {type(sample)}.")

                        # Check if we've reached the limit (only if limit is positive)
                        if limit > 0 and len(data) >= limit:
                            logger.info(f"Reached limit of {limit} samples.")
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {i+1} in {data_path}")
                    except Exception as line_e:
                        logger.error(f"Error processing line {i+1} in {data_path}: {line_e}", exc_info=True)
                        # Decide whether to continue or stop on line processing error

            if limit <= 0:
                logger.info(f"Loaded all {len(data)} test samples (no limit applied)")
            else:
                logger.info(f"Loaded {len(data)} test samples (limit: {limit})")

            if not data:
                 logger.warning(f"No valid test samples loaded from {data_path}.")

            return data

        except Exception as e:
            logger.error(f"Error loading test data from {data_path}: {str(e)}", exc_info=True)
            return []

    def save_results(self,
                   results: Union[List[Dict[str, Any]], Dict[str, Any]],
                   filename: str,
                   approach_name: str) -> str:
        """
        Save evaluation or augmented dataset results to disk.

        Args:
            results: Either a dictionary (expected for final eval results or augmented data structure)
                     or a list (legacy support, less preferred).
            filename: Base filename to use (e.g., "augmented_dataset", "evaluation_results").
            approach_name: Name of the approach for file annotation.

        Returns:
            Path to the saved results file (JSON).
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- Define subdirectories consistently ---
        augmented_datasets_dir = os.path.join(self.evaluation_dir, "augmented_datasets")
        csv_results_dir = os.path.join(self.evaluation_dir, "csv_results")
        metrics_dir = os.path.join(self.evaluation_dir, "metrics")
        # Create directories if they don't exist
        os.makedirs(augmented_datasets_dir, exist_ok=True)
        os.makedirs(csv_results_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        # Determine the correct output directory based on filename
        if filename == "augmented_dataset":
            output_dir = augmented_datasets_dir
            logger.info(f"Saving augmented dataset to subdirectory: {output_dir}")
        elif filename.startswith("evaluation_results"):
             output_dir = self.evaluation_dir
             logger.info(f"Saving evaluation results to main evaluation directory: {output_dir}")
        else:
             output_dir = self.evaluation_dir # Default to main evaluation dir for other filenames
             logger.info(f"Saving file '{filename}' to main evaluation directory: {output_dir}")

        # Format filename for main results (JSON)
        result_path = os.path.join(
            output_dir, # Use determined output directory
            f"{filename}_{approach_name}_{timestamp}.json"
        )

        # --- Process and save ---
        serializable_results = results # Start assuming input is already serializable

        # Handle dictionary case (expected for eval results and augmented data structure)
        if isinstance(results, dict):
            # Process potential DataFrame within the results dictionary
            df_key_found = None
            # Iterate over a copy of items if modifying the dict during iteration is possible
            items_to_process = list(results.items())
            for key, value in items_to_process:
                # --- FIX: Check isinstance using pd imported at the top ---
                if isinstance(value, pd.DataFrame):
                    df_key_found = key
                    # Save DataFrame to CSV only if it's evaluation results
                    if filename.startswith("evaluation_results") and key == "results":
                        csv_path = os.path.join( # Define CSV path here for clarity
                            csv_results_dir,
                            f"{filename}_{approach_name}_{timestamp}.csv"
                        )
                        try:
                            value.to_csv(csv_path, index=False)
                            logger.info(f"Saved results DataFrame to {csv_path}")
                        except ImportError:
                            logger.warning("Pandas not installed. Cannot save DataFrame to CSV.")
                        except Exception as e:
                            logger.warning(f"Could not save DataFrame to CSV {csv_path}: {e}")
                    # Replace DataFrame with list of records for JSON serialization
                    # Update the original dictionary directly or the copy being built
                    serializable_results[key] = value.to_dict(orient='records')
                    # break # Remove break if multiple dataframes are possible (unlikely for now)

            # Also save a CSV of the metrics and average scores for easy comparison
            # Only do this for evaluation results files
            if "average_scores" in results and filename.startswith("evaluation_results"):
                metrics_csv_path = os.path.join(
                    metrics_dir,
                    f"metrics_{approach_name}_{timestamp}.csv" # Use original approach name/ts
                )
                try:
                    avg_scores = results.get("average_scores", {})
                    if avg_scores and isinstance(avg_scores, dict):
                        metrics_df = pd.DataFrame({
                            "metric": list(avg_scores.keys()),
                            "score": list(avg_scores.values()),
                            "approach": approach_name,
                            "timestamp": timestamp  # Add timestamp for easier tracking
                        })
                        metrics_df.to_csv(metrics_csv_path, index=False)
                        logger.info(f"Saved metrics summary to {metrics_csv_path}")
                    else:
                         logger.info("No average scores found or invalid format, skipping metrics CSV.")
                except ImportError:
                    logger.warning("Pandas not installed. Cannot save metrics summary CSV.")
                except Exception as e:
                    logger.warning(f"Could not save metrics summary to CSV: {str(e)}")

        elif isinstance(results, list):
             # Handle legacy list format if necessary, though dict is preferred
             logger.warning("Saving results as a list. Consider using a dictionary structure.")
             serializable_results = results # Assume list items are serializable
        else:
             logger.error(f"Unsupported results type for saving: {type(results)}")
             # Optionally raise an error or return early
             return "" # Indicate failure

        # Save the processed results to JSON
        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, cls=NumpyEncoder)
            logger.info(f"Saved results JSON to {result_path}")
            return result_path
        except TypeError as e:
             logger.error(f"Failed to serialize results to JSON: {e}. Check for non-serializable data types.", exc_info=True)
             # Attempt to save a simplified version or log the problematic structure
             try:
                  with open(result_path + ".ERROR", 'w', encoding='utf-8') as f:
                       f.write(f"Error saving original results due to TypeError: {e}\n")
                       f.write(f"Problematic data structure (simplified):\n{repr(serializable_results)[:1000]}...") # Log a snippet
             except Exception as save_err:
                  logger.error(f"Could not even save error file: {save_err}")
             return "" # Indicate failure
        except Exception as e:
             logger.error(f"Failed to save results JSON to {result_path}: {e}", exc_info=True)
             return "" # Indicate failure