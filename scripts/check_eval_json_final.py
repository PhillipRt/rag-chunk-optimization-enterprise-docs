import json
import glob
import os
import logging
import math # Import math to check for NaN
import argparse # Added for command line arguments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
DEFAULT_EVALUATION_DIR = "data/evaluation/"
DEFAULT_RERUN_DIR = os.path.join(DEFAULT_EVALUATION_DIR, "reruns")  # Subdirectory for rerun files
DEFAULT_FINAL_DIR = os.path.join(DEFAULT_EVALUATION_DIR, "final")  # Final results directory
FILE_PATTERN = "evaluation_results_*.json" # Target JSON files
RESULTS_LIST_KEY = "results" # Key containing the list of samples
# Keys expected directly within each sample dictionary in the main list
METRIC_KEYS_TO_CHECK = [
    "nv_accuracy",  # NVIDIA metric (actual name in RAGAS)
    "answer_correctness",
    "factual_correctness",
    "faithfulness",
    "context_precision",
    "context_recall"
]

# Map old metric names to new ones for backward compatibility
METRIC_NAME_MAPPING = {
    "answer_accuracy": "nv_accuracy"  # Map old name to actual RAGAS name
}
# Try multiple potential identifier keys within each sample
IDENTIFIER_KEYS = ["user_input", "question", "input"]

# --- End Configuration ---

def find_identifier(sample_dict: dict, id_keys: list) -> str:
    """Tries to find a suitable identifier string from the sample dictionary."""
    for key in id_keys:
        if key in sample_dict and sample_dict[key]:
            full_id = str(sample_dict[key])
            # Truncate long identifiers for cleaner logging
            return (full_id[:75] + '...') if len(full_id) > 78 else full_id
    return "Identifier not found"


def check_json_null_nan_final(directory: str, pattern: str, results_key: str, metric_keys: list, id_keys: list):
    """
    Checks evaluation result JSON files (dict containing a 'results' list)
    for null or NaN values in specified metric keys directly within each sample dictionary.
    """
    search_path = os.path.join(directory, pattern)
    json_files = glob.glob(search_path)
    found_failures_overall = False
    failure_files_summary = {} # Store {filename: count_of_affected_samples}

    if not json_files:
        logging.warning(f"No files found matching pattern '{pattern}' in directory '{directory}'")
        return

    logging.info(f"Found {len(json_files)} evaluation result JSON files. Checking for NULL or NaN values in metric keys within '{results_key}' list...")
    print("-" * 60) # Separator for clarity

    for file_path in json_files:
        filename = os.path.basename(file_path)
        affected_samples_count = 0
        file_has_failures = False
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Load the top-level dictionary
                data = json.load(f)

            # Check if it's a dictionary and contains the results list key
            if not isinstance(data, dict) or results_key not in data:
                logging.warning(f"File {filename} is not a dictionary or missing '{results_key}' key. Skipping.")
                continue

            results_list = data[results_key]
            if not isinstance(results_list, list):
                 logging.warning(f"'{results_key}' in {filename} is not a list. Skipping detailed check.")
                 continue

            logging.info(f"Processing {filename} ({len(results_list)} samples in '{results_key}')...")

            for index, sample in enumerate(results_list):
                if not isinstance(sample, dict):
                    logging.warning(f"Item at index {index} in {filename}['{results_key}'] is not a dictionary. Skipping item.")
                    continue

                failed_metrics_for_sample = []
                # Check only the metric keys that actually exist in this sample
                existing_metrics_in_sample = []
                for mk in metric_keys:
                    if mk in sample:
                        existing_metrics_in_sample.append(mk)
                    # Check for old metric names using the mapping
                    elif mk in METRIC_NAME_MAPPING and METRIC_NAME_MAPPING[mk] in sample:
                        # Use the new name for checking but report the old name for backward compatibility
                        existing_metrics_in_sample.append(mk)

                for metric in existing_metrics_in_sample:
                    # Get the actual metric name in the sample (might be mapped)
                    actual_metric = metric
                    if metric in METRIC_NAME_MAPPING and METRIC_NAME_MAPPING[metric] in sample:
                        actual_metric = METRIC_NAME_MAPPING[metric]

                    value = sample[actual_metric]
                    # Check if the metric key's value is None (JSON null) OR NaN
                    is_failure = False
                    if value is None:
                        is_failure = True
                    elif isinstance(value, float) and math.isnan(value):
                         is_failure = True

                    if is_failure:
                        failed_metrics_for_sample.append(metric)

                if failed_metrics_for_sample:
                    found_failures_overall = True
                    file_has_failures = True
                    affected_samples_count += 1

                    sample_identifier = find_identifier(sample, id_keys)
                    if sample_identifier == "Identifier not found":
                         sample_identifier = f"Index {index}" # Fallback to index

                    # Log as WARNING because null/NaN indicates failure
                    logging.warning(f"  [FAILURE] in {filename} - Sample ({sample_identifier}): Null/NaN in [{', '.join(failed_metrics_for_sample)}]")

            if file_has_failures:
                failure_files_summary[filename] = affected_samples_count # Store summary count
                print("-" * 60) # Separator after each file with failures


        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON in file {filename}. Skipping.")
            found_failures_overall = True # Count as an issue
            failure_files_summary[filename] = 'JSON Decode Error'
        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")
            found_failures_overall = True # Count as an issue
            failure_files_summary[filename] = f'Processing Error: {e}'


    print("=" * 60)
    logging.info("JSON Null/NaN Check Complete. Summary:")
    if found_failures_overall:
        logging.warning("Failures (Null or NaN values) were found in the following JSON files:")
        for fname, count_or_error in failure_files_summary.items():
             logging.warning(f"  - {fname}: {count_or_error}")
    else:
        logging.info(f"No Null or NaN values found for the checked metric keys within the '{results_key}' list in any evaluation result JSON files.")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Check evaluation JSON files for NULL or NaN values in metrics')
    parser.add_argument('--path', type=str, default=DEFAULT_EVALUATION_DIR,
                        help=f'Path to directory containing JSON files (default: {DEFAULT_EVALUATION_DIR})')
    parser.add_argument('--pattern', type=str, default=FILE_PATTERN,
                        help=f'File pattern to match (default: {FILE_PATTERN})')
    
    args = parser.parse_args()
    
    # Ensure the directory exists or provide helpful error
    if not os.path.exists(args.path):
        logging.error(f"Directory does not exist: {args.path}")
        logging.info(f"Available options:")
        logging.info(f"  --path {DEFAULT_FINAL_DIR}  (final results)")
        logging.info(f"  --path {DEFAULT_RERUN_DIR}  (rerun results)")
        logging.info(f"  --path {DEFAULT_EVALUATION_DIR}  (main evaluation)")
        exit(1)
    
    logging.info(f"Checking directory: {args.path}")
    logging.info(f"Using pattern: {args.pattern}")
    
    # Call the function that expects the correct structure and checks null/nan
    check_json_null_nan_final(args.path, args.pattern, RESULTS_LIST_KEY, METRIC_KEYS_TO_CHECK, IDENTIFIER_KEYS)