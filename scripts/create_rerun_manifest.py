import json
import glob
import os
import logging
import math # Import math to check for NaN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
EVALUATION_DIR = "data/evaluation/"
RERUN_DIR = os.path.join(EVALUATION_DIR, "reruns")  # Subdirectory for rerun files
FILE_PATTERN = "evaluation_results_*.json" # Target JSON files
MANIFEST_FILE = "rerun_manifest.json" # Output file
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

# Create rerun directory if it doesn't exist
os.makedirs(RERUN_DIR, exist_ok=True)
# --- End Configuration ---

def create_failure_manifest(directory: str, pattern: str, results_key: str, metric_keys: list, output_manifest_path: str):
    """
    Checks evaluation result JSON files for null or NaN values in metric keys
    and creates a manifest file listing the failures.
    """
    search_path = os.path.join(directory, pattern)
    json_files = glob.glob(search_path)
    failure_manifest = {} # {filename: [{"index": index, "metrics": [failed_metric_names]}]}
    found_failures_overall = False

    if not json_files:
        logging.warning(f"No files found matching pattern '{pattern}' in directory '{directory}'")
        return

    logging.info(f"Found {len(json_files)} evaluation result JSON files. Generating failure manifest...")

    for file_path in json_files:
        filename = os.path.basename(file_path)
        failures_in_file = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict) or results_key not in data:
                logging.warning(f"File {filename} is not a dictionary or missing '{results_key}' key. Skipping.")
                continue

            results_list = data[results_key]
            if not isinstance(results_list, list):
                 logging.warning(f"'{results_key}' in {filename} is not a list. Skipping detailed check.")
                 continue

            logging.info(f"Processing {filename} ({len(results_list)} samples)...")

            for index, sample in enumerate(results_list):
                if not isinstance(sample, dict):
                    logging.warning(f"Item at index {index} in {filename} is not a dictionary. Skipping item.")
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
                    is_failure = False
                    if value is None: # Check for JSON null
                        is_failure = True
                    # Also check for float NaN, although less likely based on previous check
                    elif isinstance(value, float) and math.isnan(value):
                         is_failure = True

                    if is_failure:
                        failed_metrics_for_sample.append(metric)

                if failed_metrics_for_sample:
                    found_failures_overall = True
                    failures_in_file.append({
                        "index": index,
                        "metrics": failed_metrics_for_sample
                    })

            if failures_in_file:
                failure_manifest[filename] = failures_in_file
                logging.info(f"Found {len(failures_in_file)} samples with failures in {filename}.")


        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON in file {filename}. Skipping.")
            failure_manifest[filename] = "JSON Decode Error" # Mark file as problematic
        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")
            failure_manifest[filename] = f"Processing Error: {e}" # Mark file as problematic


    # Save the manifest
    manifest_full_path = os.path.join(directory, output_manifest_path) # Save manifest in eval dir
    try:
        os.makedirs(directory, exist_ok=True) # Ensure directory exists
        with open(manifest_full_path, 'w', encoding='utf-8') as f:
            json.dump(failure_manifest, f, indent=2)
        logging.info(f"Successfully saved failure manifest to {manifest_full_path}")
    except Exception as e:
         logging.error(f"Error saving manifest file {manifest_full_path}: {e}")

    print("=" * 60)
    logging.info("Manifest Creation Complete.")
    if not found_failures_overall:
        logging.info("No failures (Null or NaN values) found in any files.")
    print("=" * 60)


if __name__ == "__main__":
    create_failure_manifest(EVALUATION_DIR, FILE_PATTERN, RESULTS_LIST_KEY, METRIC_KEYS_TO_CHECK, MANIFEST_FILE)