import json
import glob
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
EVALUATION_DIR = "data/evaluation/"
FILE_PATTERN = "augmented_dataset_*.json"
KEYS_TO_CHECK = {
    "response": {
        "check_type": "non_empty_string"
    },
    "retrieved_docs": {
        "check_type": "non_empty_list" # Changed check type
    }
}
IDENTIFIER_COLUMN = "question" # Column to identify samples (adjust if needed)
# --- End Configuration ---

def check_augmented_data(directory: str, pattern: str, keys_config: dict, id_col: str):
    """
    Checks augmented dataset JSON files for issues with specified keys.

    Args:
        directory: The directory containing the augmented dataset JSON files.
        pattern: The glob pattern to match the JSON files.
        keys_config: A dictionary defining keys to check and their validation rules.
        id_col: The name of the key used to identify samples (e.g., 'question').
    """
    search_path = os.path.join(directory, pattern)
    json_files = glob.glob(search_path)
    found_issues_overall = False
    issue_files_summary = {} # Store {filename: count_of_affected_samples}

    if not json_files:
        logging.warning(f"No files found matching pattern '{pattern}' in directory '{directory}'")
        return

    logging.info(f"Found {len(json_files)} augmented dataset files. Checking specified keys (response, retrieved_docs)...")
    print("-" * 60) # Separator for clarity

    for file_path in json_files:
        filename = os.path.basename(file_path)
        affected_samples_count = 0
        file_has_issues = False
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                logging.warning(f"File {filename} does not contain a list as expected. Skipping.")
                continue

            logging.info(f"Processing {filename} ({len(data)} samples)...")

            for index, sample in enumerate(data):
                if not isinstance(sample, dict):
                    logging.warning(f"Item at index {index} in {filename} is not a dictionary. Skipping item.")
                    continue

                sample_issues = [] # Collect issues for this specific sample

                for key, config in keys_config.items():
                    value = sample.get(key)
                    check_type = config.get("check_type")
                    issue_description = None

                    if value is None:
                        issue_description = f"Missing '{key}' key"
                    elif check_type == "non_empty_string":
                        if not isinstance(value, str):
                            issue_description = f"'{key}' is not a string (type: {type(value).__name__})"
                        elif not value.strip():
                            issue_description = f"'{key}' is empty or whitespace"
                    elif check_type == "non_empty_list": # Check for non-empty list
                        if not isinstance(value, list):
                            issue_description = f"'{key}' is not a list (type: {type(value).__name__})"
                        elif not value: # Check if list is empty
                            issue_description = f"'{key}' list is empty"
                        # Optional: Check if all items in the list are dicts (can be added if needed)
                        # elif key == "retrieved_docs" and not all(isinstance(item, dict) for item in value):
                        #     issue_description = f"Not all items in '{key}' list are dictionaries"


                    if issue_description:
                        sample_issues.append(issue_description)

                if sample_issues:
                    found_issues_overall = True
                    file_has_issues = True
                    affected_samples_count += 1

                    # Try to get identifier
                    if id_col in sample:
                        sample_identifier_full = str(sample[id_col])
                        sample_identifier = (sample_identifier_full[:75] + '...') if len(sample_identifier_full) > 78 else sample_identifier_full
                    else:
                        sample_identifier = f"Index {index}"

                    logging.warning(f"  - Issue(s) in {filename} - Sample ({sample_identifier}): {'; '.join(sample_issues)}")

            if file_has_issues:
                issue_files_summary[filename] = affected_samples_count # Store summary count
                print("-" * 60) # Separator after each file with issues


        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON in file {filename}. Skipping.")
            found_issues_overall = True # Count as an issue
            issue_files_summary[filename] = 'JSON Decode Error'
        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")
            found_issues_overall = True # Count as an issue
            issue_files_summary[filename] = f'Processing Error: {e}'


    print("=" * 60)
    logging.info("Processing Complete. Summary:")
    if found_issues_overall:
        logging.warning("Issues found with specified keys in the following files:")
        for fname, count_or_error in issue_files_summary.items():
             logging.warning(f"  - {fname}: {count_or_error}")
    else:
        logging.info("No issues found with specified keys (response, retrieved_docs) in any augmented dataset files.")
    print("=" * 60)


if __name__ == "__main__":
    check_augmented_data(EVALUATION_DIR, FILE_PATTERN, KEYS_TO_CHECK, IDENTIFIER_COLUMN)