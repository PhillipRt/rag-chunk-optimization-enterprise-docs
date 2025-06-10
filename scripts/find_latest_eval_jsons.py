import os
import glob
import re
import shutil
from typing import List, Dict, Any, Optional

# Determine project root for cleaner output paths, assuming script is in master-thesis-rag/scripts/
try:
    SCRIPT_ABS_PATH = os.path.abspath(__file__)
    SCRIPTS_DIR = os.path.dirname(SCRIPT_ABS_PATH)
    # Assumes the script is in 'master-thesis-rag/scripts/', so parent of 'scripts' is 'master-thesis-rag'
    project_root_for_output = os.path.dirname(SCRIPTS_DIR)
    if os.path.basename(project_root_for_output) != "master-thesis-rag":
        # Fallback: try to find project root by looking for key files
        current_dir = os.getcwd()
        while current_dir != os.path.dirname(current_dir):  # not at filesystem root
            if os.path.exists(os.path.join(current_dir, "pyproject.toml")) and \
               os.path.exists(os.path.join(current_dir, "requirements.txt")):
                project_root_for_output = current_dir
                break
            current_dir = os.path.dirname(current_dir)
        else:
            # Ultimate fallback to current directory
            project_root_for_output = os.getcwd()
except NameError: # __file__ is not defined (e.g. in interactive interpreter)
    # Try to find project root by looking for key files
    current_dir = os.getcwd()
    while current_dir != os.path.dirname(current_dir):  # not at filesystem root
        if os.path.exists(os.path.join(current_dir, "pyproject.toml")) and \
           os.path.exists(os.path.join(current_dir, "requirements.txt")):
            project_root_for_output = current_dir
            break
        current_dir = os.path.dirname(current_dir)
    else:
        # Ultimate fallback to current directory
        project_root_for_output = os.getcwd()

# Define directories to scan relative to project root
DATA_DIRS_TO_SCAN = [
    os.path.join(project_root_for_output, "data", "evaluation"),
    os.path.join(project_root_for_output, "data", "evaluation", "reruns"),
    os.path.join(project_root_for_output, "data", "evaluation", "old"),
]

# Define the destination directory for final files
FINAL_DIR = os.path.join(project_root_for_output, "data", "evaluation", "final")

# Regex patterns
RERUN_PATTERN = re.compile(
    # Captures: (1) full original_run_id (e.g., evaluation_results_approach_name_20230101_120000)
    #           (2) approach_name_part (e.g., approach_name) (NOW NON-GREEDY)
    #           (3) original_timestamp (e.g., 20230101_120000)
    #           (4) the _rerun or _rerun_rerun string
    #           (5) rerun_version (e.g., 1)
    #           (6) rerun_timestamp (e.g., 20230102_130000)
    r"^(evaluation_results_(.+?)_(\d{8}_\d{6}))((?:_rerun)+)_v(\d+)_(\d{8}_\d{6})\.json$"
)
SIMPLE_RERUN_PATTERN = re.compile(
    # Captures: (1) full original_run_id (e.g., evaluation_results_approach_name_20230101_120000)
    #           (2) approach_name_part (e.g., approach_name)
    #           (3) original_timestamp (e.g., 20230101_120000)
    r"^(evaluation_results_(.+?)_(\d{8}_\d{6}))_rerun\.json$"
)
ORIGINAL_PATTERN = re.compile(
    # Captures: (1) full original_run_id (e.g., evaluation_results_approach_name_20230101_120000)
    #           (2) approach_name_part (e.g., approach_name) (NOW NON-GREEDY)
    #           (3) original_timestamp (e.g., 20230101_120000)
    r"^(evaluation_results_(.+?)_(\d{8}_\d{6}))\.json$"
)

def parse_filename(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Parses a JSON filename to extract run ID, rerun version, and timestamp.
    Returns a dictionary with parsed info, or None if the filename doesn't match patterns.
    """
    filename = os.path.basename(filepath)
    
    rerun_match = RERUN_PATTERN.match(filename)
    if rerun_match:
        return {
            "filepath": filepath,
            "original_run_id": rerun_match.group(1), # Full original run ID (base_name + orig_ts)
            "rerun_version": int(rerun_match.group(5)) + 1, # v1 -> 2, v2 -> 3, etc.
            "rerun_timestamp": rerun_match.group(6), # YYYYMMDD_HHMMSS
            "is_rerun": True,
        }
    
    simple_rerun_match = SIMPLE_RERUN_PATTERN.match(filename)
    if simple_rerun_match:
        return {
            "filepath": filepath,
            "original_run_id": simple_rerun_match.group(1), # Full original run ID
            "rerun_version": 1, # Fixed version for simple reruns
            "rerun_timestamp": "", # No specific rerun timestamp in this format
            "is_rerun": True,
        }

    original_match = ORIGINAL_PATTERN.match(filename)
    if original_match:
        return {
            "filepath": filepath,
            "original_run_id": original_match.group(1), # Full original run ID
            "rerun_version": 0, # Assign version 0 to originals for comparison
            "rerun_timestamp": "", # Originals don't have a specific rerun timestamp
            "is_rerun": False,
        }
        
    print(f"Warning: Filename '{filename}' (from path '{filepath}') did not match expected patterns. Skipping.")
    return None

def is_later(new_info: Dict[str, Any], existing_info: Dict[str, Any]) -> bool:
    """
    Compares two parsed file infos to determine if new_info represents a later version.
    """
    # Reruns are always prioritized over original (non-rerun) files
    if new_info["is_rerun"] and not existing_info["is_rerun"]:
        return True
    if not new_info["is_rerun"] and existing_info["is_rerun"]:
        return False

    # If both are reruns, compare version and then timestamp
    if new_info["is_rerun"] and existing_info["is_rerun"]:
        if new_info["rerun_version"] > existing_info["rerun_version"]:
            return True
        if new_info["rerun_version"] < existing_info["rerun_version"]:
            return False
        # Versions are the same, compare rerun timestamps (string comparison is fine for YYYYMMDD_HHMMSS format)
        if new_info["rerun_timestamp"] > existing_info["rerun_timestamp"]:
            return True
        if new_info["rerun_timestamp"] < existing_info["rerun_timestamp"]:
            return False
            
    # If both are originals (is_rerun=False for both), or if all relevant fields of reruns are identical,
    # the new one is not considered strictly later by this logic.
    return False

def find_latest_evaluation_files(scan_dirs: List[str]) -> List[str]:
    """
    Scans directories, parses JSON filenames, and identifies the latest file for each original run.
    Returns a sorted list of filepaths.
    """
    all_json_files: List[str] = []
    for data_dir in scan_dirs:
        # Normalize path for consistency (e.g., handles mixed slashes if any)
        normalized_data_dir = os.path.normpath(data_dir)
        if not os.path.isdir(normalized_data_dir):
            print(f"Warning: Directory not found: {normalized_data_dir}. Skipping.")
            continue
        all_json_files.extend(glob.glob(os.path.join(normalized_data_dir, "*.json")))

    if not all_json_files:
        print("No JSON files found in the specified directories.")
        return []

    # Stores the information of the latest file found for each original_run_id
    latest_files_info: Dict[str, Dict[str, Any]] = {}

    for f_path in all_json_files:
        parsed_info = parse_filename(f_path)
        if not parsed_info:
            continue # Skip files that don't match expected naming patterns
            
        run_id = parsed_info["original_run_id"]
        
        # If this run_id is new, or if the current file is later than the stored one
        if (run_id not in latest_files_info or
            is_later(parsed_info, latest_files_info[run_id])):
            latest_files_info[run_id] = parsed_info
            
    # Extract just the filepaths from the determined latest files
    result_paths = [info["filepath"] for info in latest_files_info.values()]
    
    result_paths.sort() # Sort for consistent and predictable output order
    
    return result_paths

def move_latest_files(latest_json_files: List[str], final_dir: str):
    """
    Moves the latest JSON files and their corresponding CSVs to the final directory,
    but only if they are newer than any existing version in the final directory.
    """
    if not latest_json_files:
        print("No latest JSON files found to move.")
        return

    os.makedirs(final_dir, exist_ok=True)

    # --- Step 1: Index files already in the final directory ---
    final_files_info: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(final_dir):
        final_json_files = glob.glob(os.path.join(final_dir, "*.json"))
        for f_path in final_json_files:
            parsed_info = parse_filename(f_path)
            if parsed_info:
                final_files_info[parsed_info["original_run_id"]] = parsed_info

    # --- Step 2: Determine which files actually need to be moved ---
    files_to_move_preview = []
    csv_not_found_list = []
    skipped_files = []

    for json_path in latest_json_files:
        candidate_info = parse_filename(json_path)
        if not candidate_info:
            continue

        run_id = candidate_info["original_run_id"]
        
        # Check if a version is already in the final directory
        if run_id in final_files_info:
            final_version_info = final_files_info[run_id]
            # If the candidate is NOT later than the final version, skip it
            if not is_later(candidate_info, final_version_info):
                skipped_files.append((os.path.basename(json_path), "version in final is same or newer"))
                continue

        # If we reach here, the file should be moved.
        if not os.path.exists(json_path):
            print(f"  WARNING: Source JSON file no longer exists, will skip: {os.path.relpath(json_path, project_root_for_output) if project_root_for_output else json_path}")
            continue
            
        json_filename = os.path.basename(json_path)
        csv_filename_stem = os.path.splitext(json_filename)[0]
        source_csv_path = os.path.join(os.path.dirname(json_path), csv_filename_stem + ".csv")
        
        files_to_move_preview.append(('JSON', json_path, json_filename))
        
        if os.path.exists(source_csv_path):
            files_to_move_preview.append(('CSV', source_csv_path, csv_filename_stem + ".csv"))
        else:
            csv_not_found_list.append(json_filename)

    # --- Step 3: Show the move preview and ask for confirmation ---
    print(f"\n{'='*60}")
    print(f"MOVE PREVIEW - Files will be moved to: {final_dir}")
    print(f"{'='*60}")
    
    if not files_to_move_preview:
        print("No new files need to be moved.")
        if skipped_files:
            print("\nFiles that were skipped (a same or newer version already exists in final):")
            for filename, reason in skipped_files:
                print(f"  - {filename}")
        print(f"\n{'='*60}")
        return
        
    print(f"\nFiles to be moved ({len(files_to_move_preview)} total):")
    for file_type, source_path, filename in files_to_move_preview:
        rel_source = os.path.relpath(source_path, project_root_for_output) if project_root_for_output else source_path
        print(f"  {file_type}: {rel_source} -> {filename}")
    
    if csv_not_found_list:
        print(f"\nCSV files NOT FOUND for the following JSON files:")
        for json_name in csv_not_found_list:
            print(f"  {json_name}")
            
    if skipped_files:
        print(f"\nFiles that will be SKIPPED (a same or newer version already exists in final):")
        for filename, reason in skipped_files:
            print(f"  - {filename}")
    
    print(f"\n{'='*60}")
    
    # Ask for confirmation
    while True:
        response = input("Proceed with moving the new files? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            break
        elif response in ['n', 'no']:
            print("Move operation cancelled by user.")
            return
        else:
            print("Please enter 'y' or 'n'")
    
    print(f"\nProceeding with file moves...")

    # --- Step 4: Execute the move ---
    moved_json_count = 0
    moved_csv_count = 0
    json_move_errors = 0
    csv_move_errors = 0
    csv_not_found_count = len(csv_not_found_list)

    for file_type, source_path, filename in files_to_move_preview:
        dest_path = os.path.join(final_dir, filename)
        try:
            print(f"  Moving {file_type}: {os.path.basename(source_path)} to {dest_path}")
            shutil.move(source_path, dest_path)
            if file_type == 'JSON':
                moved_json_count += 1
            else:  # CSV
                moved_csv_count += 1
        except Exception as e:
            print(f"  ERROR moving {file_type} {os.path.basename(source_path)}: {e}")
            if file_type == 'JSON':
                json_move_errors += 1
            else:  # CSV
                csv_move_errors += 1
        
    print("\n--- File Moving Summary ---")
    print(f"Successfully moved {moved_json_count} JSON files.")
    print(f"Successfully moved {moved_csv_count} CSV files.")
    if csv_not_found_count > 0:
        print(f"Corresponding CSVs not found for {csv_not_found_count} JSON files.")
    if json_move_errors > 0:
        print(f"Errors moving {json_move_errors} JSON files.")
    if csv_move_errors > 0:
        print(f"Errors moving {csv_move_errors} CSV files.")

if __name__ == "__main__":
    print("Identifying the latest evaluation JSON files...")
    print(f"Scanning directories: {DATA_DIRS_TO_SCAN}\n")
    
    latest_files = find_latest_evaluation_files(DATA_DIRS_TO_SCAN)
    
    if latest_files:
        print("\nFound the following latest evaluation files:")
        print("(Paths are displayed relative to project root where possible)")
        for f_path in latest_files:
            try:
                relative_path = os.path.relpath(f_path, project_root_for_output)
                print(relative_path)
            except ValueError: 
                print(f_path)
        
        move_latest_files(latest_files, FINAL_DIR)

    else:
        print("No up-to-date evaluation files could be determined, or no JSON files were found matching the patterns.") 