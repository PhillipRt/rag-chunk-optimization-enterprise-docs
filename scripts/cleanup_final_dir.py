#!/usr/bin/env python3
"""
Scans the 'final' evaluation directory to find the latest version of each experiment
and removes all other older, duplicate versions to ensure a clean dataset.
"""
import os
import glob
import re
from collections import defaultdict

# --- Configuration ---
FINAL_DIR = "data/evaluation/final"
# --- End Configuration ---

def get_experiment_base_name(filename: str) -> str:
    """
    Extracts the unique base name for an experiment run by stripping off
    any rerun information and the file extension.
    Example: '..._20250606_202127_rerun_v1_20250607_122316.json'
             -> '..._20250606_202127'
    """
    base, _ = os.path.splitext(filename)
    match = re.match(r"(.+?)(_rerun.*)", base)
    if match:
        return match.group(1)
    return base

def get_sort_key(filename: str) -> str:
    """
    Creates a sort key from the filename's timestamp to determine the latest version.
    Example: '..._rerun_v1_20250607_122316.json' -> '20250607122316'
    """
    # This regex finds the last timestamp in the filename
    timestamps = re.findall(r"(\d{8}_\d{6})", filename)
    if timestamps:
        # The latest timestamp is the one we care about
        return timestamps[-1]
    return "0" # Default for files without a timestamp

def cleanup_final_directory():
    """
    Scans the final directory, identifies the latest version of each experiment,
    and removes all other older versions.
    """
    if not os.path.isdir(FINAL_DIR):
        print(f"Error: Directory '{FINAL_DIR}' not found.")
        return

    all_files_in_dir = glob.glob(os.path.join(FINAL_DIR, "*"))
    all_files = [f for f in all_files_in_dir if os.path.isfile(f)]

    # Group files by their base experiment name
    experiments = defaultdict(list)
    for f_path in all_files:
        base_name = get_experiment_base_name(os.path.basename(f_path))
        experiments[base_name].append(f_path)

    print(f"Scanning {len(all_files)} files across {len(experiments)} unique experiments in '{FINAL_DIR}'...")

    files_to_delete = []
    for base_name, file_list in experiments.items():
        if len(file_list) <= 1:
            continue  # No duplicates to clean

        # Find the timestamp of the latest version in the group
        latest_timestamp = "0"
        for f_path in file_list:
            current_timestamp = get_sort_key(os.path.basename(f_path))
            if current_timestamp > latest_timestamp:
                latest_timestamp = current_timestamp

        # Add any file that does not have the latest timestamp to the deletion list
        for f_path in file_list:
            if get_sort_key(os.path.basename(f_path)) != latest_timestamp:
                files_to_delete.append(f_path)

    if not files_to_delete:
        print("\nDirectory is already clean. No older versions found to remove.")
        return

    # --- Preview and Confirm Deletion ---
    print(f"\n{'='*60}")
    print("CLEANUP PREVIEW - The following OLDER files will be DELETED:")
    print(f"{'='*60}\n")
    for f_path in sorted(files_to_delete):
        print(f"  - {os.path.basename(f_path)}")
    
    print(f"\nTotal files to be deleted: {len(files_to_delete)}")
    print(f"{'='*60}")
    
    while True:
        try:
            response = input("Proceed with deleting these files? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                break
            elif response in ['n', 'no']:
                print("\nCleanup operation cancelled by user.")
                return
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
        except KeyboardInterrupt:
            print("\nCleanup operation cancelled by user.")
            return

    # --- Execute Deletion ---
    print("\nProceeding with deletion...")
    deleted_count = 0
    for f_path in files_to_delete:
        try:
            os.remove(f_path)
            # print(f"  Deleted: {os.path.basename(f_path)}") # Uncomment for verbose output
            deleted_count += 1
        except Exception as e:
            print(f"  ERROR deleting {os.path.basename(f_path)}: {e}")
            
    print(f"\nCleanup complete. Deleted {deleted_count} older file version(s).")
    print(f"Your '{FINAL_DIR}' directory is now clean.")

if __name__ == "__main__":
    cleanup_final_directory() 