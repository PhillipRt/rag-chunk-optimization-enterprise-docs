import os
import sys
import argparse
import logging
import yaml
from dotenv import load_dotenv

# Add the project's parent directory to path to allow absolute package imports
# scripts_dir = os.path.dirname(os.path.abspath(__file__))  # e.g., /path/to/master-thesis-rag/scripts
# project_package_dir = os.path.dirname(scripts_dir)       # e.g., /path/to/master-thesis-rag
# project_container_dir = os.path.dirname(project_package_dir) # e.g., /path/to/master-thesis
# if project_container_dir not in sys.path:
#     sys.path.insert(0, project_container_dir)

# Simplified sys.path modification assuming standard structure
# project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Get the directory of the current script (e.g., .../master-thesis-rag/scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (e.g., .../master-thesis-rag)
project_root_dir = os.path.dirname(script_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from core.experiment import ExperimentManager

# --- New Logging Setup Function ---
def setup_logging(config_path='config/base_config.yaml', default_level=logging.INFO):
    """Setup logging configuration from YAML file."""
    # Construct the absolute path to the config file
    # config_path is relative to project_root_dir
    abs_config_path = os.path.join(project_root_dir, config_path)
    
    logging_settings = {}
    if os.path.exists(abs_config_path):
        with open(abs_config_path, 'rt') as f:
            try:
                config_from_yaml = yaml.safe_load(f.read())
                if config_from_yaml: # Ensure YAML was not empty
                    logging_settings = config_from_yaml.get('logging', {})
            except yaml.YAMLError as e:
                # Print to stderr so it's visible even if logging isn't fully set up
                print(f"Warning: Error parsing logging config file {abs_config_path}: {e}. Using defaults.", file=sys.stderr)
    else:
        print(f"Warning: Logging config file {abs_config_path} not found. Using defaults.", file=sys.stderr)

    log_level_str = logging_settings.get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, default_level)

    log_file_rel_path = logging_settings.get('file') # Relative to project_root_dir
    log_to_console = logging_settings.get('console', True)

    # Define formatters
    # File formatter: Detailed
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s')
    # Console formatter: More concise
    console_formatter = logging.Formatter('%(levelname)s: [%(name)s] %(message)s')

    root_logger = logging.getLogger()
    # Clear any existing handlers to prevent duplicates if this function is called multiple times
    # or if basicConfig was called before elsewhere for the root logger.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(log_level) # Set root logger level

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        # The handler's level will default to the root_logger's level if not set explicitly
        # console_handler.setLevel(log_level) 
        root_logger.addHandler(console_handler)

    if log_file_rel_path:
        # Ensure log_file_rel_path is treated as relative to project_root_dir
        log_file_abs_path = os.path.join(project_root_dir, log_file_rel_path)
        log_dir = os.path.dirname(log_file_abs_path)
        if log_dir and not os.path.exists(log_dir): # Ensure log_dir is not empty string if log file is in root
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_abs_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        # file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Silence noisy libraries by setting their log level higher than the root logger's effective level
    # This ensures their messages are suppressed unless the root level is DEBUG or lower.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING) # Pillow can be noisy

# --- End New Logging Setup Function ---

# logger = logging.getLogger(__name__) # Initialize per-module loggers after setup_logging

def main():
    """Run a comparison of different RAG approaches."""
    # Load environment variables from .env file
    load_dotenv()

    # Call setup_logging at the beginning of main
    # It will use 'config/base_config.yaml' by default
    setup_logging() 

    # Now that logging is configured, get the logger for this script
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run RAG Approach Comparison')
    parser.add_argument('--approaches', nargs='+', default=[],
                        help='Specific approaches to compare (e.g., standard multihop hybrid)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of samples to process (default: None, process all samples)')
    parser.add_argument('--test-data', type=str,
                        help='Path to specific test data file')
    # --- CORRECTED DEFAULT ---
    parser.add_argument('--cached-docs', type=str, default=None,
                        help='Path to cached documents file (e.g., cache/documents/cached_documents.pkl)')
    # --- END CORRECTION ---
    parser.add_argument('--output', type=str, default='data/evaluation',
                        help='Output directory for comparison results')
    parser.add_argument('--split-phases', action='store_true', default=True,
                        help='Run QA for all approaches first, then evaluation (allows shutting down embeddings server sooner)')

    args = parser.parse_args()

    # Set up experiment manager
    try:
        experiment_manager = ExperimentManager()
    except ValueError as e:
        logger.error(f"Failed to initialize ExperimentManager: {e}")
        sys.exit(1)

    # Find approaches to compare
    # Ensure approaches_dir is relative to project_root_dir if needed, or absolute
    # Assuming it's relative to where the script is run, or defined in config.
    # For consistency with logging config, let's make it relative to project_root_dir
    approaches_dir = os.path.join(project_root_dir, "config/approaches") 
    approach_files = {}

    if os.path.exists(approaches_dir):
        for file_name in os.listdir(approaches_dir): # Renamed file to file_name
            if file_name.endswith('.yaml') or file_name.endswith('.yml'):
                approach_name = file_name.split('.')[0]
                approach_files[approach_name] = os.path.join(approaches_dir, file_name)

    # Filter approaches if specified
    if args.approaches:
        filtered_approaches = {}
        for name in args.approaches:
            if name in approach_files:
                filtered_approaches[name] = approach_files[name]
            else:
                logger.warning(f"Approach not found: {name}")
        approach_files = filtered_approaches

    if not approach_files:
        logger.error("No approaches found to compare")
        return

    # Run each approach
    logger.info(f"Running comparison of {len(approach_files)} approaches: {', '.join(approach_files.keys())}")

    if args.split_phases:
        # Modified approach to run all QA first, then all evaluations
        logger.info("Running in split-phases mode: All QA first, then all evaluations")

        # Step 1: Run QA for all approaches
        qa_results = {}
        for name, config_path_val in approach_files.items(): # Renamed config_path to avoid conflict
            logger.info(f"Running QA for approach: {name}")
            try:
                # Use limit=0 to process all samples if args.limit is None
                limit_value = args.limit if args.limit is not None else 0
                qa_result = experiment_manager.run_qa_only(
                    config_path=config_path_val, # Use renamed variable
                    test_data_path=args.test_data,
                    limit=limit_value,
                    cached_docs_path=args.cached_docs
                )
                qa_results[name] = qa_result # Store None if QA failed
            except Exception as e:
                logger.error(f"Error running QA for {name}: {str(e)}", exc_info=True)
                qa_results[name] = None # Mark as failed

        logger.info("✅ All QA operations completed. You can now shut down your embeddings server if it was remote.")

        # Step 2: Run evaluation for all approaches
        results = {}
        experiment_ids = []

        for name, qa_result_data in qa_results.items():
            if qa_result_data is None: # Skip if QA phase failed
                logger.warning(f"Skipping evaluation for {name} as QA phase failed or produced no result.")
                continue
            logger.info(f"Running evaluation for approach: {name}")
            try:
                # Run just the evaluation part
                result = experiment_manager.run_evaluation_only(
                    qa_result=qa_result_data,
                    config_path=approach_files[name] # Still need path for evaluator config if merged_config missing
                )
                if result and "error" not in result: # Check if evaluation succeeded
                    results[name] = result
                    if "experiment" in result and "id" in result["experiment"]:
                        experiment_ids.append(result["experiment"]["id"])
                else:
                     logger.error(f"Evaluation phase failed for {name}.")
            except Exception as e:
                logger.error(f"Error running evaluation for {name}: {str(e)}", exc_info=True)
    else:
        # Original approach, running QA and evaluation sequentially for each approach
        results = {}
        experiment_ids = []

        for name, config_path_val in approach_files.items(): # Renamed config_path
            logger.info(f"Running approach: {name}")
            try:
                # Use limit=0 to process all samples if args.limit is None
                limit_value = args.limit if args.limit is not None else 0
                result = experiment_manager.run_experiment(
                    config_path=config_path_val, # Use renamed variable
                    test_data_path=args.test_data,
                    limit=limit_value,
                    cached_docs_path=args.cached_docs
                )
                if result and "error" not in result: # Check if experiment succeeded
                    results[name] = result
                    if "experiment" in result and "id" in result["experiment"]:
                        experiment_ids.append(result["experiment"]["id"])
                else:
                    logger.error(f"Experiment run failed for {name}.")
            except Exception as e:
                logger.error(f"Error running experiment {name}: {str(e)}", exc_info=True)

    # Generate comparison report
    if len(experiment_ids) > 1: # Check based on actual successful experiment IDs
        logger.info("Generating comparison report...")
        comparison = experiment_manager.compare_experiments(experiment_ids)

        # Print comparison summary
        if comparison:
            print("\n\n=== RAG Approaches Comparison ===")
            print("\nMetrics:")

            metrics_comparison_data = comparison.get("metrics_comparison_table", {})
            for metric_name, values in metrics_comparison_data.items():
                print(f"\n{metric_name}:")
                # Access the score directly as stored in compare_experiments
                for approach, score_value in values.items():
                    if score_value is not None:
                         print(f"  {approach}: {score_value:.4f}")
                    else:
                         print(f"  {approach}: N/A") # Indicate missing score

            # Use the evaluation_dir from the experiment_manager's data_manager
            # This ensures the correct path is reported if data_manager changes it.
            output_dir = experiment_manager.data_manager.evaluation_dir 
            print(f"\nDetailed comparison results saved to: {os.path.join(project_root_dir, output_dir)}")
        else:
            logger.warning("Comparison generation failed or produced no data.")
    elif len(results) == 1: # Changed from len(results) > 0 to len(results) == 1 for clarity
        logger.info("Only one successful experiment result. No comparison report generated.")
    else: # Handles len(results) == 0
        logger.warning("Not enough successful experiments to generate comparison report.")

if __name__ == "__main__":
    main()