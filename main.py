import os
import sys
import logging
import argparse
from dotenv import load_dotenv

from core.experiment import ExperimentManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for RAG evaluation framework."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RAG Evaluation Framework')
    parser.add_argument('--config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--all', action='store_true', help='Run all configured approaches')
    parser.add_argument('--compare', nargs='+', help='Compare results from multiple experiments (by name)')
    parser.add_argument('--limit', type=int, default=10, help='Limit the number of samples to process')
    parser.add_argument('--test-data', type=str, help='Path to specific test data file')
    parser.add_argument('--cached-docs', type=str, help='Path to cached documents file')
    
    args = parser.parse_args()
    
    # Verify at least one action is specified
    if not any([args.config, args.all, args.compare]):
        parser.print_help()
        logger.error("Please specify at least one action: --config, --all, or --compare")
        sys.exit(1)
    
    # Set up experiment manager
    experiment_manager = ExperimentManager()
    
    # Run a single experiment
    if args.config:
        logger.info(f"Running experiment with config: {args.config}")
        experiment_manager.run_experiment(
            config_path=args.config,
            test_data_path=args.test_data,
            limit=args.limit,
            cached_docs_path=args.cached_docs
        )
    
    # Run all configured approaches
    if args.all:
        logger.info("Running all configured approaches")

        # Get approaches directory from base config
        approaches_dir = experiment_manager.base_config.get("data", {}).get("approaches_dir", "config/approaches") # Fallback just in case
        logger.info(f"Looking for approach configs in: {approaches_dir}")

        if os.path.exists(approaches_dir):
            config_files = [
                os.path.join(approaches_dir, f)
                for f in os.listdir(approaches_dir) 
                if f.endswith('.yaml') or f.endswith('.yml')
            ]
            
            results = {}
            for config_file in config_files:
                logger.info(f"Running experiment with config: {config_file}")
                approach_name = os.path.basename(config_file).split('.')[0]
                
                try:
                    result = experiment_manager.run_experiment(
                        config_path=config_file,
                        test_data_path=args.test_data,
                        limit=args.limit,
                        cached_docs_path=args.cached_docs
                    )
                    results[approach_name] = result
                except Exception as e:
                    logger.error(f"Error running experiment {approach_name}: {str(e)}")
            
            # Compare results if multiple experiments were run
            if len(results) > 1:
                experiment_ids = [result.get("experiment", {}).get("id", "") for result in results.values()]
                experiment_manager.compare_experiments(experiment_ids)
        else:
            logger.error(f"Approaches directory not found: {approaches_dir}")
    
    # Compare results from previous experiments
    if args.compare:
        logger.info(f"Comparing results from experiments: {args.compare}")
        experiment_manager.compare_experiments(args.compare)
    
    logger.info("Execution complete!")

if __name__ == "__main__":
    main()
