import json
import glob # Added glob for file scanning
import os
import logging
import yaml
import asyncio
import math
import traceback
import re # Added for filename parsing
from dotenv import load_dotenv
from typing import Any, List, Dict, cast, Coroutine, Set, Optional # Added Optional
from typing_extensions import override

# Add parent directory to path to allow importing modules
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Assuming 'master_thesis_rag' is the top-level package for absolute imports
from core.evaluation import RagEvaluator # Reverted to original import style
from ragas import SingleTurnSample
from ragas.metrics.base import SingleTurnMetric, MetricWithLLM

# Configure logging - This might be overwritten by a setup_logging function if called from another script
# For standalone execution, this basicConfig is useful.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
EVALUATION_DIR = "data/evaluation/"
RERUN_DIR = os.path.join(EVALUATION_DIR, "reruns")
BASE_CONFIG_PATH = "config/base_config.yaml"
# Define patterns for evaluation files to scan
EVAL_FILE_PATTERNS = ["evaluation_results_*.json", "*_rerun_v*.json"]

METRIC_NAME_MAPPING: Dict[str, str] = {
    "answer_accuracy": "nv_accuracy"
}

os.makedirs(RERUN_DIR, exist_ok=True)
# --- End Configuration ---

class SafeNanEncoder(json.JSONEncoder):
    @override
    def default(self, o: Any) -> Any | None:
        if isinstance(o, float) and math.isnan(o):
            return None
        return super(SafeNanEncoder, self).default(o)

def load_config(config_path: str) -> dict[str, Any]:
    if not isinstance(config_path, str) or not config_path:
        logger.error("Invalid config path provided.")
        return {}
    try:
        if not os.path.exists(config_path):
            logger.error(f"Config file path does not exist: {config_path}")
            return {}
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if isinstance(config, dict) else {}
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}", exc_info=True)
        return {}

# Worker coroutine that respects the semaphore
async def execute_metric_with_semaphore(
    semaphore: asyncio.Semaphore,
    metric_object: SingleTurnMetric,
    sample_object: SingleTurnSample,
    sample_idx: int, # for logging
    metric_name_to_save: str # for logging
) -> Any:
    async with semaphore:
        logger.debug(f"Semaphore acquired for [SampleIdx {sample_idx}][{metric_name_to_save}]")
        try:
            # Initial attempt
            score = await metric_object.single_turn_ascore(sample_object)
            # Check for NaN from successful call, as Ragas metrics might return NaN for valid reasons
            if isinstance(score, float) and math.isnan(score):
                 logger.warning(f"Initial attempt for [SampleIdx {sample_idx}][{metric_name_to_save}] resulted in NaN.")
                 # We will let the main gather loop handle this as a form of initial failure to trigger retry if needed
                 # Or, we could raise a specific exception here if we want retry logic to be more contained.
                 # For now, returning NaN will make it look like a failure that might be retried if it matches error strings.
                 # This might not be ideal. Let's consider NaN from a metric as a valid (though possibly uninformative) result for now.
                 # The main loop will log it as "FAIL (NaN/None from successful call)"
            return score
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Initial attempt FAIL [SampleIdx {sample_idx}][{metric_name_to_save}]: {err_msg}")
            # Retry logic (only for specific, potentially transient errors)
            if any(e_str in err_msg for e_str in ["Rate limit", "LLM call failed", "429", "timeout", "APIConnectionError", "JSONDecodeError", "Expecting value"]):
                logger.warning(f"RETRYING [SampleIdx {sample_idx}][{metric_name_to_save}] after 2s delay...")
                await asyncio.sleep(2)
                try:
                    retry_val = await metric_object.single_turn_ascore(sample_object)
                    if isinstance(retry_val, float) and math.isnan(retry_val):
                        logger.warning(f"Retry attempt for [SampleIdx {sample_idx}][{metric_name_to_save}] resulted in NaN.")
                    return retry_val # Return even if NaN, to be handled by main loop
                except Exception as retry_e:
                    logger.error(f"Retry ERROR [SampleIdx {sample_idx}][{metric_name_to_save}]: {retry_e}", exc_info=False)
                    return retry_e # Return the exception from retry
            return e # Return original exception if not retried or retry failed

async def update_evaluation_files(eval_dir: str, rerun_dir: str, base_config: dict[str, Any]):
    """
    Scans evaluation files, reruns/adds metrics based on current config,
    and saves updated results to a new versioned file in the rerun_dir.
    """
    search_paths: List[str] = []
    for pattern in EVAL_FILE_PATTERNS:
        search_paths.append(os.path.join(eval_dir, pattern))
        search_paths.append(os.path.join(rerun_dir, pattern)) # Also scan existing rerun files

    all_found_files: Set[str] = set()
    for sp in search_paths:
        found = glob.glob(sp)
        # Filter out any directories that might accidentally match
        all_found_files.update(f for f in found if os.path.isfile(f))

    if not all_found_files:
        logger.info(f"No evaluation files found in {eval_dir} or {rerun_dir} matching patterns: {EVAL_FILE_PATTERNS}. Nothing to do.")
        return

    logger.info(f"Found {len(all_found_files)} unique evaluation files to process.")
    total_success_count = 0
    total_persistent_failure_count = 0

    MAX_CONCURRENT_LLM_SEMAPHORE = 40 # Define desired concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_SEMAPHORE)

    for original_json_path in all_found_files:
        filename_in_scan = os.path.basename(original_json_path)
        logger.info(f"\n--- Processing file: {filename_in_scan} ---")
        # Initialize approach_config_path here to ensure it has a value in all branches
        approach_config_path: Optional[str] = None

        try:
            with open(original_json_path, 'r', encoding='utf-8') as f:
                original_data: Dict[str, Any] = json.load(f)

            if not isinstance(original_data, dict) or "results" not in original_data:
                logger.error(f"Invalid structure in {filename_in_scan} (not a dict or missing 'results'). Skipping.")
                continue
            results_list_from_file = original_data.get("results")
            if not isinstance(results_list_from_file, list):
                logger.error(f"'results' key in {filename_in_scan} is not a list. Skipping.")
                continue
            if not results_list_from_file:
                logger.info(f"{filename_in_scan} has an empty 'results' list. Skipping processing for this file.")
                continue

            experiment_meta: Dict[str, Any] = original_data.get("experiment", {})
            approach_config_path_from_json = experiment_meta.get("config_path")
            approach_name_from_json = experiment_meta.get("name", "UnknownApproach")

            if not approach_config_path_from_json:
                logger.warning(f"'experiment.config_path' not found in {filename_in_scan}. Attempting to infer from filename...")
                inferred_approach_name = None
                # New regex: capture content between "evaluation_results_" and the first "_YYYYMMDD_HHMMSS"
                # This assumes the original timestamp (when the approach was first run)
                # is always present and marks the end of the base approach name segment in the filename.
                match_name_and_orig_ts = re.match(r"^evaluation_results_(.+?)_(\d{8}_\d{6})", filename_in_scan)
                if match_name_and_orig_ts:
                    inferred_approach_name = match_name_and_orig_ts.group(1)
                    # original_timestamp_from_filename = match_name_and_orig_ts.group(2) # For logging or future use
                    logger.info(f"Extracted base for approach name inference: '{inferred_approach_name}' (from filename up to first timestamp pattern)")
                # No else needed, inferred_approach_name remains None if no match

                if inferred_approach_name:
                    # logger.info(f"Inferred base approach name from filename: {inferred_approach_name}") # Already logged above with more context

                    # Attempt to find the corresponding YAML file, trying variations
                    possible_config_names = [
                        f"{inferred_approach_name}.yaml",                                      # Exact match
                        f"{inferred_approach_name.replace('_rag', '')}.yaml",                    # Try removing _rag
                        f"{inferred_approach_name.replace('_rag', '').replace('bge_m3', 'bge-m3')}.yaml", # Handle bge_m3 -> bge-m3
                        f"{inferred_approach_name.replace('bge_m3', 'bge-m3')}.yaml"                   # Handle bge_m3 -> bge-m3 if _rag was not present
                    ]
                    # Remove duplicates if any, maintaining order for preference
                    unique_possible_names = list(dict.fromkeys(possible_config_names))

                    found_config_path_str = None
                    for config_try_name in unique_possible_names:
                        potential_path = os.path.join(project_root, "config", "approaches", config_try_name)
                        if os.path.exists(potential_path):
                            approach_config_path_from_json = os.path.join("config", "approaches", config_try_name) # Store relative path for consistency if needed
                            found_config_path_str = potential_path # Store the successfully found absolute path
                            logger.info(f"Found matching config file: {approach_config_path_from_json}")
                            break

                    if not found_config_path_str:
                        logger.error(f"Could not find a matching config file for inferred base name '{inferred_approach_name}' (tried: {unique_possible_names}). Skipping {filename_in_scan}.")
                        continue
                    else:
                        approach_config_path = found_config_path_str # Assign to the main variable
                else:
                    logger.error(f"Could not infer approach name from filename {filename_in_scan} (pattern 'evaluation_results_(NAME)_TIMESTAMP_...' not matched). Skipping.")
                    continue # This continue is for the `if not approach_config_path_from_json:` block

            if approach_config_path is None and approach_config_path_from_json : # If inferred path was used, approach_config_path is already set
                 temp_path = str(approach_config_path_from_json)
                 if not os.path.isabs(temp_path):
                     approach_config_path = os.path.join(project_root, temp_path)
                 else:
                     approach_config_path = temp_path

            if approach_config_path is None or not os.path.exists(approach_config_path):
                logger.error(f"Final approach config path is invalid or not found: {approach_config_path}. Skipping {filename_in_scan}.")
                continue
            
            logger.info(f"Using approach config path: {approach_config_path}")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {original_json_path}: {e}. Skipping.", exc_info=True)
            continue
        except Exception as e:
            logger.error(f"Error loading original data or finding config path for {original_json_path}: {e}. Skipping.", exc_info=True)
            continue

        try:
            approach_config_loaded = load_config(approach_config_path) # approach_config_path is now guaranteed to be defined and exist
            if not approach_config_loaded:
                logger.error(f"Failed to load approach config from {approach_config_path}. Skipping {filename_in_scan}.")
                continue
            
            current_evaluator_config: Dict[str, Any] = {**base_config}
            for key, value in approach_config_loaded.items(): 
                if isinstance(value, dict) and isinstance(current_evaluator_config.get(key), dict):
                    current_evaluator_config[key] = {**current_evaluator_config[key], **value}
                else:
                    current_evaluator_config[key] = value
            
            for key in ["DEEPSEEK_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT_ID", "GOOGLE_CLOUD_LOCATION", "RAGAS_APP_TOKEN", "OPENAI_API_KEY", "COHERE_API_KEY"]:
                env_value = os.environ.get(key)
                if env_value:
                    if key not in current_evaluator_config or current_evaluator_config.get(key) is None:
                        current_evaluator_config[key] = env_value
                    # Also check within 'evaluation' and 'retriever' sections if they exist
                    for nested_section_name in ['evaluation', 'retriever', 'generation']:
                        nested_section = current_evaluator_config.get(nested_section_name)
                        if isinstance(nested_section, dict) and (key not in nested_section or nested_section.get(key) is None):
                            nested_section[key] = env_value

            current_config_metrics_list_raw = current_evaluator_config.get('evaluation', {}).get('metrics', [])
            current_config_metrics_list = []
            if isinstance(current_config_metrics_list_raw, list):
                for m_item in current_config_metrics_list_raw:
                    if isinstance(m_item, str):
                        current_config_metrics_list.append(m_item)
                    elif isinstance(m_item, dict) and 'name' in m_item and isinstance(m_item['name'], str): # Handle dict metric definition
                        # Reconstruct name with params if any, e.g., "metric_name(param=value)"
                        metric_name_with_params = m_item['name']
                        params_str_parts = []
                        for p_key, p_val in m_item.items():
                            if p_key != 'name':
                                params_str_parts.append(f"{p_key}={p_val}")
                        if params_str_parts:
                            metric_name_with_params = f"{metric_name_with_params}({', '.join(params_str_parts)})"
                        current_config_metrics_list.append(metric_name_with_params)
                    else:
                        logger.warning(f"Invalid metric definition found in config: {m_item}. Skipping.")
            else:
                logger.warning(f"Metrics in config is not a list: {current_config_metrics_list_raw}. Assuming no metrics from config.")

            # --- Determine metrics for RagEvaluator initialization (always use base names) ---
            metrics_for_evaluator_init: Set[str] = set() # Re-initialize or use the one from above if it's meant to accumulate. Assuming re-init.
            # 1. Add all base metrics mentioned in the data_to_process file (from 'results' keys or 'average_scores' keys)
            if results_list_from_file and isinstance(results_list_from_file, list) and results_list_from_file:
                # Check all samples for metric keys, not just the first one
                all_sample_keys = set()
                for r_item in results_list_from_file:
                    if isinstance(r_item, dict):
                        all_sample_keys.update(r_item.keys())

                for key in all_sample_keys:
                    if key not in ["user_input", "question", "response", "answer",
                                   "retrieved_contexts", "contexts", "reference", "ground_truth",
                                   "retrieved_docs", "retrieved_docs_content", "detailed_results", "metrics", "average_scores", "episode_done", "id"]: # Added common non-metric keys
                        base_metric_name = key.split("(")[0]
                        mapped_base_name = METRIC_NAME_MAPPING.get(base_metric_name, base_metric_name)
                        if mapped_base_name:
                            metrics_for_evaluator_init.add(mapped_base_name)

            if original_data.get("average_scores") and isinstance(original_data["average_scores"], dict):
                for key in original_data["average_scores"].keys():
                    base_metric_name = key.split("(")[0]
                    mapped_base_name = METRIC_NAME_MAPPING.get(base_metric_name, base_metric_name)
                    if mapped_base_name:
                        metrics_for_evaluator_init.add(mapped_base_name)

            # 2. Add all base metrics from the current configuration
            for cfg_metric_name in current_config_metrics_list:
                base_cfg_metric_name = cfg_metric_name.split("(")[0]
                mapped_base_cfg_name = METRIC_NAME_MAPPING.get(base_cfg_metric_name, base_cfg_metric_name) # Map even config names if they use an alias
                if mapped_base_cfg_name: # Ensure it's not None before adding
                    metrics_for_evaluator_init.add(mapped_base_cfg_name) # Add the mapped base name
            # --- End Determine metrics for RagEvaluator init ---

            logger.info(f"Metrics to initialize for RagEvaluator for {filename_in_scan}: {sorted(list(metrics_for_evaluator_init)) or 'None'}")

            # We pass the original configured metric names (with params) to RagEvaluator,
            # as it knows how to parse them. metrics_for_evaluator_init was for finding all *base* capabilities needed.
            eval_section_for_setup = current_evaluator_config.setdefault('evaluation', {})
            if isinstance(eval_section_for_setup, dict):
                eval_section_for_setup['metrics'] = current_config_metrics_list # Use the full names from config
            else:
                logger.error("Evaluation section in config is not a dictionary. Skipping file.")
                continue

            # Remove any problematic top-level keys that might interfere if RagEvaluator doesn't expect them
            current_evaluator_config.pop('answer_similarity', None) # Example, if this was ever a top-level key

            evaluator = RagEvaluator(current_evaluator_config) # Reverted to original usage
            evaluator.setup()

            metric_objects: Dict[str, SingleTurnMetric] = {}
            if evaluator.metrics: # evaluator.metrics should contain the instantiated Ragas metric objects
                for m_obj in evaluator.metrics:
                    if hasattr(m_obj, 'name') and isinstance(m_obj, SingleTurnMetric):
                        # Ragas metric objects have a .name which might include (params...)
                        # Or it might be the base name if no params were used or if it's an alias.
                        # We need to key metric_objects by the name RagEvaluator/Ragas uses internally.
                        # This name should match what's in current_config_metrics_list.
                        metric_objects[m_obj.name] = m_obj

            if not metric_objects and current_config_metrics_list :
                logger.error(f"No metric objects were successfully initialized by RagEvaluator for {filename_in_scan}, but current config lists metrics: {current_config_metrics_list}. Skipping.")
                continue
            logger.info(f"Initialized evaluator for {approach_name_from_json}. Available metric objects (keyed by Ragas name): {list(metric_objects.keys())}")

        except Exception as e:
            logger.error(f"Failed to initialize RagEvaluator for {approach_name_from_json}: {e}", exc_info=True)
            continue

        file_success_count, file_persistent_failure_count, needs_saving_for_file = 0, 0, False
        modified_results_list = [dict(s) for s in results_list_from_file]

        # --- Batching setup ---
        tasks_for_current_file: list[asyncio.Task[Any]] = []
        task_info_map: Dict[int, Dict[str, Any]] = {} # Map task id to its info
        # --- End Batching setup ---

        for sample_idx, sample_dict in enumerate(modified_results_list):
            if not isinstance(sample_dict, dict):
                logger.warning(f"    Data at index {sample_idx} in {filename_in_scan} is not a dictionary. Skipping.")
                continue

            metrics_to_calculate_for_this_sample: Set[str] = set()

            # A. Identify metrics from current config that need calculation for this sample
            for metric_name_from_config in current_config_metrics_list: # Full name, e.g., "nv_accuracy" or "factual_correctness(mode=f1)"
                base_metric_name_from_config = metric_name_from_config.split("(")[0]

                found_exact_config_metric_in_sample = False
                value_of_exact_config_metric_is_null = False

                # Check if the exact metric_name_from_config exists in the sample
                if metric_name_from_config in sample_dict:
                    found_exact_config_metric_in_sample = True
                    value_in_sample = sample_dict[metric_name_from_config]
                    if value_in_sample is None or (isinstance(value_in_sample, float) and math.isnan(value_in_sample)):
                        value_of_exact_config_metric_is_null = True

                if value_of_exact_config_metric_is_null:
                    # Exact metric from config is in sample but null/NaN, so add it for rerun.
                    metrics_to_calculate_for_this_sample.add(metric_name_from_config)
                    logger.debug(f"  Sample {sample_idx}: Exact metric {metric_name_from_config} is null/NaN, adding for rerun.")
                elif not found_exact_config_metric_in_sample:
                    # Exact metric_name_from_config is not in the sample at all.
                    # We need to check if any *other* key in the sample (after mapping its base) corresponds to this config metric's base.
                    # If no such key exists with a non-null value, then the config metric is considered missing.
                    is_covered_by_other_key = False
                    for key_in_sample, value_in_sample in sample_dict.items():
                        if key_in_sample == metric_name_from_config: continue # Already handled

                        base_key_in_sample_original = key_in_sample.split("(")[0]
                        mapped_base_key_in_sample = METRIC_NAME_MAPPING.get(base_key_in_sample_original, base_key_in_sample_original)

                        if mapped_base_key_in_sample == base_metric_name_from_config:
                            # A metric with the same (mapped) base name exists in the sample.
                            if not (value_in_sample is None or (isinstance(value_in_sample, float) and math.isnan(value_in_sample))):
                                # And it has a non-null value. So, the concept is covered.
                                # We assume the user wants the specific version from the config if the exact name wasn't found.
                                # However, if the config is just "factual_correctness" and sample has "factual_correctness(mode=simple)" with a value,
                                # we might NOT want to rerun "factual_correctness" if the user implicitly accepts the existing param version.
                                # For now, if exact config metric is missing, but a compatible base is present with value, we assume it might need updating to config's specific form if different.
                                # This part is tricky: if config says "m" and sample has "m(p=1)" with value, do we run "m"?
                                # Let's assume for now: if exact config metric is missing, it gets added unless another variant *already forced its addition*.
                                is_covered_by_other_key = True
                                break

                    if not is_covered_by_other_key:
                        metrics_to_calculate_for_this_sample.add(metric_name_from_config)
                        logger.debug(f"  Sample {sample_idx}: Metric {metric_name_from_config} (or its base) not found with a value, adding for calculation.")
                # If found_exact_config_metric_in_sample and not value_of_exact_config_metric_is_null, it has a value, so do nothing for this config metric.

            # B. Add any null/NaN metrics from the sample if their (mapped) base name corresponds to a base name in the current config list.
            # This ensures we re-calculate using the *config's version* of that metric.
            for key_in_sample, value_in_sample in sample_dict.items():
                if value_in_sample is None or (isinstance(value_in_sample, float) and math.isnan(value_in_sample)):
                    base_key_in_sample_original = key_in_sample.split("(")[0]
                    mapped_base_key_in_sample = METRIC_NAME_MAPPING.get(base_key_in_sample_original, base_key_in_sample_original)

                    for metric_name_from_config in current_config_metrics_list:
                        base_metric_name_from_config = metric_name_from_config.split("(")[0]
                        if mapped_base_key_in_sample == base_metric_name_from_config:
                            # The null sample metric corresponds to a metric in the current config.
                            # Add the config's version (full name with params) to ensure we use the right settings.
                            if metric_name_from_config not in metrics_to_calculate_for_this_sample:
                                metrics_to_calculate_for_this_sample.add(metric_name_from_config)
                                logger.debug(f"  Sample {sample_idx}: Existing null metric '{key_in_sample}' matches base of config metric '{metric_name_from_config}', adding '{metric_name_from_config}' for rerun.")
                            break # Found the corresponding config metric

            if not metrics_to_calculate_for_this_sample:
                logger.info(f"  Sample Index {sample_idx}: No metrics require calculation/rerun.")
                continue

            logger.info(f"  Sample Index {sample_idx}: Metrics to calculate/rerun: {list(metrics_to_calculate_for_this_sample)}")

            try:
                user_input_val = str(sample_dict.get("user_input", sample_dict.get("question", "")))
                response_val = str(sample_dict.get("response", sample_dict.get("answer", "")))
                retrieved_contexts_raw_val = sample_dict.get("retrieved_contexts", sample_dict.get("contexts", []))
                reference_val = sample_dict.get("reference", sample_dict.get("ground_truth"))
                if reference_val is not None: reference_val = str(reference_val)

                contexts_list_val: list[str] = []
                if isinstance(retrieved_contexts_raw_val, list):
                    processed_items = []
                    for item_idx, item in enumerate(retrieved_contexts_raw_val):
                        content = item.get('content', item.get('page_content')) if isinstance(item, dict) else item
                        if isinstance(content, str): processed_items.append(content)
                        elif content is not None: logger.warning(f"Context item {item_idx} non-string content. Skipping.")
                    contexts_list_val = processed_items
                elif retrieved_contexts_raw_val is not None:
                    logger.warning(f"Contexts for index {sample_idx} not a list. Treating as empty.")

                single_turn_sample_for_calc = SingleTurnSample(
                    user_input=user_input_val, response=response_val,
                    retrieved_contexts=contexts_list_val, reference=reference_val,
                )
            except Exception as e_sts:
                logger.error(f"Error preparing SingleTurnSample for index {sample_idx}: {e_sts}. Skipping sample.", exc_info=True)
                continue

            for metric_name_to_run in metrics_to_calculate_for_this_sample:
                base_metric_to_run = metric_name_to_run.split("(")[0]
                mapped_base_for_object = METRIC_NAME_MAPPING.get(base_metric_to_run, base_metric_to_run)

                metric_object = metric_objects.get(mapped_base_for_object) # This was one potential source of issues: it should look up using metric_name_to_run, not mapped_base_for_object if metric_objects is keyed by full name
                # Corrected assumption: metric_objects IS keyed by the Ragas name which might include params, so using metric_name_to_run for lookup is better.
                # However, the RagEvaluator initialization populates metric_objects using m_obj.name. If Ragas m_obj.name is just the base name (e.g., 'nv_accuracy')
                # even if config said 'nv_accuracy(mode=strict)', then using mapped_base_for_object is correct.
                # Given the setup logic for metric_objects: `metric_objects[m_obj.name] = m_obj`
                # And `m_obj.name` for Ragas is typically the base name.
                # So, let's stick to `metric_objects.get(mapped_base_for_object)` for lookup.
                # The `metric_name_to_run` is the key that should be *saved* in the results, reflecting the configuration.

                # Re-evaluating lookup: `metric_objects` should be keyed by the names Ragas uses.
                # If `current_config_metrics_list` contains `factual_correctness(mode=f1)`
                # and RagEvaluator initializes a metric whose `.name` is `factual_correctness(mode=f1)`,
                # then `metric_objects` will have `{"factual_correctness(mode=f1)": <obj>}`.
                # In this case, `metric_objects.get(metric_name_to_run)` is correct.
                # The original log for metric_object not found: "Metric object for '{mapped_base_for_object}' ... Ragas name '{metric_name_to_run}'"
                # This implies that the original intention was likely to look up by full name (metric_name_to_run).

                metric_object_to_use = metric_objects.get(metric_name_to_run) # Try full name first
                if metric_object_to_use is None:
                    metric_object_to_use = metric_objects.get(mapped_base_for_object) # Fallback to mapped base name

                if metric_object_to_use is None:
                    logger.warning(f"Metric object for Ragas name '{metric_name_to_run}' (or base '{mapped_base_for_object}') not found in initialized metric_objects for index {sample_idx}. Skipping this metric for this sample.")
                    continue

                if isinstance(metric_object_to_use, SingleTurnMetric):
                    task = asyncio.create_task(execute_metric_with_semaphore(semaphore, metric_object_to_use, single_turn_sample_for_calc, sample_idx, metric_name_to_run))
                    tasks_for_current_file.append(task)
                    task_info_map[id(task)] = {"s_idx": sample_idx, "m_name": metric_name_to_run} # Store info using task id
                else:
                    logger.warning(f"Metric '{metric_name_to_run}' (base: '{mapped_base_for_object}') is not a SingleTurnMetric. Skipping.")

            # Removed the per-sample asyncio.gather here. It will be done in batches later.

        # --- After iterating all samples, process all accumulated tasks for the file in batches ---
        if not tasks_for_current_file:
            logger.info(f"No metric calculations were required for any sample in {filename_in_scan}.")
        else:
            logger.info(f"Processing a total of {len(tasks_for_current_file)} metric calculations for {filename_in_scan} in batches of up to {MAX_CONCURRENT_LLM_SEMAPHORE}.")

            results_from_gather = await asyncio.gather(*tasks_for_current_file, return_exceptions=True)

            for i, res_exc in enumerate(results_from_gather):
                task_id = id(tasks_for_current_file[i])
                task_specific_info = task_info_map.get(task_id)
                if not task_specific_info:
                    logger.error(f"Could not find info for completed task {task_id}. This should not happen.")
                    continue
                
                s_idx_res = task_specific_info["s_idx"]
                m_name_save = task_specific_info["m_name"]

                if isinstance(res_exc, BaseException):
                    err_msg_str = str(res_exc)
                    problematic_llm_output = ""
                    if isinstance(res_exc, json.JSONDecodeError):
                        problematic_llm_output = f" Problematic text for JSONDecodeError: '{res_exc.doc}'."
                    elif "Expecting value" in err_msg_str:
                        problematic_llm_output = " (LLM output was likely malformed JSON/not parsable)."
                    logger.warning(f"FAIL [Sample {s_idx_res}][{m_name_save}]: {err_msg_str}.{problematic_llm_output}")
                    modified_results_list[s_idx_res][m_name_save] = None
                    file_persistent_failure_count += 1; needs_saving_for_file = True
                else:
                    score = res_exc
                    if isinstance(score, (int, float)) and not math.isnan(score):
                        logger.info(f"SUCCESS [Sample {s_idx_res}][{m_name_save}]: {score:.4f}")
                        modified_results_list[s_idx_res][m_name_save] = score

                        # --- Start: Added consolidation logic ---
                        # After a successful calculation, remove other variants of the same metric
                        base_name_of_saved_metric = m_name_save.split('(')[0]
                        keys_to_remove = [
                            k for k in modified_results_list[s_idx_res]
                            if k != m_name_save and k.split('(')[0] == base_name_of_saved_metric
                        ]
                        if keys_to_remove:
                            logger.info(f"    Consolidating metric '{base_name_of_saved_metric}': removing old keys {keys_to_remove}")
                            for key in keys_to_remove:
                                del modified_results_list[s_idx_res][key]
                        # --- End: Added consolidation logic ---

                        file_success_count += 1; needs_saving_for_file = True
                    else:
                        logger.warning(f"NaN/None score after potential retry [Sample {s_idx_res}][{m_name_save}]. Skipping this metric for this sample.")
                        modified_results_list[s_idx_res][m_name_save] = None
                        file_persistent_failure_count += 1; needs_saving_for_file = True # Count as persistent failure if NaN after retry path

        if needs_saving_for_file:
            try:
                import datetime
                current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name_orig, ext_orig = os.path.splitext(filename_in_scan)
                base_name_no_rerun = base_name_orig.split("_rerun_v")[0]

                version = 1
                while True:
                    rerun_fn_base = f"{base_name_no_rerun}_rerun_v{version}_{current_timestamp}"
                    new_json_path = os.path.join(RERUN_DIR, f"{rerun_fn_base}{ext_orig}")
                    if not os.path.exists(new_json_path): break
                    version += 1

                logger.info(f"Saving updated data to new file: {new_json_path}")
                original_data["results"] = modified_results_list # Update the results in the original data structure
                with open(new_json_path, 'w', encoding='utf-8') as f:
                    json.dump(original_data, f, indent=2, cls=SafeNanEncoder)

                # Save updated CSV
                csv_results_dir_path = os.path.join(eval_dir, "csv_results") # Original eval_dir for structure
                original_csv_path = os.path.join(csv_results_dir_path, f"{base_name_no_rerun}.csv")

                if os.path.exists(original_csv_path) or True: # Always try to save CSV if we have data
                    try:
                        import pandas as pd
                        if modified_results_list and all(isinstance(item, dict) for item in modified_results_list):
                            results_df = pd.DataFrame(modified_results_list)
                            new_csv_path = os.path.join(RERUN_DIR, f"{rerun_fn_base}.csv") # Save to RERUN_DIR
                            results_df.to_csv(new_csv_path, index=False)
                            logger.info(f"Saved updated CSV results to: {new_csv_path}")
                        else:
                            logger.warning(f"Results list for {filename_in_scan} is empty/invalid. Skipping CSV.")
                    except ImportError: logger.warning("Pandas not installed. Cannot save updated CSV.")
                    except Exception as csv_e: logger.warning(f"Could not save CSV for {filename_in_scan}: {csv_e}")
                # else: logger.info(f"Original CSV for '{base_name_no_rerun}' not found. No CSV updated.")

            except Exception as e_save:
                logger.error(f"Error saving updated data for {filename_in_scan}: {e_save}", exc_info=True)

        logger.info(f"--- Finished file: {filename_in_scan} | Success: {file_success_count} | Persistent Failures: {file_persistent_failure_count} ---")
        total_success_count += file_success_count
        total_persistent_failure_count += file_persistent_failure_count

    print("=" * 60)
    logger.info("Update Evaluation Files Script Finished.")
    logger.info(f"Total successful recalculations/additions: {total_success_count}")
    logger.info(f"Total persistent failures/errors: {total_persistent_failure_count}")
    print("=" * 60)

async def main():
    load_dotenv()
    required_env_vars = {
        "DEEPSEEK_API_KEY": "DeepSeek eval might fail.",
        "GOOGLE_API_KEY": "Gemini eval/gen might fail.",
        "GOOGLE_CLOUD_PROJECT_ID": "Vertex AI might fail."
    }
    for var, msg in required_env_vars.items():
        if not os.getenv(var):
            if var == "GOOGLE_API_KEY" and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                # This is an acceptable fallback, so no strong warning needed if key isn't explicitly set in .env
                logger.debug(f"{var} not set, but GOOGLE_APPLICATION_CREDENTIALS found (ADC). Will rely on ADC.")
            else:
                logger.warning(f"{var} not found. {msg}")
        # No need for an else if GOOGLE_API_KEY is found, or if ADC is used.

    base_cfg = load_config(BASE_CONFIG_PATH)
    if base_cfg:
        # Remove manifest_path from the call, it's no longer used.
        await update_evaluation_files(EVALUATION_DIR, RERUN_DIR, base_cfg)
    else:
        logger.error("Failed to load base configuration. Cannot proceed.")

if __name__ == "__main__":
    try:
        _ = asyncio.get_running_loop()
        is_running = True
    except RuntimeError:
        is_running = False

    if is_running:
        logger.info("Async event loop already running. Applying nest_asyncio.")
        try:
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(main())
        except ImportError:
            logger.error("nest_asyncio not installed. Cannot run main() in a running event loop without it.")
            # Attempting run anyway, might fail in some environments like Jupyter
            asyncio.run(main())
    else:
        asyncio.run(main())