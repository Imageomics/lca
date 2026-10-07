"""
Simple, truly algorithm-agnostic run script.
Core execution logic that works for any algorithm with step() interface.
"""

import argparse
import logging
from beta_stability.util.init_logger import init_logger
from beta_stability.util.tools import get_config, write_json
from beta_stability.algorithm_preparation import create_algorithm
import os

import tempfile
import shutil
import itertools
from beta_stability.util.tools import EmptyDataframeException, SingleAnnotationException, load_dataframe_lightweight, discover_field_values_from_df
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("beta_stability")



def get_initial_edges(common_data, config):
    """Get initial edges in unified format.

    Supports multiple initialization verifiers (edges collected from each in order)
    and threshold-based initialization where edges are selected based on:
    - top-k neighbors (existing)
    - random proportion (existing)
    - certain positives: score > classifier_threshold + upper_margin
    - certain negatives: score < classifier_threshold - lower_margin
    """
    init_verifiers = common_data.get('init_verifiers', [common_data['verifier_name']])

    # Get algorithm parameters
    algorithm_params = config.get('algorithm', {})

    # Compute thresholds from classifier threshold if available
    classifier_threshold = common_data.get('classifier_threshold')
    lower_threshold = None
    upper_threshold = None

    if classifier_threshold is not None:
        # Get margins from config (can be symmetric or asymmetric)
        threshold_margin = algorithm_params.get('threshold_margin', None)
        lower_margin = algorithm_params.get('lower_threshold_margin', threshold_margin)
        upper_margin = algorithm_params.get('upper_threshold_margin', threshold_margin)

        if lower_margin is not None:
            lower_threshold = classifier_threshold - lower_margin
            logger.info(f"Using lower_threshold={lower_threshold:.4f} (classifier={classifier_threshold:.4f} - margin={lower_margin})")

        if upper_margin is not None:
            upper_threshold = classifier_threshold + upper_margin
            logger.info(f"Using upper_threshold={upper_threshold:.4f} (classifier={classifier_threshold:.4f} + margin={upper_margin})")

    all_edges = []
    seen_pairs = set()
    for verifier_name in init_verifiers:
        verifier_embeddings = common_data['embeddings_dict'][verifier_name]
        raw_edges = list(verifier_embeddings.get_edges(
            target_edges=common_data['target_edges'],
            topk=common_data['initial_topk'],
            botk=common_data.get('initial_botk', 0),
            target_proportion=common_data['target_proportion'],
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold
        ))
        new_count = 0
        for edge in raw_edges:
            pair = (edge[0], edge[1])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                all_edges.append((*edge, verifier_name))
                new_count += 1
        logger.info(f"Init verifier '{verifier_name}': {len(raw_edges)} edges, {new_count} new")

    return all_edges


def get_human_responses(requested_edges, common_data):
    """Get human responses in unified format."""
    human_reviewer = common_data['human_reviewer']

    # Handle both 3-tuple (n0, n1, score) and 4-tuple (n0, n1, score, source) formats
    if requested_edges and len(requested_edges[0]) == 4:
        # Extract just the first 3 elements for human reviewer
        edges_for_review = [(edge[0], edge[1], edge[2]) for edge in requested_edges]
    else:
        edges_for_review = requested_edges

    human_reviews, quit = human_reviewer(edges_for_review)

    # Convert to unified format - use 'human' for all human responses for simplicity
    unified_responses = []
    for n0, n1, decision in human_reviews:
        score = 1.0 if decision else 0.0
        unified_responses.append((n0, n1, score, 'human'))

    return unified_responses, quit


def run(config):
    """
    Core algorithm execution - truly algorithm agnostic.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: Algorithm results
    """
    logger.info(f"Starting {config.get('algorithm_type', 'stability').upper()} algorithm")
    
    output_path = config.get("data", {}).get('output_path', "tmp")

    # Create algorithm
    try:
        algorithm, common_data = create_algorithm(config)
    except EmptyDataframeException as e:
        logger.error(e)
        logger.info(f"Saving empty results to {output_path}")
        write_json({}, os.path.join(output_path, 'clustering.json'))
        write_json({}, os.path.join(output_path, 'node2cid.json'))
        write_json({}, os.path.join(output_path, 'node2uuid_file.json'))
        write_json({}, os.path.join(output_path, 'graph.json'))
        return ({}, {}, {})
    except SingleAnnotationException as e:
        logger.info(e)
        logger.info(f"Saving single-annotation result to {output_path}")
        # Create a single cluster with the one annotation
        node2uuid = e.node2uuid
        node_id = list(node2uuid.keys())[0]
        uuid = node2uuid[node_id]
        clustering = {'0': [node_id]}  # Single cluster with ID 0
        node2cid = {str(node_id): '0'}
        write_json(clustering, os.path.join(output_path, 'clustering.json'))
        write_json(node2cid, os.path.join(output_path, 'node2cid.json'))
        write_json({str(k): v for k, v in node2uuid.items()}, os.path.join(output_path, 'node2uuid_file.json'))
        write_json({}, os.path.join(output_path, 'graph.json'))
        return (clustering, node2cid, node2uuid)
    
    temp_db = False
    output_path = common_data.get('output_path', 'tmp')
    # if 'timestamp' in common_data:
    #     output_path = output_path + '_' + common_data['timestamp']
    os.makedirs(output_path, exist_ok=True)

    try:
        # Get initial edges
        init_verifiers = common_data.get('init_verifiers', [common_data['verifier_name']])
        logger.info(f"Initializing graph using verifiers: {init_verifiers}")
        for iv_name in init_verifiers:
            iv_emb = common_data['embeddings_dict'][iv_name]
            if hasattr(iv_emb, 'k'):
                logger.info(f"K-means found {iv_emb.k} clusters")
        initial_edges = get_initial_edges(common_data, config)
        logger.info(f"Starting with {len(initial_edges)} initial edges")
        # Start algorithm
        requested_edges = algorithm.step(initial_edges)

        # print("requested", requested_edges[0])
        
        # Main loop - run until convergence or max iterations
        max_outer_iterations = config.get('algorithm', {}).get('max_outer_iterations', 100)
        # An algorithm that presents one review per step declares how many steps it may need
        # (manual / thresholded review); others do not set it and keep the configured cap.
        max_outer_iterations = max(max_outer_iterations,
                                   int(getattr(algorithm, 'required_outer_iterations', 0)))
        outer_iteration = 0

        while not algorithm.is_finished() and outer_iteration < max_outer_iterations:
            logger.info(f"Outer iteration {outer_iteration}")

            if requested_edges:
                # Get human responses for edges that need review
                human_responses, quit = get_human_responses(requested_edges, common_data)

                if quit:
                    logger.info("Human reviewer requested to quit")
                    break

                requested_edges = algorithm.step(human_responses)
            else:
                # No edges need human review, but algorithm not finished
                # Continue with empty step to allow further progress
                logger.info("No edges for human review, continuing algorithm")
                requested_edges = algorithm.step([])

            outer_iteration += 1

        if outer_iteration >= max_outer_iterations:
            logger.warning(f"Reached max outer iterations ({max_outer_iterations})")
        else:
            logger.info(f"Algorithm converged after {outer_iteration} outer iterations")

        logger.info("Algorithm execution completed")
        algorithm.show_stats()
        
        # Return results
        result = algorithm.get_clustering()

        (clustering_dict, node2cid_dict, G) = result
        
        logger.info(f"Saving results to {output_path}")
        write_json(clustering_dict, os.path.join(output_path, 'clustering.json'))
        write_json(node2cid_dict, os.path.join(output_path, 'node2cid.json'))
        write_json(common_data['node2uuid'], os.path.join(output_path, 'node2uuid_file.json'))
        write_json(G, os.path.join(output_path, 'graph.json'))
        return result
    finally:
        # Cleanup temp database if needed
        if temp_db and algorithm_type == 'lca':
            import tempfile
            import shutil
            # Get the temp path from the LCA instance
            if hasattr(algorithm, 'lca_config') and 'autosave_file' in algorithm.lca_config:
                autosave_path = algorithm.lca_config['autosave_file']
                temp_db_path = os.path.dirname(autosave_path)
                if temp_db_path.startswith(tempfile.gettempdir()):
                    logger.info(f"Cleaning up temp database: {temp_db_path}")
                    shutil.rmtree(temp_db_path, ignore_errors=True)

def run_for_field_values(config, field_overrides, save_dir):
    """Run algorithm for specific field values."""
    config_copy = config.copy()
    config_copy['data'] = config_copy['data'].copy()
    
    # Apply field overrides to config
    for field, value in field_overrides.items():
        list_key = f"{field}_list"
        config_copy['data'][list_key] = [value] if not isinstance(value, list) else value
    
    config_copy['data']['output_path'] = save_dir
    if 'histogram_path' in config_copy.get('logging', {}):
        config_copy['logging']['histogram_path'] = config_copy['logging']['histogram_path'] + f"_{str(field_overrides)}.png"
    # Override save directory for LCA if needed (backwards compatible)
    algorithm_type = config_copy.get('algorithm_type', 'lca')
    if algorithm_type == 'lca':
        if 'lca' not in config_copy:
            config_copy['lca'] = {}
        config_copy['lca'] = config_copy['lca'].copy()
        config_copy['lca']['db_path'] = save_dir
    
    return run(config_copy)


def run_for_viewpoints(config, viewpoint_list, save_dir):
    """Run algorithm for specific viewpoints. (Legacy compatibility)"""
    return run_for_field_values(config, {'viewpoint': viewpoint_list}, save_dir)


def main(config):
    """Main drone execution logic."""
    data_params = config['data']
    algorithm_config = config.get('lca', config.get('gc', {}))
    print(algorithm_config)
    
    # Setup output directory
    temp_db = algorithm_config.get('temp_db', False)
    
    if temp_db:
        db_path = tempfile.mkdtemp()
    elif "output_path" in data_params:
        db_path = os.path.join(data_params['output_path'], config['exp_name'])
    os.makedirs(db_path, exist_ok=True)
    
    try:
        # Handle field separation (generalized from viewpoint separation)
        separate_by_fields = data_params.get('separate_by_fields')
        if separate_by_fields is None and data_params.get('separate_viewpoints', False):
            separate_by_fields = ['viewpoint']
        
        if separate_by_fields:
            # Get field values: use config lists or discover from lightweight data loading
            field_values = {}
            
            # First collect explicitly provided field lists
            for field in separate_by_fields:
                list_key = f"{field}_list"
                if list_key in data_params:
                    field_values[field] = data_params[list_key]
            
            # For fields without explicit lists, discover values
            fields_to_discover = [f for f in separate_by_fields if f not in field_values]
            if fields_to_discover:
                df = load_dataframe_lightweight(config)
                discovered_values = discover_field_values_from_df(df, fields_to_discover)
                field_values.update(discovered_values)
                for field, values in discovered_values.items():
                    logger.info(f"Discovered {len(values)} values for '{field}': {values}")
            
            # Generate combinations for all fields
            base_output = data_params['output_path']
            field_names = list(field_values.keys())
            value_lists = [field_values[field] for field in field_names]
            
            for values in itertools.product(*value_lists):
                field_combo = dict(zip(field_names, values))
                combo_str = '_'.join(f"{field}-{value}" for field, value in field_combo.items())
                print(f"Running for: {combo_str}")
                logger.info(f"Running for field combination: {field_combo}")
                
                save_dir = os.path.join(base_output, combo_str.replace(':', '_').replace('/', '_'))
                os.makedirs(save_dir, exist_ok=True)
                run_for_field_values(config, field_combo, save_dir)
                
                # Append logs for subsequent runs
                if 'logging' in config and isinstance(config['logging'], dict):
                    config['logging']['file_mode'] = 'a'
        else:
            # Single combined run
            run(config)
    
    finally:
        if temp_db:
            shutil.rmtree(db_path, ignore_errors=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run clustering algorithm.")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--interactive', '-i',
                   action='store_true',
                   help='Enable interactive mode')
    parser.add_argument('--seed', type=int, default=None,
                   help='Override config seed (also redirects every output path to a '
                        'seed<N>/ subdir, so repeated seeds do not overwrite each other).')
    return parser.parse_args()


def apply_seed_override(config, seed):
    """Set the run seed and send all outputs to a per-seed subdir. Lets one base
    config serve many seeds without generating a config file per seed."""
    config['seed'] = seed
    sub = f'seed{seed}'
    def reseed(d, key):
        if isinstance(d, dict) and d.get(key):
            p = d[key]
            d[key] = os.path.join(os.path.dirname(p), sub, os.path.basename(p))
    reseed(config.get('data', {}), 'output_path')
    reseed(config.get('logging', {}), 'log_file')
    reseed(config.get('logging', {}), 'auto_threshold_plot_path')
    reseed(config.get('stability', {}), 'stability_track_path')
    return config

 

if __name__ == '__main__':
    init_logger()
    args = parse_args()
    if args.interactive:
        print(f"Working with config {os.path.abspath(args.config)}")
        input("Press Enter to continue...")
    config = get_config(args.config)
    if args.seed is not None:
        config = apply_seed_override(config, args.seed)
    main(config)