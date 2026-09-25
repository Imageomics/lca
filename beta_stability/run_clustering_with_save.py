#!/usr/bin/env python3
"""
Run clustering algorithm and save results in the correct format.
This script combines running the clustering and formatting the output.
Supports any clustering algorithm (HDBSCAN, GC, LCA, etc.).
"""

import argparse
import logging
import os
import sys
import json
import datetime
from collections import defaultdict
from beta_stability.util.cluster_tools import percent_and_PR, build_node_to_cluster_mapping, hungarian_cluster_matching, ceaf_cluster_matching
# Import clustering functions directly
from beta_stability.util.init_logger import init_logger
from beta_stability.run import main as run_clustering
from beta_stability.save_clustering_results import save_clustering_results, combine_field_separated_results


def deep_merge(base, override):
    """Recursively merge override dict into base dict.

    For nested dicts, values are merged recursively rather than replaced.
    For all other types, the override value replaces the base value.
    Modifies base in-place and returns it.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path, base_config_path=None):
    """Load YAML configuration file, optionally merging on top of a base config.

    Args:
        config_path: Path to the species/primary config file.
        base_config_path: Optional path to a base (universal) config file.
            If provided, the base config is loaded first and the species
            config is deep-merged on top of it.

    Returns:
        dict: The (merged) configuration dictionary.
    """
    import yaml
    if base_config_path:
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        with open(config_path, 'r') as f:
            species_config = yaml.safe_load(f)
        deep_merge(config, species_config)
        return config
    else:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


def load_ground_truth_clustering(anno_file, uuid_key='uuid', name_keys=None):
    """Load ground truth clustering from annotation file using name_keys as ground truth."""
    with open(anno_file, 'r') as f:
        data = json.load(f)

    # Use name_keys for ground truth if provided, otherwise try cluster_id
    gt_field = 'cluster_id'
    if name_keys and len(name_keys) > 0:
        gt_field = name_keys[0]  # Use first name_key as ground truth
        print(f"Using '{gt_field}' from name_keys as ground truth field")

    # Build ground truth clustering from specified field
    gt_clustering = {}
    gt_node2uuid = {}
    node_id = 0

    for ann in data.get('annotations', []):
        cluster_id = ann.get(gt_field)
        uuid = ann.get(uuid_key)

        if cluster_id is not None and uuid is not None:
            if cluster_id not in gt_clustering:
                gt_clustering[cluster_id] = []
            gt_clustering[cluster_id].append(node_id)
            gt_node2uuid[str(node_id)] = uuid
            node_id += 1

    # Convert lists to the expected format
    gt_clustering = {str(cid): nodes for cid, nodes in gt_clustering.items()}

    return gt_clustering, gt_node2uuid


def calculate_evaluation_metrics(est_clustering, est_node2uuid, gt_clustering, gt_node2uuid):
    """Calculate evaluation metrics between estimated and ground truth clusterings."""

    # Create uuid to node mappings for alignment
    uuid2est_node = {uuid: node for node, uuid in est_node2uuid.items()}
    uuid2gt_node = {uuid: node for node, uuid in gt_node2uuid.items()}

    # Find common UUIDs
    common_uuids = set(uuid2est_node.keys()) & set(uuid2gt_node.keys())

    if not common_uuids:
        return {
            'error': 'No common UUIDs between estimated and ground truth',
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'frac_correct': 0.0
        }

    # Align clusterings based on common UUIDs
    aligned_est_clustering = {}
    aligned_gt_clustering = {}
    node_mapping = {}  # Maps aligned node IDs to UUIDs

    aligned_node_id = 0
    for uuid in common_uuids:
        est_node = uuid2est_node[uuid]
        gt_node = uuid2gt_node[uuid]

        # Find clusters for this UUID
        est_cid = None
        gt_cid = None

        for cid, nodes in est_clustering.items():
            if int(est_node) in nodes or str(est_node) in [str(n) for n in nodes]:
                est_cid = cid
                break

        for cid, nodes in gt_clustering.items():
            if int(gt_node) in nodes or str(gt_node) in [str(n) for n in nodes]:
                gt_cid = cid
                break

        if est_cid and gt_cid:
            if est_cid not in aligned_est_clustering:
                aligned_est_clustering[est_cid] = []
            if gt_cid not in aligned_gt_clustering:
                aligned_gt_clustering[gt_cid] = []

            aligned_est_clustering[est_cid].append(aligned_node_id)
            aligned_gt_clustering[gt_cid].append(aligned_node_id)
            node_mapping[aligned_node_id] = uuid
            aligned_node_id += 1

    # Convert lists to sets for cluster_tools compatibility
    aligned_est_clustering = {cid: set(nodes) for cid, nodes in aligned_est_clustering.items()}
    aligned_gt_clustering = {cid: set(nodes) for cid, nodes in aligned_gt_clustering.items()}

    # Build node2cluster mappings
    est_n2c = build_node_to_cluster_mapping(aligned_est_clustering)
    gt_n2c = build_node_to_cluster_mapping(aligned_gt_clustering)

    # Calculate metrics using cluster_tools functions
    try:
        frac_correct, precision, recall, per_size, _, f1 = percent_and_PR(
            aligned_est_clustering, est_n2c, aligned_gt_clustering, gt_n2c
        )

        # Calculate Hungarian matching metrics (cluster-level)
        hungarian = hungarian_cluster_matching(aligned_est_clustering, aligned_gt_clustering)
        ceaf = ceaf_cluster_matching(aligned_est_clustering, aligned_gt_clustering)

        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'frac_correct': frac_correct,
            'ceaf_f1': ceaf['f1'],
            'ceaf_precision': ceaf['precision'],
            'ceaf_recall': ceaf['recall'],
            'hungarian_f1': hungarian['f1'],
            'hungarian_precision': hungarian['precision'],
            'hungarian_recall': hungarian['recall'],
            'hungarian_tp': hungarian['tp'],
            'hungarian_fp': hungarian['fp'],
            'hungarian_fn': hungarian['fn'],
            'num_estimated_clusters': len(aligned_est_clustering),
            'num_ground_truth_clusters': len(aligned_gt_clustering),
            'num_common_uuids': len(common_uuids),
            'per_size_metrics': per_size
        }
    except Exception as e:
        return {
            'error': str(e),
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'frac_correct': 0.0,
            'hungarian_f1': 0.0,
            'hungarian_precision': 0.0,
            'hungarian_recall': 0.0
        }

logger = logging.getLogger("beta_stability")

def append_metrics_to_config_log(log_file, metrics_summary):
    """Append evaluation metrics to the config-specified log file."""
    
    # if log_file and os.path.exists(log_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # with open(log_file, 'a') as f:
            # f.write("\n" + "="*60 + "\n")
            # f.write(f"CLUSTERING EVALUATION METRICS - {timestamp}\n")
            # f.write("="*60 + "\n")
            # f.write(metrics_summary)
            # f.write("="*60 + "\n")
    logger.info("\n" + "="*60 + "\n")
    logger.info(f"CLUSTERING EVALUATION METRICS - {timestamp}\n")
    logger.info("="*60 + "\n")
    logger.info("\n" + metrics_summary)
    logger.info("="*60 + "\n")

def evaluate_field_separated_results(output_base, anno_file, config_log_file,
                                    separate_by_fields, uuid_key, name_keys=None):
    """Evaluate metrics for field-separated clustering results and append to config log."""
    import glob
    import re

    # Build regex pattern to match and extract field values
    regex_pattern = "_".join([f"{field}-([^_]+)" for field in separate_by_fields])
    regex = re.compile(regex_pattern)

    # Find all directories matching the pattern
    glob_pattern = "_".join([f"{field}-*" for field in separate_by_fields])
    pattern_path = os.path.join(output_base, glob_pattern)

    print(f"Evaluating field-separated clustering results...")

    all_metrics = []
    metrics_text = []

    for field_dir in glob.glob(pattern_path):
        if os.path.isdir(field_dir):
            dir_name = os.path.basename(field_dir)
            match = regex.match(dir_name)

            if match:
                # Extract field values from regex groups
                field_combo = dict(zip(separate_by_fields, match.groups()))
                field_str = ', '.join(f"{k}={v}" for k, v in field_combo.items())

                # Load clustering results for this field combination
                clustering_file = os.path.join(field_dir, "clustering.json")
                node2uuid_file = os.path.join(field_dir, "node2uuid_file.json")

                if not os.path.exists(clustering_file) or not os.path.exists(node2uuid_file):
                    continue

                try:
                    with open(clustering_file, 'r') as f:
                        est_clustering = json.load(f)
                    with open(node2uuid_file, 'r') as f:
                        est_node2uuid = json.load(f)

                    # Load ground truth
                    gt_clustering, gt_node2uuid = load_ground_truth_clustering(anno_file, uuid_key, name_keys)

                    # Calculate metrics
                    metrics = calculate_evaluation_metrics(est_clustering, est_node2uuid,
                                                         gt_clustering, gt_node2uuid)

                    # Add field info to metrics
                    metrics['field_combination'] = field_combo
                    all_metrics.append(metrics)

                    # Format metrics for this field
                    num_pred = metrics.get('num_estimated_clusters', 0)
                    num_gt = metrics.get('num_ground_truth_clusters', 0)
                    field_text = f"\n{field_str}:\n"
                    field_text += f"  Number of clusters:        {num_pred}\n"
                    field_text += f"  Number of true clusters:   {num_gt}\n"
                    if num_gt > 0:
                        field_text += f"  Cluster count ratio:       {num_pred/num_gt:.4f}\n"
                    field_text += f"  F1 Score:                  {metrics.get('f1', 0):.4f}\n"
                    field_text += f"  Precision:                 {metrics.get('precision', 0):.4f}\n"
                    field_text += f"  Recall:                    {metrics.get('recall', 0):.4f}\n"
                    field_text += f"  Fraction Correct:          {metrics.get('frac_correct', 0):.4f}\n"
                    field_text += f"  Hungarian F1 Score:        {metrics.get('hungarian_f1', 0):.4f}\n"
                    field_text += f"  CEAF F1 Score:        {metrics.get('ceaf_f1', 0):.4f}\n"
                    field_text += f"  Hungarian Precision:       {metrics.get('hungarian_precision', 0):.4f}\n"
                    field_text += f"  Hungarian Recall:          {metrics.get('hungarian_recall', 0):.4f}\n"
                    metrics_text.append(field_text)

                except Exception as e:
                    print(f"  Error evaluating {field_str}: {e}")

    # Calculate and format average metrics
    if all_metrics:
        avg_metrics = {
            'f1': sum(m.get('f1', 0) for m in all_metrics) / len(all_metrics),
            'precision': sum(m.get('precision', 0) for m in all_metrics) / len(all_metrics),
            'recall': sum(m.get('recall', 0) for m in all_metrics) / len(all_metrics),
            'frac_correct': sum(m.get('frac_correct', 0) for m in all_metrics) / len(all_metrics),
            'hungarian_f1': sum(m.get('hungarian_f1', 0) for m in all_metrics) / len(all_metrics),
            'hungarian_precision': sum(m.get('hungarian_precision', 0) for m in all_metrics) / len(all_metrics),
            'hungarian_recall': sum(m.get('hungarian_recall', 0) for m in all_metrics) / len(all_metrics),
        }

        summary_text = "".join(metrics_text)
        summary_text += "\n" + "-"*40 + "\n"
        summary_text += "AVERAGE METRICS:\n"
        summary_text += f"  F1 Score:              {avg_metrics['f1']:.4f}\n"
        summary_text += f"  Precision:             {avg_metrics['precision']:.4f}\n"
        summary_text += f"  Recall:                {avg_metrics['recall']:.4f}\n"
        summary_text += f"  Fraction Correct:      {avg_metrics['frac_correct']:.4f}\n"
        summary_text += f"  Hungarian F1 Score:    {avg_metrics['hungarian_f1']:.4f}\n"
        summary_text += f"  CEAF F1 Score:    {avg_metrics.get('ceaf_f1', 0):.4f}\n"
        summary_text += f"  Hungarian Precision:   {avg_metrics['hungarian_precision']:.4f}\n"
        summary_text += f"  Hungarian Recall:      {avg_metrics['hungarian_recall']:.4f}\n"

        # Append to config log file
        append_metrics_to_config_log(config_log_file, summary_text)

        # Print summary
        print(summary_text)

        return True

    return False

def evaluate_global_clustering(output_base, anno_file, config_log_file,
                               gt_key='individual_id', pred_key='encounter_id',
                               uuid_key='uuid'):
    """Evaluate global clustering using individual_id as GT and encounter_id as predictions.

    This computes dataset-wide pairwise metrics, capturing cross-occurrence
    fragmentation that per-occurrence evaluation misses.
    """
    combined_file = os.path.join(output_base, "beta_stability_annots.json")
    if not os.path.exists(combined_file):
        print(f"Combined output file not found: {combined_file}, skipping global evaluation")
        return None

    with open(combined_file, 'r') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])

    # Build GT and predicted clustering: cluster_id -> set of uuids
    gt_clusters = defaultdict(set)
    pred_clusters = defaultdict(set)

    skipped = 0
    for ann in annotations:
        uid = ann.get(uuid_key)
        gt_id = ann.get(gt_key)
        pred_id = ann.get(pred_key)

        if uid is None or gt_id is None or pred_id is None:
            skipped += 1
            continue

        gt_clusters[str(gt_id)].add(uid)
        pred_clusters[str(pred_id)].add(uid)

    if not gt_clusters or not pred_clusters:
        print("No valid annotations for global evaluation")
        return None

    # Build uuid -> cluster mappings
    uuid_to_gt = {}
    for cid, uuids in gt_clusters.items():
        for u in uuids:
            uuid_to_gt[u] = cid

    uuid_to_pred = {}
    for cid, uuids in pred_clusters.items():
        for u in uuids:
            uuid_to_pred[u] = cid

    # Efficient pairwise metric computation using cluster sizes
    # TP = pairs in same pred cluster AND same gt cluster
    # FP = pairs in same pred cluster but different gt cluster
    # FN = pairs in same gt cluster but different pred cluster
    tp = 0
    fp = 0
    fn = 0

    # For each predicted cluster, count pairs by GT group
    for pred_cid, pred_uuids in pred_clusters.items():
        gt_group_sizes = defaultdict(int)
        for u in pred_uuids:
            if u in uuid_to_gt:
                gt_group_sizes[uuid_to_gt[u]] += 1
        n = sum(gt_group_sizes.values())
        total_pairs = n * (n - 1) // 2
        same_gt_pairs = sum(k * (k - 1) // 2 for k in gt_group_sizes.values())
        tp += same_gt_pairs
        fp += total_pairs - same_gt_pairs

    # For each GT cluster, count pairs by predicted group
    for gt_cid, gt_uuids in gt_clusters.items():
        pred_group_sizes = defaultdict(int)
        for u in gt_uuids:
            if u in uuid_to_pred:
                pred_group_sizes[uuid_to_pred[u]] += 1
        n = sum(pred_group_sizes.values())
        total_pairs = n * (n - 1) // 2
        same_pred_pairs = sum(k * (k - 1) // 2 for k in pred_group_sizes.values())
        fn += total_pairs - same_pred_pairs

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Cluster purity (fraction of annotations matching majority GT in their pred cluster)
    correct = 0
    total = 0
    for pred_cid, pred_uuids in pred_clusters.items():
        gt_counts = defaultdict(int)
        for u in pred_uuids:
            if u in uuid_to_gt:
                gt_counts[uuid_to_gt[u]] += 1
                total += 1
        if gt_counts:
            correct += max(gt_counts.values())
    purity = correct / total if total > 0 else 0.0

    # Hungarian matching (cluster-level)
    hungarian = hungarian_cluster_matching(pred_clusters, gt_clusters)

    # Format output
    text = "\n" + "=" * 60 + "\n"
    text += "GLOBAL EVALUATION (individual_id vs encounter_id)\n"
    text += "=" * 60 + "\n"
    text += f"  Number of annotations:         {len(annotations) - skipped}\n"
    text += f"  Number of true individuals:    {len(gt_clusters)}\n"
    text += f"  Number of predicted clusters:  {len(pred_clusters)}\n"
    text += f"  Cluster ratio (pred/gt):       {len(pred_clusters)/len(gt_clusters):.2f}\n"
    text += f"  Pairwise TP:                   {tp}\n"
    text += f"  Pairwise FP:                   {fp}\n"
    text += f"  Pairwise FN:                   {fn}\n"
    text += f"  Pairwise Precision:            {precision:.4f}\n"
    text += f"  Pairwise Recall:               {recall:.4f}\n"
    text += f"  Pairwise F1:                   {f1:.4f}\n"
    text += f"  Cluster Purity:                {purity:.4f}\n"
    text += f"  Hungarian TP:                  {hungarian['tp']}\n"
    text += f"  Hungarian FP:                  {hungarian['fp']}\n"
    text += f"  Hungarian FN:                  {hungarian['fn']}\n"
    text += f"  Hungarian Precision:           {hungarian['precision']:.4f}\n"
    text += f"  Hungarian Recall:              {hungarian['recall']:.4f}\n"
    text += f"  Hungarian F1:                  {hungarian['f1']:.4f}\n"
    text += "=" * 60 + "\n"

    print(text)

    # Append to log file
    append_metrics_to_config_log(config_log_file, text)

    return {
        'num_annotations': len(annotations) - skipped,
        'num_gt_individuals': len(gt_clusters),
        'num_pred_clusters': len(pred_clusters),
        'pairwise_precision': precision,
        'pairwise_recall': recall,
        'pairwise_f1': f1,
        'purity': purity,
        'hungarian_precision': hungarian['precision'],
        'hungarian_recall': hungarian['recall'],
        'hungarian_f1': hungarian['f1'],
    }


def run_clustering_with_save(config, interactive=False, config_path=None, save_dir=None):
    #  Get algorithm type
    algorithm_type = config.get('algorithm_type', 'unknown')
    print(f"Algorithm type: {algorithm_type}")

    # Get output prefix and suffix from config only
    output_config = config.get('output', {})

    # Get prefix from config (or use algorithm_type as fallback)
    if 'prefix' in output_config:
        output_prefix = output_config['prefix']
    # elif algorithm_type != 'unknown':
    #     output_prefix = algorithm_type
    else:
        output_prefix = None

    # Get suffix from config
    output_suffix = output_config.get('suffix', None)

    # Print info about prefix/suffix if they are set
    if output_prefix or output_suffix:
        print(f"Output naming: prefix='{output_prefix}', suffix='{output_suffix}'")

    # Get paths from config
    data_config = config.get('data', {})
    anno_file = data_config.get('annotation_file')
    output_base = data_config.get('output_path', 'tmp')
    uuid_key = data_config.get('id_key', 'uuid')
    separate_by_fields = data_config.get('separate_by_fields')
    name_keys = data_config.get('name_keys', [])
    output_key = data_config.get('output_key', 'cluster_id')  # New: get output_key from config
    
    # Get log file from config
    logging_config = config.get('logging', {})
    config_log_file = logging_config.get('log_file')

    if not anno_file:
        print("Error: annotation_file not specified in config")
        sys.exit(1)

    # Interactive mode
    if interactive:
        print(f"Config file: {os.path.abspath(config_path)}")
        print(f"Annotation file: {anno_file}")
        print(f"Output path: {output_base}")
        if separate_by_fields:
            print(f"Separating by fields: {separate_by_fields}")
        input("Press Enter to continue...")

    # Step 1: Run clustering algorithm
    print("\n" + "="*60)
    print(f"Step 1: Running {algorithm_type.upper()} clustering...")
    print("="*60)

    run_clustering(config)
    print(f"\n{algorithm_type.upper()} clustering completed successfully!")


    # Step 2: Save results in correct format
    print("\n" + "="*60)
    print("Step 2: Formatting and saving results...")
    print("="*60)

    save_dir = save_dir or output_base

    try:
        if separate_by_fields:
            # Handle field-separated results - combine into single file
            print(f"Combining field-separated results into single file...")
            print(f"Using output key: '{output_key}'")

            # Call combine_field_separated_results function directly
            combine_field_separated_results(
                base_path=output_base,
                anno_file=anno_file,
                output_path=save_dir,
                prefix=output_prefix,
                suffix=output_suffix,
                separate_by_fields=separate_by_fields,
                uuid_key=uuid_key,
                output_key=output_key
            )
        else:
            # Handle single result
            print(f"Processing single clustering result...")
            print(f"Using output key: '{output_key}'")

            # Call save_clustering_results function directly
            save_clustering_results(
                input_dir=output_base,
                anno_file=anno_file,
                output_path=save_dir,
                prefix=output_prefix,
                suffix=output_suffix,
                field_filters=None,
                uuid_key=uuid_key,
                output_key=output_key
            )

        print("\n" + "="*60)
        print(f"{algorithm_type.upper()} clustering and formatting completed successfully!")
        print(f"Results saved to: {save_dir}")
        print("="*60)

    except Exception as e:
        print("Error: Result formatting failed")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Calculate evaluation metrics (always run if ground truth available)
    metrics = None
    if config_log_file and name_keys:
        print("\n" + "="*60)
        print("Step 3: Calculating evaluation metrics...")
        print("="*60)

        # Handle field-separated vs single clustering results
        if separate_by_fields:
            # Evaluate all field combinations
            evaluate_field_separated_results(
                output_base, anno_file, config_log_file,
                separate_by_fields, uuid_key, name_keys
            )
        else:
            try:
                # Load the clustering results from output_base
                clustering_file = os.path.join(output_base, "clustering.json")
                node2uuid_file = os.path.join(output_base, "node2uuid_file.json")

                if os.path.exists(clustering_file) and os.path.exists(node2uuid_file):
                    with open(clustering_file, 'r') as f:
                        est_clustering = json.load(f)
                    with open(node2uuid_file, 'r') as f:
                        est_node2uuid = json.load(f)

                    # Load ground truth from annotation file
                    gt_clustering, gt_node2uuid = load_ground_truth_clustering(anno_file, uuid_key, name_keys)

                    # Calculate metrics
                    metrics = calculate_evaluation_metrics(est_clustering, est_node2uuid, gt_clustering, gt_node2uuid)

                    # Format metrics text
                    num_pred = metrics.get('num_estimated_clusters', 0)
                    num_gt = metrics.get('num_ground_truth_clusters', 0)
                    metrics_text = f"Number of clusters:            {num_pred}\n"
                    metrics_text += f"Number of true clusters:       {num_gt}\n"
                    metrics_text += f"Cluster count ratio:           {num_pred/num_gt:.4f}\n" if num_gt > 0 else ""
                    metrics_text += f"F1 Score:                      {metrics.get('f1', 0):.4f}\n"
                    metrics_text += f"Precision:                     {metrics.get('precision', 0):.4f}\n"
                    metrics_text += f"Recall:                        {metrics.get('recall', 0):.4f}\n"
                    metrics_text += f"Fraction Correct:              {metrics.get('frac_correct', 0):.4f}\n"
                    metrics_text += f"Hungarian F1 Score:            {metrics.get('hungarian_f1', 0):.4f}\n"
                    metrics_text += f"CEAF F1 Score:            {metrics.get('ceaf_f1', 0):.4f}\n"
                    metrics_text += f"CEAF Precision:           {metrics.get('ceaf_precision', 0):.4f}\n"
                    metrics_text += f"CEAF Recall:              {metrics.get('ceaf_recall', 0):.4f}\n"
                    metrics_text += f"Hungarian Precision:           {metrics.get('hungarian_precision', 0):.4f}\n"
                    metrics_text += f"Hungarian Recall:              {metrics.get('hungarian_recall', 0):.4f}\n"

                    # Append to config log file
                    append_metrics_to_config_log(config_log_file, metrics_text)

                    # Display metrics
                    print(metrics_text)
                else:
                    print("Clustering files not found, skipping evaluation.")

            except Exception as e:
                print(f"Error calculating metrics: {e}")
                import traceback
                traceback.print_exc()

    # Step 4: Global evaluation (individual_id vs encounter_id across all occurrences)
    if config_log_file and name_keys and separate_by_fields:
        print("\n" + "="*60)
        print("Step 4: Global clustering evaluation...")
        print("="*60)

        gt_key = name_keys[0] if name_keys else 'individual_id'
        evaluate_global_clustering(
            output_base, anno_file, config_log_file,
            gt_key=gt_key,
            pred_key=output_key,
            uuid_key=uuid_key
        )

    print("\n" + "="*60)
    print("All processing completed!")
    print("="*60)

    return metrics
    


def main():
    init_logger()
    parser = argparse.ArgumentParser(
        description="Run clustering algorithm and save results in the correct format"
    )

    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--base_config", type=str, default=None,
                       help="Base config file (universal). Species config is merged on top.")
    parser.add_argument("--save_dir", type=str,
                       help="Directory to save formatted results (default: same as clustering output)")
    parser.add_argument("--interactive", "-i", action='store_true',
                       help="Enable interactive mode")
    parser.add_argument("--seed", type=int, default=None,
                       help="Override config seed; also redirects every output path "
                            "to a seed<N>/ subdir so repeated seeds don't overwrite.")
    parser.add_argument("--prob_human_correct", type=float, default=None,
                       help="Override edge_weights.prob_human_correct (reviewer "
                            "accuracy); also redirects outputs to a phc<NN>/ subdir.")

    args = parser.parse_args()

    # Load config (with optional base config merging)
    config = load_config(args.config, base_config_path=args.base_config)

    # Optional reviewer-accuracy and seed overrides (imperfect-reviewer experiments),
    # redirecting outputs to phc<NN>/seed<N>/ so the sweep does not overwrite itself.
    import os as _os
    sub = ''
    if args.prob_human_correct is not None:
        config.setdefault('edge_weights', {})['prob_human_correct'] = args.prob_human_correct
        sub = _os.path.join(sub, f"phc{int(round(args.prob_human_correct * 100))}")
    if args.seed is not None:
        config['seed'] = args.seed
        sub = _os.path.join(sub, f"seed{args.seed}")
    if sub:
        def _reseed(d, key):
            if isinstance(d, dict) and d.get(key):
                p = d[key]
                np = _os.path.join(_os.path.dirname(p), sub, _os.path.basename(p))
                d[key] = np
                # create the redirected directory so the logger/output can open it
                dirp = np if key == 'output_path' else _os.path.dirname(np)
                if dirp:
                    _os.makedirs(dirp, exist_ok=True)
        _reseed(config.get('data', {}), 'output_path')
        for k in ('log_file', 'auto_threshold_plot_path'):
            _reseed(config.get('logging', {}), k)

    # Run clustering and formatting
    metrics = run_clustering_with_save(config)

    # Return metrics
    return metrics


if __name__ == "__main__":
    main()