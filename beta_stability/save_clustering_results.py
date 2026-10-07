"""
Save clustering results in the correct format for downstream processing.
Supports any clustering algorithm (HDBSCAN, GC, LCA, etc.).
"""

import argparse
import json
import os
import glob
import re


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)




def save_clustering_results(input_dir, anno_file, output_path, prefix, suffix,
                          field_filters=None, uuid_key="annot_uuid", output_key="cluster_id"):
    """
    Save clustering results with support for multiple field filtering.

    Args:
        input_dir: Directory containing clustering.json and node2uuid_file.json
        anno_file: Path to original annotation file
        output_path: Directory to save output
        prefix: Output filename prefix
        suffix: Output filename suffix
        field_filters: dict of {field_name: field_value} to filter on
        uuid_key: Key for UUID in annotations (default: annot_uuid)
        output_key: Key name for saving cluster IDs (default: cluster_id)
    """
    clustering_file = os.path.join(input_dir, "clustering.json")
    node2uuid_file = os.path.join(input_dir, "node2uuid_file.json")

    print(f"Loading annotations from: {anno_file}")
    print(f"Loading clustering from: {clustering_file}")
    print(f"Loading node2uuid from: {node2uuid_file}")

    # Load original annotation file - keep full structure
    original_data = load_json(anno_file)

    # Load clustering results
    clusters = load_json(clustering_file)
    node2uuid = load_json(node2uuid_file)

    # Build mapping from UUID to cluster ID
    uuid_to_cluster = {}
    for cluster_id, nodes in clusters.items():
        for node in nodes:
            uuid = node2uuid.get(str(node))
            if uuid:
                uuid_to_cluster[uuid] = cluster_id

    print(f"Built UUID to cluster mapping for {len(uuid_to_cluster)} UUIDs")

    # Create a copy of the original data to preserve structure
    result_dict = {}

    # Preserve categories if present
    if 'categories' in original_data:
        result_dict['categories'] = original_data['categories']

    # Preserve images if present
    if 'images' in original_data:
        result_dict['images'] = original_data['images']

    # Process annotations - preserve ALL original fields and add cluster_id
    if 'annotations' in original_data:
        annotations = original_data['annotations']

        # Filter annotations based on field filters if provided
        if field_filters:
            print(f"Applying filters: {field_filters}")
            filtered_annotations = []
            for ann in annotations:
                match = True
                for field, value in field_filters.items():
                    # Check direct field or name_<field> pattern
                    field_value = ann.get(field) or ann.get(f"name_{field}")
                    if field_value is None:
                        match = False
                        break
                    # Substring matching
                    if str(value) not in str(field_value):
                        match = False
                        break
                if match:
                    filtered_annotations.append(ann)

            filter_desc = ', '.join(f"{k}={v}" for k, v in field_filters.items())
            print(f"Filtered {len(filtered_annotations)} annotations with {filter_desc} "
                  f"out of {len(annotations)}")
        else:
            filtered_annotations = annotations

        # Add cluster ID to each annotation while preserving all original fields
        updated_annotations = []
        assigned_count = 0
        for ann in filtered_annotations:
            # Create a copy to preserve all original fields
            updated_ann = ann.copy()

            # Add cluster ID field based on UUID mapping (using output_key)
            cluster_id = uuid_to_cluster.get(ann.get(uuid_key))
            if cluster_id is not None:
                updated_ann[output_key] = cluster_id
                assigned_count += 1
            else:
                updated_ann[output_key] = None

            updated_annotations.append(updated_ann)

        print(f"Assigned cluster IDs to {assigned_count}/{len(updated_annotations)} annotations")
        result_dict['annotations'] = updated_annotations

    # Build output path
    if field_filters:
        field_str = '_'.join(f"{k}-{v}" for k, v in field_filters.items())
        # Build filename with optional prefix and suffix
        parts = []
        if prefix:
            parts.append(prefix)
        parts.append(field_str)
        if suffix:
            parts.append(suffix)
        output_filename = '_'.join(parts) + '.json' if parts else 'beta_stability_annots.json'
    else:
        # Build filename with optional prefix and suffix
        parts = []
        if prefix:
            parts.append(prefix)
        if suffix:
            parts.append(suffix)
        if parts:
            output_filename = '_'.join(parts) + '.json'
        else:
            output_filename = 'beta_stability_annots.json'

    output_path_full = os.path.join(output_path, output_filename)

    # Save final result with same structure as original file
    os.makedirs(os.path.dirname(output_path_full) if os.path.dirname(output_path_full) else '.', exist_ok=True)
    save_json(result_dict, output_path_full)
    print(f"Saved clustering results to {output_path_full}")


def combine_field_separated_results(base_path, anno_file, output_path, prefix, suffix,
                                   separate_by_fields, uuid_key="annot_uuid", output_key="cluster_id"):
    """
    Combine results that were separated by fields during clustering into a single file.
    Ensures unique cluster IDs across all field separations.

    Args:
        base_path: Base directory containing field-separated subdirectories
        anno_file: Path to original annotation file
        output_path: Directory to save output
        prefix: Output filename prefix
        suffix: Output filename suffix
        separate_by_fields: List of field names used for separation
        uuid_key: Key for UUID in annotations
        output_key: Key name for saving cluster IDs (default: cluster_id)
    """
    # Build regex pattern to match and extract field values
    regex_pattern = "_".join([f"{field}-([^_]+)" for field in separate_by_fields])
    regex = re.compile(regex_pattern)

    # Find all directories matching the pattern
    glob_pattern = "_".join([f"{field}-*" for field in separate_by_fields])
    pattern_path = os.path.join(base_path, glob_pattern)

    print(f"Looking for directories matching: {pattern_path}")

    # Load original annotation file
    original_data = load_json(anno_file)

    # Create result dict with same structure
    result_dict = {}
    if 'categories' in original_data:
        result_dict['categories'] = original_data['categories']
    if 'images' in original_data:
        result_dict['images'] = original_data['images']

    # Collect all UUID to cluster mappings with unique cluster IDs
    global_uuid_to_cluster = {}
    cluster_id_offset = 0

    # Process each field-separated directory
    dirs_processed = 0
    for input_dir in sorted(glob.glob(pattern_path)):
        if os.path.isdir(input_dir):
            dir_name = os.path.basename(input_dir)
            match = regex.match(dir_name)

            if match:
                # Extract field values from regex groups
                field_combo = dict(zip(separate_by_fields, match.groups()))
                print(f"\nProcessing: {field_combo}")

                # Load clustering results for this field combination
                clustering_file = os.path.join(input_dir, "clustering.json")
                node2uuid_file = os.path.join(input_dir, "node2uuid_file.json")

                if not (os.path.exists(clustering_file) and os.path.exists(node2uuid_file)):
                    print(f"  Warning: Missing files in {input_dir}")
                    continue

                clusters = load_json(clustering_file)
                node2uuid = load_json(node2uuid_file)

                # Build mapping with offset cluster IDs to ensure uniqueness
                for cluster_id, nodes in clusters.items():
                    # Create new unique cluster ID
                    new_cluster_id = str(cluster_id_offset)
                    cluster_id_offset += 1

                    for node in nodes:
                        uuid = node2uuid.get(str(node))
                        if uuid:
                            # Include field values in the cluster assignment for tracking
                            global_uuid_to_cluster[uuid] = new_cluster_id

                dirs_processed += 1

    print(f"\nProcessed {dirs_processed} field-separated directories")
    print(f"Total unique clusters: {cluster_id_offset}")
    print(f"Total UUIDs with cluster assignments: {len(global_uuid_to_cluster)}")

    # Apply cluster IDs to all annotations
    if 'annotations' in original_data:
        updated_annotations = []
        assigned_count = 0

        for ann in original_data['annotations']:
            # Create a copy to preserve all original fields
            updated_ann = ann.copy()

            # Add cluster ID field based on UUID mapping
            cluster_id = global_uuid_to_cluster.get(ann.get(uuid_key))
            if cluster_id is not None:
                updated_ann[output_key] = cluster_id
                assigned_count += 1
            else:
                updated_ann[output_key] = None

            updated_annotations.append(updated_ann)

        print(f"Assigned cluster IDs to {assigned_count}/{len(updated_annotations)} annotations")
        result_dict['annotations'] = updated_annotations

    # Build output filename
    parts = []
    if prefix:
        parts.append(prefix)
    if suffix:
        parts.append(suffix)
    if parts:
        output_filename = '_'.join(parts) + '.json'
    else:
        output_filename = 'beta_stability_annots.json'

    output_path_full = os.path.join(output_path, output_filename)

    # Save final result
    os.makedirs(os.path.dirname(output_path_full) if os.path.dirname(output_path_full) else '.', exist_ok=True)
    save_json(result_dict, output_path_full)
    print(f"Saved combined clustering results to {output_path_full}")


def process_field_separated_results(base_path, anno_file, output_path, prefix, suffix,
                                   separate_by_fields, uuid_key="annot_uuid", output_key="cluster_id"):
    """
    Process results that were separated by fields during clustering.
    Creates separate output files for each field combination.

    Args:
        base_path: Base directory containing field-separated subdirectories
        anno_file: Path to original annotation file
        output_path: Directory to save output
        prefix: Output filename prefix
        suffix: Output filename suffix
        separate_by_fields: List of field names used for separation
        uuid_key: Key for UUID in annotations
        output_key: Key name for saving cluster IDs (default: cluster_id)
    """
    # Build regex pattern to match and extract field values
    regex_pattern = "_".join([f"{field}-([^_]+)" for field in separate_by_fields])
    regex = re.compile(regex_pattern)

    # Find all directories matching the pattern
    glob_pattern = "_".join([f"{field}-*" for field in separate_by_fields])
    pattern_path = os.path.join(base_path, glob_pattern)

    print(f"Looking for directories matching: {pattern_path}")

    for input_dir in glob.glob(pattern_path):
        if os.path.isdir(input_dir):
            dir_name = os.path.basename(input_dir)
            match = regex.match(dir_name)

            if match:
                # Extract field values from regex groups
                field_combo = dict(zip(separate_by_fields, match.groups()))
                print(f"\nProcessing: {field_combo}")
                save_clustering_results(input_dir, anno_file, output_path, prefix, suffix,
                                      field_filters=field_combo, uuid_key=uuid_key, output_key=output_key)


def main():
    parser = argparse.ArgumentParser(
        description="Save clustering results in the correct format for downstream processing."
    )

    parser.add_argument("anno_file", type=str,
                       help="Path to the original annotation file")
    parser.add_argument("clustering_dir", type=str,
                       help="Directory containing clustering results (clustering.json, node2uuid_file.json)")
    parser.add_argument("output_dir", type=str,
                       help="Directory to save formatted output")
    parser.add_argument("--prefix", type=str, default="clustering",
                       help="Output filename prefix (default: clustering)")
    parser.add_argument("--suffix", type=str, default="results",
                       help="Output filename suffix (default: results)")
    parser.add_argument("--uuid_key", type=str, default="uuid",
                       help="Key for UUID in annotations (default: uuid)")
    parser.add_argument("--output_key", type=str, default="cluster_id",
                       help="Key name for saving cluster IDs (default: cluster_id)")

    # Field separation options
    parser.add_argument("--separate_by_fields", nargs='+',
                       help="List of fields used for separation (e.g., viewpoint encounter)")
    parser.add_argument("--field_values", type=str,
                       help="Specific field values to process (format: field1=value1,field2=value2)")

    args = parser.parse_args()

    if args.separate_by_fields:
        # Process field-separated results
        process_field_separated_results(
            args.clustering_dir,
            args.anno_file,
            args.output_dir,
            args.prefix,
            args.suffix,
            args.separate_by_fields,
            args.uuid_key,
            args.output_key
        )
    elif args.field_values:
        # Process specific field combination
        field_filters = {}
        for field_value in args.field_values.split(','):
            field, value = field_value.split('=')
            field_filters[field] = value

        save_clustering_results(
            args.clustering_dir,
            args.anno_file,
            args.output_dir,
            args.prefix,
            args.suffix,
            field_filters,
            args.uuid_key,
            args.output_key
        )
    else:
        # Process without field filtering
        save_clustering_results(
            args.clustering_dir,
            args.anno_file,
            args.output_dir,
            args.prefix,
            args.suffix,
            uuid_key=args.uuid_key,
            output_key=args.output_key
        )


if __name__ == "__main__":
    main()