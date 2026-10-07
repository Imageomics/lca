#!/usr/bin/env python3
"""
Config Format Converter

Automatically converts old config files to the new unified format while 
preserving backwards compatibility and all existing functionality.

Usage:
    python convert_config.py --input old_config.yaml --output new_config.yaml
    python convert_config.py --input old_config.yaml --output new_config.yaml --full-conversion
"""

import argparse
import yaml
import os
from copy import deepcopy


def load_config(file_path):
    """Load YAML config file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, file_path):
    """Save config to YAML file with proper formatting."""
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)


def convert_human_reviewer_format(config):
    """
    Convert old human reviewer format to new explicit format.
    
    Old: "miewid human" + simulate_human flag
    New: "miewid simulated_human" / "miewid ui_human"
    """
    conversions_made = []
    
    # Check edge_weights in lca section
    if 'lca' in config and 'edge_weights' in config['lca']:
        edge_weights = config['lca']['edge_weights']
        
        if 'augmentation_names' in edge_weights:
            aug_names = edge_weights['augmentation_names']
            
            # Convert string to list for processing
            if isinstance(aug_names, str):
                aug_list = aug_names.split()
            else:
                aug_list = aug_names
            
            # Convert "human" to specific type
            if 'human' in aug_list:
                # Check simulate_human flag to determine type
                simulate_human = edge_weights.get('simulate_human', 
                                                config.get('lca', {}).get('simulate_human', True))
                
                # Replace "human" with specific type
                new_aug_list = []
                for item in aug_list:
                    if item == 'human':
                        new_type = 'simulated_human' if simulate_human else 'ui_human'
                        new_aug_list.append(new_type)
                        conversions_made.append(f"Converted 'human' to '{new_type}'")
                    else:
                        new_aug_list.append(item)
                
                # Update the config
                edge_weights['augmentation_names'] = ' '.join(new_aug_list)
                
                # Remove old simulate_human flag if it exists
                if 'simulate_human' in edge_weights:
                    del edge_weights['simulate_human']
                    conversions_made.append("Removed 'simulate_human' flag")
    
    # Check top-level edge_weights
    if 'edge_weights' in config:
        edge_weights = config['edge_weights']
        
        if 'augmentation_names' in edge_weights:
            aug_names = edge_weights['augmentation_names']
            
            if isinstance(aug_names, str):
                aug_list = aug_names.split()
            else:
                aug_list = aug_names
            
            if 'human' in aug_list:
                # Default to simulated_human for top-level
                new_aug_list = ['simulated_human' if item == 'human' else item for item in aug_list]
                edge_weights['augmentation_names'] = ' '.join(new_aug_list)
                conversions_made.append("Converted top-level 'human' to 'simulated_human'")
    
    return conversions_made


def add_algorithm_type(config):
    """Add algorithm_type field if missing."""
    if 'algorithm_type' not in config:
        config['algorithm_type'] = 'lca'
        return ["Added algorithm_type: 'lca'"]
    return []


def add_algorithm_section(config):
    """Add algorithm section with shared parameters."""
    conversions_made = []
    
    if 'algorithm' not in config:
        config['algorithm'] = {
            'target_edges': 0,
            'initial_topk': 10
        }
        conversions_made.append("Added 'algorithm' section with shared parameters")
    
    return conversions_made


def move_to_top_level(config, full_conversion=False):
    """
    Optionally move edge_weights and logging to top level.
    Only done in full conversion mode.
    """
    conversions_made = []
    
    if not full_conversion:
        return conversions_made
    
    # Move edge_weights to top level
    if 'lca' in config and 'edge_weights' in config['lca'] and 'edge_weights' not in config:
        config['edge_weights'] = config['lca']['edge_weights']
        del config['lca']['edge_weights']
        conversions_made.append("Moved edge_weights to top level")
    
    # Move logging to top level  
    if 'lca' in config and 'logging' in config['lca'] and 'logging' not in config:
        config['logging'] = config['lca']['logging']
        del config['lca']['logging']
        conversions_made.append("Moved logging to top level")
    
    return conversions_made


def add_gc_section(config):
    """Add GC configuration section with sensible defaults."""
    conversions_made = []
    
    if 'gc' not in config:
        # Get verifier_name from LCA if available
        verifier_name = 'miewid'
        if 'lca' in config:
            verifier_name = config['lca'].get('verifier_name', 'miewid')
        
        config['gc'] = {
            'verifier_name': verifier_name,
            'theta': 0.1,
            'validation_step': 20,
        #     'flip_threshold': 0.5,
        #     'negative_threshold': 0.5,
        #     'densify_threshold': 10,
        }
        conversions_made.append("Added 'gc' section for Graph Consistency algorithm support")
    
    return conversions_made

def convert_config(input_config, full_conversion=False, add_gc_support=False):
    """
    Convert old config format to new unified format.
    
    Args:
        input_config: Dict containing the old config
        full_conversion: If True, move edge_weights and logging to top level
        add_gc_support: If True, add GC configuration section
    
    Returns:
        Tuple of (converted_config, list_of_changes_made)
    """
    config = deepcopy(input_config)
    all_conversions = []
    
    # Apply conversions
    all_conversions.extend(add_algorithm_type(config))
    all_conversions.extend(add_algorithm_section(config))
    all_conversions.extend(convert_human_reviewer_format(config))
    all_conversions.extend(move_to_top_level(config, full_conversion))
    all_conversions.extend(convert_to_verifier_names(config))      
    all_conversions.extend(add_classifier_thresholds_example(config))   
    all_conversions.extend(add_weight_thresholds_example(config))       
    
    if add_gc_support:
        all_conversions.extend(add_gc_section(config))
    
    return config, all_conversions


def main():
    parser = argparse.ArgumentParser(description="Convert old config files to new unified format")
    parser.add_argument('--input', '-i', required=True, help='Input config file path')
    parser.add_argument('--output', '-o', required=True, help='Output config file path')
    parser.add_argument('--full-conversion', action='store_true', 
                       help='Move edge_weights and logging to top level (optional)')
    parser.add_argument('--add-gc-support', action='store_true',
                       help='Add GC configuration section for algorithm switching')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without writing output file')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    # Load input config
    try:
        input_config = load_config(args.input)
        print(f"Loaded config from: {args.input}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Convert config
    converted_config, conversions = convert_config(
        input_config, 
        full_conversion=args.full_conversion,
        add_gc_support=args.add_gc_support
    )
    
    # Show changes
    if conversions:
        print("\nConversions applied:")
        for conversion in conversions:
            print(f"  ✓ {conversion}")
    else:
        print("\nNo conversions needed - config is already in new format")
    
    # Save or show output
    if args.dry_run:
        print(f"\nDry run - would save to: {args.output}")
        print("\nConverted config preview:")
        print(yaml.dump(converted_config, default_flow_style=False, indent=2))
    else:
        try:
            save_config(converted_config, args.output)
            print(f"\nSaved converted config to: {args.output}")
            
            # Add usage examples
            print(f"\nUsage examples:")
            print(f"  # Run with LCA (current setting):")
            print(f"  python run.py --config {args.output}")
            
            if args.add_gc_support:
                print(f"  # Run with Graph Consistency:")
                print(f"  # 1. Change 'algorithm_type: lca' to 'algorithm_type: gc' in {args.output}")
                print(f"  # 2. python run.py --config {args.output}")
                
        except Exception as e:
            print(f"Error saving config: {e}")
            return 1
    
    return 0



def convert_to_verifier_names(config):
    """
    Convert augmentation_names to verifier_names for better naming consistency.
    Maintains backwards compatibility.
    """
    conversions_made = []
    
    # Check edge_weights in lca section
    if 'lca' in config and 'edge_weights' in config['lca']:
        edge_weights = config['lca']['edge_weights']
        
        if 'augmentation_names' in edge_weights and 'verifier_names' not in edge_weights:
            edge_weights['verifier_names'] = edge_weights['augmentation_names']
            # Keep old parameter for backwards compatibility
            conversions_made.append("Added 'verifier_names' alongside 'augmentation_names' in LCA section")
    
    # Check edge_weights in gc section
    if 'gc' in config and 'edge_weights' in config['gc']:
        edge_weights = config['gc']['edge_weights']
        
        if 'augmentation_names' in edge_weights and 'verifier_names' not in edge_weights:
            edge_weights['verifier_names'] = edge_weights['augmentation_names']
            conversions_made.append("Added 'verifier_names' alongside 'augmentation_names' in GC section")
    
    # Check top-level edge_weights
    if 'edge_weights' in config:
        edge_weights = config['edge_weights']
        
        if 'augmentation_names' in edge_weights and 'verifier_names' not in edge_weights:
            edge_weights['verifier_names'] = edge_weights['augmentation_names']
            conversions_made.append("Added 'verifier_names' alongside 'augmentation_names' at top level")
    
    return conversions_made


def add_classifier_thresholds_example(config):
    """
    Add example classifier_thresholds section to demonstrate new feature.
    Only adds if GC algorithm is present and no thresholds exist.
    """
    conversions_made = []
    
    # Only add examples for GC configs
    if config.get('algorithm_type') != 'gc':
        return conversions_made
    
    # Find edge_weights location
    edge_weights_location = None
    if 'edge_weights' in config:
        edge_weights_location = config['edge_weights']
    elif 'gc' in config and 'edge_weights' in config['gc']:
        edge_weights_location = config['gc']['edge_weights']
    elif 'lca' in config and 'edge_weights' in config['lca']:
        edge_weights_location = config['lca']['edge_weights']
    
    if edge_weights_location and 'classifier_thresholds' not in edge_weights_location:
        # Get verifier names to suggest thresholds for
        verifier_names_str = edge_weights_location.get('verifier_names', 
                                                     edge_weights_location.get('augmentation_names', ''))
        if isinstance(verifier_names_str, str):
            verifier_names = verifier_names_str.split()
        else:
            verifier_names = verifier_names_str
        
        # Add example thresholds for non-human verifiers
        example_thresholds = {}
        for name in verifier_names:
            if name != 'human' and name in ['binary', 'lightglue']:  # Common threshold candidates
                if name == 'binary':
                    example_thresholds[name] = 0.8
                elif name == 'lightglue':
                    example_thresholds[name] = 0.6
        
        if example_thresholds:
            edge_weights_location['classifier_thresholds'] = example_thresholds
            threshold_names = list(example_thresholds.keys())
            conversions_made.append(f"Added example 'classifier_thresholds' for {threshold_names}")
    
    return conversions_made


def add_weight_thresholds_example(config):
    """
    Add example weight_thresholds section for WeighterBasedClassifier configuration.
    Only adds if there are weighter-based classifiers without thresholds.
    """
    conversions_made = []
    
    # Find edge_weights location
    edge_weights_location = None
    if 'edge_weights' in config:
        edge_weights_location = config['edge_weights']
    elif 'lca' in config and 'edge_weights' in config['lca']:
        edge_weights_location = config['lca']['edge_weights']
    elif 'gc' in config and 'edge_weights' in config['gc']:
        edge_weights_location = config['gc']['edge_weights']
    
    if edge_weights_location and 'weight_thresholds' not in edge_weights_location:
        # Get verifier names
        verifier_names_str = edge_weights_location.get('verifier_names', 
                                                     edge_weights_location.get('augmentation_names', ''))
        if isinstance(verifier_names_str, str):
            verifier_names = verifier_names_str.split()
        else:
            verifier_names = verifier_names_str
        
        # Add example weight thresholds for common weighter-based classifiers
        classifier_thresholds = edge_weights_location.get('classifier_thresholds', {})
        example_weight_thresholds = {}
        
        for name in verifier_names:
            # If classifier doesn't have threshold-based config, it might use weighter
            if name not in classifier_thresholds and name not in {'human', 'simulated_human', 'ui_human', 'no_human'}:
                if name == 'miewid':
                    example_weight_thresholds[name] = 10
        
        if example_weight_thresholds:
            edge_weights_location['weight_thresholds'] = example_weight_thresholds
            threshold_names = list(example_weight_thresholds.keys())
            conversions_made.append(f"Added example 'weight_thresholds' for weighter-based classifiers: {threshold_names}")
    
    return conversions_made


def add_comments_and_examples(config):
    """Add helpful comments about new features."""
    comments = []
    
    # Add comment about algorithm switching
    if 'algorithm_type' in config:
        comments.append("# Algorithm types: 'lca' or 'gc'")
    
    # Add comment about shared algorithm parameters
    if 'algorithm' in config:
        comments.append("# Shared algorithm parameters: tries_before_edge_done, validation_step")
    
    # Add comment about verifier naming
    if 'edge_weights' in config or ('lca' in config and 'edge_weights' in config['lca']):
        comments.append("# verifier_names (new): preferred over augmentation_names (old)")
        comments.append("# Human reviewer types: simulated_human, ui_human, no_human")
    
    # Add comment about multi-classifier support
    if config.get('algorithm_type') == 'gc':
        comments.append("# GC multi-classifier: Mix weighter-based and threshold-based classifiers")
        comments.append("# classifier_thresholds: Specify constant thresholds for specific classifiers")
        comments.append("# weight_thresholds: Custom weight thresholds for weighter-based classifiers")
    
    # Add comment about weighter calibration caching
    edge_weights_sections = []
    if 'edge_weights' in config:
        edge_weights_sections.append(config['edge_weights'])
    if 'lca' in config and 'edge_weights' in config['lca']:
        edge_weights_sections.append(config['lca']['edge_weights'])
    
    for edge_weights in edge_weights_sections:
        if 'verifier_file' in edge_weights:
            comments.append("# verifier_file: Use cached weighter calibration for faster startup")
            break
    
    return comments

if __name__ == '__main__':
    exit(main())