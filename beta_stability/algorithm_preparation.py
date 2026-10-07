"""
Unified algorithm preparation module.
Handles all common setup logic for both Beta Stability.
"""

import types
import math
import numpy as np
import random
import os
import logging
import datetime
import tempfile
import shutil
import re
from pathlib import Path


from beta_stability.preprocess import preprocess_data
from beta_stability.embeddings.embeddings import Embeddings
from beta_stability.util.tools import *
from beta_stability.util.cluster_validator import ClusterValidator
from beta_stability.classifier_system import ClassifierManager, WeighterBasedClassifier, ThresholdBasedClassifier
from beta_stability.embeddings.metadata_verifier import MetadataEmbeddings
from beta_stability.embeddings.tracking_id_verifier import TrackingIdEmbeddings
from beta_stability.util.robust_gmm_threshold import find_threshold as robust_gmm_find_threshold
from beta_stability.baselines.hdbscan_algorithm import HDBSCANAlgorithm
from beta_stability.baselines.review_algorithm import ReviewAlgorithm
from beta_stability.embeddings.kmeans_embeddings import KMeansEmbeddings
from beta_stability.stability_algorithm import BetaStabilityAlgorithm
from beta_stability.baselines.np3_aas_algorithm import NP3AASAlgorithm
from beta_stability.baselines.nis_algorithm import NISAlgorithm


def _get_fallback_percentile(config):
    """Read edge_weights.fallback_percentile from config.

    Used as the score percentile chosen as threshold when the GMM detector
    fails (NaN params, K=2 didn't converge, etc.). Default 99 — conservative
    (very few false positives but may over-fragment). For embeddings where
    p99 is too selective (e.g. GZCD + MegaDescriptor), lower it (e.g. 90-95)
    to start from a less fragmented graph.
    """
    edge_weights = config.get('edge_weights', {})
    return float(edge_weights.get('fallback_percentile', 99))

logger = logging.getLogger("beta_stability")

def parse_verifier_names(verifier_names):
    """
    Parse verifier names with metadata syntax.
    
    Examples:
    - "metadata(miewid) lightglue human" → metadata wraps miewid explicitly
    - "metadata miewid human" → metadata wraps next verifier (miewid) implicitly
    
    Returns:
    - parsed_verifiers: List of (verifier_name, base_embeddings_name) tuples
    """
    parsed_verifiers = []
    
    meta_names = ['metadata', 'tracking', 'kmeans']

    i = 0
    while i < len(verifier_names):
        current = verifier_names[i]
        meta = False
        for meta_name in meta_names:
            if current.startswith(meta_name + '(') and current.endswith(')'):
                # Explicit: metadata(miewid)
                # Remove "metadata(" prefix and ")" suffix
                base_name = current.removeprefix(meta_name + '(').removesuffix(')')
                if not base_name:
                    raise ValueError(f"Empty base verifier in {meta_name}() syntax")
                parsed_verifiers.append((meta_name, base_name))
                meta = True
                break
            elif current == meta_name:
                # Implicit: metadata uses next verifier as base
                if i + 1 >= len(verifier_names):
                    raise ValueError(f"{meta_name} verifier needs a base verifier")
                base_name = verifier_names[i + 1]
                parsed_verifiers.append((meta_name, base_name))
                i += 1
                meta = True
                break
        if not meta:
            parsed_verifiers.append((current, None))
        i += 1
    
    return parsed_verifiers

def prepare_common(config):
    """
    Handle all shared preparation tasks common to both algorithms.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Common data needed by both algorithms
    """
    seed = config.get('seed', 42)
    np.random.seed(seed)
    random.seed(seed)
    
    data_params = config['data']
    images_dir = data_params.get('images_dir', None)
    if images_dir == "None":
        images_dir = None

    data_params['images_dir'] = images_dir
    algorithm_type = config.get('algorithm_type', 'stability')
    algorithm_config = config.get(algorithm_type, {})
    
    # 1. Resolve embeddings config and load all pickles.
    #
    # Config schema (new, preferred):
    #   data:
    #     embeddings:
    #       miewid:
    #         file: /path/to/miewid.pickle
    #       megadescriptor:
    #         file: /path/to/megadescriptor.pickle
    #
    # Back-compat: legacy `data.embedding_file: <path>` is auto-promoted to
    # `data.embeddings.miewid.file = <path>` if no explicit miewid section is
    # already present. Mixing both forms is allowed; explicit sections win.
    #
    # All embeddings must cover the same uuid SET. The first one (in dict
    # insertion order) provides the canonical uuid ORDERING used for GT,
    # preprocess_data, and the all_node2uuid map.
    logger.info("Loading embeddings and preprocessing data...")

    embeddings_cfg = {}
    legacy_path = data_params.get('embedding_file')
    if legacy_path is not None:
        embeddings_cfg['miewid'] = {'file': legacy_path}
    explicit_embeddings = data_params.get('embeddings') or {}
    if not isinstance(explicit_embeddings, dict):
        raise ValueError(
            f"data.embeddings must be a dict of {{name: {{file: ...}}}}, "
            f"got {type(explicit_embeddings).__name__}"
        )
    for name, spec in explicit_embeddings.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"data.embeddings.{name} must be a dict (got {type(spec).__name__})"
            )
        if 'file' not in spec:
            raise KeyError(f"data.embeddings.{name} is missing required 'file' key")
        embeddings_cfg[name] = dict(spec)

    if not embeddings_cfg:
        raise KeyError(
            "config must specify at least one embedding in `data.embeddings` "
            "or via legacy `data.embedding_file`"
        )

    # Load each pickle exactly once.
    loaded_embeddings = {}  # name -> (np.ndarray, list[uuid])
    for name, spec in embeddings_cfg.items():
        path = spec['file']
        logger.info(f"Loading embedding '{name}' from {path}")
        embs, ids = load_pickle(path)
        loaded_embeddings[name] = (np.asarray(embs), list(ids))

    # First-loaded embedding provides the canonical uuid ordering.
    uuid_source_name = next(iter(loaded_embeddings))
    embeddings, uuids = loaded_embeddings[uuid_source_name]
    logger.info(
        f"Loaded {len(loaded_embeddings)} embedding(s): "
        f"{list(loaded_embeddings.keys())}; uuid ordering from '{uuid_source_name}'"
    )

    # All embeddings must cover the same uuid set.
    ref_uuid_set = set(uuids)
    for name, (_, other_uuids) in loaded_embeddings.items():
        if name == uuid_source_name:
            continue
        other_set = set(other_uuids)
        if other_set != ref_uuid_set:
            missing = ref_uuid_set - other_set
            extra = other_set - ref_uuid_set
            raise KeyError(
                f"Embedding '{name}' uuid set differs from '{uuid_source_name}': "
                f"{len(missing)} missing, {len(extra)} extra. "
                "All embeddings must cover the same set of uuids."
            )

    name_keys = data_params['name_keys']
    filter_key = '__'.join(name_keys)

    # Support different preprocessing formats
    format_type = data_params.get('format', 'drone')  # 'old' or 'drone'
    id_key = data_params.get('id_key', 'uuid')

    # Create field_filters from any {field}_list entries in config
    field_filters = {}
    for key in data_params:
        if key.endswith('_list'):
            field_name = key[:-5]  # Remove '_list' suffix
            field_filters[field_name] = data_params[key]

    df = preprocess_data(
        data_params['annotation_file'],
        name_keys=name_keys,
        convert_names_to_ids=True,
        n_filter_min=data_params['n_filter_min'],
        n_filter_max=data_params['n_filter_max'],
        images_dir=data_params['images_dir'],
        embedding_uuids=uuids,
        id_key=id_key,
        format=format_type,
        field_filters=field_filters,
        print_func=logger.info
    )

    print_intersect_stats(df, individual_key=filter_key)

    # 2. Setup ground truth and validation
    logger.info("Setting up ground truth and validation...")
    filtered_df = df[df[id_key].isin(uuids)]

    # node2uuid mapping for ALL primary-pickle entries (used for thresholding).
    all_node2uuid = {i: uuid for i, uuid in enumerate(uuids)}


    gt_clustering, gt_node2cid, node2uuid = generate_gt_clusters(filtered_df, filter_key, id_key)

    # Check for single annotation - no clustering needed
    if len(node2uuid) == 1:
        logger.info("Only 1 annotation found - skipping Beta Stability clustering")
        raise SingleAnnotationException(node2uuid, gt_clustering, gt_node2cid)

    cluster_validator = ClusterValidator.from_config(
        config, gt_clustering, gt_node2cid,
        default_method=algorithm_type,
        default_species=config.get('exp_name'),
        default_seed=seed,
    )

    # 3. Per-embedding processing: filter to filtered_df row order, apply PCA,
    # register under its own name in embeddings_dict and unfiltered_embeddings_dict.
    logger.info("Setting up embeddings...")
    distance_power = algorithm_config.get('distance_power', 1)

    embeddings_dict = {}
    unfiltered_embeddings_dict = {}

    for name, (raw_embeddings, raw_uuids) in loaded_embeddings.items():
        uuid_to_row = {u: i for i, u in enumerate(raw_uuids)}
        # Re-index to filtered_df order.
        filtered_arr = np.asarray(
            [raw_embeddings[uuid_to_row[uuid]] for uuid in filtered_df[id_key]]
        )
        # Re-index to primary uuid order (so all_node2uuid lookups align).
        unfiltered_arr = np.asarray(
            [raw_embeddings[uuid_to_row[uuid]] for uuid in uuids]
        )

        embeddings_dict[name] = lazy(
            (lambda fa=filtered_arr, n2u=node2uuid, dp=distance_power:
                Embeddings(fa, n2u, distance_power=dp, print_func=logger.info))
        )
        unfiltered_embeddings_dict[name] = lazy(
            (lambda ua=unfiltered_arr, n2u=all_node2uuid, dp=distance_power:
                Embeddings(ua, n2u, distance_power=dp, print_func=logger.info))
        )


    # Silhouette-based auto-selection of the stability init verifier list.
    # Triggered when stability config has verifier_name: auto. Runs k-means
    # at K = K̂ on the primary embedding and switches to a kmeans-only list
    # if the resulting silhouette is below auto_verifier_silhouette_threshold.
    if algorithm_type == 'stability' and algorithm_config.get('verifier_name') == 'auto':
        algorithm_config['verifier_name'] = _auto_select_stability_verifier(
            config=config,
            algorithm_config=algorithm_config,
            embeddings_dict=embeddings_dict,
            primary_name=algorithm_config.get('auto_verifier_primary', 'miewid'),
        )

    # 4. Setup human reviewer based on augmentation names (with backwards compatibility)
    logger.info("Setting up human reviewer...")
    
    # Backwards compatibility: edge_weights can be top-level or inside algorithm config
    edge_weights = config.get('edge_weights', algorithm_config.get('edge_weights', {}))
    algorithm_config['scorer'] = edge_weights.get('scorer', 'kde')
    algorithm_config['prob_human_correct'] = edge_weights.get('prob_human_correct', 0.98)
    
    prob_human_correct = edge_weights.get('prob_human_correct', 0.98)
    error_mode = edge_weights.get('error_mode', 'both')  # 'both'|'merge_only'|'split_only'
    # aug_names = edge_weights.get('augmentation_names', 'miewid human').split()
    verifier_name_raw = algorithm_config.get('verifier_name', 'miewid')
    if isinstance(verifier_name_raw, list):
        init_verifiers = verifier_name_raw
    else:
        init_verifiers = verifier_name_raw.split()
    verifier_name = init_verifiers[0]  # Primary init verifier (backward compat)

    verifier_names_str = edge_weights.get('verifier_names', edge_weights.get('augmentation_names', 'miewid human'))
    aug_names = verifier_names_str.split() if isinstance(verifier_names_str, str) else verifier_names_str
    parsed_verifiers = parse_verifier_names(aug_names + init_verifiers)

    # Backwards compatibility: handle old "human" + simulate_human flag
    simulate_human = algorithm_config.get('simulate_human', True)
    
    # Determine human reviewer type
    human_reviewer = None
    
    # New format: specific human types in aug_names
    for aug_name in aug_names:
        if aug_name == 'simulated_human':
            human_reviewer = call_get_reviews(df, filter_key, prob_human_correct, error_mode)
            break
        elif aug_name == 'ui_human':
            ui_db_path = data_params.get('ui_db_path')
            if ui_db_path:
                from beta_stability.util.human_db import human_db
                logger.info(f"ui_human - using UI database for human reviews at {ui_db_path}")
                human_reviewer = human_db(ui_db_path, filtered_df, node2uuid)
            else:
                logger.warning("ui_human specified but no ui_db_path provided, falling back to simulated")
                human_reviewer = call_get_reviews(df, filter_key, prob_human_correct, error_mode)
            break
        elif aug_name == 'no_human':
            logger.info("no_human - running without human reviews")
            human_reviewer = lambda _: ([], False)
            break
    
    # Backwards compatibility: handle old "human" in aug_names (from existing configs)
    if human_reviewer is None and 'human' in aug_names:
        if simulate_human:
            human_reviewer = call_get_reviews(df, filter_key, prob_human_correct, error_mode)
        else:
            # Old non-simulated case - try UI database
            ui_db_path = data_params.get('ui_db_path')
            if ui_db_path:
                from beta_stability.util.human_db import human_db
                human_reviewer = human_db(ui_db_path, filtered_df, node2uuid)
            else:
                human_reviewer = call_get_reviews(df, filter_key, prob_human_correct, error_mode)
    
    # Final fallback: default to simulated if no human type found
    if human_reviewer is None:
        logger.info("No human reviewer type specified, defaulting to no_human")
        human_reviewer = lambda _: ([], True)
    
    for name, base_name in parsed_verifiers:
        if name == 'metadata':
            # Create metadata wrapper
            base_embeddings = embeddings_dict[base_name]()
            embeddings_dict[f'metadata({base_name})'] = lazy(lambda: MetadataEmbeddings(
                base_embeddings, df, node2uuid)
            )
        elif name == 'tracking':
            # Create tracking ID wrapper
            base_embeddings = embeddings_dict[base_name]()
            embeddings_dict[f'tracking({base_name})'] = lazy(lambda: TrackingIdEmbeddings.from_embeddings(
                base_embeddings, df, node2uuid, id_key, tracking_key='tracking_id', multiplier=1)
            )
        elif name == 'kmeans':
            # Compute base embeddings threshold using classifier config
            base_embeddings = embeddings_dict[base_name]()
            cls_thresholds = edge_weights.get('classifier_thresholds', {})
            base_spec = cls_thresholds.get(base_name, 'auto(0.15)')
            # Derive max_implied_K from the same cluster-size assumption used to
            # cap K_hat downstream, so the GMM rejects fits whose threshold
            # implies K > n / kmeans_min_cluster_size (a degenerate K_hat).
            stability_cfg = config.get('stability', {})
            kmeans_min_cluster_size = stability_cfg.get('kmeans_min_cluster_size', 1)
            kmeans_fallback_cluster_size = stability_cfg.get('kmeans_fallback_cluster_size', 1)
            n_nodes = len(base_embeddings.ids)
            kmeans_max_k = n_nodes // kmeans_min_cluster_size if kmeans_min_cluster_size > 1 else None
            if isinstance(base_spec, str) and 'auto' in base_spec:
                match = re.match(r'auto\((\d+\.?\d*)\)', base_spec)
                fraction = float(match.group(1)) if match else 0.15
                thresh_emb = unfiltered_embeddings_dict.get(base_name)
                if thresh_emb is not None and callable(thresh_emb):
                    thresh_emb = thresh_emb()
                elif thresh_emb is None:
                    thresh_emb = base_embeddings
                base_threshold = robust_gmm_find_threshold(
                    np.array(thresh_emb.get_all_scores()),
                    entropy_alpha=1 - fraction, verbose=True, print_func=logger.info,
                    plot_path=config.get('logging', {}).get('auto_threshold_plot_path'),
                    fallback_percentile=_get_fallback_percentile(config),
                    max_implied_K=kmeans_max_k)
            else:
                base_threshold = float(base_spec)
            # Cache threshold and predicted F1 on base embeddings so classifier system can reuse it
            base_embeddings._cached_gmm_threshold = base_threshold
            base_embeddings._cached_gmm_predicted_f1 = getattr(robust_gmm_find_threshold, '_last_predicted_f1', None)
            def _make_kmeans(base_emb=base_embeddings, node2uuid=node2uuid,
                             threshold=base_threshold, max_k=kmeans_max_k,
                             fallback_size=kmeans_fallback_cluster_size,
                             fallback_emb=base_embeddings):
                km = KMeansEmbeddings(base_emb, node2uuid, threshold=threshold,
                                      max_k=max_k, print_func=logger.info)
                max_cluster_size = int(np.bincount(km.labels).max())
                if max_cluster_size < fallback_size:
                    logger.info(
                        f"KMeansEmbeddings: max cluster size {max_cluster_size} < "
                        f"{fallback_size}, falling back to base embeddings"
                    )
                    return fallback_emb
                return km
            embeddings_dict[f'kmeans({base_name})'] = lazy(_make_kmeans)
    logger.info("Computing and logging verifier performance statistics...")
    primary_verifier_embeddings = embeddings_dict[verifier_name]()
    
    try:
        # Log top-k accuracy statistics  
        topk_results = primary_verifier_embeddings.get_stats(filtered_df, filter_key, id_key)
        logger.info(f"Top-k Accuracy Statistics: " + ", ".join([f"{k}: {100*v:.2f}%" for (k, v) in topk_results]))

        # Log detailed top-20 matches for each individual
        # top20_results = primary_verifier_embeddings.get_top20_matches(filtered_df, filter_key)
        # for uuid, top20 in top20_results.items():
        #     logger.info(f"ID: {uuid} | TOP-20: " + ", ".join([f"{k}: {v:.2f}" for (k, v) in top20]))
            
    except Exception as e:
        logger.warning(f"Failed to compute verifier statistics: {e}")
        logger.info("Continuing without statistics logging")

    for aug_name in aug_names:
        if 'human' not in aug_name:
            embeddings_dict[aug_name] = embeddings_dict[aug_name]()
    for iv_name in init_verifiers:
        if iv_name not in aug_names:
            if callable(embeddings_dict.get(iv_name)):
                embeddings_dict[iv_name] = embeddings_dict[iv_name]()
    
    algorithm_type = config.get('algorithm_type', 'stability')

    weighters = {}
    weighters_calibration = None

    if 'output_path' in data_params:
        output_path = data_params['output_path']
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = 'tmp'

    algorithm_params = config.get('algorithm', {})
    target_edges = algorithm_params.get('target_edges', 0)
    target_proportion = algorithm_params.get('target_proportion', None)
    initial_topk = algorithm_params.get('initial_topk', 10)
    initial_botk = algorithm_params.get('initial_botk', 0)


    # 9. Return common data
    return {
        'embeddings_dict': embeddings_dict,
        'unfiltered_embeddings_dict': unfiltered_embeddings_dict,
        'gt_clustering': gt_clustering,
        'gt_node2cid': gt_node2cid,
        'node2uuid': node2uuid,
        'cluster_validator': cluster_validator,
        'human_reviewer': human_reviewer,
        'weighters': weighters,
        'weighters_calibration': weighters_calibration,
        'filtered_df': filtered_df,
        'df': df,
        'filter_key': filter_key,
        'verifier_name': verifier_name,
        'init_verifiers': init_verifiers,
        'output_path': output_path,
        'target_edges': target_edges,
        'initial_topk': initial_topk,
        'initial_botk': initial_botk,
        'target_proportion': target_proportion
    }


def generate_weighter_calibration(embeddings_dict, human_reviewer, edge_weights, verifier_name, aug_names, logger, config=None):
    """
    Generate calibration data for weighters with caching support.
    
    Args:
        config: Full configuration dict for accessing cache settings and db_path
        
    Returns:
        dict: Calibration data for all embedding methods
    """
    # Determine cache file location
    cache_file = None
    if 'verifier_file' in edge_weights:
        # Use specified cache file
        cache_file = edge_weights['verifier_file']
        logger.info(f"Using specified cache file: {cache_file}")
    logger.info(os.path.exists(cache_file))
    # Try to load cached calibration
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading cached weighter calibration from {cache_file}")
        try:
            return load_json(cache_file)
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}")
            logger.info("Proceeding with fresh calibration computation")
    
    # Compute calibration (existing logic)
    logger.info("Computing weighter calibration...")
    num_pos_needed = edge_weights.get('num_pos_needed', 300)
    num_neg_needed = edge_weights.get('num_neg_needed', 50)
    
    wgtrs_calib_dict = {}
    
    # Get verifier edges for calibration
    verifier_edges = embeddings_dict[verifier_name].get_edges()
    
    # Generate ground truth for main verifier
    logger.info(f"Generating calibration data for {verifier_name}...")
    gt_weights, pos_edges, neg_edges = process_verifier_edges(
        verifier_name, verifier_edges, human_reviewer, num_pos_needed, num_neg_needed, logger
    )
    wgtrs_calib_dict[verifier_name] = gt_weights
    
    # Generate calibration for other methods (excluding human reviewer types)
    human_types = {'simulated_human', 'ui_human', 'no_human', 'human'}  # Include old 'human' for compatibility
    for method in aug_names:
        if method in human_types or method == verifier_name:
            continue
            
        if method not in embeddings_dict:
            logger.warning(f"Embeddings for method {method} not found.")
            continue
            
        logger.info(f"Generating calibration data for {method}...")
        get_score = embeddings_dict[method].get_score
        wgtrs_calib_dict[method] = process_edges(method, pos_edges, neg_edges, get_score, logger)
    
    # Save to cache
    if cache_file:
        try:
            # Use the same save function as the old system
            def save_probs_to_db(data, output_path):
                dir_name = os.path.dirname(output_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                write_json(data, output_path)
            
            save_probs_to_db(wgtrs_calib_dict, cache_file)
            logger.info(f"Saved weighter calibration to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    return wgtrs_calib_dict


def process_verifier_edges(method, edges, human_reviewer, num_pos_needed, num_neg_needed, logger):
    """Generate ground truth random data and update the calibration dictionary."""
    pos, neg, quit = generate_ground_truth_random(edges, human_reviewer, num_pos_needed, num_neg_needed)
    logger.info(f"Method: {method}, Num pos edges: {len(pos)}, num neg edges: {len(neg)}")
    return {
        "gt_positive_probs": [p for _, _, p in pos],
        "gt_negative_probs": [p for _, _, p in neg],
    }, pos, neg


def process_edges(method, pos_edges, neg_edges, get_score, logger):
    """Generate ground truth random data and update the calibration dictionary."""
    pos, neg = generate_calib_weights(pos_edges, neg_edges, get_score)
    logger.info(f"Method: {method}, Num pos edges: {len(pos)}, num neg edges: {len(neg)}")
    return {
        "gt_positive_probs": [p for _, _, p in pos],
        "gt_negative_probs": [p for _, _, p in neg],
    }


def setup_logging(config):
    """
    Setup logging configuration with backwards compatibility.
    """
    # exp_name = config.get('exp_name', "")
    algorithm_config = {}
    
    # Backwards compatibility: logging can be top-level or inside algorithm config
    logging_config = config.get('logging', algorithm_config.get('logging', {}))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    log_level = logging_config.get('log_level', 'INFO')
    log_file = logging_config.get('log_file')
    
    if log_file is not None:
        if logging_config.get("update_log_file", True):
            log_file_name = Path(log_file)
            log_file_name = log_file_name.with_name(log_file_name.stem + f'_{timestamp}' + log_file_name.suffix)
            logging_config['log_file'] = log_file_name

        logger = logging.getLogger("beta_stability")
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

        file_mode = logging_config.get("file_mode", 'w')
        handler = logging.FileHandler(log_file, mode=file_mode)
        handler.setLevel(log_level)
        handler.setFormatter(get_formatter())
        logger.addHandler(handler)
        
        # print(f"Logging to {os.path.abspath(log_file)}")
    return log_file, timestamp


def create_algorithm(config):
    """Factory: build the appropriate algorithm instance from a config."""
    log_file, timestamp = setup_logging(config)
    if log_file is not None:
        print(f"Logging to {os.path.abspath(log_file)}")

    common_data = prepare_common(config)
    common_data['timestamp'] = timestamp

    algorithm_type = config.get('algorithm_type', 'stability')

    if algorithm_type == 'hdbscan':
        algorithm = prepare_hdbscan(common_data, config)
    elif algorithm_type == 'manual_review':
        algorithm = prepare_manual_review(common_data, config)
    elif algorithm_type == 'thresholded_review':
        algorithm = prepare_thresholded_review(common_data, config)
    elif algorithm_type == 'stability':
        algorithm = prepare_stability(common_data, config)
    elif algorithm_type == 'np3_aas':
        algorithm = prepare_np3_aas(common_data, config)
    elif algorithm_type == 'nis':
        algorithm = prepare_nis(common_data, config)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")

    logger.info(f"Created {algorithm_type.upper()} algorithm instance")
    return algorithm, common_data


def prepare_hdbscan(common_data, config):
    """
    Prepare HDBSCAN algorithm instance.

    Args:
        common_data: Common data from prepare_common
        config: Configuration dictionary

    Returns:
        HDBSCANAlgorithm: Configured HDBSCAN algorithm instance
    """
    # Get all nodes from node2uuid mapping
    all_nodes = list(common_data['node2uuid'].keys())

    # Get embeddings dictionary
    embeddings_dict = common_data['embeddings_dict']

    # Create and return HDBSCAN instance
    hdbscan_instance = HDBSCANAlgorithm(config, all_nodes, embeddings_dict)

    return hdbscan_instance


def prepare_manual_review(common_data, config):
    """
    Prepare Manual Review algorithm instance.

    Args:
        common_data: Common data from prepare_common
        config: Configuration dictionary

    Returns:
        ReviewAlgorithm: review baseline with no auto-accept threshold
    """
    # Get manual_review config section if it exists, otherwise use defaults
    manual_config = config.get('manual_review', {})

    # Create and return Manual Review instance
    return ReviewAlgorithm(manual_config, common_data)


def prepare_thresholded_review(common_data, config):
    """
    Prepare Thresholded Review algorithm instance.

    Args:
        common_data: Common data from prepare_common
        config: Configuration dictionary

    Returns:
        ReviewAlgorithm: review baseline with an auto-accept threshold
    """
    # Get thresholded_review config section if it exists, otherwise use defaults
    thresholded_config = config.get('thresholded_review', {})

    # Create and return Thresholded Review instance
    return ReviewAlgorithm(thresholded_config, common_data)


def estimate_num_individuals_from_topk(embeddings, threshold, topk=10):
    """
    Estimate the number of individuals by counting connected components
    of the top-k graph after thresholding.

    Builds the same top-k neighbor graph used by the algorithm, classifies
    edges as positive (above threshold) or negative, then counts the
    connected components of positive edges. This gives a much better
    estimate than pi_positive because it accounts for graph structure.

    Args:
        embeddings: Embeddings object with get_edges() method
        threshold: Classifier threshold for positive/negative classification
        topk: Number of nearest neighbors (should match initial_topk)

    Returns:
        tuple: (estimated_num_individuals, predicted_f1_or_none)
    """
    import networkx as nx

    # Get top-k edges with scores
    edges = list(embeddings.get_edges(topk=topk, target_edges=0, target_proportion=0))
    if not edges:
        n = len(embeddings.ids) if hasattr(embeddings, 'ids') else 0
        logger.info(f"Estimated num_individuals from top-k graph: {n} (no edges)")
        return n

    # Build graph of positive edges only
    G = nx.Graph()
    all_nodes = set()
    n_positive = 0
    n_negative = 0
    for edge in edges:
        n0, n1 = edge[0], edge[1]
        score = edge[2] if len(edge) > 2 else 0.0
        all_nodes.add(n0)
        all_nodes.add(n1)
        if score > threshold:
            G.add_edge(n0, n1)
            n_positive += 1
        else:
            n_negative += 1

    # Add isolated nodes (no positive edges)
    for node in all_nodes:
        if node not in G:
            G.add_node(node)

    num_components = nx.number_connected_components(G)
    n_total = len(all_nodes)
    logger.info(f"Estimated num_individuals from top-k graph: {num_components} "
                f"({n_positive} positive, {n_negative} negative edges, "
                f"{n_total} nodes, threshold={threshold:.4f})")
    return num_components


def count_pccs_from_all_positives(embeddings, threshold):
    """Count positive-connected components using ALL classifier-positive
    pairs (score > threshold), independent of any top-K choice.

    This is what makes the auto-K derivation non-circular: K̂ comes from
    the classifier's decisions on all pairs, not from a top-K subgraph.

    Args:
        embeddings: Embeddings instance.
        threshold: Classifier threshold τ. Pairs with score > τ are positive.

    Returns:
        int: Number of positive-connected components (K̂).
    """
    import networkx as nx

    # get_edges(upper_threshold=τ) returns every pair with score > τ,
    # regardless of top-K. On a strong embedding this is a sparse set
    # (~21k edges for GZCD/MiewID out of ~7M pairs).
    edges = embeddings.get_edges(
        topk=0, botk=0, target_edges=0, target_proportion=0,
        upper_threshold=threshold,
    )

    n_nodes = len(embeddings.ids) if hasattr(embeddings, 'ids') else 0
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    n_positive = 0
    for edge in edges:
        n0, n1 = edge[0], edge[1]
        score = edge[2] if len(edge) > 2 else 0.0
        if score > threshold:
            G.add_edge(n0, n1)
            n_positive += 1
    num_components = nx.number_connected_components(G)
    logger.info(
        f"K̂ from all-positives graph: {num_components} PCCs "
        f"({n_positive} positive edges over {n_nodes} nodes, "
        f"threshold={threshold:.4f})"
    )
    return num_components


def _auto_select_stability_verifier(config, algorithm_config, embeddings_dict,
                                    primary_name='miewid'):
    """Silhouette-based auto-selection of the stability init verifier list.

    Runs k-means at K = K̂ on the primary embedding matrix and measures the
    resulting silhouette. Low silhouette = the embedding space does not
    cleanly cluster at K̂, so pairwise cosine cannot recover cluster structure
    on its own — the k-means-only verifier list is used. High silhouette = the
    space already clusters cleanly, so the default combined list (primary +
    kmeans wrapper) is used.

    Threshold configurable via stability.auto_verifier_silhouette_threshold
    (default 0.15).

    Returns:
        list[str]: verifier_name list to use.
    """
    silhouette_threshold = algorithm_config.get('auto_verifier_silhouette_threshold', 0.15)
    # High-silhouette branch = the code/config baseline: just the primary
    # verifier. Low-silhouette branch = kmeans-only init.
    default_list = [primary_name]
    kmeans_only_list = [f'kmeans({primary_name})']

    if primary_name not in embeddings_dict:
        logger.warning(
            f"AUTO VERIFIER: primary '{primary_name}' not in embeddings_dict; "
            f"falling back to {default_list}"
        )
        return default_list

    try:
        emb_slot = embeddings_dict[primary_name]
        emb_obj = emb_slot() if callable(emb_slot) else emb_slot
        E = np.asarray(emb_obj.embeddings, dtype=np.float32)
    except Exception as e:
        logger.warning(
            f"AUTO VERIFIER: failed to load '{primary_name}' embeddings ({e}); "
            f"falling back to {default_list}"
        )
        return default_list

    N = E.shape[0]
    if N < 4:
        logger.info(f"AUTO VERIFIER: N={N} too small for silhouette; using {default_list}")
        return default_list

    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sim = 0.5 + 0.5 * (En @ En.T)
    iu, iv = np.triu_indices(N, k=1)
    scores = sim[iu, iv].astype(np.float32)

    try:
        tau = robust_gmm_find_threshold(
            scores, verbose=False, print_func=logger.info,
            fallback_percentile=_get_fallback_percentile(config),
        )
    except Exception as e:
        logger.warning(f"AUTO VERIFIER: GMM threshold failed ({e}); falling back to {default_list}")
        return default_list

    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(N))
    pos_mask = scores > tau
    if pos_mask.any():
        G.add_edges_from(zip(iu[pos_mask].tolist(), iv[pos_mask].tolist()))
    K_hat = nx.number_connected_components(G)
    logger.info(
        f"AUTO VERIFIER: primary='{primary_name}', τ={tau:.4f}, K̂={K_hat}, "
        f"{int(pos_mask.sum())} positive edges over {N} nodes"
    )

    K_use = max(2, min(K_hat, N - 1))
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        km = KMeans(n_clusters=K_use, n_init=5, random_state=0).fit(En)
        silhouette = float(silhouette_score(
            En, km.labels_, metric='cosine',
            sample_size=min(2000, N), random_state=0,
        ))
    except Exception as e:
        logger.warning(f"AUTO VERIFIER: silhouette failed ({e}); falling back to {default_list}")
        return default_list

    logger.info(
        f"AUTO VERIFIER: silhouette at K={K_use} on '{primary_name}' = {silhouette:.4f} "
        f"(threshold={silhouette_threshold})"
    )
    if silhouette < silhouette_threshold:
        logger.info(
            f"AUTO VERIFIER: silhouette < {silhouette_threshold} → embedding space "
            f"not clusterable at K̂; using kmeans-only verifier: {kmeans_only_list}"
        )
        return kmeans_only_list
    logger.info(
        f"AUTO VERIFIER: silhouette ≥ {silhouette_threshold} → embedding space "
        f"clusters cleanly; using default combined verifier: {default_list}"
    )
    return default_list


def coverage_auto_topk(k_hat, coverage=0.9, n_nodes=None):
    """Auto-K under the coupon-collector cluster-coverage model.

    Under uniform sampling, the fraction of *other* clusters that each
    node's top-K neighbours cover is approximately

        coverage(K) = 1 - exp(-K / K̂)

    Solving for K at a target coverage gives

        K = -ln(1 - coverage) · K̂

    coverage=0.90 gives K = 2.30·K̂ (the 90 %/10 %-of-max-slope elbow);
    coverage=0.95 gives 3.00·K̂; coverage=0.99 gives 4.61·K̂.

    Args:
        k_hat: Number of PCCs (from count_pccs_from_all_positives).
        coverage: Target cluster-coverage fraction in (0, 1). See docstring.
        n_nodes: Optional cap so K ≤ n_nodes - 1.

    Returns:
        int: Recommended initial_topk.
    """
    if k_hat <= 1:
        return max(10, k_hat)
    if not (0.0 < coverage < 1.0):
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    k = int(math.ceil(-math.log(1.0 - coverage) * k_hat))
    k = max(10, k)
    if n_nodes is not None:
        k = min(k, n_nodes - 1)
    return k




def auto_compute_stability_params(stability_config, algorithm_config, num_nodes, num_individuals, predicted_f1=None,
                                  gmm_components=None, classifier_threshold=None, prob_human_correct=None):
    """
    Auto-compute stability parameters marked with 'auto' in the config.

    Uses dataset statistics (num_nodes, num_individuals) to derive sensible
    values for parameters that depend on data characteristics.

    Args:
        stability_config: Stability section of the config (modified in-place)
        algorithm_config: Algorithm section of the config (modified in-place)
        num_nodes: Number of annotations (nodes in the graph)
        num_individuals: Estimated number of individuals (from top-k graph components)
        predicted_f1: GMM threshold's predicted F1 (classifier quality indicator)
    """
    avg_sightings = num_nodes / max(1, num_individuals)
    logger.info(f"Auto-computing stability parameters: num_nodes={num_nodes}, "
                f"num_individuals(estimated)={num_individuals}, avg_sightings={avg_sightings:.2f}")

    # initial_topk: auto uses the coupon-collector K = -ln(1-coverage)·K̂,
    # with K̂ from count_pccs_from_all_positives (independent of top-K).
    # Falls back to the legacy avg_sightings heuristic if K̂ is unavailable.
    if algorithm_config.get('initial_topk') == 'auto':
        coverage = algorithm_config.get('auto_topk_coverage', 0.9)
        if num_individuals and num_individuals > 1:
            val = coverage_auto_topk(num_individuals, coverage=coverage,
                                     n_nodes=num_nodes)
            logger.info(
                f"  Auto initial_topk = {val}  "
                f"(coupon-collector: -ln(1-{coverage:.2f})·K̂ = "
                f"{-math.log(1 - coverage):.2f}·{num_individuals})"
            )
        else:
            val = max(10, math.ceil(20 / max(avg_sightings, 1e-6)))
            logger.info(
                f"  Auto initial_topk = {val}  (legacy avg-sightings "
                f"heuristic; K̂ unavailable)"
            )
        algorithm_config['initial_topk'] = val

    # max_densify_edges: auto = min(1000, num_nodes * avg_sightings)
    if stability_config.get('max_densify_edges') == 'auto':
        val = min(1000, int(num_nodes * avg_sightings))
        stability_config['max_densify_edges'] = val
        logger.info(f"  Auto max_densify_edges = {val}")

    # sparsify_on_init: auto = (avg_sightings > 3)
    if stability_config.get('sparsify_on_init') == 'auto':
        val = (avg_sightings > 3)
        stability_config['sparsify_on_init'] = val
        logger.info(f"  Auto sparsify_on_init = {val}")

    # max_phase0_iterations: auto = max(20, num_nodes // 100)
    if stability_config.get('max_phase0_iterations') == 'auto':
        val = max(20, num_nodes // 100)
        stability_config['max_phase0_iterations'] = val
        logger.info(f"  Auto max_phase0_iterations = {val}")

    # review_confidence (c_h): auto = one verdict's share of the evidence needed to
    # settle an edge. Confidence is accumulated evidence: a reviewer of reliability
    # p = edge_weights.prob_human_correct contributes log-odds evidence ln(p/(1-p))
    # per verdict. An edge is "settled" (its label decided) once its accumulated
    # evidence clears the graph-wide decision threshold ln(N), where N is the number
    # of candidate edges — the Bonferroni bar that keeps the expected number of
    # noise-flipped edges across the whole graph below one. Hence one verdict advances
    # the [0,1] confidence by its share of that threshold:
    #     c_h = ln(p/(1-p)) / ln(N)
    #   p -> 1   => c_h -> 1  (one verdict settles the edge; reproduces the perfect baseline)
    #   p = 0.5  => c_h = 0   (a chance reviewer carries no evidence)
    # Because ln(N) ~ 10 for any real graph, c_h ~ 0.1 * ln(p/(1-p)), stable across
    # datasets with nothing tuned.
    if stability_config.get('review_confidence') == 'auto':
        p = float(prob_human_correct)
        # N = number of candidate edges ~ num_nodes * top-K (K resolved just above).
        try:
            topk = int(algorithm_config.get('initial_topk', 10))
        except (TypeError, ValueError):
            topk = 10
        num_edges = max(int(num_nodes) * max(topk, 1), 3)

        if p >= 1.0:
            val = 1.0
        elif p <= 0.5:
            val = 0.0
        else:
            val = math.log(p / (1.0 - p)) / math.log(float(num_edges))
            val = min(1.0, max(val, 0.0))
        logger.info(f"  Auto review_confidence = {val:.3f} "
                    f"(evidence ln(p/(1-p))/ln(N); p={p:.4f}, N={num_edges}, "
                    f"ln(N)={math.log(num_edges):.2f})")
        stability_config['review_confidence'] = val



def prepare_stability(common_data, config):
    """
    Prepare Beta Stability algorithm instance.

    This algorithm implements stability-driven clustering with active review.
    It treats GC behavior as alpha=0 stability and iteratively increases
    stability by selecting lowest-stability cases for human review.

    Args:
        common_data: Common data from prepare_common
        config: Configuration dictionary

    Returns:
        BetaStabilityAlgorithm: Configured stability algorithm instance
    """
    # Get stability-specific config, falling back to gc config for compatibility
    stability_config = config.get('stability', config.get('gc', {})).copy()

    # Merge edge_weights settings
    edge_weights = config.get('edge_weights', {})
    stability_config['prob_human_correct'] = edge_weights.get('prob_human_correct', 0.98)

    # Get parameters from various config sections
    stability_config['target_alpha'] = stability_config.get('target_alpha', 0.5)  # [0, 1] scale
    stability_config['max_human_reviews'] = stability_config.get('max_human_reviews', 1000)
    stability_config['review_confidence'] = stability_config.get('review_confidence', 0.97)  # [0, 1] scale
    stability_config['edges_per_review_batch'] = stability_config.get('edges_per_review_batch', 200)
    stability_config['validation_step'] = stability_config.get('validation_step', 100)
    stability_config['max_densify_edges'] = stability_config.get('max_densify_edges', 2000)
    stability_config['cross_pcc_max_edges'] = stability_config.get('cross_pcc_max_edges', 10000)

    # Phase 0 aggressiveness controls
    stability_config['phase0_alpha'] = stability_config.get('phase0_alpha', 0.0)  # 0.0 = strict, -0.1 = lenient
    stability_config['max_phase0_iterations'] = stability_config.get('max_phase0_iterations', 20)
    stability_config['densify_prioritize_negatives'] = stability_config.get('densify_prioritize_negatives', False)

    algorithm_params = config.get("algorithm", {})

    # Auto-compute parameters marked with 'auto' based on dataset statistics
    num_nodes = len(common_data['node2uuid'])

    # Estimate num_individuals from top-k graph connected components (no ground truth needed).
    # Use the user's chosen primary verifier (first entry in verifier_name) so this
    # works for any embedding (miewid, megadescriptor, etc.) — falls back to 'miewid'
    # for legacy configs.
    primary_verifier = common_data.get('verifier_name', 'miewid')
    primary_embeddings = common_data['embeddings_dict'].get(primary_verifier)
    if primary_embeddings is None:
        primary_embeddings = common_data['embeddings_dict'].get('miewid')
    predicted_f1 = None
    if primary_embeddings is not None:
        if callable(primary_embeddings):
            primary_embeddings = primary_embeddings()
        # Get classifier threshold. If it's auto(...) and not already cached,
        # fit the GMM now — earlier code paths only fit this during the
        # kmeans-verifier setup, which meant configs without kmeans hit the
        # 0.7 fallback below and got a completely wrong threshold.
        cls_thresholds = config.get('edge_weights', {}).get('classifier_thresholds', {})
        threshold_spec = cls_thresholds.get(primary_verifier, cls_thresholds.get('miewid', 0.7))
        if isinstance(threshold_spec, str) and 'auto' in threshold_spec:
            cached = getattr(primary_embeddings, '_cached_gmm_threshold', None)
            if cached is not None:
                threshold_val = float(cached)
            else:
                # Fit the GMM ourselves, exactly like the kmeans-branch and
                # the classifier-setup fallback do. Cache the result so
                # downstream classifier setup reuses it.
                match = re.match(r'auto\((\d+\.?\d*)\)', threshold_spec)
                threshold_fraction = float(match.group(1)) if match else 0.15
                unfiltered_embeddings_dict = common_data.get('unfiltered_embeddings_dict', {})
                thresh_embeddings = unfiltered_embeddings_dict.get(primary_verifier, primary_embeddings)
                if callable(thresh_embeddings):
                    thresh_embeddings = thresh_embeddings()
                if hasattr(primary_embeddings, 'get_base'):
                    thresh_embeddings = primary_embeddings.get_base()
                _kmeans_min_cluster_size = stability_config.get('kmeans_min_cluster_size', 1)
                _n_nodes = len(primary_embeddings.ids) if hasattr(primary_embeddings, 'ids') else None
                _max_implied_K = (_n_nodes // _kmeans_min_cluster_size
                                  if (_kmeans_min_cluster_size > 1 and _n_nodes) else None)
                logger.info(
                    f"Fitting GMM threshold for {primary_verifier} "
                    f"(spec={threshold_spec}) — not cached yet"
                )
                threshold_val = robust_gmm_find_threshold(
                    np.array(thresh_embeddings.get_all_scores()),
                    entropy_alpha=1 - threshold_fraction,
                    verbose=True, print_func=logger.info,
                    plot_path=config.get('logging', {}).get('auto_threshold_plot_path'),
                    fallback_percentile=_get_fallback_percentile(config),
                    max_implied_K=_max_implied_K,
                )
                primary_embeddings._cached_gmm_threshold = threshold_val
                primary_embeddings._cached_gmm_predicted_f1 = getattr(
                    robust_gmm_find_threshold, '_last_predicted_f1', None
                )
        else:
            threshold_val = float(threshold_spec)
        # Get predicted F1 from GMM (cached during threshold computation)
        predicted_f1 = getattr(primary_embeddings, '_cached_gmm_predicted_f1', None)
        topk = algorithm_params.get('initial_topk', 10)
        if topk == 'auto':
            # K̂ from all classifier-positive pairs — no top-K choice needed
            # (breaks the circularity of "need K to estimate K̂ to pick K").
            num_individuals = count_pccs_from_all_positives(
                primary_embeddings, threshold_val
            )
        else:
            num_individuals = estimate_num_individuals_from_topk(
                primary_embeddings, threshold_val, topk=topk
            )
    else:
        num_individuals = num_nodes  # fallback: assume all singletons

    # Expose the primary verifier's fitted GMM (2-component) model so the
    # LR-calibrated review_confidence model can place a verdict on its scale.
    _gmm_components = getattr(robust_gmm_find_threshold, '_last_components', None)
    _classifier_threshold = threshold_val if primary_embeddings is not None else None
    _prob_human_correct = config.get('edge_weights', {}).get('prob_human_correct', 1.0)
    auto_compute_stability_params(stability_config, algorithm_params, num_nodes, num_individuals,
                                  predicted_f1=predicted_f1,
                                  gmm_components=_gmm_components,
                                  classifier_threshold=_classifier_threshold,
                                  prob_human_correct=_prob_human_correct)

    # Update common_data with potentially auto-computed initial_topk
    if 'initial_topk' in algorithm_params:
        common_data['initial_topk'] = algorithm_params['initial_topk']

    logger.info(f"Stability algorithm config:")
    logger.info(f"  target_alpha: {stability_config['target_alpha']}")
    logger.info(f"  phase0_alpha: {stability_config['phase0_alpha']}")
    logger.info(f"  max_human_reviews: {stability_config['max_human_reviews']}")
    logger.info(f"  review_confidence: {stability_config['review_confidence']}")

    logger.info(f"  densify_prioritize_negatives: {stability_config['densify_prioritize_negatives']}")
    # Build classifier manager (reuse GC's classifier setup)
    verifier_names_str = edge_weights.get('verifier_names', edge_weights.get('augmentation_names', 'miewid human'))
    verifier_names = verifier_names_str.split() if isinstance(verifier_names_str, str) else verifier_names_str

    classifier_thresholds = edge_weights.get('classifier_thresholds', {})
    classifier_units = {}

    do_robust_plot = "auto_threshold_plot_path" in config.get("logging", {})
    robust_plot_path = config.get("logging", {}).get("auto_threshold_plot_path", "dist.png")

    # K-degeneracy guard: reject GMM thresholds whose implied K exceeds
    # n / kmeans_min_cluster_size (the same cluster-size assumption used to
    # cap k-means K_hat). Triggers on weakly-separated embeddings where the
    # GMM hallucinates a microscopic right-tail positive component.
    _kmeans_min_cluster_size = stability_config.get('kmeans_min_cluster_size', 1)

    def _max_implied_K_for(embeddings):
        if _kmeans_min_cluster_size <= 1:
            return None
        try:
            n_nodes = len(embeddings.ids)
        except AttributeError:
            return None
        return n_nodes // _kmeans_min_cluster_size

    # Separate thresholds into direct values and references
    direct_thresholds = {}  # name -> (embeddings, threshold_spec)
    reference_thresholds = {}  # name -> (embeddings, reference_name)

    for name in classifier_thresholds:
        if name not in common_data['embeddings_dict']:
            continue
        embeddings = common_data['embeddings_dict'][name]
        if isinstance(embeddings, types.FunctionType):
            embeddings = embeddings()

        threshold = classifier_thresholds[name]
        # Check if it's a reference to another classifier (string that's not 'auto(...)')
        if isinstance(threshold, str) and "auto" not in threshold:
            reference_thresholds[name] = (embeddings, threshold)
        else:
            direct_thresholds[name] = (embeddings, threshold)

    # First pass: process direct thresholds (numbers and auto())
    for name, (embeddings, threshold) in direct_thresholds.items():
        learnable = False  # constant thresholds stay fixed
        if isinstance(threshold, str) and "auto" in threshold:
            learnable = True  # auto-fit thresholds support refit
            # Check if threshold was already computed (e.g. by KMeansEmbeddings setup)
            if hasattr(embeddings, '_cached_gmm_threshold'):
                threshold = embeddings._cached_gmm_threshold
                logger.info(f"Reusing cached GMM threshold for {name}: {threshold}")
            else:
                match = re.match(r'auto\((\d+\.?\d*)\)', threshold)
                threshold_fraction = float(match.group(1)) if match else 0.15

                unfiltered_embeddings_dict = common_data.get('unfiltered_embeddings_dict', {})
                if name in unfiltered_embeddings_dict:
                    thresh_embeddings = unfiltered_embeddings_dict[name]
                    if isinstance(thresh_embeddings, types.FunctionType):
                        thresh_embeddings = thresh_embeddings()
                else:
                    thresh_embeddings = embeddings

                if hasattr(embeddings, 'get_base'):
                    thresh_embeddings = embeddings.get_base()

                threshold = robust_gmm_find_threshold(
                    np.array(thresh_embeddings.get_all_scores()),
                    entropy_alpha=1-threshold_fraction,
                    verbose=True,
                    print_func=logger.info,
                    plot_path=robust_plot_path,
                    fallback_percentile=_get_fallback_percentile(config),
                    max_implied_K=_max_implied_K_for(thresh_embeddings),
                )
                # Cache predicted F1 for auto review_confidence computation
                embeddings._cached_gmm_predicted_f1 = getattr(robust_gmm_find_threshold, '_last_predicted_f1', None)

        classifier = ThresholdBasedClassifier(threshold, learnable=learnable)
        logger.info(f"Created threshold-based classifier for {name} with threshold {threshold} (learnable={learnable})")
        classifier_units[name] = (embeddings, classifier)

    # Second pass: process reference thresholds
    for name, (embeddings, ref_name) in reference_thresholds.items():
        if ref_name in classifier_units:
            classifier_units[name] = (embeddings, classifier_units[ref_name][1])
            logger.info(f"Created threshold-based classifier for {name} referencing {ref_name}")
        else:
            logger.warning(f"Skipping {name}: referenced classifier '{ref_name}' not found")

    for name in verifier_names:
        if 'human' in name:
            continue
        if name not in classifier_units:
            embeddings = common_data['embeddings_dict'].get(name)
            if embeddings is None:
                continue

            unfiltered_embeddings_dict = common_data.get('unfiltered_embeddings_dict', {})
            if name in unfiltered_embeddings_dict:
                thresh_embeddings = unfiltered_embeddings_dict[name]
                if isinstance(thresh_embeddings, types.FunctionType):
                    thresh_embeddings = thresh_embeddings()
            else:
                thresh_embeddings = embeddings

            threshold = robust_gmm_find_threshold(
                np.array(thresh_embeddings.get_all_scores()),
                verbose=True,
                print_func=logger.info,
                plot_path=robust_plot_path,
                fallback_percentile=_get_fallback_percentile(config),
                max_implied_K=_max_implied_K_for(thresh_embeddings),
            )
            classifier = ThresholdBasedClassifier(threshold, learnable=True)
            logger.warning(f"No threshold for {name}, using default auto threshold {threshold} (learnable=True)")
            classifier_units[name] = (embeddings, classifier)

    # Ensure all init verifiers have classifiers (needed for labeling initial edges)
    for init_verifier in common_data.get('init_verifiers', []):
        if init_verifier not in classifier_units:
            init_embeddings = common_data['embeddings_dict'].get(init_verifier)
            if init_embeddings is not None:
                if isinstance(init_embeddings, types.FunctionType):
                    init_embeddings = init_embeddings()
                # If a kmeans wrapper fell back to base embeddings, reuse the base classifier
                if not isinstance(init_embeddings, KMeansEmbeddings) and init_verifier.startswith('kmeans('):
                    base_name = init_verifier[len('kmeans('):-1]
                    if base_name in classifier_units:
                        logger.info(
                            f"Init verifier '{init_verifier}' fell back to base embeddings; "
                            f"reusing '{base_name}' classifier"
                        )
                        classifier_units[init_verifier] = (init_embeddings, classifier_units[base_name][1])
                        continue
                # kmeans wrappers use 0.5 as natural threshold by design
                default_threshold = 0.5
                classifier = ThresholdBasedClassifier(default_threshold)
                logger.info(f"Created default classifier for init verifier '{init_verifier}' with threshold {default_threshold}")
                classifier_units[init_verifier] = (init_embeddings, classifier)

    classifier_manager = ClassifierManager(
        verifier_names=verifier_names,
        classifier_units=classifier_units
    )

    # Create stability algorithm instance.
    # Ablation runs opt into the isolated ablation module living in the
    # repo-root `internal/` directory (kept out of the shipped package but
    # loadable at runtime when explicitly requested).
    if stability_config.get('use_ablation_module', False):
        import sys as _sys
        _internal_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'internal')
        )
        if _internal_dir not in _sys.path:
            _sys.path.insert(0, _internal_dir)
        from stability_algorithm_ablation import BetaStabilityAlgorithm as _StabilityAlgo
        logger.info(
            f"Using ABLATION stability module from {_internal_dir}"
        )
    else:
        _StabilityAlgo = BetaStabilityAlgorithm
    stability_instance = _StabilityAlgo(
        stability_config,
        classifier_manager=classifier_manager,
        cluster_validator=common_data['cluster_validator']
    )

    return stability_instance


def prepare_np3_aas(common_data, config):
    """
    Prepare NP3+AAS algorithm instance.

    Args:
        common_data: Common data from prepare_common
        config: Configuration dictionary

    Returns:
        NP3AASAlgorithm: Configured algorithm instance
    """
    np3_config = config.get('np3_aas', {}).copy()

    # Merge edge_weights settings
    edge_weights = config.get('edge_weights', {})
    np3_config['prob_human_correct'] = edge_weights.get('prob_human_correct', 0.98)

    # Set defaults
    np3_config.setdefault('max_human_reviews', 5000)
    np3_config.setdefault('edges_per_review_batch', 200)
    np3_config.setdefault('validation_step', 100)
    np3_config.setdefault('dbscan_eps', 0.5)
    np3_config.setdefault('dbscan_min_samples', 2)
    np3_config.setdefault('dbscan_metric', 'cosine')
    np3_config.setdefault('s_min', 0.3)
    np3_config.setdefault('k_max', 5)
    np3_config.setdefault('epsilon', 0.6)
    np3_config.setdefault('verifier_name', common_data['verifier_name'])

    # Build classifier manager (same pattern as prepare_stability)
    verifier_names_str = edge_weights.get('verifier_names',
                                          edge_weights.get('augmentation_names', 'miewid human'))
    verifier_names = verifier_names_str.split() if isinstance(verifier_names_str, str) else verifier_names_str

    classifier_thresholds = edge_weights.get('classifier_thresholds', {})
    classifier_units = {}

    do_robust_plot = "auto_threshold_plot_path" in config.get("logging", {})
    robust_plot_path = config.get("logging", {}).get("auto_threshold_plot_path", "dist.png")

    # Process direct thresholds
    for name in classifier_thresholds:
        if name not in common_data['embeddings_dict']:
            continue
        embeddings = common_data['embeddings_dict'][name]
        if isinstance(embeddings, types.FunctionType):
            embeddings = embeddings()

        threshold = classifier_thresholds[name]
        if isinstance(threshold, str) and "auto" in threshold:
            match = re.match(r'auto\((\d+\.?\d*)\)', threshold)
            threshold_fraction = float(match.group(1)) if match else 0.15

            unfiltered_embeddings_dict = common_data.get('unfiltered_embeddings_dict', {})
            if name in unfiltered_embeddings_dict:
                thresh_embeddings = unfiltered_embeddings_dict[name]
                if isinstance(thresh_embeddings, types.FunctionType):
                    thresh_embeddings = thresh_embeddings()
            else:
                thresh_embeddings = embeddings

            if hasattr(embeddings, 'get_base'):
                thresh_embeddings = embeddings.get_base()

            threshold = robust_gmm_find_threshold(
                np.array(thresh_embeddings.get_all_scores()),
                entropy_alpha=1 - threshold_fraction,
                verbose=True,
                print_func=logger.info,
                plot_path=robust_plot_path,
                fallback_percentile=_get_fallback_percentile(config),
            )

        classifier = ThresholdBasedClassifier(threshold)
        logger.info(f"Created threshold-based classifier for {name} with threshold {threshold}")
        classifier_units[name] = (embeddings, classifier)

    # Fallback for verifiers without explicit thresholds
    for name in verifier_names:
        if 'human' in name:
            continue
        if name not in classifier_units:
            embeddings = common_data['embeddings_dict'].get(name)
            if embeddings is None:
                continue

            unfiltered_embeddings_dict = common_data.get('unfiltered_embeddings_dict', {})
            if name in unfiltered_embeddings_dict:
                thresh_embeddings = unfiltered_embeddings_dict[name]
                if isinstance(thresh_embeddings, types.FunctionType):
                    thresh_embeddings = thresh_embeddings()
            else:
                thresh_embeddings = embeddings

            threshold = robust_gmm_find_threshold(
                np.array(thresh_embeddings.get_all_scores()),
                verbose=True,
                print_func=logger.info,
                plot_path=robust_plot_path,
                fallback_percentile=_get_fallback_percentile(config),
            )
            classifier = ThresholdBasedClassifier(threshold)
            classifier_units[name] = (embeddings, classifier)

    classifier_manager = ClassifierManager(
        verifier_names=verifier_names,
        classifier_units=classifier_units
    )

    # Pass the classifier threshold to the algorithm config so it can
    # derive dbscan_eps='auto' from it (similarity -> cosine distance)
    verifier_name = np3_config.get('verifier_name', 'miewid')
    if verifier_name in classifier_units:
        _, classifier_obj = classifier_units[verifier_name]
        np3_config['classifier_threshold'] = classifier_obj.threshold
        logger.info(f"Classifier threshold for {verifier_name}: {classifier_obj.threshold:.4f}")

    logger.info(f"NP3+AAS algorithm config:")
    logger.info(f"  dbscan_eps: {np3_config['dbscan_eps']}")
    logger.info(f"  dbscan_min_samples: {np3_config['dbscan_min_samples']}")
    logger.info(f"  s_min: {np3_config['s_min']}")
    logger.info(f"  k_max: {np3_config['k_max']}")
    logger.info(f"  epsilon: {np3_config['epsilon']}")
    logger.info(f"  max_human_reviews: {np3_config['max_human_reviews']}")
    logger.info(f"  edges_per_review_batch: {np3_config['edges_per_review_batch']}")

    return NP3AASAlgorithm(
        np3_config,
        classifier_manager=classifier_manager,
        cluster_validator=common_data['cluster_validator']
    )


def prepare_nis(common_data, config):
    """
    Prepare NIS + k-means algorithm instance.

    Uses Nested Importance Sampling to estimate the number of clusters K,
    then runs k-means with k=K_hat. Based on Perez et al. (ECCV 2024).

    Args:
        common_data: Common data from prepare_common
        config: Configuration dictionary

    Returns:
        NISAlgorithm: Configured algorithm instance
    """
    nis_config = config.get('nis', {}).copy()

    # Merge edge_weights settings
    edge_weights = config.get('edge_weights', {})
    nis_config['prob_human_correct'] = edge_weights.get('prob_human_correct', 0.98)

    # Set defaults
    nis_config.setdefault('n_sampled_vertices', 50)
    nis_config.setdefault('m_neighbors_per_vertex', 100)
    nis_config.setdefault('temperature', 0.5)
    nis_config.setdefault('max_human_reviews', 5000)
    nis_config.setdefault('edges_per_review_batch', 200)
    nis_config.setdefault('validation_step', 100)
    nis_config.setdefault('verifier_name', common_data['verifier_name'])

    # Build classifier manager (same pattern as prepare_np3_aas)
    verifier_names_str = edge_weights.get('verifier_names',
                                          edge_weights.get('augmentation_names', 'miewid human'))
    verifier_names = verifier_names_str.split() if isinstance(verifier_names_str, str) else verifier_names_str

    classifier_thresholds = edge_weights.get('classifier_thresholds', {})
    classifier_units = {}

    robust_plot_path = config.get("logging", {}).get("auto_threshold_plot_path", "dist.png")

    # Process direct thresholds
    for name in classifier_thresholds:
        if name not in common_data['embeddings_dict']:
            continue
        embeddings = common_data['embeddings_dict'][name]
        if isinstance(embeddings, types.FunctionType):
            embeddings = embeddings()

        threshold = classifier_thresholds[name]
        if isinstance(threshold, str) and "auto" in threshold:
            match = re.match(r'auto\((\d+\.?\d*)\)', threshold)
            threshold_fraction = float(match.group(1)) if match else 0.15

            unfiltered_embeddings_dict = common_data.get('unfiltered_embeddings_dict', {})
            if name in unfiltered_embeddings_dict:
                thresh_embeddings = unfiltered_embeddings_dict[name]
                if isinstance(thresh_embeddings, types.FunctionType):
                    thresh_embeddings = thresh_embeddings()
            else:
                thresh_embeddings = embeddings

            if hasattr(embeddings, 'get_base'):
                thresh_embeddings = embeddings.get_base()

            threshold = robust_gmm_find_threshold(
                np.array(thresh_embeddings.get_all_scores()),
                entropy_alpha=1 - threshold_fraction,
                verbose=True,
                print_func=logger.info,
                plot_path=robust_plot_path,
                fallback_percentile=_get_fallback_percentile(config),
            )

        classifier = ThresholdBasedClassifier(threshold)
        logger.info(f"Created threshold-based classifier for {name} with threshold {threshold}")
        classifier_units[name] = (embeddings, classifier)

    # Fallback for verifiers without explicit thresholds
    for name in verifier_names:
        if 'human' in name:
            continue
        if name not in classifier_units:
            embeddings = common_data['embeddings_dict'].get(name)
            if embeddings is None:
                continue

            unfiltered_embeddings_dict = common_data.get('unfiltered_embeddings_dict', {})
            if name in unfiltered_embeddings_dict:
                thresh_embeddings = unfiltered_embeddings_dict[name]
                if isinstance(thresh_embeddings, types.FunctionType):
                    thresh_embeddings = thresh_embeddings()
            else:
                thresh_embeddings = embeddings

            threshold = robust_gmm_find_threshold(
                np.array(thresh_embeddings.get_all_scores()),
                verbose=True,
                print_func=logger.info,
                plot_path=robust_plot_path,
                fallback_percentile=_get_fallback_percentile(config),
            )
            classifier = ThresholdBasedClassifier(threshold)
            classifier_units[name] = (embeddings, classifier)

    classifier_manager = ClassifierManager(
        verifier_names=verifier_names,
        classifier_units=classifier_units
    )

    logger.info(f"NIS algorithm config:")
    logger.info(f"  n_sampled_vertices: {nis_config['n_sampled_vertices']}")
    logger.info(f"  m_neighbors_per_vertex: {nis_config['m_neighbors_per_vertex']}")
    logger.info(f"  temperature: {nis_config['temperature']}")
    logger.info(f"  max_human_reviews: {nis_config['max_human_reviews']}")
    logger.info(f"  edges_per_review_batch: {nis_config['edges_per_review_batch']}")

    return NISAlgorithm(
        nis_config,
        classifier_manager=classifier_manager,
        cluster_validator=common_data['cluster_validator']
    )


def call_verifier_alg(embeddings):
    """Helper function to create verifier algorithm from embeddings."""
    def verifier_alg(edge_nodes):
        logger = logging.getLogger("beta_stability")
        scores = [embeddings.get_score(n0, n1) for n0, n1 in edge_nodes]
        logger.info(f'Scores  {scores} ')
        return scores
    return verifier_alg
