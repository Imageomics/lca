import pickle
import yaml
import json
import random
import logging
import cluster_tools as ct
import networkx as nx
from init_logger import get_formatter

import matplotlib.pyplot as plt
import json

import numpy as np

from weighter import weighter

import numpy as np

class EmptyDataframeException(Exception):
    pass

class SingleAnnotationException(Exception):
    """Raised when only one annotation remains after filtering - no clustering needed."""
    def __init__(self, node2uuid, gt_clustering=None, gt_node2cid=None):
        self.node2uuid = node2uuid
        self.gt_clustering = gt_clustering
        self.gt_node2cid = gt_node2cid
        super().__init__("Only one annotation - skipping LCA clustering")

def kth_diag_indices(a, k):
    r, c = a.shape
    i_start = max(0, -k)
    i_end = min(r, c - k)
    rows = np.arange(i_start, i_end)
    cols = rows + k
    return rows, cols




def rosin_threshold(data, bins=256):
    """
    Implement Rosin (Triangle) threshold manually
    """
    # Create histogram
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find the peak (mode) of the histogram
    peak_idx = np.argmax(hist)
    
    # Find the last non-zero bin
    last_nonzero = np.where(hist > 0)[0][-1]
    
    # Create line from peak to last non-zero bin
    x1, y1 = peak_idx, hist[peak_idx]
    x2, y2 = last_nonzero, hist[last_nonzero]
    
    # Calculate perpendicular distances from line to histogram
    max_distance = 0
    threshold_idx = peak_idx
    
    for i in range(peak_idx, last_nonzero + 1):
        if hist[i] > 0:
            # Distance from point to line formula
            # Line: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
            # Distance = |ax + by + c| / sqrt(a² + b²)
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x1
            
            distance = abs(a * i + b * hist[i] + c) / np.sqrt(a**2 + b**2)
            
            if distance > max_distance:
                max_distance = distance
                threshold_idx = i
    
    return bin_centers[threshold_idx]

def order_edge(u, v):
    return (min(u, v), max(u, v))

def lazy(func):
    func.__val = None 
    def wrapper():
        computed = func.__val
        if computed:
            return computed
        computed = func()
        func.__val = computed
        return computed
    return wrapper

def load_yaml(file_path):
    logger = logging.getLogger('lca')
    logger.info(f"Loading config from path: {file_path}")
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    return config_dict



def get_config(file_path):

    config_dict = load_yaml(file_path)

    return config_dict


def get_review(node_1, node_2, df, name_key, rate=0.98):

    is_similar = False
    if df.iloc[node_1][name_key] == df.iloc[node_2][name_key]:
        is_similar=True
    
    return is_similar if random.random() < rate else not is_similar

def call_get_reviews(df, name_key, rate):
    def get_reviews(edge_nodes, rate=rate):
        logger = logging.getLogger('lca')
        reviews = [(n0, n1, get_review(n0, n1, df, name_key, rate)) for n0, n1, _ in edge_nodes]
        # quit_lca = random.random() < 0.4
        quit_lca = False
        return reviews, quit_lca
    return get_reviews
    # return reviews, quit_lca

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


class SetEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, nx.Graph):
            return nx.cytoscape_data(obj)
        else:
            return json.JSONEncoder.default(self, obj)
        

def load_json(path):
    with open(path, "r") as file:
        json_object = json.load(file)
    return json_object


def write_json(data, out_path):
    json_object = json.dumps(data, indent=4, cls=SetEncoder)
    with open(out_path, "w") as outfile:
        outfile.write(json_object)
        


def load_pickle(file):
    with open(file, 'rb') as f_file:
        result = pickle.load(f_file)
    return result

def save_pickle(x, file):
    with open(file, 'wb') as f_file:
        result = pickle.dump(x, f_file)
    return result

def load_dataframe_lightweight(config):
    """Load dataframe without embeddings for field discovery."""
    from preprocess import preprocess_data
    
    data_params = config['data']
    
    # Load embeddings just to get UUIDs
    embeddings, uuids = load_pickle(data_params['embedding_file'])
    
    name_keys = data_params['name_keys']
    format_type = data_params.get('format', 'drone')
    id_key = data_params.get('id_key', 'uuid')
    
    df = preprocess_data(
        data_params['annotation_file'],
        name_keys=name_keys,
        convert_names_to_ids=True,
        n_filter_min=data_params['n_filter_min'],
        n_filter_max=data_params['n_filter_max'],
        images_dir=data_params.get('images_dir'),
        embedding_uuids=uuids,
        id_key=id_key,
        format=format_type,
        print_func=lambda x: None  # Silent loading
    )
    
    # Filter to available UUIDs
    filtered_df = df[df[id_key].isin(uuids)]
    return filtered_df


def discover_field_values_from_df(df, fields):
    """Discover unique values for fields from dataframe."""
    field_values = {}
    for field in fields:
        if field in df.columns:
            unique_values = df[field].dropna().unique().tolist()
        else:
            # Try name_<field> pattern
            name_field = f"name_{field}"
            if name_field in df.columns:
                unique_values = df[name_field].dropna().unique().tolist()
            else:
                continue
        field_values[field] = sorted(unique_values, key=str)
    return field_values


def generate_ga_params(config):
    
    ga_params = dict()

    phc = float(config['edge_weights']['prob_human_correct'])
    assert 0 < phc <= 1
    ga_params['prob_human_correct'] = phc
    ga_params['simulate_human'] = config.get('edge_weights', True).get('simulate_human', True)
    s = config['edge_weights']['augmentation_names']
    ga_params['aug_names'] = s

    ga_params['num_pos_needed'] = config['edge_weights']['num_pos_needed']
    ga_params['num_neg_needed'] = config['edge_weights']['num_neg_needed']

    ga_params['distance_power'] = config['distance_power']

    
    ga_params['scorer'] = config['edge_weights']['scorer']

    mult = float(config['iterations']['min_delta_converge_multiplier'])
    ga_params['min_delta_converge_multiplier'] = mult

    s = float(config['iterations']['min_delta_stability_ratio'])
    assert s > 1
    ga_params['min_delta_stability_ratio'] = s

    n = int(config['iterations']['num_per_augmentation'])
    assert n >= 1
    ga_params['num_per_augmentation'] = n

    n = int(config['iterations']['tries_before_edge_done'])
    assert n >= 1
    ga_params['tries_before_edge_done'] = n

    ga_params['max_human_decisions'] = config['iterations']['max_human_decisions']

    i = int(config['iterations']['ga_iterations_before_return'])
    assert i >= 1
    ga_params['ga_iterations_before_return'] = i

    mw = int(config['iterations']['ga_max_num_waiting'])
    assert mw >= 1
    ga_params['ga_max_num_waiting'] = mw

    should_densify_str = str(config['iterations'].get('should_densify', False)).lower()
    ga_params['should_densify'] = should_densify_str == 'true'
    ga_params['densify_min_number_human'] = config['iterations']['densify_min_number_human']

    n = int(config['iterations'].get('densify_min_edges', 1))
    assert n >= 1
    ga_params['densify_min_edges'] = n

    df = float(config['iterations'].get('densify_frac', 0.0))
    assert 0 <= df <= 1
    ga_params['densify_frac'] = df

    log_level = config['logging']['log_level']
    ga_params['log_level'] = log_level
    log_file = config['logging']['log_file']
    ga_params['log_file'] = log_file

    draw_iterations_str = str(config['drawing'].get('draw_iterations', False)).lower()
    ga_params['draw_iterations'] = draw_iterations_str == 'true'
    ga_params['drawing_prefix'] = config['drawing']['drawing_prefix']

    

    if log_file is not None:
        logger = logging.getLogger('lca')
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

        file_mode = config['logging'].get("file_mode", 'w')
        
        handler = logging.FileHandler(log_file, mode=file_mode)
        handler.setLevel(log_level)
        handler.setFormatter(get_formatter())
        logger.addHandler(handler)

    return ga_params

def generate_gt_clusters(df, name_key, id_key='uuid'):
    gt_node2cid = {}
    cids_unique = df[name_key].unique()
    cids = [f'ct{idx}' for idx in range(len(cids_unique))]
    gt_clustering = {cid: [] for cid in cids}
    ids = df.index.tolist()
    node2uuid = {}
    
    for i, row in df.iterrows():
        cid = cids[list(cids_unique).index(row[name_key])]
        gt_clustering[cid].append(i)
        gt_node2cid[i] = cid
        node2uuid[i] = row[id_key]

    gt_clustering = {
        cid: set(cluster) for cid, cluster in gt_clustering.items()
    }

    return gt_clustering, gt_node2cid, node2uuid


def print_intersect_stats(df,individual_key="name"):
    logger = logging.getLogger('lca')
    logger.info(f"** Dataset statistcs **")
    logger.info(f' - Counts: ')
    names = df[individual_key].unique()
    
    
    logger.info(f" ---- number of individuals: {len(names)}" )
    logger.info(f" ---- number of annotations: {len(df)}")
    avg_ratio = len(df) / max(1, len(names))
    logger.info(f" ---- average number of annotations per individual: {avg_ratio:.2f}")

    annotation_counts = df[individual_key].value_counts()
    single_annotation_count = (annotation_counts == 1).sum()
    
    logger.info(f" ---- number of singletons: {single_annotation_count}")


def get_pos_neg_histogram(pos, neg, wgtr, species, timestamp, label):
    plt.figure()

    # print(pos)
    pos_weights = [wgtr.wgt(x)/10 for x in pos]
    neg_weights = [wgtr.wgt(x)/10 for x in neg]
 

    # plt.hist(pos_scores_filtered, bins=35, density=True, alpha=0.6, color='g')
    # plt.hist(neg_scores_filtered, bins=35, density=True, alpha=0.6, color='r')
    plt.hist(pos, bins=35, density=True, alpha=0.6, color='g')
    plt.hist(neg, bins=35, density=True, alpha=0.6, color='r')
    # plt.hist(wgtrs_calib_dict['gt_negative_probs'], bins=35, density=True, alpha=0.6, color='r')
    # plt.hist(wgtrs_calib_dict['gt_positive_probs'], bins=35, density=True, alpha=0.6, color='g')


    # plt.hist(ss, bins=35, density=True, alpha=0.6, color='b')

    xs = np.linspace(0, 1, 100)

    pos_ys = [wgtr.scorer.get_pos_neg(x)[0] for x in xs]
    neg_ys = [wgtr.scorer.get_pos_neg(x)[1] for x in xs]


    plt.plot(xs, pos_ys, color='g')
    plt.plot(xs, neg_ys, color='r')


    # wgtr = weighter.weighter(scorer, config["lca"]["edge_weights"]['prob_human_correct'])
    wgtr.max_weight = 10
    
    plt.plot(xs, [wgtr.wgt_smooth(x) for x in xs], 'k-')

    plt.xlabel("Score")
    plt.ylabel("Probability density function")
    plt.title(species)

    plt.savefig(f'/ekaterina/work/src/lca/lca/visualisations/{species}_{timestamp}_{label}.png')

def get_histogram(initial_edges, wgtr, species, timestamp, wgtrs_calib_dict):
    plt.figure()
    print(len(initial_edges))


    pos = [s for (a0, a1, s, w, c) in initial_edges if c]
    neg = [s for (a0, a1, s, w, c) in initial_edges if not c]

    ss = [s for (a0, a1, s, w, c) in initial_edges]

    pos_weights = [w/10 for (a0, a1, s, w, c) in initial_edges if c]
    neg_weights = [w/10 for (a0, a1, s, w, c) in initial_edges if not c]
 

    # plt.hist(pos_scores_filtered, bins=35, density=True, alpha=0.6, color='g')
    # plt.hist(neg_scores_filtered, bins=35, density=True, alpha=0.6, color='r')
    plt.hist(pos, bins=35, density=True, alpha=0.6, color='g')
    plt.hist(neg, bins=35, density=True, alpha=0.6, color='r')
    # plt.hist(wgtrs_calib_dict['gt_negative_probs'], bins=35, density=True, alpha=0.6, color='r')
    # plt.hist(wgtrs_calib_dict['gt_positive_probs'], bins=35, density=True, alpha=0.6, color='g')


    # plt.hist(ss, bins=35, density=True, alpha=0.6, color='b')

    xs = np.linspace(0, 1, 100)

    pos_ys = [wgtr.scorer.get_pos_neg(x)[0] for x in xs]
    neg_ys = [wgtr.scorer.get_pos_neg(x)[1] for x in xs]


    plt.plot(xs, pos_ys, color='g')
    plt.plot(xs, neg_ys, color='r')


    # wgtr = weighter.weighter(scorer, config["lca"]["edge_weights"]['prob_human_correct'])
    weighter_max = weighter.max_weight
    weighter.max_weight = 10
    
    plt.plot(xs, [wgtr.wgt_smooth(x) for x in xs], 'k-')

    weighter.max_weight = weighter_max

    plt.xlabel("Score")
    plt.ylabel("Probability density function")
    plt.title(species)

    plt.savefig(f'/ekaterina/work/src/lca/lca/visualisations/{species}__{timestamp}.png')


def connect_disconnected_clusters_strongest_node(G, node2cid):
    """
    Connects clusters by identifying the most "important" nodes (highest edge confidence)
    and linking disconnected clusters through them.

    Args:
        G (nx.Graph): The graph after final clustering.
        node2cid (dict): Mapping from node to cluster ID.

    Returns:
        G (nx.Graph): Modified graph with added edges between clusters.
        added_edges (list): List of newly added edges between clusters.
    """
    from collections import defaultdict
    logger = logging.getLogger('lca')
    cluster_to_nodes = defaultdict(list)
    node_max_conf = defaultdict(float)

    # Gather max confidence per node
    for u, v, data in G.edges(data=True):
        conf = data.get("confidence", 0)
        node_max_conf[u] = max(node_max_conf[u], conf)
        node_max_conf[v] = max(node_max_conf[v], conf)

    # Group nodes by cluster
    for node, cid in node2cid.items():
        cluster_to_nodes[cid].append(node)

    # Sort nodes in each cluster by descending edge confidence
    cluster_representatives = {
        cid: sorted(nodes, key=lambda x: -node_max_conf.get(x, 0))[0]
        for cid, nodes in cluster_to_nodes.items()
    }

    cluster_ids = list(cluster_representatives.keys())
    added_edges = set([])

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            c1, c2 = cluster_ids[i], cluster_ids[j]
            u, v = cluster_representatives[c1], cluster_representatives[c2]

            # Check if an edge already exists between any node pair from c1 and c2
            connected = any(G.has_edge(n1, n2) for n1 in cluster_to_nodes[c1] for n2 in cluster_to_nodes[c2])
            if not connected:
                # G.add_edge(u, v, label="added", confidence=0.0)
                # if G.has_edge(u, v):
                #     print("WTF")
                added_edges.add(tuple(sorted((u, v))))
                # print(f"Added edge between Cluster {c1} (node {u}) and Cluster {c2} (node {v})")
                # logger.info(f"Added edge between Cluster {c1} (node {u}) and Cluster {c2} (node {v})")
    logger.info(f"Missing {len(added_edges)} edges between clusters")
    return added_edges

