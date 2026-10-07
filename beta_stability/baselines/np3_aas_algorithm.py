"""
NP3 + AAS Clustering Algorithm

Implements the algorithm from:
"Active Learning for Animal Re-Identification with Ambiguity-Aware Sampling"
(Sani, Khurana, Anand - AAAI 2026)

Two components:
1. AAS (Ambiguity-Aware Sampling): Selects informative pairs for human review
   by leveraging disagreements between DBSCAN and FINCH clusterings.
2. NP3 (Non-Parametric Plug-and-Play): Refines clustering using must-link
   and cannot-link constraints from human feedback via graph coloring.
"""

import networkx as nx
import numpy as np
import logging
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment

from beta_stability.util.cluster_tools import build_node_to_cluster_mapping
from beta_stability.util.tools import resolve_checkpoints, CHECKPOINT_LABEL

logger = logging.getLogger("beta_stability")


# ---------------------------------------------------------------------------
# FINCH clustering (Sarfraz et al., CVPR 2019)
# ---------------------------------------------------------------------------

def finch_clustering(embedding_matrix, distance_metric='cosine'):
    """
    First Integer Neighbor Clustering Hierarchy.

    Iteratively merges points based on first (nearest) neighbor links.
    Returns list of partitions from finest to coarsest.

    Args:
        embedding_matrix: numpy array (n_samples, n_features)
        distance_metric: 'cosine' or 'euclidean'

    Returns:
        list[np.ndarray]: partitions[level] is an array of cluster labels.
                          Level 0 is finest, last level is coarsest.
    """
    n = embedding_matrix.shape[0]
    if n <= 1:
        return [np.zeros(n, dtype=int)]

    partitions = []
    current_embeddings = embedding_matrix.copy()
    # Map from current-level indices back to original sample indices
    current_to_original = list(range(n))

    while True:
        m = current_embeddings.shape[0]
        if m <= 1:
            # Everything in one cluster
            labels = np.zeros(n, dtype=int)
            partitions.append(labels)
            break

        # Pairwise distances
        dist = pairwise_distances(current_embeddings, metric=distance_metric)
        np.fill_diagonal(dist, np.inf)

        # First neighbor for each point
        first_neighbor = np.argmin(dist, axis=1)

        # Build adjacency graph from first-neighbor links (undirected)
        G = nx.Graph()
        G.add_nodes_from(range(m))
        for i in range(m):
            G.add_edge(i, first_neighbor[i])

        # Connected components = clusters at this level
        components = list(nx.connected_components(G))

        if len(components) == m:
            # No merges happened -- force stop to avoid infinite loop
            labels = np.zeros(n, dtype=int)
            for comp_id, comp in enumerate(components):
                for idx in comp:
                    if isinstance(current_to_original[idx], list):
                        for orig in current_to_original[idx]:
                            labels[orig] = comp_id
                    else:
                        labels[current_to_original[idx]] = comp_id
            partitions.append(labels)
            break

        # Map back to original indices and record partition
        labels = np.zeros(n, dtype=int)
        new_to_original = []
        new_embeddings = []

        for comp_id, comp in enumerate(components):
            orig_indices = []
            for idx in comp:
                if isinstance(current_to_original[idx], list):
                    orig_indices.extend(current_to_original[idx])
                else:
                    orig_indices.append(current_to_original[idx])

            for orig in orig_indices:
                labels[orig] = comp_id

            new_to_original.append(orig_indices)
            # Centroid of this component
            comp_embs = embedding_matrix[orig_indices]
            new_embeddings.append(comp_embs.mean(axis=0))

        partitions.append(labels)

        if len(components) == 1:
            break

        # Prepare for next level
        current_embeddings = np.array(new_embeddings)
        current_to_original = new_to_original

    return partitions


# ---------------------------------------------------------------------------
# AAS: Regions of Uncertainty
# ---------------------------------------------------------------------------

def compute_regions_of_uncertainty(clustering_a, clustering_b):
    """
    Identify regions of uncertainty from disagreements between two clusterings.

    A region of uncertainty is a connected component of nodes whose cluster
    assignments are inconsistent across the two methods (partial overlap).

    Args:
        clustering_a: dict {cid: set of node_ids}
        clustering_b: dict {cid: set of node_ids}

    Returns:
        list of sets: each set contains node IDs in one region of uncertainty
    """
    node2cid_a = build_node_to_cluster_mapping(clustering_a)
    node2cid_b = build_node_to_cluster_mapping(clustering_b)

    # Build overlap graph: which A-clusters overlap which B-clusters
    # For each (cA, cB) pair with partial overlap, all their nodes are uncertain
    uncertain_nodes = set()
    overlap_edges = []  # edges between nodes that share uncertain clusters

    # Map B-cluster to its nodes for quick lookup
    b_nodes = {}
    for cid_b, nodes_b in clustering_b.items():
        b_nodes[cid_b] = nodes_b

    for cid_a, nodes_a in clustering_a.items():
        # Find all B-clusters that overlap with this A-cluster
        b_overlaps = defaultdict(set)
        for node in nodes_a:
            if node in node2cid_b:
                b_overlaps[node2cid_b[node]].add(node)

        for cid_b, overlap_nodes in b_overlaps.items():
            nodes_b = b_nodes.get(cid_b, set())
            intersection = len(overlap_nodes)
            union = len(nodes_a | nodes_b)

            if union == 0:
                continue

            iou = intersection / union
            if 0 < iou < 1:
                # Partial overlap -- all nodes in both clusters are uncertain
                all_uncertain = nodes_a | nodes_b
                uncertain_nodes.update(all_uncertain)

    if not uncertain_nodes:
        return []

    # Build connectivity graph among uncertain nodes:
    # Two nodes are connected if they share a cluster in EITHER clustering
    uncertain_graph = nx.Graph()
    uncertain_graph.add_nodes_from(uncertain_nodes)

    for clustering in [clustering_a, clustering_b]:
        for cid, nodes in clustering.items():
            uncertain_in_cluster = nodes & uncertain_nodes
            if len(uncertain_in_cluster) > 1:
                nodes_list = list(uncertain_in_cluster)
                # Connect all pairs in this cluster (star pattern for efficiency)
                hub = nodes_list[0]
                for i in range(1, len(nodes_list)):
                    uncertain_graph.add_edge(hub, nodes_list[i])

    regions = [comp for comp in nx.connected_components(uncertain_graph)
               if len(comp) > 1]

    return regions


# ---------------------------------------------------------------------------
# AAS: Sampling Pool Construction
# ---------------------------------------------------------------------------

def compute_medoid(embedding_matrix, node_ids, id_to_idx):
    """
    Compute the medoid of a set of nodes.

    Args:
        embedding_matrix: full embedding matrix (n_total, dim)
        node_ids: iterable of node IDs
        id_to_idx: dict mapping node_id -> index in embedding_matrix

    Returns:
        int: node_id of the medoid
    """
    ids = list(node_ids)
    if len(ids) == 1:
        return ids[0]

    indices = [id_to_idx[nid] for nid in ids]
    sub_emb = embedding_matrix[indices]

    dist = pairwise_distances(sub_emb, metric='cosine')
    total_dist = dist.sum(axis=1)
    medoid_local = np.argmin(total_dist)
    return ids[medoid_local]


def build_sampling_pool(regions, clustering_a, clustering_b,
                        embeddings, s_min=0.3, k_max=5, epsilon=0.6):
    """
    Build the AAS sampling pool U = U_os union U_us.

    Args:
        regions: list of sets of node_ids (regions of uncertainty)
        clustering_a: DBSCAN clustering
        clustering_b: FINCH clustering
        embeddings: Embeddings object with .get_score(), .embeddings, .id_to_idx
        s_min: minimum similarity for over-segmentation medoid pairs
        k_max: max nearest medoid neighbors
        epsilon: weight for over-segmentation pool (1-epsilon for under-seg)

    Returns:
        list of (node_i, node_j, weight) tuples
    """
    node2cid_a = build_node_to_cluster_mapping(clustering_a)
    node2cid_b = build_node_to_cluster_mapping(clustering_b)

    embedding_matrix = embeddings.embeddings
    id_to_idx = embeddings.id_to_idx

    pool_os = []  # over-segmentation pairs
    pool_us = []  # under-segmentation pairs

    # --- U_os: Over-segmentation pairs (across regions) ---
    # Compute medoid for each region
    medoids = []
    for region in regions:
        medoid = compute_medoid(embedding_matrix, region, id_to_idx)
        medoids.append(medoid)

    if len(medoids) > 1:
        # Find nearest medoid neighbors
        medoid_indices = [id_to_idx[m] for m in medoids]
        medoid_emb = embedding_matrix[medoid_indices]
        medoid_dist = pairwise_distances(medoid_emb, metric='cosine')
        np.fill_diagonal(medoid_dist, np.inf)

        for i in range(len(medoids)):
            # Get k_max nearest neighbors
            neighbors = np.argsort(medoid_dist[i])[:k_max]
            for j in neighbors:
                if j <= i:
                    continue  # avoid duplicates
                sim = embeddings.get_score(medoids[i], medoids[j])
                if sim >= s_min:
                    pool_os.append((medoids[i], medoids[j], sim))

    # --- U_us: Under-segmentation pairs (within regions) ---
    for region in regions:
        region_nodes = list(region)
        if len(region_nodes) < 2:
            continue

        # Find clusters from A and B that intersect this region
        clusters_a_in_region = defaultdict(set)
        clusters_b_in_region = defaultdict(set)
        for node in region_nodes:
            if node in node2cid_a:
                clusters_a_in_region[node2cid_a[node]].add(node)
            if node in node2cid_b:
                clusters_b_in_region[node2cid_b[node]].add(node)

        # For each clustering, find closest inter-cluster pairs
        for clusters_in_region in [clusters_a_in_region, clusters_b_in_region]:
            cluster_ids = list(clusters_in_region.keys())
            for ci_idx in range(len(cluster_ids)):
                for cj_idx in range(ci_idx + 1, len(cluster_ids)):
                    ci_nodes = list(clusters_in_region[cluster_ids[ci_idx]])
                    cj_nodes = list(clusters_in_region[cluster_ids[cj_idx]])

                    # Find closest pair between the two clusters
                    best_pair = None
                    best_sim = -1
                    for ni in ci_nodes:
                        for nj in cj_nodes:
                            sim = embeddings.get_score(ni, nj)
                            if sim > best_sim:
                                best_sim = sim
                                best_pair = (ni, nj)

                    if best_pair is not None:
                        # Check if this pair is inconsistent
                        ni, nj = best_pair
                        same_a = (node2cid_a.get(ni) == node2cid_a.get(nj))
                        same_b = (node2cid_b.get(ni) == node2cid_b.get(nj))
                        if same_a != same_b:
                            # Inconsistent pair -- add to pool
                            pool_us.append((ni, nj, best_sim))

    # --- Combine with epsilon weighting ---
    pool = []

    # Normalize and weight U_os
    if pool_os:
        os_sims = np.array([w for _, _, w in pool_os])
        os_probs = os_sims / os_sims.sum() if os_sims.sum() > 0 else np.ones(len(os_sims)) / len(os_sims)
        for idx, (ni, nj, _) in enumerate(pool_os):
            pool.append((ni, nj, epsilon * os_probs[idx]))

    # Normalize and weight U_us
    if pool_us:
        us_sims = np.array([w for _, _, w in pool_us])
        us_probs = us_sims / us_sims.sum() if us_sims.sum() > 0 else np.ones(len(us_sims)) / len(us_sims)
        for idx, (ni, nj, _) in enumerate(pool_us):
            pool.append((ni, nj, (1 - epsilon) * us_probs[idx]))

    # Deduplicate (keep highest weight)
    seen = {}
    for ni, nj, w in pool:
        key = (min(ni, nj), max(ni, nj))
        if key not in seen or w > seen[key][2]:
            seen[key] = (key[0], key[1], w)

    return list(seen.values())


# ---------------------------------------------------------------------------
# NP3: Union-Find for must-link enforcement
# ---------------------------------------------------------------------------

class UnionFind:
    """Simple Union-Find / Disjoint-Set for must-link merging."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ---------------------------------------------------------------------------
# NP3: Constrained Clustering Refinement
# ---------------------------------------------------------------------------

def apply_must_link_constraints(clustering, node2cid, must_links):
    """
    Enforce must-link constraints by merging clusters.

    Args:
        clustering: dict {cid: set of node_ids}
        node2cid: dict {node_id: cid}
        must_links: list of (node_i, node_j) pairs

    Returns:
        (updated_clustering, updated_node2cid)
    """
    if not must_links:
        return clustering, node2cid

    # Use Union-Find on cluster IDs
    uf = UnionFind()
    for cid in clustering:
        uf.find(cid)

    for ni, nj in must_links:
        if ni in node2cid and nj in node2cid:
            cid_i = node2cid[ni]
            cid_j = node2cid[nj]
            uf.union(cid_i, cid_j)

    # Build merged clustering
    merged = defaultdict(set)
    for cid, nodes in clustering.items():
        root = uf.find(cid)
        merged[root].update(nodes)

    new_clustering = dict(merged)
    new_node2cid = build_node_to_cluster_mapping(new_clustering)
    return new_clustering, new_node2cid


def find_impure_clusters(clustering, node2cid, cannot_links):
    """
    Identify clusters containing cannot-link violations.

    Args:
        clustering: dict {cid: set of node_ids}
        node2cid: dict {node_id: cid}
        cannot_links: list of (node_i, node_j) pairs

    Returns:
        dict {cid: list of (node_i, node_j) CL pairs within the cluster}
    """
    impure = defaultdict(list)
    for ni, nj in cannot_links:
        if ni in node2cid and nj in node2cid:
            if node2cid[ni] == node2cid[nj]:
                impure[node2cid[ni]].append((ni, nj))
    return dict(impure)


def resolve_impure_cluster(cluster_nodes, must_links, cannot_links, embeddings):
    """
    Resolve an impure cluster using NP3: graph coloring + Hungarian matching.

    Steps:
    1. Build ML-groups (connected components of must-link edges)
    2. Build conflict graph: nodes = ML-groups, edges = CL between groups
    3. Graph coloring to partition respecting CL constraints
    4. Hungarian matching for unconstrained singleton groups

    Args:
        cluster_nodes: set of node_ids
        must_links: ML pairs within this cluster
        cannot_links: CL pairs within this cluster
        embeddings: Embeddings object

    Returns:
        list of sets: sub-clusters after resolution
    """
    if len(cluster_nodes) <= 1:
        return [cluster_nodes]

    # Step 1: Build ML-groups
    ml_graph = nx.Graph()
    ml_graph.add_nodes_from(cluster_nodes)
    for ni, nj in must_links:
        if ni in cluster_nodes and nj in cluster_nodes:
            ml_graph.add_edge(ni, nj)

    ml_groups = list(nx.connected_components(ml_graph))
    group_id_map = {}  # node -> group_id
    for gid, group in enumerate(ml_groups):
        for node in group:
            group_id_map[node] = gid

    # Step 2: Build conflict graph among ML-groups
    conflict = nx.Graph()
    conflict.add_nodes_from(range(len(ml_groups)))
    for ni, nj in cannot_links:
        gi = group_id_map.get(ni)
        gj = group_id_map.get(nj)
        if gi is not None and gj is not None and gi != gj:
            conflict.add_edge(gi, gj)

    # Step 3: Graph coloring
    if conflict.number_of_edges() == 0:
        # No conflicts -- everything stays together
        return [set(cluster_nodes)]

    coloring = nx.greedy_color(conflict, strategy='largest_first')

    # Group ML-groups by color
    color_to_groups = defaultdict(list)
    for gid, color in coloring.items():
        color_to_groups[color].append(gid)

    # Step 4: Build sub-clusters from coloring
    # First, assign constrained groups
    constrained_groups = set()
    for ni, nj in cannot_links:
        gi = group_id_map.get(ni)
        gj = group_id_map.get(nj)
        if gi is not None:
            constrained_groups.add(gi)
        if gj is not None:
            constrained_groups.add(gj)
    for ni, nj in must_links:
        gi = group_id_map.get(ni)
        if gi is not None:
            constrained_groups.add(gi)

    # Sub-clusters from coloring (for constrained groups)
    sub_clusters = defaultdict(set)
    unconstrained = []

    for gid in range(len(ml_groups)):
        if gid in constrained_groups:
            color = coloring[gid]
            sub_clusters[color].update(ml_groups[gid])
        else:
            unconstrained.append(gid)

    # Assign unconstrained singleton groups to closest sub-cluster
    if unconstrained and sub_clusters:
        embedding_matrix = embeddings.embeddings
        id_to_idx = embeddings.id_to_idx

        # Compute centroid of each sub-cluster
        sc_ids = list(sub_clusters.keys())
        centroids = []
        for sc_id in sc_ids:
            sc_nodes = list(sub_clusters[sc_id])
            indices = [id_to_idx[n] for n in sc_nodes]
            centroids.append(embedding_matrix[indices].mean(axis=0))
        centroids = np.array(centroids)

        for gid in unconstrained:
            group_nodes = list(ml_groups[gid])
            group_indices = [id_to_idx[n] for n in group_nodes]
            group_centroid = embedding_matrix[group_indices].mean(axis=0)

            # Find closest sub-cluster
            dists = pairwise_distances(
                group_centroid.reshape(1, -1), centroids, metric='cosine'
            )[0]
            closest = sc_ids[np.argmin(dists)]
            sub_clusters[closest].update(group_nodes)
    elif unconstrained and not sub_clusters:
        # No constrained groups -- all unconstrained, keep as one cluster
        all_nodes = set()
        for gid in unconstrained:
            all_nodes.update(ml_groups[gid])
        return [all_nodes]

    result = [nodes for nodes in sub_clusters.values() if len(nodes) > 0]
    return result if result else [set(cluster_nodes)]


# ---------------------------------------------------------------------------
# Main Algorithm Class
# ---------------------------------------------------------------------------

class NP3AASAlgorithm:
    """
    NP3 + AAS clustering algorithm.

    Implements the standard algorithm interface (step, is_finished, get_clustering).

    Phases:
    - INIT: Run DBSCAN + FINCH, build AAS sampling pool
    - ACTIVE_REVIEW: Apply NP3 with human feedback, re-sample
    - FINISHED: Budget exhausted or no disagreements
    """

    def __init__(self, config: Dict, classifier_manager, cluster_validator=None):
        self.config = config
        self.classifier_manager = classifier_manager
        self.cluster_validator = cluster_validator

        # Algorithm state
        self.phase = "INIT"

        # Embeddings (populated during first step)
        self.embeddings = None
        self.embedding_matrix = None
        self.node_ids = None
        self.id_to_idx = None

        # Clusterings
        self.clustering = {}
        self.node2cid = {}
        self.clustering_dbscan = None
        self.clustering_finch = None

        # Constraint sets
        self.must_links = set()
        self.cannot_links = set()

        # Sampling pool
        self.sampling_pool = []

        # Budget
        self.num_human_reviews = 0
        self.max_human_reviews = config.get('max_human_reviews', 5000)
        self.edges_per_review_batch = config.get('edges_per_review_batch', 200)

        # DBSCAN parameters
        dbscan_eps_raw = config.get('dbscan_eps', 0.5)
        if dbscan_eps_raw == 'auto':
            classifier_threshold = config.get('classifier_threshold')
            if classifier_threshold is None:
                raise ValueError(
                    "dbscan_eps='auto' requires 'classifier_threshold' in config. "
                    "Ensure prepare_np3_aas passes the threshold."
                )
            # Convert similarity threshold to cosine distance
            self.dbscan_eps = 1.0 - classifier_threshold
            logger.info(f"  Auto dbscan_eps: classifier_threshold={classifier_threshold:.4f} "
                        f"-> eps={self.dbscan_eps:.4f}")
        else:
            self.dbscan_eps = float(dbscan_eps_raw)
        self.dbscan_min_samples = config.get('dbscan_min_samples', 2)
        self.dbscan_metric = config.get('dbscan_metric', 'cosine')

        # AAS parameters
        self.s_min = config.get('s_min', 0.3)
        self.k_max = config.get('k_max', 5)
        self.epsilon = config.get('epsilon', 0.6)

        # Validation
        self.validation_step = config.get('validation_step', 100)
        self.validation_initialized = False
        # Exact evaluation checkpoints (util/tools.py); resolved once the node count is known.
        self._checkpoints = set()

        logger.info("NP3+AAS Algorithm initialized")
        logger.info(f"  DBSCAN: eps={self.dbscan_eps}, min_samples={self.dbscan_min_samples}")
        logger.info(f"  AAS: s_min={self.s_min}, k_max={self.k_max}, epsilon={self.epsilon}")
        logger.info(f"  Budget: max_human_reviews={self.max_human_reviews}")
        logger.info(f"  Batch size: {self.edges_per_review_batch}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, new_edges: List[Tuple]) -> List[Tuple[int, int, float]]:
        """
        Perform one step of the algorithm.

        Args:
            new_edges: List of (n0, n1, score, verifier_name) tuples

        Returns:
            List of edges needing human review: [(n0, n1, score), ...]
        """
        if self.phase == "INIT":
            return self._init_step(new_edges)
        elif self.phase == "ACTIVE_REVIEW":
            return self._active_review_step(new_edges)
        else:
            return []

    def is_finished(self) -> bool:
        if self.phase == "FINISHED":
            return True
        if self.num_human_reviews >= self.max_human_reviews:
            return True
        return False

    def get_clustering(self) -> Tuple[Dict, Dict, nx.Graph]:
        """Return (clustering_dict, node2cid, G)."""
        G = nx.Graph()
        if self.node_ids:
            G.add_nodes_from(self.node_ids)

        # Add edges within clusters (nearest neighbors to keep it sparse)
        if self.embeddings is not None:
            for cid, nodes in self.clustering.items():
                nodes_list = list(nodes)
                if len(nodes_list) <= 10:
                    # Small cluster: add all pairs
                    for i in range(len(nodes_list)):
                        for j in range(i + 1, len(nodes_list)):
                            score = self.embeddings.get_score(nodes_list[i], nodes_list[j])
                            G.add_edge(nodes_list[i], nodes_list[j],
                                       label='positive', score=score)
                else:
                    # Large cluster: add nearest-neighbor edges
                    indices = [self.id_to_idx[n] for n in nodes_list]
                    sub_emb = self.embedding_matrix[indices]
                    dist = pairwise_distances(sub_emb, metric='cosine')
                    np.fill_diagonal(dist, np.inf)
                    for i in range(len(nodes_list)):
                        nn = np.argmin(dist[i])
                        score = self.embeddings.get_score(nodes_list[i], nodes_list[nn])
                        G.add_edge(nodes_list[i], nodes_list[nn],
                                   label='positive', score=score)

        return self.clustering, self.node2cid, G

    def show_stats(self):
        """Log algorithm statistics."""
        n_clusters = len(self.clustering)
        cluster_sizes = [len(c) for c in self.clustering.values()]
        n_singletons = sum(1 for s in cluster_sizes if s == 1)

        logger.info("=" * 60)
        logger.info("NP3 + AAS Algorithm Statistics")
        logger.info("=" * 60)
        logger.info(f"Phase: {self.phase}")
        logger.info(f"Human reviews: {self.num_human_reviews}")
        logger.info(f"Must-link constraints: {len(self.must_links)}")
        logger.info(f"Cannot-link constraints: {len(self.cannot_links)}")
        logger.info(f"Clusters: {n_clusters}")
        logger.info(f"Singletons: {n_singletons}")
        if cluster_sizes:
            logger.info(f"Cluster sizes: min={min(cluster_sizes)}, "
                        f"max={max(cluster_sizes)}, "
                        f"mean={np.mean(cluster_sizes):.2f}")
        logger.info(f"Sampling pool remaining: {len(self.sampling_pool)}")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase: INIT
    # ------------------------------------------------------------------

    def _init_step(self, new_edges):
        """Initialize: run DBSCAN + FINCH, build AAS pool, return first batch."""
        # Extract embeddings from classifier_manager
        first_classifier = self.classifier_manager.algo_classifiers[0]
        embeddings_obj, _ = self.classifier_manager.classifier_units[first_classifier]
        self.embeddings = embeddings_obj
        self.node_ids = list(embeddings_obj.ids)
        self.id_to_idx = dict(embeddings_obj.id_to_idx)
        self.embedding_matrix = embeddings_obj.embeddings

        n = len(self.node_ids)
        logger.info(f"Running NP3+AAS on {n} nodes")
        self._checkpoints = set(resolve_checkpoints(self.config.get('report_at_reviews', ['N']), n))

        # Run DBSCAN
        logger.info(f"Running DBSCAN (eps={self.dbscan_eps}, "
                     f"min_samples={self.dbscan_min_samples})...")
        dbscan = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric=self.dbscan_metric
        )
        dbscan_labels = dbscan.fit_predict(self.embedding_matrix)
        self.clustering_dbscan = self._labels_to_clustering(dbscan_labels)
        logger.info(f"DBSCAN: {len(self.clustering_dbscan)} clusters")

        # Run FINCH
        logger.info("Running FINCH clustering...")
        finch_partitions = finch_clustering(self.embedding_matrix,
                                            distance_metric='cosine')
        # Use the finest partition (level 0)
        self.clustering_finch = self._labels_to_clustering(finch_partitions[0])
        logger.info(f"FINCH: {len(self.clustering_finch)} clusters "
                     f"({len(finch_partitions)} levels)")

        # Use DBSCAN as working clustering
        self.clustering = {cid: set(nodes) for cid, nodes in
                           self.clustering_dbscan.items()}
        self.node2cid = build_node_to_cluster_mapping(self.clustering)

        # Log initial clustering stats
        self._log_clustering_comparison()

        # Build AAS sampling pool
        self._build_aas_sampling_pool()

        # Validation
        self._handle_validation()

        if not self.sampling_pool:
            logger.info("No disagreements between DBSCAN and FINCH -- finished")
            self.phase = "FINISHED"
            return []

        self.phase = "ACTIVE_REVIEW"
        logger.info("=== Starting Active Review Phase ===")
        return self._sample_next_batch()

    # ------------------------------------------------------------------
    # Phase: ACTIVE_REVIEW
    # ------------------------------------------------------------------

    def _active_review_step(self, new_edges):
        """Process human feedback, apply NP3, re-sample."""
        # Parse human responses into ML/CL constraints
        for edge in new_edges:
            if len(edge) < 4:
                continue
            n0, n1, score, verifier = edge[:4]
            if 'human' in str(verifier):
                self.num_human_reviews += 1
                ordered = (min(n0, n1), max(n0, n1))
                if score > 0.5:
                    self.must_links.add(ordered)
                    self.cannot_links.discard(ordered)
                else:
                    self.cannot_links.add(ordered)
                    self.must_links.discard(ordered)
                if self.num_human_reviews in self._checkpoints:
                    self._log_checkpoint()

        logger.info(f"Constraints: {len(self.must_links)} ML, "
                     f"{len(self.cannot_links)} CL "
                     f"({self.num_human_reviews} total reviews)")

        # Apply NP3 constrained clustering refinement
        self._apply_np3()

        # Rebuild AAS sampling pool with updated clustering
        self._build_aas_sampling_pool()

        # Validation
        self._handle_validation()

        # Check termination
        if self.num_human_reviews >= self.max_human_reviews:
            logger.info(f"Reached max human reviews ({self.max_human_reviews})")
            self.phase = "FINISHED"
            return []

        if not self.sampling_pool:
            logger.info("No more uncertain pairs -- finished")
            self.phase = "FINISHED"
            return []

        return self._sample_next_batch()

    # ------------------------------------------------------------------
    # NP3 Refinement
    # ------------------------------------------------------------------

    def _apply_np3(self):
        """Apply NP3 constrained clustering refinement."""
        # Step 1: Enforce must-link constraints
        self.clustering, self.node2cid = apply_must_link_constraints(
            self.clustering, self.node2cid, list(self.must_links)
        )

        # Step 2: Find impure clusters
        impure = find_impure_clusters(
            self.clustering, self.node2cid, list(self.cannot_links)
        )

        if not impure:
            logger.info("No impure clusters after ML enforcement")
            return

        logger.info(f"Resolving {len(impure)} impure clusters...")

        # Step 3: Resolve each impure cluster
        new_clusters_to_add = {}
        cids_to_remove = []

        for cid, cl_pairs in impure.items():
            cluster_nodes = self.clustering[cid]

            ml_in_cluster = [(i, j) for (i, j) in self.must_links
                             if i in cluster_nodes and j in cluster_nodes]

            sub_clusters = resolve_impure_cluster(
                cluster_nodes, ml_in_cluster, cl_pairs, self.embeddings
            )

            if len(sub_clusters) > 1:
                cids_to_remove.append(cid)
                for sub in sub_clusters:
                    new_cid = max(
                        list(self.clustering.keys()) +
                        list(new_clusters_to_add.keys()),
                        default=-1
                    ) + 1
                    new_clusters_to_add[new_cid] = sub

        # Apply changes
        for cid in cids_to_remove:
            del self.clustering[cid]
        self.clustering.update(new_clusters_to_add)

        # Rebuild node2cid
        self.node2cid = build_node_to_cluster_mapping(self.clustering)

        if cids_to_remove:
            logger.info(f"Split {len(cids_to_remove)} clusters into "
                        f"{len(new_clusters_to_add)} sub-clusters")

    # ------------------------------------------------------------------
    # AAS Sampling
    # ------------------------------------------------------------------

    def _build_aas_sampling_pool(self):
        """Build AAS sampling pool from clustering disagreements."""
        regions = compute_regions_of_uncertainty(
            self.clustering_dbscan, self.clustering_finch
        )

        if not regions:
            self.sampling_pool = []
            logger.info("No regions of uncertainty found")
            return

        self.sampling_pool = build_sampling_pool(
            regions, self.clustering_dbscan, self.clustering_finch,
            self.embeddings, self.s_min, self.k_max, self.epsilon
        )

        # Filter out already-reviewed pairs
        reviewed = self.must_links | self.cannot_links
        self.sampling_pool = [
            (i, j, w) for (i, j, w) in self.sampling_pool
            if (min(i, j), max(i, j)) not in reviewed
        ]

        logger.info(f"AAS sampling pool: {len(self.sampling_pool)} pairs "
                     f"({len(regions)} uncertainty regions)")

    def _sample_next_batch(self):
        """Sample next batch from AAS pool."""
        if not self.sampling_pool:
            return []

        batch_size = min(self.edges_per_review_batch, len(self.sampling_pool))

        # Weighted sampling
        weights = np.array([w for (_, _, w) in self.sampling_pool])
        total = weights.sum()
        if total > 0:
            probs = weights / total
        else:
            probs = np.ones(len(weights)) / len(weights)

        # Handle case where batch_size > pool size
        batch_size = min(batch_size, len(self.sampling_pool))

        indices = np.random.choice(
            len(self.sampling_pool), size=batch_size,
            replace=False, p=probs
        )

        batch = []
        for idx in sorted(indices, reverse=True):
            ni, nj, _ = self.sampling_pool[idx]
            score = self.embeddings.get_score(ni, nj)
            batch.append((ni, nj, score))
            self.sampling_pool.pop(idx)

        logger.info(f"Sampled {len(batch)} pairs for human review")
        return batch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _labels_to_clustering(self, labels):
        """Convert sklearn-style label array to clustering dict."""
        clustering = {}
        next_cid = int(max(labels)) + 1 if len(labels) > 0 else 0
        for idx, label in enumerate(labels):
            node_id = self.node_ids[idx]
            label = int(label)
            if label == -1:
                # DBSCAN noise: each gets own cluster
                clustering[next_cid] = {node_id}
                next_cid += 1
            else:
                if label not in clustering:
                    clustering[label] = set()
                clustering[label].add(node_id)
        return clustering

    def _log_clustering_comparison(self):
        """Log comparison between DBSCAN and FINCH clusterings."""
        a_sizes = sorted([len(c) for c in self.clustering_dbscan.values()],
                         reverse=True)
        b_sizes = sorted([len(c) for c in self.clustering_finch.values()],
                         reverse=True)

        logger.info(f"DBSCAN: {len(a_sizes)} clusters, "
                     f"sizes: {a_sizes[:10]}{'...' if len(a_sizes) > 10 else ''}")
        logger.info(f"FINCH:  {len(b_sizes)} clusters, "
                     f"sizes: {b_sizes[:10]}{'...' if len(b_sizes) > 10 else ''}")

    def _log_checkpoint(self):
        """Log the clustering after exactly `num_human_reviews` reviews: the NP3 refinement
        applied to the constraints received so far, as it would be at the end of a batch.
        Runs on copies of the clustering and constraint sets; the refinement draws no random
        numbers, so the live run (including its sampling RNG) is left untouched."""
        if not (self.cluster_validator and self.validation_initialized):
            return
        import copy
        live = (self.clustering, self.node2cid, self.must_links, self.cannot_links)
        try:
            self.clustering = copy.deepcopy(live[0])
            self.node2cid = dict(live[1])
            self.must_links = set(live[2])
            self.cannot_links = set(live[3])
            self._apply_np3()
            self.cluster_validator.incremental_stats(
                self.num_human_reviews, self.clustering, self.node2cid,
                self.cluster_validator.gt_clustering, self.cluster_validator.gt_node2cid,
                CHECKPOINT_LABEL)
        finally:
            self.clustering, self.node2cid, self.must_links, self.cannot_links = live

    def _handle_validation(self):
        """Validate against ground truth (same pattern as stability algorithm)."""
        if not self.cluster_validator:
            return

        if not self.validation_initialized:
            clustering, node2cid, G = self.get_clustering()
            self.cluster_validator.trace_start_human(
                clustering, node2cid, G, self.num_human_reviews
            )
            self.validation_initialized = True
            return

        if hasattr(self.cluster_validator, 'prev_num_human'):
            if (self.num_human_reviews - self.cluster_validator.prev_num_human
                    >= self.validation_step):
                clustering, node2cid, G = self.get_clustering()
                self.cluster_validator.trace_iter_compare_to_gt(
                    clustering, node2cid, self.num_human_reviews, G
                )
