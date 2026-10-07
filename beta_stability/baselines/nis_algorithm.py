"""
NIS (Nested Importance Sampling) clustering algorithm.

Exact implementation of Algorithm 1 from:
"Human-in-the-Loop Visual Re-ID for Population Size Estimation"
(Perez et al., ECCV 2024, arXiv:2312.05287)

Reference: https://github.com/cvl-umass/counting-clusters

The algorithm:
1. Computes cosine similarity matrix s_ij between embeddings
2. Computes approximate degree n_hat(u) = sum_v s_ij(u,v)
3. Vertex proposal Q(u) ∝ 1/n_hat(u) (biases toward isolated nodes)
4. Neighbor proposal q_u(v) ∝ s_ij(u,v) (biases toward likely matches)
5. Samples N_v vertices from Q, M neighbors per vertex from q_u
   (self always included as first neighbor)
6. Uses human oracle responses to estimate K via doubly IS-weighted formula
7. Produces clustering by running k-means with k = round(K_hat)
"""

import logging
import networkx as nx
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from beta_stability.util.cluster_tools import build_node_to_cluster_mapping

logger = logging.getLogger("beta_stability")


class NISAlgorithm:
    """
    NIS (Nested Importance Sampling) clustering algorithm.

    Matches the reference implementation at:
    https://github.com/cvl-umass/counting-clusters/blob/main/src/methods.py

    Phases:
    - INIT: Compute cosine sim, build proposals, pre-sample all pairs
    - SAMPLING: Serve pairs for human review, accumulate oracle responses
    - FINISHED: Compute final K_hat and run k-means
    """

    def __init__(self, config: Dict, classifier_manager, cluster_validator=None):
        self.config = config
        self.classifier_manager = classifier_manager
        self.cluster_validator = cluster_validator

        # Algorithm state
        self.phase = "INIT"

        # Embeddings (populated during _init_step)
        self.embeddings = None
        self.embedding_matrix = None
        self.node_ids = None
        self.id_to_idx = None
        self.n_nodes = 0

        # NIS parameters (N_v = vertices, N_n = neighbors per vertex)
        self.N_v = config.get('n_sampled_vertices', 50)
        self.N_n = config.get('m_neighbors_per_vertex', 100)

        # Budget
        self.num_human_reviews = 0
        self.max_human_reviews = config.get('max_human_reviews', 5000)
        self.edges_per_review_batch = config.get('edges_per_review_batch', 200)

        # Validation
        self.validation_step = config.get('validation_step', 100)
        self.validation_initialized = False

        # NIS distributions (populated in _init_step)
        self.Q_prob = None            # Vertex proposal Q(u), array of size n
        self.n_hat = None             # Approximate degree n_hat(u) = sum_v s(u,v)
        self.s_ij = None              # Raw cosine similarity matrix

        # Sampling plan (populated in _init_step)
        self.sampled_vertices = None  # List of N_v vertex indices
        self.sampled_neighbors = None # List of N_v lists, each N_n neighbor indices
        self.q_all = None             # List of N_v arrays, q_u distributions

        # Edge queue: only non-self pairs need human review (deduplicated)
        self.all_pairs = []           # List of (node_u, node_v, score) - unique pairs
        # Maps (node_u, node_v) -> list of (vertex_idx, neighbor_idx) slots
        # Multiple slots can share the same pair (WITH-replacement sampling)
        self.pair_to_locations = defaultdict(list)
        self.pair_idx = 0             # Next pair to serve

        # Oracle responses: gt_s[i][j] = human response for vertex i, neighbor j
        # None = not yet reviewed, 1.0 = same, 0.0 = different
        self.oracle_responses = {}    # (vertex_sample_idx, neighbor_sample_idx) -> 0/1

        # K estimates
        self.K_hat = None
        self.K_hat_ci_lower = None
        self.K_hat_ci_upper = None
        self.K_hat_history = []

        # Clustering
        self.clustering = {}
        self.node2cid = {}
        self._last_k = None  # Track last k used for k-means to avoid redundant runs
        self._normalized_emb = None  # Cached L2-normalized embeddings

        logger.info("NIS Algorithm initialized (reference implementation)")
        logger.info(f"  N_v={self.N_v}, N_n={self.N_n}")
        logger.info(f"  Budget: max_human_reviews={self.max_human_reviews}")
        logger.info(f"  Batch size: {self.edges_per_review_batch}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, new_edges: List[Tuple]) -> List[Tuple[int, int, float]]:
        if self.phase == "INIT":
            return self._init_step(new_edges)
        elif self.phase == "SAMPLING":
            return self._sampling_step(new_edges)
        else:
            return []

    def is_finished(self) -> bool:
        return (self.phase == "FINISHED" or
                self.num_human_reviews >= self.max_human_reviews)

    def get_clustering(self) -> Tuple[Dict, Dict, nx.Graph]:
        """Return (clustering_dict, node2cid, G)."""
        G = nx.Graph()
        if self.node_ids:
            G.add_nodes_from(self.node_ids)

        if self.embeddings is not None:
            for cid, nodes in self.clustering.items():
                nodes_list = list(nodes)
                if len(nodes_list) <= 10:
                    for i in range(len(nodes_list)):
                        for j in range(i + 1, len(nodes_list)):
                            score = self.embeddings.get_score(
                                nodes_list[i], nodes_list[j])
                            G.add_edge(nodes_list[i], nodes_list[j],
                                       label='positive', score=score)
                else:
                    indices = [self.id_to_idx[n] for n in nodes_list]
                    sub_emb = self.embedding_matrix[indices]
                    dist = pairwise_distances(sub_emb, metric='cosine')
                    np.fill_diagonal(dist, np.inf)
                    for i in range(len(nodes_list)):
                        nn = np.argmin(dist[i])
                        score = self.embeddings.get_score(
                            nodes_list[i], nodes_list[nn])
                        G.add_edge(nodes_list[i], nodes_list[nn],
                                   label='positive', score=score)

        return self.clustering, self.node2cid, G

    def show_stats(self):
        n_clusters = len(self.clustering)
        cluster_sizes = [len(c) for c in self.clustering.values()]
        n_singletons = sum(1 for s in cluster_sizes if s == 1)

        logger.info("=" * 60)
        logger.info("NIS Algorithm Statistics (reference implementation)")
        logger.info("=" * 60)
        logger.info(f"Phase: {self.phase}")
        logger.info(f"Human reviews: {self.num_human_reviews}")
        logger.info(f"NIS parameters: N_v={self.N_v}, N_n={self.N_n}")
        if self.K_hat is not None:
            logger.info(f"K_hat: {self.K_hat:.2f} "
                        f"(95% CI: [{self.K_hat_ci_lower:.2f}, "
                        f"{self.K_hat_ci_upper:.2f}])")
            k_used = max(1, min(self.n_nodes, round(self.K_hat)))
            logger.info(f"k used for k-means: {k_used}")
        logger.info(f"Clusters: {n_clusters}")
        logger.info(f"Singletons: {n_singletons}")
        if cluster_sizes:
            logger.info(f"Cluster sizes: min={min(cluster_sizes)}, "
                        f"max={max(cluster_sizes)}, "
                        f"mean={np.mean(cluster_sizes):.2f}")
        total_pairs = len(self.all_pairs) if self.all_pairs else 0
        logger.info(f"Non-self pairs reviewed: {self.pair_idx}/{total_pairs}")
        if self.K_hat_history:
            logger.info("K_hat history:")
            for (reviews, k, ci_lo, ci_hi, n_complete) in self.K_hat_history:
                logger.info(f"  {reviews} reviews: K={k:.2f} "
                            f"[{ci_lo:.2f}, {ci_hi:.2f}] "
                            f"({n_complete} vertices complete)")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase: INIT
    # ------------------------------------------------------------------

    def _init_step(self, new_edges):
        """Initialize: compute similarities, build proposals, pre-sample pairs.

        Follows the reference implementation exactly:
        1. s_ij = cosine similarity (raw, no softmax)
        2. n_hat(u) = sum_v s_ij(u,v) (approximate degree including self)
        3. Q(u) ∝ 1/n_hat(u) (vertex proposal)
        4. q_u(v) ∝ s_ij(u,v) (neighbor proposal)
        5. Sample WITH replacement
        6. Self (u) always included as first neighbor
        """
        # Extract embeddings from classifier_manager
        first_classifier = self.classifier_manager.algo_classifiers[0]
        embeddings_obj, _ = self.classifier_manager.classifier_units[
            first_classifier]
        self.embeddings = embeddings_obj
        self.node_ids = list(embeddings_obj.ids)
        self.id_to_idx = dict(embeddings_obj.id_to_idx)
        self.embedding_matrix = embeddings_obj.embeddings
        self.n_nodes = len(self.node_ids)

        logger.info(f"Running NIS on {self.n_nodes} nodes")

        # Step 1: Compute raw cosine similarity matrix
        # Reference: s_ij = 1 - cosine_distance, diagonal=1, clip negatives
        logger.info("Computing cosine similarity matrix...")
        self.s_ij = 1.0 - pairwise_distances(
            self.embedding_matrix, metric='cosine')
        np.fill_diagonal(self.s_ij, 1.0)
        self.s_ij[self.s_ij < 0] = 0

        # Step 2: Compute approximate degree n_hat(u) = sum_v s(u,v)
        # Reference: n_hat.append(np.sum(s_ij[i]))
        self.n_hat = np.sum(self.s_ij, axis=1)

        logger.info(f"n_hat range: [{self.n_hat.min():.2f}, "
                    f"{self.n_hat.max():.2f}], "
                    f"mean={self.n_hat.mean():.2f}")

        # Step 3: Vertex proposal Q(u) ∝ 1/n_hat(u)
        # Reference: Q = (1/np.array(n_hat)) / np.sum((1/np.array(n_hat)))
        inv_n_hat = 1.0 / self.n_hat
        self.Q_prob = inv_n_hat / inv_n_hat.sum()

        logger.info(f"Q(u) range: [{self.Q_prob.min():.6f}, "
                    f"{self.Q_prob.max():.6f}], "
                    f"ratio max/min: "
                    f"{self.Q_prob.max() / max(self.Q_prob.min(), 1e-15):.2f}")

        # Adjust N_v, N_n for dataset size and budget
        self.N_v = min(self.N_v, self.n_nodes)
        self.N_n = min(self.N_n, self.n_nodes)
        total_human_budget = self.N_v * (self.N_n - 1)
        if total_human_budget > self.max_human_reviews:
            ratio = self.config.get('nis_nv_nn_ratio', 7)
            self.N_v = max(1, int(np.sqrt(self.max_human_reviews / ratio)))
            self.N_v = min(self.N_v, self.n_nodes, self.max_human_reviews)
            self.N_n = max(2, min(self.n_nodes,
                           self.max_human_reviews // self.N_v + 1))
            total_human_budget = self.N_v * (self.N_n - 1)
            logger.info(f"NIS: Budget-adjusted N_v={self.N_v}, N_n={self.N_n} "
                        f"(ratio={ratio})")
        logger.info(f"NIS: N_v={self.N_v}, N_n={self.N_n}, "
                    f"total_human_pairs={total_human_budget}")

        # Step 4: Sample N_v vertices from Q (WITH replacement, matching reference)
        # Reference: np.random.choice(..., N_v, p=Q, replace=True)
        logger.info(f"Sampling {self.N_v} vertices from Q(u) "
                    f"(with replacement)...")
        self.sampled_vertices = list(np.random.choice(
            self.n_nodes, size=self.N_v, p=self.Q_prob, replace=True))

        # Step 5: For each vertex, compute q_u and sample N_n-1 neighbors
        # Then prepend self as first neighbor
        # Reference: q = (s_ij[v_i]) / np.sum((s_ij[v_i]))
        #            [v_i] + list(np.random.choice(..., N_n-1, p=q, replace=True))
        logger.info(f"Sampling {self.N_n} neighbors per vertex "
                    f"(self + {self.N_n - 1} from q_u, with replacement)...")
        self.sampled_neighbors = []
        self.q_all = []

        for v_i in self.sampled_vertices:
            # q_u(v) ∝ s_ij(u,v) for ALL v (including self)
            # Reference: q = (s_ij[v_i]) / np.sum((s_ij[v_i]))
            q = self.s_ij[v_i].copy()
            q_sum = q.sum()
            if q_sum > 0:
                q = q / q_sum
            else:
                q = np.ones(self.n_nodes) / self.n_nodes
            self.q_all.append(q)

            # Self always first, then N_n-1 sampled neighbors (with replacement)
            neighbors = [v_i] + list(np.random.choice(
                self.n_nodes, size=self.N_n - 1, p=q, replace=True))
            self.sampled_neighbors.append(neighbors)

        # Step 6: Build flat pair list for human review (non-self pairs only)
        # Self-pairs (j=0) have known oracle response = 1 (same individual)
        # Build mapping from unique (u_id, v_id) pairs to all (i, j) slots.
        # With WITH-replacement sampling, many slots can share the same pair.
        logger.info("Building pair list for human review...")
        self.pair_to_locations = defaultdict(list)
        seen_pairs = set()

        total_slots = 0
        for i in range(self.N_v):
            u_idx = self.sampled_vertices[i]
            u_id = self.node_ids[u_idx]

            # Self response is always 1 (same individual)
            self.oracle_responses[(i, 0)] = 1.0

            # Non-self neighbors need human review
            for j in range(1, self.N_n):
                v_idx = self.sampled_neighbors[i][j]
                v_id = self.node_ids[v_idx]
                total_slots += 1

                # If u==v (self sampled again), auto-fill as 1
                if u_idx == v_idx:
                    self.oracle_responses[(i, j)] = 1.0
                    continue

                # Map this slot; deduplicate the pair list
                pair_key = (u_id, v_id)
                self.pair_to_locations[pair_key].append((i, j))
                if pair_key not in seen_pairs:
                    score = self.embeddings.get_score(u_id, v_id)
                    self.all_pairs.append((u_id, v_id, score))
                    seen_pairs.add(pair_key)

        self.pair_idx = 0
        n_auto = sum(1 for k in self.oracle_responses)
        logger.info(f"NIS: {total_slots} total slots, "
                    f"{len(self.all_pairs)} unique pairs for human review, "
                    f"{n_auto} auto-filled (self-pairs)")

        # Initial K estimate using approximate similarities (0 human reviews)
        # K_0 = Σ_u 1/n_hat(u) = Σ_u Q(u) * Z where Z = Σ_w 1/n_hat(w)
        # Since n_hat(u) = 1 + Σ_{v≠u} s(u,v), this estimates K from embeddings alone
        K_0 = float(np.sum(1.0 / self.n_hat))
        K_0 = max(1.0, min(float(self.n_nodes), K_0))
        self.K_hat = K_0
        self.K_hat_ci_lower = 1.0
        self.K_hat_ci_upper = float(self.n_nodes)
        self.K_hat_history.append((0, K_0, 1.0, float(self.n_nodes), 0))
        logger.info(f"NIS: Initial K_hat_0 = {K_0:.2f} (from approximate similarity, 0 reviews)")

        # Run initial k-means with K_0
        self._run_kmeans()
        self.node2cid = build_node_to_cluster_mapping(self.clustering)

        # Validate initial state
        self._handle_validation()

        # Transition to sampling
        self.phase = "SAMPLING"
        logger.info("=== Starting NIS Sampling Phase ===")
        return self._serve_next_batch()

    # ------------------------------------------------------------------
    # Phase: SAMPLING
    # ------------------------------------------------------------------

    def _sampling_step(self, new_edges):
        """Process human feedback, update K estimate, run k-means."""
        # Parse human responses
        for edge in new_edges:
            if len(edge) < 4:
                continue
            n0, n1, score, verifier = edge[:4]
            if 'human' in str(verifier):
                self.num_human_reviews += 1

                # Binary response: score > 0.5 means same individual
                s_val = 1.0 if score > 0.5 else 0.0

                # Map to ALL vertex/neighbor slots that need this pair
                pair_key = None
                if (n0, n1) in self.pair_to_locations:
                    pair_key = (n0, n1)
                elif (n1, n0) in self.pair_to_locations:
                    pair_key = (n1, n0)
                if pair_key is not None:
                    for i, j in self.pair_to_locations[pair_key]:
                        self.oracle_responses[(i, j)] = s_val

        # Update K estimate from fully-reviewed vertices
        self._update_k_estimate()

        # Run k-means only if k changed (k-means is the main bottleneck)
        if self.K_hat is not None:
            new_k = max(1, min(self.n_nodes, round(self.K_hat)))
            if new_k != self._last_k:
                self._run_kmeans()
                self._handle_validation()

        # Check termination
        if self.num_human_reviews >= self.max_human_reviews:
            logger.info(f"NIS: Reached max human reviews "
                        f"({self.max_human_reviews})")
            self._finalize()
            return []

        if self.pair_idx >= len(self.all_pairs):
            logger.info("NIS: All pairs reviewed")
            self._finalize()
            return []

        return self._serve_next_batch()

    # ------------------------------------------------------------------
    # NIS K Estimation (exact reference implementation)
    # ------------------------------------------------------------------

    def _update_k_estimate(self):
        """
        Compute K_hat using the reference per-vertex estimator.

        Reference implementation (Algorithm 1):
            n_bar(u_i) = (1/N_n) * Σ_{j=0}^{N_n-1} gt_s(u_i,v_j) / q(v_j)
            term(u_i)  = (1 / n_bar(u_i)) * (1 / Q(u_i))
            CC_NIS     = (1/N_v) * Σ_i term(u_i)
            CI         = CC_NIS ± 1.96 * std(terms) / sqrt(N_complete)

        Self (j=0) is included in the sum with q including self.
        This matches: https://github.com/cvl-umass/counting-clusters
        """
        terms = []

        for i in range(self.N_v):
            # Check if all N_n responses are available for this vertex
            all_reviewed = True
            for j in range(self.N_n):
                if (i, j) not in self.oracle_responses:
                    all_reviewed = False
                    break
            if not all_reviewed:
                continue

            u_idx = self.sampled_vertices[i]

            # IS estimate: n_bar = (1/N_n) * Σ_j gt_s / q
            # j=0 is self (gt_s=1, q=q(self)), included in sum
            sum_is = 0.0
            for j in range(self.N_n):
                v_idx = self.sampled_neighbors[i][j]
                gt_s = self.oracle_responses[(i, j)]
                q_val = self.q_all[i][v_idx]
                if q_val > 0:
                    sum_is += gt_s / q_val

            n_bar = sum_is / self.N_n

            Q_u = self.Q_prob[u_idx]
            if Q_u > 0 and n_bar > 0:
                term = (1.0 / n_bar) * (1.0 / Q_u)
                terms.append(term)

        N_complete = len(terms)
        if N_complete == 0:
            return

        terms = np.array(terms)

        # K_hat = mean of per-vertex terms
        self.K_hat = float(np.mean(terms))

        # 95% CI
        if N_complete > 1:
            term_std = float(np.std(terms, ddof=1))
            ci = 1.96 * term_std / np.sqrt(N_complete)
            self.K_hat_ci_lower = self.K_hat - ci
            self.K_hat_ci_upper = self.K_hat + ci
        else:
            self.K_hat_ci_lower = 1.0
            self.K_hat_ci_upper = float(self.n_nodes)

        # Clamp to valid range
        self.K_hat = max(1.0, min(float(self.n_nodes), self.K_hat))
        self.K_hat_ci_lower = max(1.0, self.K_hat_ci_lower)
        self.K_hat_ci_upper = min(float(self.n_nodes), self.K_hat_ci_upper)

        self.K_hat_history.append((
            self.num_human_reviews, self.K_hat,
            self.K_hat_ci_lower, self.K_hat_ci_upper, N_complete
        ))

        logger.info(f"NIS: K_hat = {self.K_hat:.2f} "
                    f"(95% CI: [{self.K_hat_ci_lower:.2f}, "
                    f"{self.K_hat_ci_upper:.2f}]) "
                    f"from {N_complete}/{self.N_v} fully-reviewed vertices "
                    f"after {self.num_human_reviews} reviews")

    # ------------------------------------------------------------------
    # k-means Clustering
    # ------------------------------------------------------------------

    def _run_kmeans(self):
        """Run k-means with current K_hat on L2-normalized embeddings."""
        k = max(1, min(self.n_nodes, round(self.K_hat)))

        if self._normalized_emb is None:
            self._normalized_emb = normalize(self.embedding_matrix, norm='l2')

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self._normalized_emb)

        self.clustering = self._labels_to_clustering(labels)
        self.node2cid = build_node_to_cluster_mapping(self.clustering)
        self._last_k = k

        logger.info(f"NIS: k-means with k={k} produced "
                    f"{len(self.clustering)} clusters")

    # ------------------------------------------------------------------
    # Batch Management
    # ------------------------------------------------------------------

    def _serve_next_batch(self):
        """Serve the next batch of non-self pairs for human review."""
        remaining = len(self.all_pairs) - self.pair_idx
        batch_size = min(self.edges_per_review_batch, remaining)

        if batch_size == 0:
            return []

        batch = self.all_pairs[self.pair_idx:self.pair_idx + batch_size]
        self.pair_idx += batch_size

        logger.info(f"NIS: Serving batch of {len(batch)} pairs "
                    f"({self.pair_idx}/{len(self.all_pairs)} total, "
                    f"{self.num_human_reviews} reviews so far)")

        return batch

    def _finalize(self):
        """Finalize: compute final K_hat and run k-means."""
        self._update_k_estimate()

        if self.K_hat is not None:
            self._run_kmeans()
        else:
            logger.warning("NIS: No K estimate obtained, "
                           "using singleton clustering")
            self.clustering = {i: {self.node_ids[i]}
                               for i in range(self.n_nodes)}
            self.node2cid = build_node_to_cluster_mapping(self.clustering)

        self._handle_validation()
        self.phase = "FINISHED"
        logger.info(f"NIS: Algorithm finished. Final K_hat={self.K_hat}, "
                    f"clusters={len(self.clustering)}")

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
                clustering[next_cid] = {node_id}
                next_cid += 1
            else:
                if label not in clustering:
                    clustering[label] = set()
                clustering[label].add(node_id)
        return clustering

    def _handle_validation(self):
        """Validate against ground truth."""
        if not self.cluster_validator:
            return

        if not self.validation_initialized:
            clustering, node2cid, G = self.get_clustering()
            self.cluster_validator.trace_start_human(
                clustering, node2cid, G, self.num_human_reviews)
            self.validation_initialized = True
            return

        if hasattr(self.cluster_validator, 'prev_num_human'):
            if (self.num_human_reviews - self.cluster_validator.prev_num_human
                    >= self.validation_step):
                clustering, node2cid, G = self.get_clustering()
                self.cluster_validator.trace_iter_compare_to_gt(
                    clustering, node2cid, self.num_human_reviews, G)
