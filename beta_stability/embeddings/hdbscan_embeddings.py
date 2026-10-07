import numpy as np
import hdbscan
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_chunked
import time
import functools
from beta_stability.util.tools import kth_diag_indices
import logging

logger = logging.getLogger("beta_stability")


class HDBSCANEmbeddings(object):
    """
    Embeddings class that uses HDBSCAN clustering probabilities for scoring.
    Follows the same interface as Embeddings class but scores are based on
    HDBSCAN's cluster membership probabilities instead of raw distances.
    """

    def __init__(self, embeddings, ids, distance_power=1, print_func=print, min_cluster_size=2, metric='euclidean'):
        self.embeddings = np.array(embeddings)
        self.uuids = ids
        self.ids = list(ids.keys())
        self.distance_power = distance_power
        self.print_func = print_func
        self.min_cluster_size = min_cluster_size
        self.metric = metric

        # Train HDBSCAN clusterer with prediction_data=True to enable soft clustering
        self.print_func(f"Training HDBSCAN with min_cluster_size={self.min_cluster_size}, metric={self.metric}")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric=self.metric,
            cluster_selection_method='eom',
            prediction_data=True,
            gen_min_span_tree=True,
            alpha=1.0
        )
        self.labels = self.clusterer.fit_predict(self.embeddings)

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        self.print_func(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points")

        # Precompute mutual reachability distances from HDBSCAN
        self._precompute_mutual_reachability()

        # Also precompute soft cluster memberships for backward compatibility
        self._precompute_memberships()

    def _precompute_mutual_reachability(self):
        """Precompute mutual reachability distance matrix from HDBSCAN's internal graph."""
        from sklearn.neighbors import NearestNeighbors

        n = len(self.embeddings)
        self.mutual_reachability_matrix = np.full((n, n), np.inf)

        # Compute core distances: distance to k-th nearest neighbor
        # where k = min_cluster_size (this is the standard HDBSCAN approach)
        k = self.min_cluster_size
        self.print_func(f"Computing core distances using k={k} nearest neighbors...")

        nbrs = NearestNeighbors(n_neighbors=k+1, metric=self.metric).fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)
        # Core distance is the distance to the k-th nearest neighbor (index k, since index 0 is the point itself)
        core_distances = distances[:, k]

        self.print_func(f"Core distances range: [{np.min(core_distances):.4f}, {np.max(core_distances):.4f}]")

        # Compute mutual reachability for all pairs in the same cluster
        # Mutual reachability distance = max(core_distance[i], core_distance[j], distance[i,j])
        self.print_func("Computing mutual reachability distances for all cluster pairs...")
        pair_count = 0
        for i in range(n):
            for j in range(i+1, n):
                # Only compute for points in the same cluster (not noise)
                if self.labels[i] == self.labels[j] and self.labels[i] != -1:
                    # Compute raw distance
                    if self.metric == 'euclidean':
                        raw_dist = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
                    elif self.metric == 'cosine':
                        raw_dist = cosine(self.embeddings[i], self.embeddings[j])
                    else:
                        # Use scipy's distance for other metrics
                        from scipy.spatial.distance import pdist
                        raw_dist = pdist([self.embeddings[i], self.embeddings[j]], metric=self.metric)[0]

                    # Mutual reachability distance
                    mr_dist = max(core_distances[i], core_distances[j], raw_dist)
                    self.mutual_reachability_matrix[i, j] = mr_dist
                    self.mutual_reachability_matrix[j, i] = mr_dist
                    pair_count += 1

        self.print_func(f"Computed mutual reachability for {pair_count} pairs in same clusters")

        # Count finite edges (excluding diagonal)
        finite_count = np.sum(np.isfinite(self.mutual_reachability_matrix)) - n  # subtract diagonal
        self.print_func(f"Mutual reachability matrix: {finite_count/2} finite edges")

        # Set diagonal to 0
        np.fill_diagonal(self.mutual_reachability_matrix, 0)

        # Normalize to [0, 1] range: convert distances to similarities
        # Finite distances become scores, infinite distances become 0 (no connection)
        # Exclude diagonal when finding max
        finite_mask = np.isfinite(self.mutual_reachability_matrix)
        off_diag_mask = ~np.eye(n, dtype=bool)
        finite_off_diag = finite_mask & off_diag_mask

        if np.any(finite_off_diag):
            max_finite_dist = np.max(self.mutual_reachability_matrix[finite_off_diag])
            min_finite_dist = np.min(self.mutual_reachability_matrix[finite_off_diag])
            self.max_reachability_dist = max_finite_dist if max_finite_dist > 0 else 1.0
            self.print_func(f"Mutual reachability distances range: [{min_finite_dist:.4f}, {max_finite_dist:.4f}]")
        else:
            self.max_reachability_dist = 1.0
            self.print_func("Warning: No finite off-diagonal mutual reachability distances found!")

        # Convert to similarity scores: high distance = low score
        # Score = 1 - (distance / max_distance) for finite distances
        # Score = 0 for infinite distances (not connected)
        self.similarity_matrix = np.zeros_like(self.mutual_reachability_matrix)
        self.similarity_matrix[finite_mask] = 1.0 - (self.mutual_reachability_matrix[finite_mask] / self.max_reachability_dist)
        np.fill_diagonal(self.similarity_matrix, 1.0)  # Self-similarity is 1.0

        non_zero_scores = np.sum((self.similarity_matrix > 0) & off_diag_mask)
        self.print_func(f"Similarity scores: {non_zero_scores/2} non-zero scores, range: [{np.min(self.similarity_matrix[off_diag_mask]):.4f}, {np.max(self.similarity_matrix):.4f}]")

    def _precompute_memberships(self):
        """Precompute membership vectors for all points."""
        self.membership_vectors = []
        if hasattr(self.clusterer, 'prediction_data_'):
            try:
                for i in range(len(self.embeddings)):
                    membership = hdbscan.membership_vector(self.clusterer, self.embeddings[i:i+1])[0]
                    self.membership_vectors.append(membership)
            except:
                # Fall back to hard clustering
                n_clusters = len(set(self.labels[self.labels != -1]))
                for i in range(len(self.embeddings)):
                    membership = np.zeros(n_clusters)
                    if self.labels[i] != -1:
                        membership[self.labels[i]] = 1.0
                    self.membership_vectors.append(membership)
        else:
            # Hard clustering only
            n_clusters = len(set(self.labels[self.labels != -1]))
            for i in range(len(self.embeddings)):
                membership = np.zeros(n_clusters)
                if self.labels[i] != -1:
                    membership[self.labels[i]] = 1.0
                self.membership_vectors.append(membership)

    def _reduce_func(self, distmat, start):
        """Extract similarity scores from precomputed matrix (range [0, 1])."""
        # Use precomputed similarity matrix (already in [0, 1] range)
        scores = np.zeros_like(distmat)

        for i in range(distmat.shape[0]):
            for j in range(distmat.shape[1]):
                node_i = i + start
                node_j = j

                if node_i < self.similarity_matrix.shape[0] and node_j < self.similarity_matrix.shape[1]:
                    # Use similarity score (1 = connected, 0 = not connected)
                    # Convert to distance: distance = 1 - similarity
                    scores[i, j] = 1.0 - self.similarity_matrix[node_i, node_j]
                else:
                    scores[i, j] = 1.0  # Maximum distance for out of bounds

        # Set diagonal to maximum distance (to avoid self-matching)
        rng = np.arange(min(scores.shape[0], scores.shape[1] - start))
        if len(rng) > 0 and start < scores.shape[1]:
            scores[rng, rng + start] = 1.0

        return scores

    @functools.cache
    def _calculate_distance_matrix(self, flags=None):
        """Calculate distance matrix using HDBSCAN membership probabilities."""
        if flags is None:
            embeddings = self.embeddings
            ids = self.ids
        else:
            embeddings = [self.embeddings[i] for (i, f) in enumerate(flags) if f]
            ids = [self.ids[i] for (i, f) in enumerate(flags) if f]

        # Use pairwise_distances_chunked with our custom reduce function
        chunks = pairwise_distances_chunked(
            embeddings,
            metric=self.metric,
            reduce_func=self._reduce_func,
            n_jobs=-1
        )
        return list(chunks), ids

    def get_score(self, id1, id2):
        """
        Get HDBSCAN similarity score between two nodes (range [0, 1]).

        Score interpretation:
        - score > 0: Points have a finite mutual reachability distance (connected by HDBSCAN)
        - score = 0: Points have infinite mutual reachability distance (not connected by HDBSCAN)

        Higher scores indicate stronger connections (smaller mutual reachability distances).
        """
        idx1 = self.ids.index(id1)
        idx2 = self.ids.index(id2)

        # Use precomputed similarity matrix based on mutual reachability
        if idx1 < self.similarity_matrix.shape[0] and idx2 < self.similarity_matrix.shape[1]:
            return float(self.similarity_matrix[idx1, idx2])

        # Fallback: not connected
        return 0.0

    def is_connected(self, id1, id2):
        """
        Check if two nodes are connected according to HDBSCAN.
        Returns True if they have a finite mutual reachability distance (score > 0).
        """
        idx1 = self.ids.index(id1)
        idx2 = self.ids.index(id2)

        # Check if there's a finite mutual reachability distance
        if idx1 < self.mutual_reachability_matrix.shape[0] and idx2 < self.mutual_reachability_matrix.shape[1]:
            return np.isfinite(self.mutual_reachability_matrix[idx1, idx2])

        return False

    def get_embeddings_score(self, embedding1, embedding2):
        """Get score between two embeddings (requires finding their indices). Range [0, 1]."""
        # This is less efficient - try to use get_score with IDs instead
        idx1 = None
        idx2 = None

        for i, emb in enumerate(self.embeddings):
            if np.allclose(emb, embedding1):
                idx1 = i
            if np.allclose(emb, embedding2):
                idx2 = i
            if idx1 is not None and idx2 is not None:
                break

        if idx1 is not None and idx2 is not None:
            return float(self.similarity_matrix[idx1, idx2])

        return 0.0

    def sign_power(self, x, power):
        return np.sign(x) * np.power(np.abs(x), power)

    def get_score_from_cosine_distance(self, cosine_dist):
        """Convert cosine distance to score (kept for compatibility)."""
        return 1 - self.sign_power(cosine_dist, self.distance_power) * 0.5

    def get_topk_acc(self, labels_q, labels_db, dists, topk):
        return sum(self.get_topk_hits(labels_q, labels_db, dists, topk)) / len(labels_q)

    def get_topk_hits(self, labels_q, labels_db, dists, topk):
        indices = np.argsort(dists, axis=1)
        top_labels = np.array(labels_db)[indices[:, :topk]]
        hits = (top_labels.T == labels_q).T
        return np.sum(hits[:, :topk+1], axis=1) > 0

    def get_top_ks(self, q_pids, distmat, ks=[1, 3, 5, 10]):
        return [(k, self.get_topk_acc(q_pids, q_pids, distmat, k)) for k in ks]

    def get_stats(self, df, filter_key, id_key='uuid'):
        start_time = time.time()
        self.print_func("Calculating distances...")
        self.print_func(f"{len(self.embeddings)}/{len(self.ids)}")

        chunks, ids = self._calculate_distance_matrix()
        distmat = np.concatenate(list(chunks), axis=0)

        labels = [df.loc[df[id_key] == self.uuids[id], filter_key].values[0] for id in ids]
        top1, top3, top5, top10 = self.get_top_ks(labels, distmat, ks=[1, 3, 5, 10])

        return top1, top3, top5, top10

    def get_all_scores(self):
        """Get all pairwise HDBSCAN similarity scores (range [0, 1])."""
        chunks, ids = self._calculate_distance_matrix()
        distmat = np.concatenate(list(chunks), axis=0)
        all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=1)
        # distmat contains distances (1 - similarity), so convert back to similarity
        scores = [1-distmat[y, x] for y,x in zip(all_inds_y, all_inds_x)]
        return scores

    def get_distance_matrix(self):
        chunks, ids = self.get_distmat_chunks()
        distmat = np.stack(list(chunks), axis=0)
        return distmat

    def get_uuids(self):
        return [self.uuids[id] for id in self.ids]

    def get_distmat_chunks(self, uuids_filter=None):
        if uuids_filter is not None:
            flags = tuple(self.uuids[id] in uuids_filter for id in self.ids)
        else:
            flags = None

        return self._calculate_distance_matrix(flags)

    def get_edges(self, topk=5, target_edges=10000, target_proportion=None, uuids_filter=None):
        """Get edges based on HDBSCAN cluster membership probabilities."""
        self.print_func("Calculating HDBSCAN-based edges...")
        start_time = time.time()
        chunks, ids = self.get_distmat_chunks(uuids_filter=uuids_filter)
        self.print_func(f"Calculate chunks time: {time.time() - start_time:.6f} seconds")
        start_time = time.time()

        result = []
        start = 0
        embeds_num = len(ids)
        total_edges = (embeds_num * embeds_num - embeds_num) / 2
        if target_proportion is None:
            target_proportion = np.clip(target_edges / max(1, total_edges), 0, 1)
        else:
            target_edges = int(total_edges * target_proportion)

        self.print_func(f"Target: {target_edges}/{total_edges}")

        for distmat in chunks:
            sorted_dists = distmat.argsort(axis=1).argsort(axis=1) < topk

            all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=start+1)

            chunk_len = len(all_inds_y)
            order = np.random.permutation(chunk_len)[:int(chunk_len * target_proportion)]
            all_inds_y = all_inds_y[order]
            all_inds_x = all_inds_x[order]

            selected_dists = np.full(distmat.shape, False)
            selected_dists[all_inds_y, all_inds_x] = True

            diag_y, diag_x = kth_diag_indices(distmat, start)

            filtered = np.logical_or(sorted_dists, selected_dists)
            filtered[diag_y, diag_x] = False
            inds_y, inds_x = np.nonzero(filtered)

            result.extend([
                (*sorted([ids[ind1+start], ids[ind2]]), 1-distmat[ind1, ind2])
                for (ind1, ind2) in zip(inds_y, inds_x)
            ])

            self.print_func(f"Chunk result: {time.time() - start_time:.6f} seconds")
            start_time = time.time()
            start += filtered.shape[0]

        self.print_func(f"Calculated distances: {time.time() - start_time:.6f} seconds")
        self.print_func(f"{len(set(result))}")
        self.print_func(result[:50])
        return set(result)