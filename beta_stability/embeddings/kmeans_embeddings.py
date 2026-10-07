"""K-means based embeddings for cluster-aware graph initialization."""
import numpy as np
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger("beta_stability")


class KMeansEmbeddings:
    """Embeddings class that uses K-means clustering for scoring.

    Scores respect the invariant: >0.5 = connected, <0.5 = disconnected.
      - Same cluster: 0.5 + 0.5 * cosine_sim(xi, xj)  -> (0.5, 1.0]
      - Different cluster: 0.5 * cosine_sim(centroid_a, centroid_b) -> [0, 0.5)

    Estimates K from the GMM threshold (K = 1/pi_positive), runs spherical
    k-means, then builds the pairwise similarity matrix.
    """

    def __init__(self, base_embeddings, node2uuid, threshold, max_k=None, print_func=print):
        self.embeddings = np.array(base_embeddings.embeddings)
        self.ids = base_embeddings.ids
        self.uuids = node2uuid
        self.id_to_idx = base_embeddings.id_to_idx
        self.distance_power = base_embeddings.distance_power
        self.print_func = print_func

        n = len(self.ids)

        # Estimate K from provided threshold (from base embeddings classifier config)
        all_scores = np.array(base_embeddings.get_all_scores())
        pi_positive = np.mean(all_scores > threshold)
        if pi_positive > 0:
            k_hat = max(1, round(1.0 / pi_positive))
        else:
            k_hat = n
        k_hat = min(k_hat, n - 1)
        if max_k is not None:
            k_hat = min(k_hat, max_k)
            print_func(f"KMeansEmbeddings: pi_positive={pi_positive:.6f}, K_hat={k_hat} (capped at max_k={max_k})")
        else:
            print_func(f"KMeansEmbeddings: pi_positive={pi_positive:.6f}, K_hat={k_hat}")

        # Normalize and run spherical k-means
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = self.embeddings / norms

        self.k = k_hat
        km = KMeans(n_clusters=k_hat, random_state=42, n_init=10)
        self.labels = km.fit_predict(normalized)
        centroids = km.cluster_centers_

        # Centroid cosine similarities (for cross-cluster scores)
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        c_norms[c_norms == 0] = 1
        norm_centroids = centroids / c_norms
        centroid_sims = np.clip(norm_centroids @ norm_centroids.T, 0, 1)

        # Point cosine similarities (for within-cluster scores)
        point_sims = np.clip(normalized @ normalized.T, 0, 1)

        # Build similarity matrix:
        #   same cluster:      0.5 + 0.5 * cos_sim(xi, xj)     -> (0.5, 1.0]
        #   different cluster:  0.5 * cos_sim(centroid_a, centroid_b)  -> [0, 0.5)
        same_mask = self.labels[:, None] == self.labels[None, :]
        cross_scores = centroid_sims[self.labels][:, self.labels]

        self.similarity_matrix = np.where(
            same_mask,
            0.5 + 0.5 * point_sims,
            0.5 * cross_scores
        ).astype(np.float32)

        n = len(self.ids)
        same_scores = self.similarity_matrix[same_mask & ~np.eye(n, dtype=bool)]
        diff_scores = self.similarity_matrix[~same_mask]
        cluster_sizes = np.bincount(self.labels)
        print_func(f"KMeansEmbeddings: K={k_hat}, n={n}")
        print_func(f"KMeansEmbeddings: same-cluster scores: "
                   f"[{same_scores.min():.4f}, {same_scores.max():.4f}], mean={same_scores.mean():.4f}")
        print_func(f"KMeansEmbeddings: cross-cluster scores: "
                   f"[{diff_scores.min():.4f}, {diff_scores.max():.4f}], mean={diff_scores.mean():.4f}")
        print_func(f"KMeansEmbeddings: cluster sizes: [{cluster_sizes.min()}, {cluster_sizes.max()}]")

    def get_score(self, id1, id2):
        """Score reflecting k-means cluster membership. >0.5 = same cluster."""
        idx1 = self.id_to_idx[id1]
        idx2 = self.id_to_idx[id2]
        return float(self.similarity_matrix[idx1, idx2])

    def get_all_scores(self):
        """All pairwise k-means similarity scores."""
        n = len(self.ids)
        all_inds_y, all_inds_x = np.triu_indices(n, k=1)
        return self.similarity_matrix[all_inds_y, all_inds_x].tolist()

    def get_edges(self, topk=5, **kwargs):
        """Top-k edges per node based on k-means similarity scores."""
        self.print_func("Calculating k-means-based edges...")
        n = len(self.ids)

        edges = set()
        for i in range(n):
            sims = self.similarity_matrix[i].copy()
            sims[i] = -np.inf
            k_nn = min(topk, n - 1)
            for j in np.argsort(-sims)[:k_nn]:
                n0, n1 = sorted([self.ids[i], self.ids[j]])
                score = float(self.similarity_matrix[i, j])
                edges.add((n0, n1, score))

        self.print_func(f"KMeansEmbeddings: {len(edges)} initial edges")
        return edges

    def get_uuids(self):
        return [self.uuids[id] for id in self.ids]
