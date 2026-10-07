"""Jaccard-of-kNN embeddings: graph-topology verifier.

For each pair (i, j) computes the Jaccard similarity of their top-k
cosine-neighbour sets. On weakly-separated embeddings (megadescriptor),
this gives ~4x better positive/negative separation than raw cosine (the
data is far more bimodal in Jaccard space: most negatives have 0 shared
neighbours, most positives have at least some).

Mirrors KMeansEmbeddings's interface so it plugs into the existing
meta-verifier machinery (`jaccard(base_name)`).
"""
import numpy as np
import logging

logger = logging.getLogger("beta_stability")


class JaccardEmbeddings:
    """Pairwise score = |kNN(i) ∩ kNN(j)| / |kNN(i) ∪ kNN(j)| over the
    base embedding's cosine top-k neighbourhoods.

    Score range: [0, 1]. Zero = no shared neighbours (overwhelmingly the
    negative class on the wildlife re-ID datasets). The framework's
    auto-threshold mechanism finds the cutoff via GMM on the empirical
    distribution.
    """

    def __init__(self, base_embeddings, node2uuid, topk=10, print_func=print):
        self.embeddings = np.array(base_embeddings.embeddings)
        self.ids = base_embeddings.ids
        self.uuids = node2uuid
        self.id_to_idx = base_embeddings.id_to_idx
        self.distance_power = base_embeddings.distance_power
        self.print_func = print_func
        self.topk = topk

        n = len(self.ids)

        # L2-normalize and compute pairwise cosines on the base embedding
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = self.embeddings / norms
        cos = (normalized @ normalized.T).astype(np.float32)
        np.fill_diagonal(cos, -np.inf)

        # Per-node top-k neighbour set (excluding self via -inf diagonal)
        knn_idx = np.argpartition(-cos, topk, axis=1)[:, :topk]
        membership = np.zeros((n, n), dtype=np.int8)
        rows = np.repeat(np.arange(n), topk)
        cols = knn_idx.ravel()
        membership[rows, cols] = 1

        # Pairwise intersection: membership @ membership.T gives, for each
        # (i, j), the number of k such that k is in both kNN(i) and kNN(j).
        intersect = (membership.astype(np.int32) @ membership.T.astype(np.int32))
        sizes = membership.sum(axis=1).astype(np.int32)  # == topk for all i
        union = sizes[:, None] + sizes[None, :] - intersect
        union = np.maximum(union, 1)
        jaccard = (intersect / union).astype(np.float32)
        np.fill_diagonal(jaccard, 0.0)

        # Build similarity matrix following the KMeansEmbeddings convention
        # so the framework's default threshold of 0.5 is a meaningful cut:
        #   shared-kNN pair (jaccard > 0):  0.5 + 0.5 * jaccard  -> (0.5, 1.0]
        #   no-shared pair (jaccard == 0):  0.5 * cos            -> [0, 0.5)
        cos_clipped = np.clip(cos, 0, 1).astype(np.float32)
        np.fill_diagonal(cos_clipped, 0.0)
        has_overlap = jaccard > 0
        self.similarity_matrix = np.where(
            has_overlap,
            0.5 + 0.5 * jaccard,
            0.5 * cos_clipped,
        ).astype(np.float32)

        triu = np.triu_indices(n, k=1)
        all_scores = self.similarity_matrix[triu]
        above = all_scores[all_scores > 0.5]
        print_func(f"JaccardEmbeddings: topk={topk}, n={n}")
        print_func(f"JaccardEmbeddings: pairs with shared-kNN > 0 "
                   f"(score > 0.5): {len(above)}/{len(all_scores)} "
                   f"({100*len(above)/len(all_scores):.2f}%)")
        if len(above):
            print_func(f"JaccardEmbeddings: above-0.5 scores: "
                       f"min={above.min():.4f} mean={above.mean():.4f} "
                       f"max={above.max():.4f}")

    def get_score(self, id1, id2):
        idx1 = self.id_to_idx[id1]
        idx2 = self.id_to_idx[id2]
        return float(self.similarity_matrix[idx1, idx2])

    def get_all_scores(self):
        n = len(self.ids)
        triu = np.triu_indices(n, k=1)
        return self.similarity_matrix[triu].tolist()

    def get_edges(self, topk=5, **kwargs):
        """Top-k edges per node by Jaccard score. Skips zero-score pairs
        (no shared neighbours — no signal)."""
        self.print_func("Calculating jaccard-based edges...")
        n = len(self.ids)
        edges = set()
        for i in range(n):
            scores = self.similarity_matrix[i].copy()
            scores[i] = -np.inf
            k_nn = min(topk, n - 1)
            for j in np.argsort(-scores)[:k_nn]:
                sc = float(self.similarity_matrix[i, j])
                if sc <= 0:
                    continue
                n0, n1 = sorted([self.ids[i], self.ids[j]])
                edges.add((n0, n1, sc))
        self.print_func(f"JaccardEmbeddings: {len(edges)} initial edges")
        return edges

    def get_uuids(self):
        return [self.uuids[id] for id in self.ids]
