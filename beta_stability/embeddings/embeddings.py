import numpy as np
from numpy.linalg import norm 
import time
import scipy
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import functools
from beta_stability.util.tools import kth_diag_indices

class Embeddings(object):
    def __init__(self, embeddings, ids, distance_power=1, print_func=print):
        self.embeddings = np.array(embeddings)
        self.uuids = ids
        self.ids = list(ids.keys())
        self.id_to_idx = {id: idx for idx, id in enumerate(self.ids)}  # O(1) lookup
        self.distance_power = distance_power
        self.print_func = print_func

    def _reduce_func(self, distmat, start):
        """Common distance matrix reduction function."""
        distmat = 1 - self.get_score_from_cosine_distance(distmat)
        rng = np.arange(distmat.shape[0])
        distmat[rng, rng + start] = np.inf
        return distmat
    
    @functools.cache
    def _calculate_distance_matrix(self, flags=None):
        """Calculate distance matrix using chunked computation."""
        if flags is None:
            embeddings = self.embeddings
            ids = self.ids
        else:
            embeddings = [self.embeddings[i] for (i, f) in enumerate(flags) if f]
            ids = [self.ids[i] for (i, f) in enumerate(flags) if f]
        chunks = pairwise_distances_chunked(
            embeddings,
            metric='cosine',
            reduce_func=self._reduce_func,
            n_jobs=-1
        )
        return list(chunks), ids

    @functools.lru_cache(maxsize=1000000)
    def get_score(self, id1, id2):
        embedding1 = self.embeddings[self.id_to_idx[id1]]
        embedding2 = self.embeddings[self.id_to_idx[id2]]
        return self.get_embeddings_score(embedding1, embedding2)
    
    def get_embeddings_score(self, embedding1, embedding2):
        return self.get_score_from_cosine_distance(cosine(embedding1, embedding2))
    
    def sign_power(self, x, power):
        return np.sign(x) * np.power(np.abs(x), power)
    
    def get_score_from_cosine_distance(self, cosine_dist):
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
        chunks, ids = self._calculate_distance_matrix()
        distmat = np.concatenate(list(chunks), axis=0)
        all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=1)
        scores = [1-distmat[y, x] for y,x in zip(all_inds_y, all_inds_x)]
        return scores

    def get_top20_matches(self, df, filter_key):
        start_time = time.time()
        self.print_func("Calculating distances...")
        self.print_func(f"{len(self.embeddings)}/{len(self.ids)}")
        
        chunks, ids = self._calculate_distance_matrix()
        distmat = np.concatenate(list(chunks), axis=0)
        
        top20_results = {}
        for i, row in enumerate(distmat):
            top20_indices = np.argsort(row)[:20]
            top20_scores = row[top20_indices]
            top20_uuids = [ids[idx] for idx in top20_indices]
            top20_results[ids[i]] = [(uuid, 1-score) for uuid, score in zip(top20_uuids, top20_scores)]
        
        self.print_func(f"Finished calculating distances in {time.time() - start_time:.2f} seconds.")
        return top20_results

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

    def get_edges(self, topk=5, botk=0, target_edges=10000, target_proportion=None, uuids_filter=None,
                  lower_threshold=None, upper_threshold=None):
        """
        Get candidate edges based on various selection criteria.

        Args:
            topk: Number of top neighbors to include for each node (most similar)
            botk: Number of bottom neighbors to include for each node (least similar)
            target_edges: Target number of random edges to include
            target_proportion: Proportion of all possible edges to include randomly
            uuids_filter: Optional filter for specific UUIDs
            lower_threshold: Include all edges with score < lower_threshold (certain negatives)
            upper_threshold: Include all edges with score > upper_threshold (certain positives)

        Returns:
            set: Set of (node1, node2, score) tuples
        """
        self.print_func("Calculating distances...")
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
        if lower_threshold is not None or upper_threshold is not None:
            self.print_func(f"Thresholds: lower={lower_threshold}, upper={upper_threshold}")

        for distmat in chunks:
            # Get topk (most similar - lowest distances) and botk (least similar - highest distances)
            sorted_ranks = distmat.argsort(axis=1).argsort(axis=1)
            topk_mask = sorted_ranks < topk
            botk_mask = sorted_ranks >= (distmat.shape[1] - botk) if botk > 0 else np.zeros_like(sorted_ranks, dtype=bool)
            sorted_dists = np.logical_or(topk_mask, botk_mask)

            all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=start+1)

            valid_dists = np.full(distmat.shape, True)
            # valid_dists[all_inds_y, all_inds_x] = True

            chunk_len = len(all_inds_y)
            order = np.random.permutation(chunk_len)[:int(chunk_len * target_proportion)]
            all_inds_y = all_inds_y[order]
            all_inds_x = all_inds_x[order]

            selected_dists = np.full(distmat.shape, False)
            selected_dists[all_inds_y, all_inds_x] = True

            diag_y, diag_x = kth_diag_indices(distmat, start)

            filtered = np.logical_or(np.logical_and(sorted_dists, valid_dists), selected_dists)

            # Add threshold-based selection (certain positives and certain negatives)
            # Note: distmat contains distances (1 - score), so:
            # - score > upper_threshold means distmat < (1 - upper_threshold)
            # - score < lower_threshold means distmat > (1 - lower_threshold)
            if upper_threshold is not None:
                # Certain positives: high scores (low distances)
                certain_positives = distmat < (1 - upper_threshold)
                filtered = np.logical_or(filtered, certain_positives)

            if lower_threshold is not None:
                # Certain negatives: low scores (high distances)
                certain_negatives = distmat > (1 - lower_threshold)
                filtered = np.logical_or(filtered, certain_negatives)

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
        return set(result)

    def get_baseline_edges(self, topk=10, distance_threshold=0.5):
        start_time = time.time()
        self.print_func("Calculating distances...")
        self.print_func(f"{len(self.embeddings)}/{len(self.ids)}")
        
        chunks, ids = self.get_distmat_chunks()
        result = []
        start = 0
        
        for distmat in chunks:
            sorted_indices = np.argsort(distmat, axis=1)
            
            sorted_dists_mask = np.zeros_like(distmat, dtype=bool)
            for i in range(distmat.shape[0]):
                topk_indices = sorted_indices[i, :topk]
                valid_topk_indices = topk_indices[distmat[i, topk_indices] <= distance_threshold]
                sorted_dists_mask[i, valid_topk_indices] = True
            
            inds_y, inds_x = np.nonzero(sorted_dists_mask)
            result.extend([
                (*sorted([ids[ind1 + start], ids[ind2]]), 1 - distmat[ind1, ind2]) 
                for (ind1, ind2) in zip(inds_y, inds_x)
            ])
            
            self.print_func(f"Chunk processed in: {time.time() - start_time:.6f} seconds")
            start_time = time.time()
            start += sorted_dists_mask.shape[0]
        
        self.print_func(f"Calculated distances in: {time.time() - start_time:.6f} seconds")
        return set(result)