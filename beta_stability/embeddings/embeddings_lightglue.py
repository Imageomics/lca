import numpy as np
from numpy.linalg import norm 
import time
import scipy
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
from beta_stability.util.tools import load_pickle


class LightglueEmbeddings(object):
    def __init__(self,
                 ids,
                 pickle_path="lightglue_scores_superpoint.pickle"):
        self.uuids_dict = ids
        results = load_pickle("lightglue_scores_superpoint.pickle")
        self.scores = results["scores"]
        self.uuids = results["uuids"]
        inverse_dict = {v: k for k, v in self.uuids_dict.items()}
        self.ids = [inverse_dict[uuid] for uuid in self.uuids]


    def get_score(self, id1, id2):
        idx1 = self.ids.index(id1)
        idx2 = self.ids.index(id2)
        return self.scores[idx1][idx2]
    
    def get_topk_acc(self, labels_q, labels_db, dists, topk):
        return sum(self.get_topk_hits(labels_q, labels_db, dists, topk)) / len(labels_q)


    def get_topk_hits(self,labels_q, labels_db, dists, topk):
        indices = np.argsort(dists, axis=1)
        top_labels = np.array(labels_db)[indices[:, :topk]]
        hits = (top_labels.T == labels_q).T
        return np.sum(hits[:, :topk+1], axis=1) > 0

    def get_top_ks(self,q_pids, distmat, ks=[1,3,5,10]):
        return [(k, self.get_topk_acc(q_pids, q_pids, distmat, k)) for k in ks]


    def get_stats(self, df, filter_key, id_key='uuid'):
        
        print("Calculating distances...")
        
        distmat = self.scores
        labels = [df.loc[df[id_key] == self.uuids_dict[id], filter_key].values[0] for id in self.ids]
        
        top1, top3, top5, top10 = self.get_top_ks(labels, distmat, ks=[1,3,5,10])


        return top1, top3, top5, top10


    def get_top20_matches(self, df, filter_key, id_key='uuid'):
        
        distmat = self.scores
        labels = [df.loc[df[id_key] == self.uuids_dict[id], filter_key].values[0] for id in self.ids]

        top20_results = {}

        for i, row in enumerate(distmat):
            top20_indices = np.argsort(row)[:20]  # Get the top 20 closest indices
            top20_scores = row[top20_indices]  # Get their corresponding scores

            # Fetch the actual UUIDs corresponding to the indices
            top20_uuids =[self.ids[idx] for idx in top20_indices]

            # Build the dictionary for each UUID with top 20 matches and their scores
            top20_results[self.ids[i]] = [(uuid, 1-score) for uuid, score in zip(top20_uuids, top20_scores)]
        
        return top20_results


    
    def get_edges(self, topk=5, target_edges=10000, uuids_filter=None):
        
        if uuids_filter is not None:
            inds = np.array([self.uuids[id] in uuids_filter for id in self.ids])
            scores = np.array(self.scores)[inds, inds]
            ids = [id for id in self.uuids.keys() if self.uuids[id] in uuids_filter]
        else:
            scores = np.array(self.scores)
            ids = self.ids
        distmat = 1 - scores
        np.fill_diagonal(distmat, np.inf)


        print("Calculating distances...")
        
        result = []
        start = 0
        embeds_num = scores.shape[0]
        total_edges = (embeds_num * embeds_num - embeds_num)/2
        target_proportion = np.clip(target_edges/total_edges, 0, 1)
        
        sorted_dists = distmat.argsort(axis=1).argsort(axis=1) < topk
        
        all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=start+1)
        chunk_len = len(all_inds_y)
        order = np.random.permutation(chunk_len)[:int(chunk_len*target_proportion)]
        all_inds_y = all_inds_y[order]
        all_inds_x = all_inds_x[order]
        selected_dists = np.full(distmat.shape, False)
        selected_dists[all_inds_y, all_inds_x] = True

        filtered = np.logical_or(sorted_dists, selected_dists)
        inds_y, inds_x = np.nonzero(filtered)
        result.extend([(*sorted([ids[ind1+start], ids[ind2]]), 1-distmat[ind1, ind2]) for (ind1, ind2) in zip(inds_y, inds_x)])
        start += filtered.shape[0]
        # print(f"Max dist: {np.max(distmat[np.where(distmat != np.inf)])}")
        print(f"Calculated distances")
        print(f"{len(set(result))}")
        return set(result)


    def get_baseline_edges(self, topk=10, distance_threshold=0.5):
        def reduce_func(distmat, start):
            distmat = 1 - self.get_score_from_cosine_distance(distmat)
            rng = np.arange(distmat.shape[0])
            distmat[rng, rng+start] = np.inf
            return distmat

        start_time = time.time()
        print("Calculating distances...")
        print(f"{len(self.embeddings)}/{len(self.ids)}")
        
        chunks = pairwise_distances_chunked(
            self.embeddings,
            metric='cosine',
            reduce_func=reduce_func,
            n_jobs=-1
        )
        
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
            result.extend([(*sorted([self.ids[ind1 + start], self.ids[ind2]]), 1 - distmat[ind1, ind2]) for (ind1, ind2) in zip(inds_y, inds_x)])
            
            print(f"Chunk processed in: {time.time() - start_time:.6f} seconds")
            start_time = time.time()
            start += sorted_dists_mask.shape[0]
        
        print(f"Calculated distances in: {time.time() - start_time:.6f} seconds")
        
        return set(result)



