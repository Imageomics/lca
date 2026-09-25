# -*- coding: utf-8 -*-

import logging
import math
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


logger = logging.getLogger("beta_stability")


def cids_from_range(n, prefix='c'):
    k = math.ceil(math.log10(n))
    cids = [prefix + str(i).zfill(k) for i in range(n)]
    return cids


def build_clustering(node2cluster):
    """
    node2cluster: mapping from node id to cluster id
    clustering: mapping from cluster id to set of node ids
    """
    clustering = {}
    for n, c in node2cluster.items():
        if c not in clustering:
            clustering[c] = {n}
        else:
            clustering[c].add(n)
    return clustering


def build_clustering_from_clusters(cids, clusters):
    """
    From an iterator through the clusters, each of which can be a list
    or a set of nodes, build a clustering.  This maps from cluster ids
    to sets of nodes. Each node must appear in at most one list and
    only once in that list. An assert failure occurs if either of
    these conditions is violated.
    """
    num_nodes = sum([len(c) for c in clusters])
    as_sets = [set(c) for c in clusters]
    num_unique = len(set.union(*as_sets))
    assert num_nodes == num_unique
    clustering = {id: c for id, c in zip(cids, as_sets)}
    return clustering


def build_node_to_cluster_mapping(clustering):
    # print('cluster:', clustering)
    """
    clustering: mapping from cluster id to set of node ids
    node2cluster: mapping from node id to cluster id
    """
    node2cluster = {n: c for c in clustering for n in clustering[c]}
    return node2cluster


def sign_from_clusters(n1, n2, node2cid):
    """
    Returns 1 if the two nodes are in the same cluster and -1 otherwise.
    """
    if node2cid[n1] == node2cid[n2]:
        return 1
    else:
        return -1


def cid_list_score(G, clustering, node2cid, cid_list):
    """
    Return the objective function score of the clusters indicated in
    the cid_list
    """
    score = 0
    for cid0 in cid_list:
        for n0 in clustering[cid0]:
            for n1 in G[n0]:
                if n0 >= n1:  # only count when n0 < n1
                    continue
                cid_n1 = node2cid[n1]
                if cid0 == cid_n1:
                    score += G[n0][n1]['weight']
                elif cid_n1 in cid_list:
                    score -= G[n0][n1]['weight']
    return score


def clustering_score(G, node2cid):
    """
    Return the objective function score for the entire graph.
    """
    score = sum(
        [
            sign_from_clusters(n1, n2, node2cid) * G[n1][n2]['weight']
            for n1 in G
            for n2 in G[n1]
            if n1 < n2
        ]
    )
    return score


def get_weight_lists(G, sort_positive=True):
    """
    Return a pair of lists of the negative and positive edges
    in graph G.  Each list contains triples of (ni, nj, wgt).
    The negative edge list is unordered.
    The positive edge list is in decreasing order if requested.
    """
    weight_list = [(n1, n2, G[n1][n2]['weight']) for n1 in G for n2 in G[n1] if n1 < n2]
    negatives = [t for t in weight_list if t[2] < 0]
    positives = [t for t in weight_list if t[2] >= 0]
    if sort_positive:
        positives.sort(key=lambda x: x[2], reverse=True)
    return negatives, positives


def weights_between(c0, c1, G):
    """
    Return an unordered list of the weights between two clusters
    """
    return [G[n0][n1]['weight'] for n0 in c0 for n1 in G[n0] if n1 in c1]


def has_edges_between_them(G, c0, c1):
    """
    Return true iff there is an edge joining two clusters
    """
    if len(c0) > len(c1):
        c0, c1 = c1, c0
    for n in c0:
        for m in G[n]:
            if m in c1:
                return True
    return False


def replace_clusters(old_cids, add_clustering, clustering, node2cid):
    """
    Remove the old_cids clusters from the clustering dictionary,
    add the new ones, and change the mapping of the nodes in the
    new clusters. It is assumed without checking that the nodes
    in the old clusters and the nodes in the new clusters are
    the same.
    """
    for cid in old_cids:
        del clustering[cid]
    clustering.update(add_clustering)
    for cid, nodes in add_clustering.items():
        for n in nodes:
            node2cid[n] = cid


def merge_clusters(cid0, cid1, G, clustering, node2cid):
    """
    Merge the smaller cluster into the larger.  This includes
    removing the cluster, reassigning the node-to-cluster-id
    mapping, and returning the change in score.
    """
    cluster0 = clustering[cid0]
    cluster1 = clustering[cid1]
    if len(cluster0) < len(cluster1):
        c_small, c_large = cluster0, cluster1
        cid_small, cid_large = cid0, cid1
    else:
        c_small, c_large = cluster1, cluster0
        cid_small, cid_large = cid1, cid0

    adjoining_wgts = weights_between(c_small, c_large, G)
    delta_score = 2 * sum(adjoining_wgts)
    c_large |= c_small
    for n in c_small:
        node2cid[n] = cid_large
    del clustering[cid_small]
    return delta_score


def score_delta_after_merge(cid0, cid1, G, clustering):
    """
    Return the change in score that would occur IF the two clusters
    were merge.  This is done without performing the actual merge.
    """
    cluster0 = clustering[cid0]
    cluster1 = clustering[cid1]
    if len(cluster0) < len(cluster1):
        c_small, c_large = cluster0, cluster1
    else:
        c_small, c_large = cluster1, cluster0

    adjoining_wgts = weights_between(c_small, c_large, G)
    delta_score = 2 * sum(adjoining_wgts)
    return delta_score


def shift_between_clusters(cid0, nodes_to_move, cid1, clustering, node2cid):
    """
    Move nodes from cluster 0 to cluster 1, changing the clustering
    and the node-to-cluster mapping.  This move must be such that at
    least one node is left in cluster 0.
    """
    cluster0, cluster1 = clustering[cid0], clustering[cid1]
    assert len(nodes_to_move) < len(cluster0)

    for n in nodes_to_move:
        node2cid[n] = cid1
    cluster0 -= nodes_to_move
    cluster1 |= nodes_to_move


def form_connected_cluster_pairs(G, clustering, node2cid, new_cids=None):
    """
    Return a sorted list of all pairs of cluster ids such that at least one
    cluster is in the new_cids list (defaults to the entire list of
    clusters) and there is an edge connecting the clusters. Each id pair
    is ordered smaller to larger, and the entire list is ordered.
    """
    if new_cids is None:
        cids = sorted(list(clustering.keys()))
    else:
        cids = sorted(list(new_cids))

    cid_pairs = set()
    for cid_n in cids:
        for n in clustering[cid_n]:
            for m in G[n]:
                cid_m = node2cid[m]
                if cid_n == cid_m:
                    continue
                elif cid_n < cid_m:
                    cid_pairs.add((cid_n, cid_m))
                else:
                    cid_pairs.add((cid_m, cid_n))

    return sorted(cid_pairs)


def print_structures(G, clustering, node2cid, score):
    """
    Output the graph, the clusters, and the node-to-cluster mapping.
    """
    logger.info('Graph:')
    for n in sorted(G.nodes()):
        logger.info('    %a: ' % n, end='')
        for m in sorted(G[n]):
            logger.info(' %a/%a' % (m, G[n][m]['weight']), end='')
        logger.info()
    logger.info('Clusters:')
    for c in sorted(clustering):
        logger.info('    %d:' % c, end='')
        for n in sorted(clustering[c]):
            logger.info(' %a' % n, end='')
        logger.info()
    logger.info('Node to cluster:')
    for n in sorted(G.nodes()):
        logger.info('    %a: %d ' % (n, node2cid[n]))
    logger.info(
        'input score %a, actual score %a' % (score, clustering_score(G, node2cid))
    )


def same_clustering(clustering0, clustering1, output_differences=False):
    """
    Return True i.f.f. the two clusterings have the same sets of clusters (even if
    the cluster ids are different).  If the output_difference flag is set then
    print the clusters that appear only in one clustering.
    """
    same = True
    if len(clustering0) != len(clustering1):
        same = False
        if output_differences:
            logger.info(
                'clustering lengths are different %d vs. %d'
                % (len(clustering0), len(clustering1))
            )

    for c in clustering0.values():
        if c not in clustering1.values():
            same = False
            if output_differences:
                logger.info('cluster %s not in 2nd clustering' % (sorted(c),))

    for c in clustering1.values():
        if c not in clustering0.values():
            same = False
            if output_differences:
                logger.info('cluster %s not in 1st clustering' % (sorted(c),))

    return same


def extract_subclustering(nodes, clustering, n2c=None):
    if n2c is None:
        n2c = build_node_to_cluster_mapping(clustering)
    new_clustering = dict()
    for n in nodes:
        c = n2c[n]
        if c in new_clustering:
            new_clustering[c].add(n)
        else:
            new_clustering[c] = set([n])
    return new_clustering


def intersection_over_union(setA, setB):
    if setA == setB:
        return 1
    else:
        return len(setA & setB) / len(setA | setB)


def compare_by_lengths(est, est_n2c, gt):
    """
    Examine each ground truth cluster to see how it is represented
    in the estimated clustering.  To do this, find all estimated
    clustering containing the ground truth cluster nodes, and then
    find which of these has the max overlap (in terms of IOU) with the
    set of ground truth nodes.
    """
    if est_n2c is None:
        est_n2c = build_node_to_cluster_mapping(est)

    gt_iou_dict = dict()
    for gt_cid, gt_nodes in gt.items():
        est_cids = {est_n2c[n] for n in gt_nodes}
        best_iou = 0
        for cid in est_cids:
            iou = intersection_over_union(gt_nodes, est[cid])
            best_iou = max(iou, best_iou)
        gt_lng = len(gt_nodes)
        if gt_lng in gt_iou_dict:
            gt_iou_dict[gt_lng].append(best_iou)
        else:
            gt_iou_dict[gt_lng] = [best_iou]

    logger.info(
        'cluster length, number of clusters, number of exact matches, '
        + 'fraction exact, average IOU'
    )
    for gt_lng in sorted(gt_iou_dict.keys()):
        val = gt_iou_dict[gt_lng]
        num_exact = val.count(1)  # how many have exact overlap
        frac_exact = num_exact / len(val)
        avg_iou = sum(val) / len(val)
        logger.info(
            '%d, %d, %d, %1.2f, %1.3f'
            % (gt_lng, len(val), num_exact, frac_exact, avg_iou)
        )


def count_equal_clustering(from_clustering, to_clustering, to_n2c):
    """
    Return the number of clusters that appear exactly in both clusterings.
    """
    n = 0
    for from_c in from_clustering.values():
        node = from_c.pop()  # ugly, but need to get a node from the
        from_c.add(node)  # cluster set and put it back
        if (node not in to_n2c):
            # print(f"NO {node}")
            continue
        
        to_c = to_clustering[to_n2c[node]]

        if from_c == to_c:
            n += 1

    return n

def get_nonequal_clustering(from_clustering, to_clustering, to_n2c):
    """
    Return the clusters which are different from gt
    """
    result = []
    for from_c in from_clustering.values():
        node = from_c.pop()  
        from_c.add(node) 
        if (node not in to_n2c):
            # print(f"NO {node}")
            continue
        to_c = to_clustering[to_n2c[node]]

        if from_c != to_c:
            result.append({'Beta Stability clustering': from_c, 'GT clustering': to_c})

    return result


def precision_recall_per_size(est, est_n2c, gt, gt_n2c):
    """
    Calculate precision and recall for each cluster size individually.

    :param est: Estimated clusters (dictionary of cluster ID to set of node IDs)
    :param est_n2c: Estimated node-to-cluster mapping (dictionary of node ID to cluster ID)
    :param gt: Ground truth clusters (dictionary of cluster ID to set of node IDs)
    :param gt_n2c: Ground truth node-to-cluster mapping (dictionary of node ID to cluster ID)
    :return: Dictionary with cluster sizes as keys and tuples of (precision, recall) as values
    """
    
    tp_counts = defaultdict(int)
    fp_counts = defaultdict(int)
    fn_counts = defaultdict(int)
    
    # Calculate TP and FP for each estimated cluster size
    for est_c in est.values():
        cluster_size = len(est_c)
        est_c_list = list(est_c)
        for i, ni in enumerate(est_c_list):
            for j in range(i + 1, len(est_c_list)):
                nj = est_c_list[j]
                if gt_n2c[ni] == gt_n2c[nj]:
                    tp_counts[cluster_size] += 1
                else:
                    fp_counts[cluster_size] += 1
    
    # Calculate FN for each ground truth cluster size
    for gt_c in gt.values():
        cluster_size = len(gt_c)
        gt_c_list = list(gt_c)
        for i, ni in enumerate(gt_c_list):
            # if ni not in est_n2c:
            #     continue
            for j in range(i + 1, len(gt_c_list)):
                nj = gt_c_list[j]
                if (ni not in est_n2c) or (nj not in est_n2c) or (est_n2c[ni] != est_n2c[nj]):
                    fn_counts[cluster_size] += 1

    # Calculate precision and recall for each cluster size
    precision_recall_per_size = {}
    all_sizes = set(tp_counts.keys()).union(fp_counts.keys()).union(fn_counts.keys())

    for size in all_sizes:
        tp = tp_counts[size]
        fp = fp_counts[size]
        fn = fn_counts[size]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_recall_per_size[size] = (precision, recall)

    return precision_recall_per_size


def precision_recall(est, est_n2c, gt, gt_n2c):
    """Pairwise precision/recall/F1 of the estimated clustering against GT.

    A pair of nodes sharing an estimated cluster is a TP if they also share a GT
    cluster, else a FP; a pair sharing a GT cluster but not an estimated one is a FN.

    Counted from the cluster-overlap table rather than by enumerating pairs. Let
    n_ij = |est_i n gt_j|; then TP = sum C(n_ij, 2), FP = sum C(|est_i|, 2) - TP and
    FN = sum C(|gt_j|, 2) - TP -- identical numbers, but O(N) instead of
    O(sum |c|^2). The pair enumeration cost ~14.7M comparisons for a single
    5429-node cluster and ran on every validation point.
    """
    pairs = defaultdict(int)
    for cid, est_c in est.items():
        for nd in est_c:
            g = gt_n2c[nd]          # KeyError here matches the old behaviour
            pairs[(cid, g)] += 1
    tp = 0
    for n_ij in pairs.values():
        tp += n_ij * (n_ij - 1) // 2

    est_pairs = 0
    for est_c in est.values():
        m = len(est_c)
        est_pairs += m * (m - 1) // 2
    gt_pairs = 0
    for gt_c in gt.values():
        m = len(gt_c)
        gt_pairs += m * (m - 1) // 2

    fp = est_pairs - tp
    # A GT pair is a FN unless BOTH nodes are in est_n2c and share an estimated
    # cluster -- and that co-clustered count is exactly tp.
    fn = gt_pairs - tp

    if tp + fp > 0:
        precision = tp / max((tp + fp), 1)
        recall = tp / max((tp + fn), 1)

        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
    else:
        precision = 0
        recall = 0
        f1_score = 0
    return (precision, recall, f1_score)


def percent_and_PR(est, est_n2c, gt, gt_n2c, want_detail=False):
    """`want_detail=False` (default) skips the two per-cluster breakdowns.

    `get_nonequal_clustering` and `precision_recall_per_size` are discarded by BOTH
    callers (commented out of the validator's result dict, absent from
    run_clustering_with_save's). precision_recall_per_size in particular repeats the
    same O(sum |c|^2) pair enumeration as precision_recall, so computing it doubled
    the cost of every validation point for a result nobody reads. Pass
    want_detail=True to get them back.
    """
    num_eq = count_equal_clustering(est, gt, gt_n2c)
    non_eq = get_nonequal_clustering(est, gt, gt_n2c) if want_detail else []
    pr, rec, f1 = precision_recall(est, est_n2c, gt, gt_n2c)
    per_size = precision_recall_per_size(est, est_n2c, gt, gt_n2c) if want_detail else {}
    lng = len(est)
    lng = 1 if lng == 0 else lng
    return (num_eq / lng, pr, rec, per_size, non_eq, f1)


def _overlap_table(gt_ids, gt_sets, est_ids, est_sets):
    """Cluster-overlap counts |A n B| for every OVERLAPPING (gt, est) pair, from a
    single pass over the nodes.

    The dense `for i in gt: for j in est: len(A & B)` build is O(C_gt * C_est) set
    intersections -- ~1.5-3M of them per call on whale shark 2023 (982 GT x ~1500-3000
    estimated clusters), and it was run twice per validation (Hungarian + CEAF), 54
    times per run. Every intersection is instead accumulated by walking each estimated
    cluster once: O(N) total, and the result is identical because a node contributes to
    exactly one (gt, est) pair.

    Returns (rows, cols, counts) index arrays into gt_ids / est_ids, plus the
    per-cluster sizes as float arrays.
    """
    gt_pos = {c: i for i, c in enumerate(gt_ids)}
    node_gt = {}
    for c in gt_ids:
        i = gt_pos[c]
        for nd in gt_sets[c]:
            node_gt[nd] = i
    counts = defaultdict(int)
    for j, c in enumerate(est_ids):
        for nd in est_sets[c]:
            i = node_gt.get(nd)
            if i is not None:
                counts[(i, j)] += 1
    if counts:
        keys = np.fromiter((k[0] for k in counts), dtype=np.int64, count=len(counts))
        cols = np.fromiter((k[1] for k in counts), dtype=np.int64, count=len(counts))
        vals = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    else:
        keys = np.zeros(0, dtype=np.int64); cols = np.zeros(0, dtype=np.int64)
        vals = np.zeros(0, dtype=np.float64)
    gsz = np.array([len(gt_sets[c]) for c in gt_ids], dtype=np.float64)
    esz = np.array([len(est_sets[c]) for c in est_ids], dtype=np.float64)
    return keys, cols, vals, gsz, esz


def hungarian_cluster_matching(est, gt):
    """
    Match estimated clusters to ground truth clusters using the Hungarian algorithm
    based on Jaccard similarity. This provides a "softer" evaluation metric that
    doesn't require exact cluster matches.

    Process:
    1. Compute Jaccard similarity between each GT cluster and each estimated cluster
    2. Run Hungarian algorithm to find optimal one-to-one assignment (maximizing Jaccard)
    3. Exclude assignments where Jaccard = 0 (no overlap)
    4. Compute TP, FP, FN from the matched node sets

    Args:
        est: Estimated clustering (dict: cluster_id -> set of nodes)
        gt: Ground truth clustering (dict: cluster_id -> set of nodes)

    Returns:
        dict with keys: 'tp', 'fp', 'fn', 'precision', 'recall', 'f1',
                       'num_matched_gt', 'num_matched_est', 'assignments'
    """
    # Convert to sets if needed (some callers pass lists)
    gt_sets = {k: set(v) if not isinstance(v, set) else v for k, v in gt.items()}
    est_sets = {k: set(v) if not isinstance(v, set) else v for k, v in est.items()}

    gt_ids = list(gt_sets.keys())
    est_ids = list(est_sets.keys())

    n_gt = len(gt_ids)
    n_est = len(est_ids)

    if n_gt == 0 or n_est == 0:
        # Edge case: empty clustering
        total_gt_nodes = sum(len(c) for c in gt.values())
        total_est_nodes = sum(len(c) for c in est.values())
        return {
            'tp': 0, 'fp': total_est_nodes, 'fn': total_gt_nodes,
            'precision': 0, 'recall': 0, 'f1': 0,
            'num_matched_gt': 0, 'num_matched_est': 0, 'assignments': []
        }

    # Jaccard similarity matrix, filled from the O(N) overlap table instead of
    # C_gt x C_est set intersections. Non-overlapping pairs are 0 by construction,
    # which is exactly what intersection_over_union returns for them.
    jaccard_matrix = np.zeros((n_gt, n_est))
    _r, _c, _v, _gsz, _esz = _overlap_table(gt_ids, gt_sets, est_ids, est_sets)
    if _r.size:
        jaccard_matrix[_r, _c] = _v / (_gsz[_r] + _esz[_c] - _v)

    # Use negative Jaccard as cost for minimization
    cost_matrix = -jaccard_matrix

    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter out assignments with 0 Jaccard (no overlap)
    valid_assignments = []
    for gt_idx, est_idx in zip(row_ind, col_ind):
        if jaccard_matrix[gt_idx, est_idx] > 0:
            valid_assignments.append((gt_ids[gt_idx], est_ids[est_idx]))

    # Calculate cluster-level TP, FP, FN from assignments
    # TP = number of successfully matched cluster pairs
    # FP = number of EST clusters that remained unmatched
    # FN = number of GT clusters that remained unmatched

    tp = len(valid_assignments)  # Number of matched cluster pairs
    fn = n_gt - tp  # Unmatched GT clusters
    fp = n_est - tp  # Unmatched EST clusters

    # Compute precision, recall, F1 at cluster level
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_gt_clusters': n_gt,
        'num_est_clusters': n_est,
        'assignments': valid_assignments
    }


def ceaf_cluster_matching(est, gt, sim='dice'):
    """CEAF (Constrained Entity-Alignment F-measure, Luo 2005).

    Same optimal 1-1 cluster alignment as hungarian_cluster_matching, but the
    matched pairs are weighted by a set-similarity instead of thresholded at
    Jaccard>0. This makes it purity- and retrieval-aware: an impure match
    (contamination) or a fragmented/incomplete match scores below 1.

        sim='dice'    -> phi4 = 2|A∩B|/(|A|+|B|)   (canonical CEAF-phi4)
        sim='jaccard' -> |A∩B| / |A∪B|

    Phi = sum of similarity over the optimal alignment. Precision = Phi/#est
    (penalizes spurious/impure predicted clusters), Recall = Phi/#gt (penalizes
    missed/fragmented individuals), F1 = harmonic mean. Drop-in replacement for
    the Hungarian precision/recall/f1 triple.
    """
    gt_sets = {k: set(v) if not isinstance(v, set) else v for k, v in gt.items()}
    est_sets = {k: set(v) if not isinstance(v, set) else v for k, v in est.items()}
    gt_ids = list(gt_sets)
    est_ids = list(est_sets)
    n_gt, n_est = len(gt_ids), len(est_ids)
    if n_gt == 0 or n_est == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'phi': 0.0,
                'num_gt_clusters': n_gt, 'num_est_clusters': n_est, 'sim': sim}

    # Same optimal alignment, but the similarity matrix is filled from the O(N)
    # overlap table rather than C_gt x C_est set intersections.
    S = np.zeros((n_gt, n_est))
    _r, _c, _v, _gsz, _esz = _overlap_table(gt_ids, gt_sets, est_ids, est_sets)
    if _r.size:
        if sim == 'jaccard':
            S[_r, _c] = _v / (_gsz[_r] + _esz[_c] - _v)
        else:  # dice / phi4
            S[_r, _c] = 2 * _v / (_gsz[_r] + _esz[_c])

    row, col = linear_sum_assignment(-S)
    phi = float(S[row, col].sum())
    precision = phi / n_est
    recall = phi / n_gt
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'phi': phi,
            'num_gt_clusters': n_gt, 'num_est_clusters': n_est, 'sim': sim}
