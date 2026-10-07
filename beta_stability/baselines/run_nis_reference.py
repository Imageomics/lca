"""
Standalone reference NIS (Nested Importance Sampling) estimator.

Runs the EXACT algorithm from:
  "Human-in-the-Loop Visual Re-ID for Population Size Estimation"
  (Perez et al., ECCV 2024, arXiv:2312.05287)
  https://github.com/cvl-umass/counting-clusters

This loads our data (pickle embeddings + JSON annotations), applies the
same filtering as the LCA pipeline, then calls nested_is() directly
with ground-truth oracle (no batching, no LCA framework overhead).

Usage:
  python3 run_nis_reference.py --dataset beluga
  python3 run_nis_reference.py --dataset GZCD
  python3 run_nis_reference.py --dataset beluga --runs 20 --N_v 50 --N_n 100

  # Sweep mode: run NIS at multiple budget levels (0 to 5000)
  python3 -u run_nis_reference.py --dataset beluga --sweep
  python3 -u run_nis_reference.py --dataset beluga --sweep --budget_max 5000 --budget_steps 10
"""

import argparse
import yaml
from pathlib import Path
import json
import os
import pickle
import sys
import time
import numpy as np
import numpy.core as _nc
# Compat shim: pickles saved with numpy 2.x reference numpy._core
sys.modules.setdefault('numpy._core', _nc)
sys.modules.setdefault('numpy._core.multiarray', _nc.multiarray)
import pandas as pd
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, normalize

from beta_stability.util.tools import resolve_checkpoints


# ---------------------------------------------------------------
# Logging helper: prints to stdout AND writes to log file
# ---------------------------------------------------------------

_log_file = None

def log(msg=""):
    print(msg, flush=True)
    if _log_file is not None:
        _log_file.write(msg + "\n")
        _log_file.flush()


def write_metrics_row(metrics_file, row):
    if not metrics_file:
        return
    metrics_dir = os.path.dirname(metrics_file)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------
# Reference NIS implementation (verbatim from Perez et al.)
# with GT query counting added around lookups
# ---------------------------------------------------------------

def _safe_normalize(weights, name='distribution'):
    """Normalize a weight vector to a valid probability distribution.

    Replaces non-finite/negative entries and falls back to uniform if the
    sanitized weights are all zero. Returns (probs, was_sanitized).
    """
    w = np.asarray(weights, dtype=float).copy()
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        if len(w) == 0:
            return w, True
        return np.full(len(w), 1.0 / len(w)), True
    probs = w / total
    # Float drift: renormalize so np.random.choice doesn't reject p
    s = probs.sum()
    if s != 1.0:
        probs = probs / s
    return probs, False


def nested_is(gt_s_ij, s_ij, N_v, N_n, n_hat=None, ci=False):
    """
    Estimate number of categories using nested importance sampling.
    Based on reference: github.com/cvl-umass/counting-clusters

    Hardened against degenerate inputs (zero similarity rows, NaN/inf in n_hat,
    isolated nodes that produce zero proposal mass) — substitutes uniform
    fallbacks for unrecoverable distributions and skips terms that would
    divide by zero in the estimator.

    Returns additional stats dict with GT query counts.
    """
    OBSERVATIONS = len(gt_s_ij)
    eps = 1e-12

    if not n_hat:
        n_hat = []
        for i in range(OBSERVATIONS):
            n_hat.append(np.sum(s_ij[i]))

    # Sanitize n_hat: zero/NaN/inf entries collapse the proposal. Treat such
    # nodes as singletons (n_hat = 1) — they contribute uniformly to Q.
    n_hat = np.asarray(n_hat, dtype=float)
    bad_n_hat = ~np.isfinite(n_hat) | (n_hat <= 0)
    if np.any(bad_n_hat):
        log(f"  ⚠ {int(bad_n_hat.sum())} node(s) have invalid n_hat (zero/NaN/inf); "
            f"treating as singletons")
        n_hat = n_hat.copy()
        n_hat[bad_n_hat] = 1.0

    # Proposal distribution Q(u) ∝ 1/n_hat(u)
    inv_n_hat = 1.0 / np.maximum(n_hat, eps)
    Q, Q_uniform = _safe_normalize(inv_n_hat, 'Q')
    if Q_uniform:
        log(f"  ⚠ Proposal Q collapsed; falling back to uniform proposal")

    sampled_vertices = list(np.random.choice(
        list(range(OBSERVATIONS)), N_v, p=Q, replace=True))

    sampled_neighbors = []
    q_all = []
    n_neighbor_uniform = 0
    for v_i in sampled_vertices:
        q, q_uniform = _safe_normalize(s_ij[v_i], 'neighbor')
        if q_uniform:
            n_neighbor_uniform += 1
        q_all.append(q)
        sampled_neighbors.append(
            [v_i] + list(np.random.choice(
                list(range(len(s_ij[v_i]))), N_n - 1, p=q, replace=True)))
    if n_neighbor_uniform > 0:
        log(f"  ⚠ Neighbor distribution collapsed for {n_neighbor_uniform} sampled vertex/vertices; "
            f"used uniform fallback")

    # --- Count GT queries ---
    gt_lookups = 0          # total gt_s_ij[v_i][v_j] accesses
    self_lookups = 0        # lookups where v_i == v_j (self-comparison)
    positive_lookups = 0    # lookups where gt_s_ij == 1 (same individual)
    unique_pairs = set()    # unique (v_i, v_j) pairs queried

    # Estimate K — skip terms where dividing by q or Q would blow up.
    sum_cc = 0.0
    n_terms = 0
    for i, v_i in enumerate(sampled_vertices):
        sum_n_bar = 0.0
        for j, v_j in enumerate(sampled_neighbors[i]):
            gt_val = gt_s_ij[v_i][v_j]
            gt_lookups += 1
            if v_i == v_j:
                self_lookups += 1
            if gt_val == 1.0:
                positive_lookups += 1
            unique_pairs.add((min(v_i, v_j), max(v_i, v_j)))
            q_v_j = q_all[i][v_j]
            if q_v_j > 0:
                sum_n_bar += gt_val / q_v_j
        n_bar = sum_n_bar / max(1, len(sampled_neighbors[i]))
        if n_bar > 0 and Q[v_i] > 0 and np.isfinite(n_bar) and np.isfinite(Q[v_i]):
            sum_cc += (1.0 / n_bar) * (1.0 / Q[v_i])
            n_terms += 1
    if n_terms == 0:
        log(f"  ⚠ All {N_v} sampled vertices produced degenerate estimator terms; "
            f"K_hat undefined — returning 1.0 as a placeholder")
        f_hat = 1.0
    else:
        f_hat = sum_cc / n_terms
    if not np.isfinite(f_hat):
        log(f"  ⚠ K_hat is non-finite ({f_hat}); clamping to 1.0")
        f_hat = 1.0

    stats = {
        'gt_lookups': gt_lookups,
        'self_lookups': self_lookups,
        'non_self_lookups': gt_lookups - self_lookups,
        'positive_lookups': positive_lookups,
        'unique_pairs': len(unique_pairs),
        'unique_vertices': len(set(sampled_vertices)),
    }

    if not ci:
        return f_hat, n_hat, stats
    else:
        # CI recomputes same values — no new GT information used
        w_ci = 0.0
        n_ci_terms = 0
        for i, v_i in enumerate(sampled_vertices):
            sum_n_bar = 0.0
            for v_j in sampled_neighbors[i]:
                q_v_j = q_all[i][v_j]
                if q_v_j > 0:
                    sum_n_bar += gt_s_ij[v_i][v_j] / q_v_j
            n_bar = sum_n_bar / max(1, len(sampled_neighbors[i]))
            if n_bar > 0 and Q[v_i] > 0 and np.isfinite(n_bar) and np.isfinite(Q[v_i]):
                w_ci += ((1.0 / n_bar) * (1.0 / Q[v_i]) - f_hat) ** 2
                n_ci_terms += 1
        var_hat = w_ci / max(1, n_ci_terms)
        ci_val = 1.96 * (np.sqrt(var_hat / max(1, N_v)))
        if not np.isfinite(ci_val):
            ci_val = 0.0
        return f_hat, ci_val, n_hat, stats


# ---------------------------------------------------------------
# Data loading (replicates LCA pipeline filtering)
# ---------------------------------------------------------------

def _load_dataset_registry(path=None):
    """Dataset paths for --dataset, kept out of the package.

    Resolution order: explicit `path`, then $BETA_STABILITY_NIS_DATASETS, then
    `configs/nis_datasets.yaml` next to the package. Site-specific paths are not
    hardcoded here; a missing registry is an error, never a guess.
    """
    # An explicitly requested registry must exist; only the implicit locations fall through.
    for c, explicit in ((path, True),
                        (os.environ.get('BETA_STABILITY_NIS_DATASETS'), True),
                        (Path(__file__).resolve().parents[1] / 'configs' / 'nis_datasets.yaml', False)):
        if not c:
            continue
        if Path(c).is_file():
            with open(c) as f:
                return yaml.safe_load(f)
        if explicit:
            raise SystemExit(f'Dataset registry not found: {c}')
    raise SystemExit(
        'No NIS dataset registry found. Pass --datasets <file.yaml>, set '
        'BETA_STABILITY_NIS_DATASETS, or create configs/nis_datasets.yaml '
        '(mapping dataset name -> annotation_file, embedding_file, name_keys, ...).')


DATASET_CONFIGS = None   # populated from the registry in main()


def load_and_filter(dataset_name):
    """Load embeddings and annotations, apply same filtering as LCA pipeline."""
    cfg = DATASET_CONFIGS[dataset_name]

    # Load embeddings pickle: (embedding_array, uuid_list_or_dict)
    with open(cfg['embedding_file'], 'rb') as f:
        embeddings_raw, uuids_raw = pickle.load(f)

    if isinstance(uuids_raw, dict):
        uuid_list = list(uuids_raw.keys())
    else:
        uuid_list = list(uuids_raw)

    # Load annotations JSON
    with open(cfg['annotation_file'], 'r') as f:
        data = json.load(f)

    dfa = pd.DataFrame(data['annotations'])
    dfi = pd.DataFrame(data['images']).drop_duplicates(subset=['uuid'])

    if cfg['format'] == 'standard':
        dfn = pd.DataFrame(data['individuals'])
        dfc = pd.DataFrame(data['categories'])

    # Merge annotations with images
    if 'image_uuid' in dfa.columns and 'uuid' in dfi.columns:
        df = dfa.merge(dfi, left_on='image_uuid', right_on='uuid',
                       suffixes=('', '_y'))
    else:
        df = dfa.merge(dfi, left_on='image_id', right_on='id',
                       suffixes=('', '_y'))

    if cfg['format'] == 'standard':
        df = df.merge(dfn, left_on='individual_uuid', right_on='uuid',
                      suffixes=('', '_y'))
        df = df.merge(dfc, left_on='category_id', right_on='id',
                      suffixes=('', '_y'))

    id_key = cfg['id_key']
    name_keys = cfg['name_keys']
    filter_key = '__'.join(name_keys)

    # Create composite name key
    df[filter_key] = df[name_keys].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)

    # Filter 1: only annotations with embeddings
    df = df[df[id_key].isin(uuid_list)]
    log(f"  After UUID filter: {len(df)} annotations")

    # Filter 2: viewpoint
    if cfg['viewpoint_list']:
        for vp_val in [cfg['viewpoint_list']]:
            def matches(value):
                if pd.isna(value):
                    return False
                return str(value) in [str(v) for v in vp_val]
            df = df[df['viewpoint'].apply(matches)]
    log(f"  After viewpoint filter: {len(df)} annotations")

    # Filter 3: min count per individual
    if cfg['n_filter_min']:
        df = df.groupby(filter_key).filter(
            lambda g: len(g) >= cfg['n_filter_min'])
    log(f"  After min filter: {len(df)} annotations")

    # Filter 4: max count per individual
    if cfg['n_filter_max']:
        df = df.groupby(filter_key, as_index=False).apply(
            lambda g: g.sample(frac=1, random_state=0).head(
                cfg['n_filter_max'])).droplevel(level=0)
    log(f"  After max filter: {len(df)} annotations")

    df = df.reset_index(drop=True)

    # Extract filtered embeddings
    filtered_uuids = df[id_key].tolist()
    filtered_embeddings = np.array([
        embeddings_raw[uuid_list.index(uuid)] for uuid in filtered_uuids])

    # Ground truth labels
    le = LabelEncoder()
    gt_labels = le.fit_transform(df[filter_key].values)
    true_K = len(le.classes_)

    log(f"  Dataset: {len(filtered_uuids)} nodes, {true_K} true clusters")
    sizes = np.bincount(gt_labels)
    log(f"  Cluster sizes: min={sizes.min()}, max={sizes.max()}, "
        f"mean={sizes.mean():.1f}, singletons={np.sum(sizes == 1)}")

    return filtered_embeddings, gt_labels, true_K


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------

def hungarian_metrics(pred_labels, gt_labels):
    """
    Hungarian (optimal assignment) cluster matching.
    Same as cluster_tools.hungarian_cluster_matching:
    1. Build Jaccard similarity matrix between GT and predicted clusters
    2. Run Hungarian algorithm for optimal one-to-one assignment
    3. Count matched/unmatched clusters as TP/FP/FN
    """
    # Build cluster sets
    gt_clusters = defaultdict(set)
    pred_clusters = defaultdict(set)
    for i, (g, p) in enumerate(zip(gt_labels, pred_labels)):
        gt_clusters[g].add(i)
        pred_clusters[p].add(i)

    gt_ids = list(gt_clusters.keys())
    pred_ids = list(pred_clusters.keys())
    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    if n_gt == 0 or n_pred == 0:
        return {'h_precision': 0, 'h_recall': 0, 'h_f1': 0}

    # Jaccard similarity matrix
    jaccard = np.zeros((n_gt, n_pred))
    for i, gid in enumerate(gt_ids):
        gs = gt_clusters[gid]
        for j, pid in enumerate(pred_ids):
            ps = pred_clusters[pid]
            inter = len(gs & ps)
            union = len(gs | ps)
            jaccard[i, j] = inter / union if union > 0 else 0

    # Hungarian assignment (minimize negative Jaccard)
    row_ind, col_ind = linear_sum_assignment(-jaccard)

    # Count valid matches (Jaccard > 0)
    tp = sum(1 for r, c in zip(row_ind, col_ind) if jaccard[r, c] > 0)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # CEAF (phi4 / Dice): same optimal 1-1 cluster alignment, but matches are
    # weighted by Dice similarity instead of thresholded at Jaccard>0.
    dice = np.zeros((n_gt, n_pred))
    for i, gid in enumerate(gt_ids):
        gs = gt_clusters[gid]
        for j, pid in enumerate(pred_ids):
            ps = pred_clusters[pid]
            inter = len(gs & ps)
            denom = len(gs) + len(ps)
            dice[i, j] = 2 * inter / denom if denom > 0 else 0
    r2, c2 = linear_sum_assignment(-dice)
    phi = float(dice[r2, c2].sum())
    ceaf_p = phi / n_pred if n_pred else 0
    ceaf_r = phi / n_gt if n_gt else 0
    ceaf_f1 = 2 * ceaf_p * ceaf_r / (ceaf_p + ceaf_r) if (ceaf_p + ceaf_r) > 0 else 0

    return {'h_precision': precision, 'h_recall': recall, 'h_f1': f1,
            'ceaf_precision': ceaf_p, 'ceaf_recall': ceaf_r, 'ceaf_f1': ceaf_f1}


_KMEANS_EVAL_CACHE = {}


def evaluate_kmeans(embeddings, gt_labels, k, true_K, n_init=10):
    """Run k-means and compute pairwise + Hungarian + CEAF metrics.

    The result depends ONLY on the integer k (embeddings, gt_labels and the
    KMeans random_state are fixed within a run), so it is memoized by k. During
    the budget sweep the same k recurs across runs and budgets -- especially once
    the NIS estimate converges at higher budgets -- so the cache turns hundreds of
    identical k-means fits into a few dozen, with byte-identical results.
    """
    k = max(1, min(len(embeddings), round(k)))
    cache_key = (k, n_init)
    cached = _KMEANS_EVAL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    norm_emb = normalize(embeddings, norm='l2')
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
    pred_labels = kmeans.fit_predict(norm_emb)

    ari = adjusted_rand_score(gt_labels, pred_labels)

    # Pairwise precision/recall/f1
    n = len(gt_labels)
    gt_same = defaultdict(set)
    pred_same = defaultdict(set)
    for i in range(n):
        gt_same[gt_labels[i]].add(i)
        pred_same[pred_labels[i]].add(i)

    tp_gt = sum(len(s) * (len(s) - 1) // 2 for s in gt_same.values())
    tp_pred = sum(len(s) * (len(s) - 1) // 2 for s in pred_same.values())

    tp = 0
    for gt_c in gt_same.values():
        counts = defaultdict(int)
        for node in gt_c:
            counts[pred_labels[node]] += 1
        tp += sum(c * (c - 1) // 2 for c in counts.values())

    total_pairs = n * (n - 1) // 2
    fp = tp_pred - tp
    fn = tp_gt - tp

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-10, precision + recall)

    # Hungarian metrics
    h = hungarian_metrics(pred_labels, gt_labels)

    result = {
        'k_used': k,
        'ari': ari,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'h_precision': h['h_precision'],
        'h_recall': h['h_recall'],
        'h_f1': h['h_f1'],
        'ceaf_precision': h.get('ceaf_precision', 0.0),
        'ceaf_recall': h.get('ceaf_recall', 0.0),
        'ceaf_f1': h.get('ceaf_f1', 0.0),
        'n_clusters': len(set(pred_labels)),
    }
    _KMEANS_EVAL_CACHE[cache_key] = result
    return result


def find_optimal_ratio(gt_s_synth, s_ij, n_hat, n_nodes, budget, K_synth,
                       n_trials=50, workers=1):
    """Find optimal N_n/N_v ratio by simulating NIS with synthetic oracle.

    Creates ~20 candidate (N_v, N_n) allocations for the given budget,
    runs NIS n_trials times for each, and returns the ratio that minimizes
    MSE of K_hat vs K_synth.

    With workers > 1 the trials run in parallel. Each trial reseeds the global RNG
    (100000 + t) immediately before nested_is, so every trial's K_hat is identical to
    the serial loop's; results are aggregated in the original (candidate, trial) order,
    and any log output a trial produces is captured in the worker and replayed into the
    log in that same order.
    """
    # Generate candidate N_v values (log-spaced for wide coverage)
    max_nv = min(budget // 2, n_nodes)
    if max_nv < 2:
        return 7.0  # fallback
    nv_candidates = sorted(set(
        max(2, int(x))
        for x in np.logspace(np.log10(2), np.log10(max_nv), 20)
    ))

    plan = []
    for nv in nv_candidates:
        nn = max(2, budget // nv)
        if nn > n_nodes:
            nn = n_nodes
        plan.append((nv, nn))

    if workers > 1:
        def _trial(task):
            import contextlib, io
            global _log_file
            nv_, nn_, t_ = task
            _log_file = None
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                np.random.seed(100000 + t_)
                f_hat_, _, _, _ = nested_is(gt_s_synth, s_ij, nv_, nn_, n_hat=n_hat, ci=True)
            return f_hat_, buf.getvalue()
        tasks = [(nv, nn, t) for nv, nn in plan for t in range(n_trials)]
        outs = _fork_map(_trial, tasks, workers, threads=1)
        per_task = dict(zip(tasks, outs))

    best_mse = float('inf')
    best_ratio = 7.0
    results = []

    for nv, nn in plan:
        k_hats = []
        for t in range(n_trials):
            if workers > 1:
                f_hat, captured = per_task[(nv, nn, t)]
                if captured:
                    log(captured[:-1] if captured.endswith('\n') else captured)
            else:
                np.random.seed(100000 + t)
                f_hat, _, _, _ = nested_is(
                    gt_s_synth, s_ij, nv, nn, n_hat=n_hat, ci=True)
            k_hats.append(f_hat)
        k_arr = np.array(k_hats)
        mse = float(np.mean((k_arr - K_synth) ** 2))
        ratio = nn / max(1, nv)
        results.append((nv, nn, ratio, mse, k_arr.mean(), k_arr.std()))
        if mse < best_mse:
            best_mse = mse
            best_ratio = ratio

    return best_ratio, results


def budget_to_nv_nn(budget, n_nodes, ratio=7.0):
    """Convert total GT lookup budget to (N_v, N_n) pair.

    N_v = sqrt(budget / ratio), N_n = budget / N_v.
    The ratio should be computed by find_optimal_ratio().
    """
    if budget <= 0:
        return 0, 0
    N_v = max(1, int(np.sqrt(budget / ratio)))
    N_v = min(N_v, n_nodes, budget)
    N_n = max(2, budget // max(1, N_v))
    N_n = min(N_n, n_nodes)
    return N_v, N_n


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def _fork_map(fn, tasks, workers, threads=1, parent_task=None):
    """Evaluate fn(task) for every task in forked worker processes, returning results in
    task order. fn is inherited through fork and never pickled, so this works no matter
    how the script was launched (plain run, `python -m cProfile`, ...). Only tasks and
    results cross process boundaries. A failing or dying worker raises; nothing is
    skipped.

    parent_task, if given, runs in this (parent) process after the workers are forked
    and before results are collected. Multithreaded OpenMP work must happen here and
    not in a forked child: the parent has already used OpenMP (k-means), and libgomp
    is not fork-safe, so a forked child that opens a multi-thread team hangs."""
    import multiprocessing as mp
    import queue as _queue
    ctx = mp.get_context('fork')
    q_in, q_out = ctx.Queue(), ctx.Queue()

    def _loop():
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=threads):
            while True:
                item = q_in.get()
                if item is None:
                    return
                i, task = item
                try:
                    q_out.put((i, True, fn(task)))
                except BaseException as e:            # report, never swallow
                    q_out.put((i, False, repr(e)))

    procs = [ctx.Process(target=_loop, daemon=True) for _ in range(max(1, min(workers, len(tasks))))]
    for pr in procs:
        pr.start()
    for i, t in enumerate(tasks):
        q_in.put((i, t))
    for _ in procs:
        q_in.put(None)
    out, got = [None] * len(tasks), 0
    try:
        if parent_task is not None:
            parent_task()
        while got < len(tasks):
            try:
                i, ok, res = q_out.get(timeout=30)
            except _queue.Empty:
                dead = [pr.exitcode for pr in procs if pr.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f'worker process died (exit codes {dead})')
                continue
            if not ok:
                raise RuntimeError(f'worker failed on task {tasks[i]!r}: {res}')
            out[i] = res
            got += 1
    finally:
        for pr in procs:
            if pr.is_alive():
                pr.join(timeout=5)
            if pr.is_alive():
                pr.terminate()
    return out


def prefill_kmeans_cache(embeddings, gt_labels, true_K, ks, n_init, workers, parent_task=None):
    """Evaluate every distinct k once, in parallel, and store it in _KMEANS_EVAL_CACHE.

    Exact: evaluate_kmeans depends only on k (fixed data, KMeans random_state=42), and
    k-means labels were verified identical across 1/4/16 threads on real data, so a
    worker returns exactly what the serial sweep computes. The sweep loop is unchanged;
    it simply finds each evaluation already cached. Hundreds of full-dataset k-means
    fits per sweep were the dominant cost (nested_is itself is ~0.02 s per call).
    """
    todo = sorted({k for k in ks if (k, n_init) not in _KMEANS_EVAL_CACHE})
    if workers <= 1 or len(todo) <= 1:
        return 0
    res = _fork_map(lambda k: evaluate_kmeans(embeddings, gt_labels, k, true_K, n_init=n_init),
                    todo, workers, threads=1, parent_task=parent_task)
    for k, r in zip(todo, res):
        _KMEANS_EVAL_CACHE[(k, n_init)] = r
    return len(todo)


def main():
    global _log_file

    # The registry defines the valid --dataset values, so it is read first by a
    # pre-parser; `parents=` then re-exposes --datasets in the real help text.
    # allow_abbrev=False: without it '--dataset X' is taken as an abbreviation
    # of '--datasets' and the registry path becomes the dataset name.
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument('--datasets', default=None,
                     help='Dataset registry YAML (default: $BETA_STABILITY_NIS_DATASETS '
                          'or configs/nis_datasets.yaml)')
    pre_args, _ = pre.parse_known_args()
    global DATASET_CONFIGS
    DATASET_CONFIGS = _load_dataset_registry(pre_args.datasets)

    parser = argparse.ArgumentParser(
        description='Reference NIS estimator on LCA data', parents=[pre])
    parser.add_argument('--dataset', required=True,
                        choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument('--N_v', type=int, default=50,
                        help='Number of sampled vertices')
    parser.add_argument('--N_n', type=int, default=100,
                        help='Number of neighbors per vertex (incl. self)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of independent runs')
    parser.add_argument('--workers', type=int, default=1,
                        help='Processes for evaluating distinct k-means k values in parallel '
                             'during --sweep (1 = serial). Results are identical either way.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prob_human_correct', type=float, default=1.0,
                        help='Reviewer accuracy for the pairwise same/different '
                             'oracle queries. 1.0 = perfect oracle (default). '
                             '<1 flips each queried pair-label with probability '
                             '(1-p), modelling an imperfect reviewer — same error '
                             'model and review budget as the other methods. Only '
                             'the queries are corrupted; evaluation uses true labels.')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to save log output (default: '
                             'tmp/<dataset>/output/nis_reference.log)')
    parser.add_argument('--metrics_file', type=str, default=None,
                        help='Path to save review-curve metrics as JSONL')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep GT budgets from 0 to budget_max')
    parser.add_argument('--budget_max', type=int, default=5000,
                        help='Max total GT lookups for sweep (default: 5000)')
    parser.add_argument('--budget_steps', type=int, default=10,
                        help='Number of evenly-spaced budget levels (default: 10)')
    parser.add_argument('--report_at_reviews', default='N',
                        help="Comma-separated extra sweep budgets evaluated in the same run "
                             "(exact checkpoints); 'N' = number of annotations. Their rows carry "
                             "'checkpoint': true. Budgets above --budget_max are not evaluated.")
    parser.add_argument('--embedding', default='miewid',
                        help='Embedding name; rewrites embedding_file path and '
                             'adds suffix to output log (default: miewid).')
    args = parser.parse_args()

    # Override embedding file for non-miewid embeddings
    cfg = DATASET_CONFIGS[args.dataset]
    if args.embedding != 'miewid':
        import re
        cfg['embedding_file'] = re.sub(
            r'/embeddings_([^/]+)\.pickle',
            rf'/{args.embedding}_embeddings_\1.pickle',
            cfg['embedding_file'])

    # Setup log file
    if args.log_file is None:
        log_dir = os.path.join('tmp', args.dataset, 'output')
        os.makedirs(log_dir, exist_ok=True)
        if args.embedding == 'miewid':
            suffix = 'nis_reference_sweep.log' if args.sweep else 'nis_reference.log'
        else:
            suffix = (f'nis_reference_sweep_{args.embedding}.log'
                      if args.sweep else f'nis_reference_{args.embedding}.log')
        args.log_file = os.path.join(log_dir, suffix)
    else:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    if args.metrics_file:
        metrics_dir = os.path.dirname(args.metrics_file)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        open(args.metrics_file, 'w').close()

    _log_file = open(args.log_file, 'w')

    log(f"=== Reference NIS on {args.dataset} ===")
    log(f"N_v={args.N_v}, N_n={args.N_n}, runs={args.runs}, seed={args.seed}")
    log(f"Log file: {args.log_file}")
    log()

    # Load data
    log("Loading data...")
    embeddings, gt_labels, true_K = load_and_filter(args.dataset)
    n = len(embeddings)
    log()

    # Adjust N_v, N_n for dataset size
    N_v = min(args.N_v, n)
    N_n = min(args.N_n, n)
    log(f"N_v={N_v}, N_n={N_n}")
    log(f"True K = {true_K}")
    log()

    # Explain GT query budget
    log("--- GT query budget ---")
    log(f"Per run:")
    log(f"  Total GT lookups:     N_v * N_n = {N_v} * {N_n} = {N_v * N_n}")
    log(f"  Self lookups (j=0):   N_v * 1   = {N_v}  (always gt=1, answer is known)")
    log(f"  Non-self lookups:     N_v * (N_n-1) = {N_v} * {N_n - 1} = {N_v * (N_n - 1)}")
    log(f"  (non-self may include duplicates from with-replacement sampling)")
    log()

    # Build similarity matrix
    log("Computing similarity matrix...")
    t0 = time.time()
    s_ij = cosine_similarity(embeddings)
    s_ij[s_ij < 0] = 0
    log(f"  s_ij computed in {time.time() - t0:.1f}s, shape={s_ij.shape}")

    # Build ground truth similarity matrix
    log("Building ground truth matrix...")
    t0 = time.time()
    gt_labels_arr = np.array(gt_labels)
    gt_s_ij = (gt_labels_arr[:, None] == gt_labels_arr[None, :]).astype(np.float64)
    log(f"  gt_s_ij computed in {time.time() - t0:.1f}s")

    # Imperfect reviewer: the human answering NIS's pairwise same/different queries
    # is wrong with probability (1 - prob_human_correct). We corrupt the QUERY matrix
    # once per seed (each pair gets a fixed, possibly-wrong answer — consistent if the
    # same pair is queried more than once), symmetric with the self-diagonal preserved.
    # Evaluation still uses the true gt_labels; only the oracle queries are corrupted,
    # matching the error model and review budget used by the other methods.
    phc = float(args.prob_human_correct)
    if phc < 1.0:
        rng = np.random.default_rng(args.seed)
        Nn = gt_s_ij.shape[0]
        iu, iv = np.triu_indices(Nn, k=1)
        flip = rng.random(iu.shape[0]) >= phc  # wrong with prob (1 - phc)
        fi, fj = iu[flip], iv[flip]
        gt_s_ij[fi, fj] = 1.0 - gt_s_ij[fi, fj]
        gt_s_ij[fj, fi] = gt_s_ij[fi, fj]  # keep symmetric
        log(f"  Imperfect reviewer (prob_human_correct={phc}): flipped "
            f"{int(flip.sum())}/{iu.shape[0]} pair-labels (seed={args.seed})")

    # Precompute n_hat and K_hat_0 (zero-review estimate)
    n_hat = list(np.sum(s_ij, axis=1))
    n_hat_arr = np.array(n_hat)
    K_hat_0 = float(np.sum(1.0 / n_hat_arr))
    K_hat_0 = max(1.0, min(float(n), K_hat_0))
    log(f"  K_hat_0 = {K_hat_0:.2f} (from embeddings, 0 reviews)")

    # Find optimal N_n/N_v ratio via synthetic oracle simulation
    log("\nFinding optimal ratio via synthetic oracle simulation...")
    t0 = time.time()
    K_synth = max(1, round(K_hat_0))
    synth_labels = KMeans(
        n_clusters=K_synth, random_state=42, n_init=10
    ).fit_predict(normalize(embeddings.reshape(n, -1), norm='l2'))
    gt_s_synth = (synth_labels[:, None] == synth_labels[None, :]).astype(
        np.float64)
    K_synth_actual = len(set(synth_labels))
    budget_ref = args.budget_max if args.sweep else N_v * N_n
    # Worker processes for the parallel phases. During the k-means prefill the parent
    # process computes the oracle k-means (true K, reported after the sweep) with the
    # same 4 threads the serial run uses, so 4 cores are kept free for it there.
    _pool_workers = max(1, args.workers - 4) if args.workers > 4 else args.workers
    optimal_ratio, ratio_results = find_optimal_ratio(
        gt_s_synth, s_ij, n_hat, n, budget_ref, K_synth_actual,
        workers=args.workers)
    log(f"  Synthetic K = {K_synth_actual} (k-means with k={K_synth})")
    log(f"  Reference budget = {budget_ref}")
    log(f"  Optimal ratio = {optimal_ratio:.2f} "
        f"(computed in {time.time() - t0:.1f}s)")
    log(f"  {'N_v':>5} | {'N_n':>5} | {'ratio':>6} | {'MSE':>10} | "
        f"{'K_hat mean':>10} | {'K_hat std':>10}")
    for nv, nn, r, mse, mean, std in ratio_results:
        marker = " <-- best" if abs(r - optimal_ratio) < 0.01 else ""
        log(f"  {nv:>5} | {nn:>5} | {r:>6.2f} | {mse:>10.1f} | "
            f"{mean:>10.2f} | {std:>10.2f}{marker}")

    # ---------------------------------------------------------------
    # Sweep mode: run NIS at multiple budget levels
    # ---------------------------------------------------------------
    if args.sweep:
        budgets = np.linspace(0, args.budget_max,
                              args.budget_steps + 1).astype(int).tolist()
        # Exact checkpoints (e.g. one review per annotation) are extra budgets evaluated in
        # this same run. Every budget reseeds before sampling, so adding one leaves each grid
        # row unchanged, and the ratio stays tuned to --budget_max.
        checkpoint_budgets = {b for b in resolve_checkpoints(
            [x for x in str(args.report_at_reviews).split(',') if x.strip()], n)
            if b <= args.budget_max}
        budgets = sorted(set(budgets) | checkpoint_budgets)

        log(f"\n=== Budget sweep: {len(budgets)} levels, "
            f"{args.runs} runs each ===")
        log(f"Budgets: {budgets}")
        log()

        def pm(a):
            """Format mean±std."""
            return f"{a.mean():.3f}±{a.std():.3f}"

        # Table header
        log(f"{'Budget':>6} | {'N_v':>4} | {'N_n':>4} | {'Non-self':>8} | "
            f"{'K_hat':>14} | {'K/K_true':>8} | "
            f"{'F1':>14} | {'ARI':>14} | "
            f"{'H_F1':>14} | {'H_Prec':>14} | {'H_Rec':>14}")
        log("-" * 140)

        if args.workers > 1:
            # Pass 1: every K_hat the loop below will produce. Each run reseeds the
            # global RNG immediately before nested_is, so these are the same values;
            # output is suppressed so nis.log is not altered.
            import contextlib, io
            _ks = set()
            _saved_log_file = _log_file
            _log_file = None
            with contextlib.redirect_stdout(io.StringIO()):
                for _b in budgets:
                    if _b == 0:
                        continue
                    _nv, _nn = budget_to_nv_nn(_b, n, ratio=optimal_ratio)
                    for _r in range(args.runs):
                        np.random.seed(args.seed + _r)
                        _fh, _, _, _ = nested_is(gt_s_ij, s_ij, _nv, _nn, n_hat=n_hat, ci=True)
                        _ks.add(max(1, min(n, round(_fh))))
            _log_file = _saved_log_file
            _t_pf = time.time()
            def _oracle_in_parent():
                from threadpoolctl import threadpool_limits
                with threadpool_limits(limits=4):
                    evaluate_kmeans(embeddings, gt_labels, true_K, true_K)
            _nfit = prefill_kmeans_cache(embeddings, gt_labels, true_K, _ks, n_init=3, workers=_pool_workers,
                                         parent_task=_oracle_in_parent)
            print(f"[prefill] {_nfit} distinct k-means evaluations on {args.workers} workers "
                  f"in {time.time() - _t_pf:.1f}s", flush=True)

        for budget in budgets:
            if budget == 0:
                # Singleton baseline: each node is its own cluster
                pred = np.arange(n)
                ari_val = adjusted_rand_score(gt_labels, pred)
                h = hungarian_metrics(pred, gt_labels)
                write_metrics_row(args.metrics_file, {
                    'method': 'nis',
                    'species': args.dataset,
                    'seed': args.seed,
                    'reviews': 0,
                    'h_f1': h['h_f1'],
                    'ceaf_f1': h.get('ceaf_f1', None),
                    'pcc_f1': 0.0,
                    'h_precision': h['h_precision'],
                    'h_recall': h['h_recall'],
                    'pcc_precision': 0.0,
                    'pcc_recall': 0.0,
                    'budget': 0,
                    'actual_budget': 0,
                    'runs': args.runs,
                    # every node is its own cluster
                    'n_clusters': int(n),
                    'true_clusters': int(true_K),
                    'cluster_count_ratio': float(n / true_K),
                    'per_run': [],
                })
                log(f"{0:>6} | {'--':>4} | {'--':>4} | {0:>8} | "
                    f"{'singletons':>14} | {n / true_K:>8.3f} | "
                    f"{'0.000':>14} | {ari_val:>14.3f} | "
                    f"{h['h_f1']:>14.3f} | {h['h_precision']:>14.3f} | "
                    f"{h['h_recall']:>14.3f}")
                continue

            N_v_b, N_n_b = budget_to_nv_nn(budget, n, ratio=optimal_ratio)
            actual = N_v_b * N_n_b
            non_self = N_v_b * (N_n_b - 1)

            k_hats_b, metrics_b = [], []
            for run_i in range(args.runs):
                np.random.seed(args.seed + run_i)
                f_hat, ci_val, _, _ = nested_is(
                    gt_s_ij, s_ij, N_v_b, N_n_b, n_hat=n_hat, ci=True)
                k_hats_b.append(f_hat)
                m = evaluate_kmeans(
                    embeddings, gt_labels, f_hat, true_K, n_init=3)
                metrics_b.append(m)

            k_arr = np.array(k_hats_b)
            f1s = np.array([m['f1'] for m in metrics_b])
            aris = np.array([m['ari'] for m in metrics_b])
            precs = np.array([m['precision'] for m in metrics_b])
            recs = np.array([m['recall'] for m in metrics_b])
            hf1 = np.array([m['h_f1'] for m in metrics_b])
            hp = np.array([m['h_precision'] for m in metrics_b])
            hr = np.array([m['h_recall'] for m in metrics_b])
            cf1 = np.array([m.get('ceaf_f1', 0.0) for m in metrics_b])
            cp = np.array([m.get('ceaf_precision', 0.0) for m in metrics_b])
            cr = np.array([m.get('ceaf_recall', 0.0) for m in metrics_b])
            ncl = np.array([m['n_clusters'] for m in metrics_b])

            write_metrics_row(args.metrics_file, {
                'method': 'nis',
                'species': args.dataset,
                'seed': args.seed,
                'reviews': int(non_self),
                'h_f1': float(hf1.mean()),
                'pcc_f1': float(f1s.mean()),
                'h_precision': float(hp.mean()),
                'h_recall': float(hr.mean()),
                'pcc_precision': float(precs.mean()),
                'pcc_recall': float(recs.mean()),
                'ceaf_f1': float(cf1.mean()),
                'ceaf_precision': float(cp.mean()),
                'ceaf_recall': float(cr.mean()),
                'budget': int(budget),
                **({'checkpoint': True} if budget in checkpoint_budgets else {}),
                'actual_budget': int(actual),
                'runs': int(args.runs),
                # Cluster count of the clustering actually evaluated (k-means with
                # k = round(K_hat), per run). Previously computed in evaluate_kmeans
                # and discarded; K/K_true in the text log is the estimate, not this.
                'n_clusters': float(ncl.mean()),
                'n_clusters_std': float(ncl.std()),
                'true_clusters': int(true_K),
                'cluster_count_ratio': float(ncl.mean() / true_K),
                'k_hat_mean': float(k_arr.mean()),
                'k_hat_std': float(k_arr.std()),
                # Raw per-run values, so any aggregate can be recomputed offline.
                'per_run': [{
                    'run': int(i),
                    'seed': int(args.seed + i),
                    'k_hat': float(k_hats_b[i]),
                    'k_used': int(metrics_b[i]['k_used']),
                    'n_clusters': int(metrics_b[i]['n_clusters']),
                    'ceaf_f1': float(metrics_b[i].get('ceaf_f1', 0.0)),
                    'ceaf_precision': float(metrics_b[i].get('ceaf_precision', 0.0)),
                    'ceaf_recall': float(metrics_b[i].get('ceaf_recall', 0.0)),
                    'h_f1': float(metrics_b[i]['h_f1']),
                    'h_precision': float(metrics_b[i]['h_precision']),
                    'h_recall': float(metrics_b[i]['h_recall']),
                    'pcc_f1': float(metrics_b[i]['f1']),
                    'pcc_precision': float(metrics_b[i]['precision']),
                    'pcc_recall': float(metrics_b[i]['recall']),
                    'ari': float(metrics_b[i]['ari']),
                } for i in range(len(metrics_b))],
            })

            log(f"{actual:>6} | {N_v_b:>4} | {N_n_b:>4} | {non_self:>8} | "
                f"{k_arr.mean():>7.1f}±{k_arr.std():<5.1f} | "
                f"{k_arr.mean() / true_K:>8.3f} | "
                f"{pm(f1s):>14} | {pm(aris):>14} | "
                f"{pm(hf1):>14} | {pm(hp):>14} | {pm(hr):>14}")

        log("-" * 140)

        # Oracle baseline
        log(f"\n--- Oracle baseline: k-means with true K={true_K} ---")
        np.random.seed(args.seed)
        oracle = evaluate_kmeans(embeddings, gt_labels, true_K, true_K)
        log(f"  Pairwise:   F1={oracle['f1']:.3f}  "
            f"Prec={oracle['precision']:.3f}  "
            f"Rec={oracle['recall']:.3f}  "
            f"ARI={oracle['ari']:.3f}")
        log(f"  Hungarian:  F1={oracle['h_f1']:.3f}  "
            f"Prec={oracle['h_precision']:.3f}  "
            f"Rec={oracle['h_recall']:.3f}")

        log(f"\nLog saved to: {args.log_file}")
        _log_file.close()
        return

    # Run (single-budget mode)
    log(f"\nRunning {args.runs} independent NIS estimations...")
    log("-" * 130)
    log(f"{'Run':>4} | {'K_hat':>8} | {'CI':>8} | {'k_used':>6} | "
        f"{'ARI':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | "
        f"{'H_Prec':>6} | {'H_Rec':>6} | {'H_F1':>6} | "
        f"{'GT_look':>7} | {'Non-self':>8} | {'Uniq':>5} | {'Pos':>5}")
    log("-" * 130)

    k_hats = []
    cis = []
    metrics_all = []
    stats_all = []

    for run in range(args.runs):
        np.random.seed(args.seed + run)

        f_hat, ci_val, _, stats = nested_is(
            gt_s_ij, s_ij, N_v, N_n, n_hat=n_hat, ci=True)

        k_hats.append(f_hat)
        cis.append(ci_val)
        stats_all.append(stats)

        metrics = evaluate_kmeans(embeddings, gt_labels, f_hat, true_K, n_init=3)
        metrics_all.append(metrics)

        log(f"{run + 1:>4} | {f_hat:>8.1f} | {ci_val:>8.1f} | "
            f"{metrics['k_used']:>6} | {metrics['ari']:>6.3f} | "
            f"{metrics['precision']:>6.3f} | {metrics['recall']:>6.3f} | "
            f"{metrics['f1']:>6.3f} | "
            f"{metrics['h_precision']:>6.3f} | {metrics['h_recall']:>6.3f} | "
            f"{metrics['h_f1']:>6.3f} | "
            f"{stats['gt_lookups']:>7} | "
            f"{stats['non_self_lookups']:>8} | "
            f"{stats['unique_pairs']:>5} | "
            f"{stats['positive_lookups']:>5}")

    # Summary
    log("-" * 130)
    k_hats = np.array(k_hats)
    cis = np.array(cis)
    f1s = np.array([m['f1'] for m in metrics_all])
    aris = np.array([m['ari'] for m in metrics_all])
    precs = np.array([m['precision'] for m in metrics_all])
    recs = np.array([m['recall'] for m in metrics_all])
    h_f1s = np.array([m['h_f1'] for m in metrics_all])
    h_precs = np.array([m['h_precision'] for m in metrics_all])
    h_recs = np.array([m['h_recall'] for m in metrics_all])

    log(f"\nSummary over {args.runs} runs:")
    log(f"  True K:     {true_K}")
    log(f"  K_hat:      {k_hats.mean():.1f} +/- {k_hats.std():.1f} "
        f"(range: [{k_hats.min():.1f}, {k_hats.max():.1f}])")
    log(f"  CI (mean):  +/- {cis.mean():.1f}")
    log(f"  K_hat/K:    {k_hats.mean() / true_K:.3f}")
    log(f"  Pairwise metrics:")
    log(f"    F1:         {f1s.mean():.3f} +/- {f1s.std():.3f}")
    log(f"    ARI:        {aris.mean():.3f} +/- {aris.std():.3f}")
    log(f"    Precision:  {precs.mean():.3f} +/- {precs.std():.3f}")
    log(f"    Recall:     {recs.mean():.3f} +/- {recs.std():.3f}")
    log(f"  Hungarian metrics:")
    log(f"    H_F1:       {h_f1s.mean():.3f} +/- {h_f1s.std():.3f}")
    log(f"    H_Prec:     {h_precs.mean():.3f} +/- {h_precs.std():.3f}")
    log(f"    H_Recall:   {h_recs.mean():.3f} +/- {h_recs.std():.3f}")

    # GT usage summary
    log()
    log("--- GT usage summary (per run) ---")
    avg_stats = {k: np.mean([s[k] for s in stats_all]) for k in stats_all[0]}
    log(f"  Total GT lookups:      {avg_stats['gt_lookups']:.0f}")
    log(f"    Self (v_i == v_j):   {avg_stats['self_lookups']:.0f}  "
        f"(always gt=1, trivial)")
    log(f"    Non-self:            {avg_stats['non_self_lookups']:.0f}  "
        f"(actual 'human questions')")
    log(f"  Unique (i,j) pairs:    {avg_stats['unique_pairs']:.0f}  "
        f"(deduplicated, both directions)")
    log(f"  Positive (gt=1):       {avg_stats['positive_lookups']:.0f}  "
        f"({100*avg_stats['positive_lookups']/avg_stats['gt_lookups']:.1f}% of lookups)")
    log(f"  Unique vertices:       {avg_stats['unique_vertices']:.0f}  "
        f"(out of {N_v} sampled, with replacement)")
    log()
    log(f"  In the reference code, each gt_s_ij[v_i][v_j] lookup = 1 oracle query.")
    log(f"  The algorithm queries GT exactly N_v * N_n = {N_v * N_n} times per run.")
    log(f"  Of these, N_v = {N_v} are self-comparisons (j=0, answer known a priori).")
    log(f"  The remaining {N_v * (N_n - 1)} are the 'human review' equivalent.")

    # Oracle baseline
    log(f"\n--- Oracle baseline: k-means with true K={true_K} ---")
    np.random.seed(args.seed)
    oracle_metrics = evaluate_kmeans(embeddings, gt_labels, true_K, true_K)
    log(f"  Pairwise:   F1={oracle_metrics['f1']:.3f}  "
        f"Prec={oracle_metrics['precision']:.3f}  "
        f"Rec={oracle_metrics['recall']:.3f}  "
        f"ARI={oracle_metrics['ari']:.3f}")
    log(f"  Hungarian:  F1={oracle_metrics['h_f1']:.3f}  "
        f"Prec={oracle_metrics['h_precision']:.3f}  "
        f"Rec={oracle_metrics['h_recall']:.3f}")

    log(f"\nLog saved to: {args.log_file}")
    _log_file.close()


if __name__ == '__main__':
    main()
