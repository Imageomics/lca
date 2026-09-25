"""Sequential review baseline, shared by manual and thresholded review.

Both baselines are the same procedure over the same candidate list: take each
node's top-k matches by embedding score, present the pairs to a reviewer one at a
time in decreasing score order, and skip any pair whose two annotations are
already connected by confirmed matches, since transitivity implies the answer.
They differ in one parameter. With `high_threshold` set, matches scoring above it
are accepted without review (thresholded review) and count as connections for the
transitivity skip; without it, every candidate is reviewed (manual review).

Evaluation is delegated to ClusterValidator, so these baselines report exactly what
every other method reports, in the same format.
"""

import logging
import os
import time
from collections import defaultdict

import networkx as nx
import numpy as np

from beta_stability.util.tools import resolve_checkpoints, write_json, CHECKPOINT_LABEL

logger = logging.getLogger("beta_stability")


class ReviewAlgorithm:
    """Top-k candidates reviewed sequentially, with an optional auto-accept threshold."""

    def __init__(self, config, common_data):
        self.config = config
        self.common_data = common_data

        # None => review every candidate (manual review). A value => accept above it
        # without review (thresholded review).
        self.threshold = config.get('high_threshold')
        self.topk = config.get('topk', 10)
        self.max_reviews = config.get('max_reviews', None)
        self.clustering_method = config.get('clustering_method', 'connected_components')
        self.review_batch_size = config.get('review_batch_size', 100)
        # Evaluation is periodic: every `eval_every_reviews` reviews (default:
        # review_batch_size), plus once at the end, plus any exact checkpoints.
        self.eval_every = int(config.get('eval_every_reviews', self.review_batch_size))
        self._last_eval_reviews = None
        self._queue_pos = 0
        self.required_outer_iterations = 0   # read by run.py: one review per step

        self.node2uuid = common_data['node2uuid']
        self._checkpoints = set(resolve_checkpoints(config.get('report_at_reviews', ['N']),
                                                    len(self.node2uuid)))
        self.verifier_name = common_data['verifier_name']
        self.embeddings = common_data['embeddings_dict'][self.verifier_name]
        self.validator = common_data.get('cluster_validator')

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.node2uuid.keys())

        self.auto_accepted = []
        self.human_reviewed = []
        self.positive_edges = []
        self.negative_edges = []
        self.reviewed_pairs = set()
        self.num_reviews = 0
        self.finished = False

        self.uncertain_edges = []
        self.current_batch_idx = 0
        # Skip a pair once a confirmed path already connects its two annotations.
        self.skip_redundant = bool(config.get('stop_at_match', True))
        self._parent = {}
        self.remaining = None

        self.stats = {
            'num_nodes': len(self.node2uuid),
            'num_auto_accepted': 0,
            'num_human_reviews': 0,
            'num_human_positive': 0,
            'num_human_negative': 0,
            'start_time': time.time(),
            'review_history': []
        }

        logger.info("Initialized Review Algorithm")
        logger.info(f"  Auto-accept threshold: "
                    f"{'none (every candidate reviewed)' if self.threshold is None else self.threshold}")
        logger.info(f"  Top-k: {self.topk}, Max reviews: {self.max_reviews}")
        logger.info(f"  Number of nodes: {len(self.node2uuid)}")

    def _candidates(self):
        """Top-k candidate pairs, split into auto-accepted and to-review.

        Returns (auto_accepted, to_review), both lists of (n1, n2, score, verifier).
        With no threshold every candidate goes to review.
        """
        logger.info("Getting candidate edges...")
        all_edges = self.embeddings.get_edges(topk=self.topk, target_edges=0,
                                              target_proportion=None)
        edge_set = set()
        node_neighbor_count = defaultdict(int)
        cands = []
        for n1, n2, score in all_edges:
            if node_neighbor_count[n1] >= self.topk and node_neighbor_count[n2] >= self.topk:
                continue
            edge = tuple(sorted([n1, n2]))
            if edge in edge_set:
                continue
            edge_set.add(edge)
            node_neighbor_count[n1] += 1
            node_neighbor_count[n2] += 1
            cands.append((n1, n2, score))

        auto_accepted, to_review = [], []
        for n1, n2, score in cands:
            e = (n1, n2, score, self.verifier_name)
            (auto_accepted if (self.threshold is not None and score > self.threshold)
             else to_review).append(e)
        # Review the most-likely matches first (precision-first, recall-building).
        to_review.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Candidate edges: {len(cands)} total, "
                    f"{len(auto_accepted)} auto-accepted, {len(to_review)} for review")
        return auto_accepted, to_review

    def _find(self, x):
        p = self._parent
        r = x
        while p.get(r, r) != r:
            r = p[r]
        while p.get(x, x) != r:
            x, p[x] = p.get(x, x), r
        return r

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[ra] = rb

    def step(self, edge_responses):
        """Apply any human answers, then return the next single pair to review."""
        if edge_responses and edge_responses[0][3] != self.verifier_name:
            self._process_human_responses(edge_responses)

        if not self.uncertain_edges and not edge_responses:
            auto_accepted, to_review = self._candidates()
            for n1, n2, score, _ in auto_accepted:
                self.auto_accepted.append((n1, n2))
                self.positive_edges.append((n1, n2))
                self.graph.add_edge(n1, n2, weight=score)
                self.stats['num_auto_accepted'] += 1
                if self.skip_redundant:
                    self._union(n1, n2)   # an accepted match joins the two clusters

            self.uncertain_edges = to_review
            self.remaining = list(to_review)
            self.required_outer_iterations = min(
                len(to_review), self.max_reviews or len(to_review)) + 2
            # Score the graph before any review so the curve starts at x=0 like every
            # other method: the empty graph for manual review, the auto-accepted edges
            # alone for thresholded review.
            self._evaluate()

        if self.max_reviews and self.num_reviews >= self.max_reviews:
            logger.info(f"Reached maximum review limit of {self.max_reviews}")
            self._finish()
            return []

        # One pair per step, so every earlier answer is applied before the next check.
        # A pair already reviewed, or already implied by confirmed matches, is dropped;
        # clusters only ever merge, so a dropped pair stays implied.
        while self._queue_pos < len(self.remaining):
            edge = self.remaining[self._queue_pos]
            self._queue_pos += 1
            n1, n2 = edge[0], edge[1]
            if tuple(sorted([n1, n2])) in self.reviewed_pairs:
                continue
            if self.skip_redundant and self._find(n1) == self._find(n2):
                continue
            return [edge]
        logger.info("All candidate pairs have been reviewed or implied")
        self._finish()
        return []

    def _finish(self):
        """Stop requesting reviews; evaluate the final state unless the last periodic
        evaluation already covered it."""
        if not self.finished and self._last_eval_reviews != self.stats['num_human_reviews']:
            self._evaluate()
        self.finished = True

    def _process_human_responses(self, edge_responses):
        """
        Process human review responses for uncertain edges.

        Args:
            edge_responses: List of (n1, n2, score, source) tuples
        """
        batch_positive = 0
        batch_negative = 0

        for n1, n2, score, source in edge_responses:
            self.num_reviews += 1
            self.stats['num_human_reviews'] += 1
            self.human_reviewed.append((n1, n2, score))
            self.reviewed_pairs.add(tuple(sorted([n1, n2])))

            if score >= 0.5:  # Positive match
                self.positive_edges.append((n1, n2))
                self.graph.add_edge(n1, n2, weight=score)
                batch_positive += 1
                self.stats['num_human_positive'] += 1
                if self.skip_redundant:
                    self._union(n1, n2)
            else:  # Negative match
                self.negative_edges.append((n1, n2))
                # human verdict overrides the classifier: if this edge was
                # auto-accepted (uncertainty mode places s>tau edges in the graph),
                # remove it. Harmless in band modes, where reviewed edges are never
                # auto-added.
                if self.graph.has_edge(n1, n2):
                    self.graph.remove_edge(n1, n2)
                batch_negative += 1
                self.stats['num_human_negative'] += 1
            # exact checkpoint: this answer is fully applied (edge added or removed, clusters merged)
            if self.stats['num_human_reviews'] in self._checkpoints:
                self._evaluate(checkpoint=True)

        # Log batch statistics
        if edge_responses:
            # Update review history
            self.stats['review_history'].append({
                'batch_size': len(edge_responses),
                'positive': batch_positive,
                'negative': batch_negative,
                'total_reviews': self.stats['num_human_reviews'],
                'timestamp': time.time() - self.stats['start_time']
            })

            # Periodic statistics and evaluation (one review per step)
            if self.stats['num_human_reviews'] % self.eval_every == 0:
                logger.info(f"Total human reviews so far: {self.stats['num_human_reviews']} "
                           f"({self.stats['num_human_positive']} positive, "
                           f"{self.stats['num_human_negative']} negative)")
                self._evaluate()

    def _evaluate(self, checkpoint=False):
        """Report the current clustering through ClusterValidator.

        A checkpoint is logged only; it never enters the run's evaluation history.
        """
        if self.validator is None:
            return
        clustering, node2cid = {}, {}
        for cid, component in enumerate(nx.connected_components(self.graph)):
            clustering[str(cid)] = set(component)
            for node in component:
                node2cid[node] = str(cid)
        next_cid = len(clustering)
        for node in self.node2uuid:
            if node not in node2cid:                      # isolated: its own cluster
                clustering[str(next_cid)] = {node}
                node2cid[node] = str(next_cid)
                next_cid += 1
        self.validator.incremental_stats(
            self.stats['num_human_reviews'], clustering, node2cid,
            self.validator.gt_clustering, self.validator.gt_node2cid,
            CHECKPOINT_LABEL if checkpoint else 'Basic stats')
        if not checkpoint:
            self._last_eval_reviews = self.stats['num_human_reviews']

    def is_finished(self):
        """Check if algorithm has finished reviewing all edges."""
        return self.finished

    def show_stats(self):
        elapsed = time.time() - self.stats['start_time']
        logger.info("Review Algorithm Statistics:")
        logger.info("=" * 60)
        logger.info(f"Threshold: {self.threshold if self.threshold is not None else 'none'}")
        logger.info(f"Auto-accepted edges: {self.stats['num_auto_accepted']}")
        logger.info(f"Human reviews: {self.stats['num_human_reviews']} "
                    f"({self.stats['num_human_positive']} positive, "
                    f"{self.stats['num_human_negative']} negative)")
        logger.info(f"Elapsed: {elapsed:.1f}s")

    def get_clustering(self):
        """
        Get the final clustering from the constructed graph.

        Returns:
            tuple: (clustering, node2cid, graph_dict) where:
                - clustering: dict mapping cluster_id to list of node_ids
                - node2cid: dict mapping node_id to cluster_id
                - graph_dict: dict representation of the graph
        """
        logger.info(f"Generating clustering using method: {self.clustering_method}")

        if self.clustering_method == 'connected_components':
            # Use connected components for clustering
            components = list(nx.connected_components(self.graph))

            clustering = {}
            node2cid = {}

            for cid, component in enumerate(components):
                clustering[str(cid)] = list(component)
                for node in component:
                    node2cid[node] = str(cid)

            logger.info(f"Found {len(components)} connected components")

        elif self.clustering_method == 'community':
            # Use community detection (Louvain)
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(self.graph)

                clustering = defaultdict(list)
                node2cid = {}

                for node, cid in partition.items():
                    clustering[str(cid)].append(node)
                    node2cid[node] = str(cid)

                clustering = dict(clustering)
                logger.info(f"Found {len(clustering)} communities")

            except ImportError:
                logger.warning("python-louvain not installed, falling back to connected components")
                self.clustering_method = 'connected_components'
                return self.get_clustering()

        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        # Add singleton clusters for isolated nodes
        for node in self.node2uuid.keys():
            if node not in node2cid:
                cid = str(len(clustering))
                clustering[cid] = [node]
                node2cid[node] = cid

        # Create graph dict representation
        graph_dict = {
            'nodes': list(self.graph.nodes()),
            'edges': [(int(u), int(v), float(data.get('weight', 1.0)))
                     for u, v, data in self.graph.edges(data=True)],
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges()
        }

        return clustering, node2cid, graph_dict

    def save_results(self, output_path):
        """
        Save algorithm results to files.

        Args:
            output_path: Directory to save results
        """
        os.makedirs(output_path, exist_ok=True)

        # Get clustering
        clustering, node2cid, graph_dict = self.get_clustering()

        # Save clustering
        write_json(clustering, os.path.join(output_path, 'clustering.json'))
        write_json(node2cid, os.path.join(output_path, 'node2cid.json'))
        write_json(self.node2uuid, os.path.join(output_path, 'node2uuid_file.json'))

        # Save graph
        write_json(graph_dict, os.path.join(output_path, 'graph.json'))

        # Save statistics
        self.stats['end_time'] = time.time()
        self.stats['total_time'] = self.stats['end_time'] - self.stats['start_time']
        self.stats['num_clusters'] = len(clustering)
        self.stats['num_edges'] = len(self.positive_edges)

        # Calculate efficiency metrics
        total_evaluated = (self.stats['num_auto_accepted'] +
                          self.stats['num_human_reviews'])
        self.stats['human_review_rate'] = (self.stats['num_human_reviews'] /
                                           max(1, total_evaluated))
        self.stats['auto_decision_rate'] = 1.0 - self.stats['human_review_rate']

        # Calculate cluster size distribution
        cluster_sizes = [len(nodes) for nodes in clustering.values()]
        self.stats['cluster_size_distribution'] = {
            'min': min(cluster_sizes) if cluster_sizes else 0,
            'max': max(cluster_sizes) if cluster_sizes else 0,
            'mean': np.mean(cluster_sizes) if cluster_sizes else 0,
            'median': np.median(cluster_sizes) if cluster_sizes else 0,
            'num_singletons': sum(1 for s in cluster_sizes if s == 1)
        }

        write_json(self.stats, os.path.join(output_path, 'review_stats.json'))

        logger.info(f"Results saved to {output_path}")
        logger.info(f"Final statistics: {self.stats['num_human_reviews']} human reviews, "
                   f"{self.stats['num_auto_accepted']} auto-accepted, "
                   f"{len(clustering)} clusters")

        return clustering, node2cid, graph_dict
