"""
HDBSCAN clustering algorithm implementation following the standard algorithm interface.
"""

import numpy as np
import hdbscan
import networkx as nx
import logging
import beta_stability.util.cluster_tools as ct

logger = logging.getLogger("beta_stability")


class HDBSCANAlgorithm:
    """
    HDBSCAN clustering algorithm that follows the standard step() interface.
    Non-iterative algorithm that performs clustering immediately.
    """

    def __init__(self, config, all_nodes, embeddings_dict):
        """
        Initialize HDBSCAN algorithm and perform clustering immediately.

        Args:
            config: Configuration dictionary with HDBSCAN parameters
            all_nodes: List of all node IDs to cluster
            embeddings_dict: Dictionary of embeddings by verifier name
        """
        self.config = config
        self.all_nodes = all_nodes
        self.embeddings_dict = embeddings_dict
        self.finished = False
        self.clustering = {}
        self.node2cid = {}
        self.G = nx.Graph()

        # Get HDBSCAN-specific config
        hdbscan_config = config.get('hdbscan', {})
        self.min_cluster_size = hdbscan_config.get('min_cluster_size', 2)
        self.min_samples = hdbscan_config.get('min_samples', None)
        self.metric = hdbscan_config.get('metric', 'euclidean')
        self.cluster_selection_method = hdbscan_config.get('cluster_selection_method', 'eom')
        self.cluster_selection_epsilon = hdbscan_config.get('cluster_selection_epsilon', 0.0)
        self.allow_single_cluster = hdbscan_config.get('allow_single_cluster', False)

        # Get embeddings to use (default to first available verifier)
        self.verifier_name = hdbscan_config.get('verifier_name')
        if not self.verifier_name:
            # Use first non-human verifier
            for name in embeddings_dict.keys():
                if name != 'human':
                    self.verifier_name = name
                    break

        if not self.verifier_name:
            raise ValueError("No verifier embeddings available for HDBSCAN")

        logger.info(f"HDBSCAN using verifier: {self.verifier_name}")
        logger.info(f"HDBSCAN parameters: min_cluster_size={self.min_cluster_size}, "
                   f"min_samples={self.min_samples}, method={self.cluster_selection_method}, "
                   f"epsilon={self.cluster_selection_epsilon}, allow_single={self.allow_single_cluster}")

        # Perform clustering immediately
        self._perform_clustering()

    def _perform_clustering(self):
        """Perform HDBSCAN clustering on initialization."""
        # Get embeddings for all nodes
        verifier_embeddings = self.embeddings_dict[self.verifier_name]

        # Extract embedding vectors for our nodes
        embedding_vectors = []
        valid_nodes = []

        # Check if it's a function (lazy loading) and call it
        if callable(verifier_embeddings):
            verifier_embeddings = verifier_embeddings()

        # Access embeddings based on the Embeddings class structure
        for node_id in self.all_nodes:
            try:
                # Find index of node_id in the embeddings ids list
                idx = verifier_embeddings.ids.index(node_id)
                embedding = verifier_embeddings.embeddings[idx]
                embedding_vectors.append(embedding)
                valid_nodes.append(node_id)
            except (ValueError, IndexError, AttributeError):
                logger.warning(f"No embedding found for node {node_id}")

        if not embedding_vectors:
            logger.warning("No valid embeddings found, creating singleton clusters")
            self._create_singleton_clusters()
            return

        # Handle single node case - HDBSCAN needs at least 2 points
        if len(embedding_vectors) == 1:
            logger.info(f"Only one node found, creating single cluster")
            self.clustering = {0: {valid_nodes[0]}}
            self.node2cid = ct.build_node_to_cluster_mapping(self.clustering)
            self._build_graph()
            self.finished = True
            return

        embedding_matrix = np.array(embedding_vectors)
        logger.info(f"Clustering {len(valid_nodes)} nodes with embeddings of shape {embedding_matrix.shape}")

        # Apply HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            allow_single_cluster=self.allow_single_cluster
        )
        labels = clusterer.fit_predict(embedding_matrix)

        # Build clustering dictionary
        self.clustering = {}
        next_singleton_id = 0

        # First, find the max cluster label to continue numbering from there
        max_label = max(labels) if len(labels) > 0 and max(labels) >= 0 else -1
        next_singleton_id = max_label + 1

        for node_id, label in zip(valid_nodes, labels):
            # HDBSCAN uses -1 for noise points, we'll put them in singleton clusters
            if label == -1:
                cluster_id = next_singleton_id
                self.clustering[cluster_id] = {node_id}
                next_singleton_id += 1
            else:
                cluster_id = label
                if cluster_id not in self.clustering:
                    self.clustering[cluster_id] = set()
                self.clustering[cluster_id].add(node_id)

        # Add any nodes without embeddings as singletons
        embedded_nodes = set(valid_nodes)
        for node_id in self.all_nodes:
            if node_id not in embedded_nodes:
                cluster_id = next_singleton_id
                self.clustering[cluster_id] = {node_id}
                next_singleton_id += 1

        # Build node to cluster ID mapping
        self.node2cid = ct.build_node_to_cluster_mapping(self.clustering)

        # Build graph representation
        self._build_graph()

        # Count actual clusters (with more than 1 member) vs singletons
        n_clusters = len([c for c, members in self.clustering.items() if len(members) > 1])
        n_singletons = len([c for c, members in self.clustering.items() if len(members) == 1])
        logger.info(f"HDBSCAN found {n_clusters} clusters and {n_singletons} singleton/noise points")

        self.finished = True

    def _create_singleton_clusters(self):
        """Create singleton clusters for all nodes when no embeddings are available."""
        for i, node_id in enumerate(self.all_nodes):
            cluster_id = i
            self.clustering[cluster_id] = {node_id}

        self.node2cid = ct.build_node_to_cluster_mapping(self.clustering)
        self.G.add_nodes_from(self.all_nodes)
        self.finished = True

    def _build_graph(self):
        """Build graph representation from clustering."""
        self.G = nx.Graph()
        self.G.add_nodes_from(self.all_nodes)

        # Add edges between nodes in the same cluster
        for cluster_nodes in self.clustering.values():
            nodes_list = list(cluster_nodes)
            for i in range(len(nodes_list)):
                for j in range(i + 1, len(nodes_list)):
                    self.G.add_edge(nodes_list[i], nodes_list[j])

    def step(self, new_edges):
        """
        Process new edges (compatibility with iterative interface).
        HDBSCAN is non-iterative, so this just returns empty list.

        Args:
            new_edges: List of new edges (ignored for HDBSCAN)

        Returns:
            list: Empty list (no human review needed)
        """
        return []

    def is_finished(self):
        """
        Check if algorithm is finished.
        HDBSCAN finishes immediately after initialization.

        Returns:
            bool: Always True after initialization
        """
        return self.finished

    def get_clustering(self):
        """
        Get the final clustering result.

        Returns:
            tuple: (clustering_dict, node2cid_dict, graph)
        """
        # Convert sets to lists for JSON serialization
        clustering_dict = {
            str(cid): list(nodes) for cid, nodes in self.clustering.items()
        }

        node2cid_dict = {
            str(node): str(cid) for node, cid in self.node2cid.items()
        }

        # Convert graph to edge list format
        graph_dict = {
            'nodes': list(self.G.nodes()),
            'edges': list(self.G.edges())
        }

        return clustering_dict, node2cid_dict, graph_dict

    def show_stats(self):
        """Display clustering statistics."""
        n_clusters = len([c for c, members in self.clustering.items() if len(members) > 1])
        n_singletons = len([c for c, members in self.clustering.items() if len(members) == 1])
        n_total_nodes = len(self.all_nodes)

        logger.info("HDBSCAN Clustering Statistics:")
        logger.info(f"  Total nodes: {n_total_nodes}")
        logger.info(f"  Clusters found: {n_clusters}")
        logger.info(f"  Singleton/noise points: {n_singletons}")

        # Show cluster size distribution for non-singleton clusters
        cluster_sizes = [len(nodes) for cid, nodes in self.clustering.items()
                        if len(nodes) > 1]
        if cluster_sizes:
            logger.info(f"  Cluster sizes: min={min(cluster_sizes)}, "
                       f"max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.2f}")