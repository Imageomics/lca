========================
Beta Stability
========================

Stability-driven clustering with active review for wildlife re-identification.
Given re-identification embeddings and an annotation file, the algorithm builds
an edge graph, deactivates positive edges to reach an initial ``0``-stable
state (Phase 0), and then queries a human reviewer for the least-stable edges
until either a target ``α``-stability is reached or the review budget is spent.

The algorithm selects between top-K and k-means-based initial verifiers
automatically, based on a silhouette test on the primary embedding at
the coupon-collector cluster count ``K̂``.


Requirements
------------

* Python ≥ 3.11
* Dependencies managed via ``uv`` (preferred) or ``pip``.


Installation
------------

Using `uv <https://docs.astral.sh/uv/>`_::

    uv sync

Using ``pip`` (for those not on ``uv``)::

    pip install -r requirements.txt
    pip install -e .


Quick start
-----------

Run against a species using the reference method config and your data config::

    python -m beta_stability.run_clustering_with_save \\
        --base_config examples/beta_stability_config.yaml \\
        --config path/to/your_data_config.yaml

The reference method config is at
`examples/beta_stability_config.yaml <examples/beta_stability_config.yaml>`_.
Data configs specify the annotation JSON, embedding pickle, output path, and
any species-specific preprocessing overrides — the ``data:`` section in the
reference config documents the schema.

Inputs the data config must provide:

* ``annotation_file`` — a JSON with one entry per annotation (``uuid``,
  ``name``, ``viewpoint``, etc.). ``name_keys`` in the config controls which
  fields form the ground-truth identity.
* ``embedding_file`` — a pickle of ``(embedding_matrix, uuid_list)`` from the
  re-identification model.
* ``output_path`` — directory where clustering results are written.

Outputs written to ``output_path``:

* ``clustering.json`` — clusters as ``{cluster_id: [node_id, ...]}``.
* ``node2cid.json`` — inverse mapping ``{node_id: cluster_id}``.
* ``node2uuid_file.json`` — node index → annotation UUID.
* ``beta_stability_annots.json`` — annotations enriched with the assigned
  ``cluster_id``.
* Evaluation metrics (Hungarian F1, precision, recall) if ground-truth is
  present.


How it works
------------

Beta Stability treats clustering as a series of local decisions about which
edges in a graph are trustworthy positives, then iteratively strengthens the
ones a human confirms. The loop:

1. **Init graph** — build initial edges from an embedding-based verifier. The
   auto-selector runs k-means at ``K = K̂`` on the primary embedding; if the
   resulting silhouette is above the threshold, pairwise cosine is used;
   otherwise the algorithm switches to k-means-based edges.
2. **Phase 0** — deactivate positive edges until the graph is
   ``0``-stable (every edge decision is consistent under the current
   classifier's confidence). No human input yet.
3. **Active review** — iteratively pick the least-stable edges, send them
   to a human reviewer (or a simulated one drawn from ground truth for
   experiments), and re-stabilize after each verdict.
4. **Terminate** when the target stability ``α`` is reached or the review
   budget is exhausted.

The reference implementation lives in
`beta_stability/stability_algorithm.py <beta_stability/stability_algorithm.py>`_.


Usage example
-------------

End-to-end run against a whaleshark dataset.

**1. Prepare data files** (outside the repo, on your data store):

* ``annotations_whaleshark.json`` — a coco-style annotation file with a
  ``name`` field for identity and (optionally) a ``viewpoint`` field. Each
  annotation has a unique ``uuid``.
* ``embeddings_whaleshark.pickle`` — a tuple
  ``(embedding_matrix, uuid_list)`` from your re-identification model. The
  ``uuid_list`` must match the annotation file.

**2. Write a data config** (``configs/whaleshark_data.yaml``):

.. code-block:: yaml

    exp_name: whaleshark
    species: Whale shark
    data:
      viewpoint_list: [left]        # keep only left-view annotations
      name_keys: [name]             # ground-truth identity comes from this field
      annotation_file: /data/whaleshark/annotations_whaleshark.json
      embedding_file: /data/whaleshark/embeddings_whaleshark.pickle
      output_path: /results/whaleshark/beta_stability/

**3. Run**::

    python -m beta_stability.run_clustering_with_save \
        --base_config examples/beta_stability_config.yaml \
        --config configs/whaleshark_data.yaml

**4. What you see in the log** (excerpt)::

    AUTO VERIFIER: primary='miewid', τ=0.6590, K̂=183, 1924 positive edges over 997 nodes
    AUTO VERIFIER: silhouette at K=183 on 'miewid' = 0.0997 (threshold=0.15)
    AUTO VERIFIER: silhouette < 0.15 → embedding space not clusterable at K̂;
                   using kmeans-only verifier: ['kmeans(miewid)']
    Beta Stability Algorithm initialized
      Target alpha: 0.5
      Max human reviews: 5000
    ...
    Phase: FINISHED
    Hungarian F1 Score: 0.7609

The auto-verifier's silhouette check decides whether pairwise cosine can
recover cluster structure. Whaleshark's ``miewid`` embedding is a hard
case (silhouette ≪ 0.15), so the algorithm switches to k-means edges.

**5. Consume the results** (in ``output_path/``):

.. code-block:: python

    import json

    with open("/results/whaleshark/beta_stability/clustering.json") as f:
        clustering = json.load(f)
    # clustering: {"0": [node_id, node_id, …], "1": [...], …}

    with open("/results/whaleshark/beta_stability/node2uuid_file.json") as f:
        node2uuid = json.load(f)
    # node2uuid: {node_id: annotation_uuid, …}

    for cluster_id, nodes in clustering.items():
        uuids = [node2uuid[str(n)] for n in nodes]
        print(f"Cluster {cluster_id}: {len(uuids)} annotations")


Repository layout
-----------------

.. code-block:: text

    beta_stability/           # the package
    ├── stability_algorithm.py    # the Beta Stability algorithm
    ├── stability_graph.py        # its graph data structure
    ├── algorithm_preparation.py  # top-level orchestration + auto-verifier
    ├── classifier_system.py      # classifier + threshold framework
    ├── preprocess.py             # data loader / filters
    ├── run.py, run_clustering_with_save.py, save_clustering_results.py
    ├── embeddings/               # embedding backends (miewid, k-means, hdbscan, …)
    ├── baselines/                # alternative clustering algorithms
    │                              # (hdbscan, nis, np3_aas, manual_review, thresholded_review)
    └── util/                     # tools, cluster utilities, GMM threshold, …

    examples/
    └── beta_stability_config.yaml  # public reference method config

    pyproject.toml            # project metadata + entry points
    uv.lock                   # locked dependency resolution


Configuration essentials
------------------------

* ``algorithm_type: stability`` (default) selects Beta Stability. Alternative
  algorithm types are ``hdbscan``, ``nis``, ``np3_aas``, ``manual_review``,
  ``thresholded_review``.
* ``initial_topk: auto`` uses the coupon-collector formula
  ``K = -ln(1 - coverage) · K̂`` where ``K̂`` is the number of connected
  components in the positive-edge graph.
* ``verifier_name: auto`` enables the silhouette-based init selector.
  ``auto_verifier_silhouette_threshold`` (default ``0.15``) is the decision
  boundary — below it the algorithm switches to a k-means-only init.
* ``classifier_thresholds`` uses ``auto(fraction)`` to fit a two-component
  GMM to the pair-score histogram and pick a threshold. See the reference
  config for the schema.


Citation
--------

If you use this code or its models in your research, please cite:

.. code:: text

    TBD
