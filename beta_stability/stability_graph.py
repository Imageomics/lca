"""
Stability Graph Module for Beta Stability (stability-driven) algorithm.

Implements the graph structure and stability computations as defined in:
"LCA V2 Formulation" by Charles Stewart, January 2026

Key definitions from the PDF:
- Edge labels: positive, positive-inactive, negative, incomparable
- PCCs: Connected components using ONLY positive edges (not positive-inactive)
- MSP: Maximum strength path = path with highest minimum edge confidence
- Internal stability(u,v) = MSP(u,v) if no negative edge, else MSP(u,v) - neg_conf
- External stability(A,B) = max_neg_conf - max_pos_inactive_conf
- alpha-stable: min(internal, external) >= alpha

CRITICAL: Only positive edges can be deactivated (become positive-inactive).
Negative edges are NEVER deactivated.
"""

import os
import networkx as nx
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from beta_stability.util.tools import order_edge

logger = logging.getLogger("beta_stability")


def _build_krt(mst, idx):
    """Kruskal reconstruction tree over `mst` (edges added in DECREASING weight).
    Internal node = one merge, carrying that merge's weight and edge. Bottleneck
    of any pair = weight at their LCA, so all queries reduce to one LCA pass."""
    n = len(idx)
    edges = sorted(mst.edges(data='weight'), key=lambda t: t[2], reverse=True)
    M = 2 * n
    parent = np.full(M, -1, dtype=np.int64)
    wt = np.zeros(M, dtype=np.float64)
    eu = np.full(M, -1, dtype=np.int64); ev = np.full(M, -1, dtype=np.int64)
    uf = list(range(n)); top = list(range(n))

    def find(x):
        r = x
        while uf[r] != r: r = uf[r]
        while uf[x] != r: uf[x], x = r, uf[x]
        return r

    nxt = n
    for a, b, w in edges:
        ra, rb = find(idx[a]), find(idx[b])
        if ra == rb: continue
        p = nxt; nxt += 1
        parent[top[ra]] = p; parent[top[rb]] = p
        wt[p] = w; eu[p] = a; ev[p] = b
        uf[ra] = rb; top[rb] = p
    K = nxt
    par = parent[:K]
    depth = np.zeros(K, dtype=np.int64); root = np.arange(K)
    for v in range(K - 1, -1, -1):
        p = par[v]
        if p >= 0:
            depth[v] = depth[p] + 1; root[v] = root[p]
    LOG = max(1, int(np.ceil(np.log2(max(K, 2)))) + 1)
    up = np.empty((LOG, K), dtype=np.int64)
    up[0] = np.where(par >= 0, par, np.arange(K))
    for k in range(1, LOG):
        up[k] = up[k - 1][up[k - 1]]
    return wt, eu, ev, depth, root, up, LOG


def _bottleneck_lca(mst, idx, nu, nv):
    """Bottleneck (weakest edge weight) on the tree path for many pairs at once.

    Returns (qi, w) where qi indexes the pairs that are CONNECTED in `mst` and w
    holds their bottleneck weights. Replaces per-pair `nx.shortest_path` walks:
    one Kruskal reconstruction tree plus a vectorised binary-lifting LCA answers
    every query, and the bottleneck of a tree path is unique so the values are
    identical to walking each path.
    """
    wt, eu, ev, depth, root, up, LOG = _build_krt(mst, idx)
    live = (root[nu] == root[nv]) & (nu != nv)
    qi = np.flatnonzero(live)
    if qi.size == 0:
        return qi, np.zeros(0, dtype=np.float64)
    a = nu[qi].copy(); b = nv[qi].copy()
    swap = depth[a] < depth[b]
    a[swap], b[swap] = b[swap].copy(), a[swap].copy()
    diff = depth[a] - depth[b]
    for k in range(LOG):
        sel = ((diff >> k) & 1).astype(bool)
        if sel.any():
            a[sel] = up[k][a[sel]]
    same = a == b
    for k in range(LOG - 1, -1, -1):
        ua, ub = up[k][a], up[k][b]
        m = (~same) & (ua != ub)
        if m.any():
            a[m] = ua[m]; b[m] = ub[m]
    lca = np.where(same, a, up[0][a])
    return qi, wt[lca]


def _edges_to_cut(mst, idx, inv, nu, nv, nconf, alpha):
    """Vectorised replacement for `_bottleneck_edges` + the caller's cut loop.

    Returns {(a, b): (u, v)} in ORIGINAL node ids -- the MST edges to deactivate
    this round, keyed normalised, valued by the FIRST negative pair (in
    `negatives` order) that demands the cut, matching `to_cut.setdefault` exactly.
    `inv` maps index -> original id (eu/ev already hold original ids).
    Never materialises a per-pair dict: the cut test is one boolean reduction.
    """
    wt, eu, ev, depth, root, up, LOG = _build_krt(mst, idx)
    live = (root[nu] == root[nv]) & (nu != nv)      # still connected in the tree
    qi = np.flatnonzero(live)
    if qi.size == 0:
        return {}
    a = nu[qi].copy(); b = nv[qi].copy()
    swap = depth[a] < depth[b]
    a[swap], b[swap] = b[swap].copy(), a[swap].copy()
    diff = depth[a] - depth[b]
    for k in range(LOG):
        sel = ((diff >> k) & 1).astype(bool)
        if sel.any(): a[sel] = up[k][a[sel]]
    same = a == b
    for k in range(LOG - 1, -1, -1):
        ua, ub = up[k][a], up[k][b]
        m = (~same) & (ua != ub)
        if m.any(): a[m] = ua[m]; b[m] = ub[m]
    lca = np.where(same, a, up[0][a])

    need = (wt[lca] - nconf[qi]) < alpha          # the cut condition, vectorised
    sel = qi[need]
    if sel.size == 0:
        return {}
    ea, eb = eu[lca[need]], ev[lca[need]]
    lo = np.minimum(ea, eb); hi = np.maximum(ea, eb)
    order = np.argsort(sel, kind='stable')        # first-in-negatives-order wins
    lo, hi, sel = lo[order], hi[order], sel[order]
    key = lo.astype(np.int64) * (int(max(hi.max(), lo.max())) + 1) + hi
    _, first = np.unique(key, return_index=True)
    pu = inv[nu[sel[first]]]; pv = inv[nv[sel[first]]]
    return {(int(lo[i]), int(hi[i])): (int(x), int(y))
            for i, x, y in zip(first, pu, pv)}

def _bottleneck_edges(mst, negatives):
    """Weakest edge on the max-strength (tree) path for every connected negative
    pair, in one Kruskal pass instead of a per-pair tree walk.

    `mst` is a maximum spanning forest (a tree per component). Adding its edges in
    DECREASING weight, the edge that first unites the components of u and v is
    exactly the minimum-weight edge on their unique path -- i.e. MSP(u,v) and the
    edge to cut. Small-to-large endpoint matching keeps it O(E log E + K log K),
    so a giant PCC no longer costs one full tree traversal per negative per round.

    Returns {(u, v): (min_conf, min_edge)} for pairs connected in `mst`; pairs in
    different components are omitted (already separated).
    """
    from collections import defaultdict
    q_at = defaultdict(list)                      # node -> [(query_id, other_endpoint)]
    for i, (u, v, _n) in enumerate(negatives):
        if u != v and mst.has_node(u) and mst.has_node(v):
            q_at[u].append((i, v))
            q_at[v].append((i, u))
    if not q_at:
        return {}
    parent = {n: n for n in mst.nodes()}
    live = {n: ([n] if n in q_at else []) for n in mst.nodes()}   # query endpoints per root

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    answered = {}
    for a, b, w in sorted(mst.edges(data='weight'), key=lambda t: t[2], reverse=True):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if len(live[ra]) > len(live[rb]):
            ra, rb = rb, ra                       # ra = smaller side
        for node in live[ra]:
            for qi, other in q_at[node]:
                if qi not in answered and find(other) == rb:
                    answered[qi] = (w, (a, b))
        live[rb].extend(live[ra])
        parent[ra] = rb
    return {(negatives[qi][0], negatives[qi][1]): val for qi, val in answered.items()}


class EdgeLabel(Enum):
    """Edge label types."""
    POSITIVE = "positive"
    POSITIVE_INACTIVE = "positive-inactive"
    NEGATIVE = "negative"


_LABEL_CODE = {EdgeLabel.POSITIVE: 0, EdgeLabel.POSITIVE_INACTIVE: 1, EdgeLabel.NEGATIVE: 2}
_CODE_NEGATIVE = _LABEL_CODE[EdgeLabel.NEGATIVE]


@dataclass
class EdgeData:
    """Edge attributes."""
    label: EdgeLabel
    confidence: float  # [0, 1]
    score: float = 0.0  # Original embedding score
    ranker: str = ""  # Source of classification
    deactivator: Optional[Tuple[int, int]] = None  # Edge that caused deactivation (for positive-inactive)


@dataclass
class StabilityCandidate:
    """A candidate for human review."""
    candidate_type: str  # "INTERNAL" or "EXTERNAL"
    stability: float
    review_edge: Tuple[int, int]  # Edge to present to human
    structural_impact: float = 1.0  # min(subtree_left, subtree_right) for splits, min(|pcc_a|, |pcc_b|) for merges
    # For INTERNAL: the unstable pair
    node_pair: Optional[Tuple[int, int]] = None
    pcc_id: Optional[int] = None
    # For EXTERNAL: the two PCCs
    pcc_pair: Optional[Tuple[int, int]] = None


class StabilityGraph:
    """
    Graph structure for Beta Stability stability-driven clustering.

    Per PDF spec:
    - Only POSITIVE edges can be deactivated (become positive-inactive)
    - Negative edges are NEVER deactivated
    - PCCs use only active positive edges
    """

    def __init__(self):
        self.G = nx.Graph()
        self._pcc_cache_valid = False
        self._pccs: List[Set[int]] = []
        self._node_to_pcc: Dict[int, int] = {}

        # MST forest - built explicitly when needed, no caching
        self._mst_forest: Optional[nx.Graph] = None

        # Incremental edge label counts (avoids iterating all edges)
        self._edge_counts = {'positive': 0, 'positive_inactive': 0, 'negative': 0}

        # Track positive-inactive edges for fast external stability computation
        self._pos_inactive_edges: Set[Tuple[int, int]] = set()
        # Columnar mirror of the edge table, built lazily and kept in step by
        # _touch_edge() at every label/confidence write. Consumers that would
        # otherwise walk the whole adjacency in Python (~3 us per edge visit, i.e.
        # ~90 s per pass over 14.5M edges) read these instead. `None` = not built;
        # any structural change (a NEW edge) drops it so it is rebuilt on demand.
        self._earr_idx: Dict[Tuple[int, int], int] = {}
        self._earr_u = None
        self._earr_v = None
        self._earr_lab = None
        self._earr_conf = None
        # rank of hi within G._adj[lo] for each mirror edge (lo < hi); built once,
        # structure-dependent only, dropped together with the mirror
        self._earr_rank = None

    def add_node(self, node_id: int):
        if node_id not in self.G:
            self.G.add_node(node_id)
            self._pcc_cache_valid = False

    def add_edge(self, u: int, v: int, label: EdgeLabel, confidence: float,
                 score: float = 0.0, ranker: str = ""):
        """Add or update an edge."""
        self.add_node(u)
        self.add_node(v)
        confidence = np.clip(confidence, 0, 1)

        if self.G.has_edge(u, v):
            old_data = self.G[u][v].get('data')
            if old_data:
                # Decrement old label count
                self._decrement_edge_count(old_data.label)
                # If label or confidence changed significantly, may need to reactivate
                if old_data.label == EdgeLabel.POSITIVE_INACTIVE:
                    if label == EdgeLabel.POSITIVE:
                        # Reactivating - clear deactivator
                        new_data = EdgeData(label=label, confidence=confidence,
                                          score=score, ranker=ranker, deactivator=None)
                    else:
                        new_data = EdgeData(label=label, confidence=confidence,
                                          score=score, ranker=ranker)
                else:
                    new_data = EdgeData(label=label, confidence=confidence,
                                      score=score, ranker=ranker)
                self.G[u][v]['data'] = new_data
                self._touch_edge(u, v, new_data)
            else:
                new_data = EdgeData(label=label, confidence=confidence,
                                               score=score, ranker=ranker)
                self.G[u][v]['data'] = new_data
                self._touch_edge(u, v, new_data)
        else:
            new_data = EdgeData(label=label, confidence=confidence,
                                               score=score, ranker=ranker)
            self.G.add_edge(u, v, data=new_data)
            self._invalidate_edge_arrays()   # new edge -> mirror must be rebuilt
        # Increment new label count and maintain pos-inactive tracking
        self._increment_edge_count(label)
        key = (min(u, v), max(u, v))
        if label == EdgeLabel.POSITIVE_INACTIVE:
            self._pos_inactive_edges.add(key)
        else:
            self._pos_inactive_edges.discard(key)
        self._pcc_cache_valid = False

    def get_edge(self, u: int, v: int) -> Optional[EdgeData]:
        if self.G.has_edge(u, v):
            return self.G[u][v].get('data')
        return None

    def has_edge(self, u: int, v: int) -> bool:
        return self.G.has_edge(u, v)

    def get_confidence(self, u: int, v: int) -> float:
        if self.G.has_edge(u, v):
            data = self.G[u][v].get('data')
            return data.confidence if data else 0.0
        return 0.0

    def _increment_edge_count(self, label: EdgeLabel):
        if label == EdgeLabel.POSITIVE:
            self._edge_counts['positive'] += 1
        elif label == EdgeLabel.POSITIVE_INACTIVE:
            self._edge_counts['positive_inactive'] += 1
        elif label == EdgeLabel.NEGATIVE:
            self._edge_counts['negative'] += 1

    def _decrement_edge_count(self, label: EdgeLabel):
        if label == EdgeLabel.POSITIVE:
            self._edge_counts['positive'] -= 1
        elif label == EdgeLabel.POSITIVE_INACTIVE:
            self._edge_counts['positive_inactive'] -= 1
        elif label == EdgeLabel.NEGATIVE:
            self._edge_counts['negative'] -= 1

    # ------------------------------------------------------------------ arrays
    def _ensure_edge_arrays(self):
        """Materialise the columnar edge mirror (one O(E) pass, then maintained)."""
        if self._earr_u is not None:
            return
        items = []
        for u, v, attr in self.G.edges(data=True):
            d = attr.get('data')
            if d is None:
                continue
            items.append((u, v, int(d.label.value_code) if hasattr(d.label, 'value_code')
                          else _LABEL_CODE[d.label], float(d.confidence)))
        n = len(items)
        self._earr_u = np.empty(n, dtype=np.int64)
        self._earr_v = np.empty(n, dtype=np.int64)
        self._earr_lab = np.empty(n, dtype=np.int8)
        self._earr_conf = np.empty(n, dtype=np.float64)
        self._earr_idx = {}
        for i, (u, v, lab, conf) in enumerate(items):
            self._earr_u[i] = u; self._earr_v[i] = v
            self._earr_lab[i] = lab; self._earr_conf[i] = conf
            self._earr_idx[(u, v) if u <= v else (v, u)] = i

    def _invalidate_edge_arrays(self):
        self._earr_u = self._earr_v = self._earr_lab = self._earr_conf = None
        self._earr_idx = {}
        self._earr_rank = None

    def _touch_edge(self, u, v, edge_data):
        """Propagate one edge's label/confidence into the mirror. No-op until built."""
        if self._earr_u is None:
            return
        i = self._earr_idx.get((u, v) if u <= v else (v, u))
        if i is None:
            self._invalidate_edge_arrays()      # new edge -> rebuild on next use
            return
        self._earr_lab[i] = _LABEL_CODE[edge_data.label]
        self._earr_conf[i] = float(edge_data.confidence)

    def verify_edge_arrays(self):
        """Rebuild from the graph and compare -- catches any unhooked write site."""
        keep = (self._earr_u, self._earr_v, self._earr_lab, self._earr_conf, self._earr_idx)
        self._invalidate_edge_arrays()
        self._ensure_edge_arrays()
        fresh = (self._earr_u, self._earr_v, self._earr_lab, self._earr_conf, self._earr_idx)
        self._earr_u, self._earr_v, self._earr_lab, self._earr_conf, self._earr_idx = keep
        if keep[0] is None:
            return True
        ok = (len(keep[4]) == len(fresh[4]))
        if ok:
            for k, i in fresh[4].items():
                j = keep[4].get(k)
                if j is None or fresh[2][i] != keep[2][j] or abs(fresh[3][i] - keep[3][j]) > 1e-12:
                    ok = False
                    break
        return ok

    def deactivate_positive(self, edge: Tuple[int, int], deactivator: Tuple[int, int] = None):
        """
        Deactivate a POSITIVE edge (make it positive-inactive).
        Per PDF: Only positive edges can be deactivated. Negative edges are never deactivated.
        """
        u, v = edge
        if not self.G.has_edge(u, v):
            return

        edge_data = self.G[u][v].get('data')
        if not edge_data:
            return

        # Only deactivate POSITIVE edges
        if edge_data.label != EdgeLabel.POSITIVE:
            logger.warning(f"Cannot deactivate non-positive edge ({u}, {v}) with label {edge_data.label}")
            return

        self._edge_counts['positive'] -= 1
        self._edge_counts['positive_inactive'] += 1
        edge_data.label = EdgeLabel.POSITIVE_INACTIVE
        edge_data.deactivator = deactivator
        self._touch_edge(u, v, edge_data)
        self._pos_inactive_edges.add((min(u, v), max(u, v)))
        self._pcc_cache_valid = False



    def flip_negative_to_positive(self, u: int, v: int, confidence: float):
        """Contract: flip a NEGATIVE edge to POSITIVE at a chosen confidence
        (merges the two PCCs). Used by the merge-dual."""
        if not self.G.has_edge(u, v):
            return
        ed = self.G[u][v].get('data')
        if not ed or ed.label != EdgeLabel.NEGATIVE:
            return
        self._decrement_edge_count(EdgeLabel.NEGATIVE)
        ed.label = EdgeLabel.POSITIVE
        ed.confidence = confidence
        self._touch_edge(u, v, ed)
        self._increment_edge_count(EdgeLabel.POSITIVE)
        self._pcc_cache_valid = False

    def restore_to_negative(self, u: int, v: int, confidence: float):
        """Restore an edge (that we flipped and that the re-cut demoted) back to
        NEGATIVE, so a human's confirmed 'different' is never overruled."""
        if not self.G.has_edge(u, v):
            return
        ed = self.G[u][v].get('data')
        if not ed or ed.label == EdgeLabel.NEGATIVE:
            return
        self._decrement_edge_count(ed.label)
        ed.label = EdgeLabel.NEGATIVE
        ed.confidence = confidence
        self._touch_edge(u, v, ed)
        self._increment_edge_count(EdgeLabel.NEGATIVE)
        self._pos_inactive_edges.discard((min(u, v), max(u, v)))
        self._pcc_cache_valid = False


    def _invalidate_cache(self):
        """Invalidate PCC cache. MST is rebuilt explicitly when needed."""
        self._pcc_cache_valid = False

    def _positive_edges_scan_order(self):
        """(u, v, conf) arrays of POSITIVE edges in exact G.edges(data=True) order."""
        self._ensure_edge_arrays()
        if self._earr_u is None or self._earr_u.size == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z, np.zeros(0, dtype=np.float64)
        i = np.flatnonzero(self._earr_lab == _LABEL_CODE[EdgeLabel.POSITIVE])
        return self._earr_u[i], self._earr_v[i], self._earr_conf[i]

    def _ensure_pcc_cache(self):
        if not self._pcc_cache_valid:
            self._compute_pccs()
            self._pcc_cache_valid = True

    def _compute_pccs(self):
        """
        Compute PCCs using ONLY positive edges (not positive-inactive).
        Per PDF: "PCCs (positive edges only, not positive-inactive) correspond to individuals"
        """
        # Positive edges come from the columnar mirror instead of a Python scan of
        # all edges. The mirror is built by iterating G.edges(data=True) and nothing
        # ever removes an edge from G, so its positive entries are the same (u, v)
        # pairs, same orientation, same order as the old list comprehension. The
        # networkx graph below is therefore built identically, and connected
        # components come out in the same order with sets of the same iteration
        # order -- pcc ids, and anything that iterates a PCC set, are unchanged.
        # This ran on every PCC-cache invalidation, i.e. after every label change.
        _pu, _pv, _ = self._positive_edges_scan_order()
        positive_edges = list(zip(_pu.tolist(), _pv.tolist()))

        positive_graph = nx.Graph()
        positive_graph.add_nodes_from(self.G.nodes())
        positive_graph.add_edges_from(positive_edges)

        components = list(nx.connected_components(positive_graph))
        self._pccs = [set(comp) for comp in components]
        self._node_to_pcc = {}
        for pcc_id, pcc in enumerate(self._pccs):
            for node in pcc:
                self._node_to_pcc[node] = pcc_id

        if os.environ.get('BETA_SELFCHECK'):
            _ref_edges = [(u, v) for u, v, data in self.G.edges(data=True)
                          if data.get('data') and data['data'].label == EdgeLabel.POSITIVE]
            _rg = nx.Graph(); _rg.add_nodes_from(self.G.nodes()); _rg.add_edges_from(_ref_edges)
            _ref = [set(c) for c in nx.connected_components(_rg)]
            if (_ref_edges != positive_edges
                    or [list(x) for x in _ref] != [list(x) for x in self._pccs]):
                logger.error("SELFCHECK FAIL _compute_pccs")
                raise AssertionError('_compute_pccs mismatch')
            logger.info(f"SELFCHECK ok: compute_pccs {len(self._pccs)} PCCs")

    def _build_mst_forest(self):
        """
        Build MST forest for all active positive edges.
        Called explicitly at the start of phases that need MST.
        """
        # Same mirror-backed positive edge list as _compute_pccs (identical order,
        # orientation and weights), so the forest and its tie-breaking are unchanged.
        # This full scan ran at the start of every candidate-generation call.
        _pu, _pv, _pc = self._positive_edges_scan_order()
        positive_edges = [(u, v, {'weight': c})
                          for u, v, c in zip(_pu.tolist(), _pv.tolist(), _pc.tolist())]

        self._mst_forest = nx.Graph()
        self._mst_forest.add_nodes_from(self.G.nodes())
        self._mst_forest.add_edges_from(positive_edges)

        # Build MST forest (one MST per connected component)
        self._mst_forest = nx.maximum_spanning_tree(self._mst_forest, weight='weight')

        if os.environ.get('BETA_SELFCHECK'):
            _re = []
            for u, v, data in self.G.edges(data=True):
                ed = data.get('data')
                if ed and ed.label == EdgeLabel.POSITIVE:
                    _re.append((u, v, {'weight': ed.confidence}))
            _rg = nx.Graph(); _rg.add_nodes_from(self.G.nodes()); _rg.add_edges_from(_re)
            _rm = nx.maximum_spanning_tree(_rg, weight='weight')
            if list(_rm.edges(data='weight')) != list(self._mst_forest.edges(data='weight')):
                logger.error("SELFCHECK FAIL _build_mst_forest")
                raise AssertionError('_build_mst_forest mismatch')
            logger.info(f"SELFCHECK ok: mst_forest {self._mst_forest.number_of_edges()} edges")


    def get_pccs(self) -> List[Set[int]]:
        self._ensure_pcc_cache()
        return self._pccs.copy()

    def get_node_pcc(self, node: int) -> Optional[int]:
        self._ensure_pcc_cache()
        return self._node_to_pcc.get(node)

    def nodes_in_same_pcc(self, u: int, v: int) -> bool:
        self._ensure_pcc_cache()
        pcc_u = self._node_to_pcc.get(u)
        pcc_v = self._node_to_pcc.get(v)
        return pcc_u is not None and pcc_u == pcc_v

    def get_msp_strength(self, u: int, v: int) -> Optional[Tuple[float, List[int], Tuple[int, int]]]:
        """
        Get MSP (Maximum Strength Path) between two nodes in same PCC.

        Returns: (strength, path, min_edge) or None if not in same PCC.

        Per PDF: "The maximum strength path (MSP) between two vertices in same PCC
        is the maximum strength simple path."

        Computed via Maximum Spanning Tree - the path in MST is the widest path.
        """
        self._ensure_pcc_cache()

        pcc_u = self._node_to_pcc.get(u)
        pcc_v = self._node_to_pcc.get(v)

        if pcc_u is None or pcc_u != pcc_v:
            return None

        if u == v:
            return (float('inf'), [u], (u, u))

        # Build MST for this PCC using positive edges
        pcc = self._pccs[pcc_u]
        pcc_graph = nx.Graph()

        for a in pcc:
            for b in self.G.neighbors(a):
                if b in pcc and a < b:
                    edge_data = self.G[a][b].get('data')
                    if edge_data and edge_data.label == EdgeLabel.POSITIVE:
                        pcc_graph.add_edge(a, b, weight=edge_data.confidence)

        if pcc_graph.number_of_edges() == 0:
            return None

        mst = nx.maximum_spanning_tree(pcc_graph, weight='weight')

        if not mst.has_node(u) or not mst.has_node(v):
            return None

        try:
            path = nx.shortest_path(mst, u, v)
        except nx.NetworkXNoPath:
            return None

        if len(path) < 2:
            return None

        # Find minimum confidence edge on path
        min_conf = float('inf')
        min_edge = None
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            conf = mst[a][b]['weight']
            if conf < min_conf:
                min_conf = conf
                min_edge = (a, b)

        return (min_conf, path, min_edge)

    def compute_internal_stability(self, u: int, v: int) -> Optional[float]:
        """
        Compute internal stability for vertex pair (u, v) in same PCC.

        Per PDF:
        - MSP(u, v) if there is no negative edge between u and v
        - MSP(u, v) - conf(u, v) if there is a negative edge
        """
        msp_result = self.get_msp_strength(u, v)
        if msp_result is None:
            return None

        msp_strength = msp_result[0]

        # Check for negative edge between u and v
        edge_data = self.get_edge(u, v)
        if edge_data and edge_data.label == EdgeLabel.NEGATIVE:
            return msp_strength - edge_data.confidence

        return msp_strength

    def per_pcc_internal_stability(self):
        """Per-PCC internal stability, for tracking over time.

        Returns a list of (anchor_node, size, internal_stability), one per PCC.
        anchor_node = min node id (a stable-ish identity across iterations).
        internal_stability = min over within-PCC negative pairs of
        (MSP - neg_conf); None if the PCC has no internal negative (nothing can
        destabilize it). Negative value = unstable (a cut is pending).
        """
        self._ensure_pcc_cache()
        # Intra-PCC edges come from the columnar mirror instead of probing every
        # node PAIR (14.7M has_edge() calls for a 5429-node PCC), and the per-pair
        # `nx.shortest_path` walk -- one per intra-PCC negative, ~8.17M of them on
        # whale shark 2023 -- is replaced by a single vectorised bottleneck-LCA
        # pass. This runs on EVERY phase-0 iteration and throughout active review,
        # and logs nothing itself, so its cost was previously invisible in the log.
        intra = self._intra_pcc_edges()
        out = []
        for pcc_id, pcc in enumerate(self._pccs):
            pcc_list = sorted(pcc)
            anchor = pcc_list[0]
            if len(pcc_list) < 2:
                out.append((anchor, len(pcc_list), None))
                continue
            neg, pos = intra.get(pcc_id, ([], []))
            if not neg or not pos:
                out.append((anchor, len(pcc_list), None))
                continue
            pg = nx.Graph()
            for u, v, c in pos:
                pg.add_edge(u, v, weight=c)
            mst = nx.maximum_spanning_tree(pg, weight='weight')
            nodes = list(mst.nodes()); nidx = {n: i for i, n in enumerate(nodes)}
            keep = [(u, v, nc) for (u, v, nc) in neg if u in nidx and v in nidx]
            if not keep:
                out.append((anchor, len(pcc_list), None))
                continue
            nu = np.fromiter((nidx[e[0]] for e in keep), dtype=np.int64, count=len(keep))
            nv = np.fromiter((nidx[e[1]] for e in keep), dtype=np.int64, count=len(keep))
            nc = np.fromiter((e[2] for e in keep), dtype=np.float64, count=len(keep))
            qi, w = _bottleneck_lca(mst, nidx, nu, nv)
            if qi.size == 0:
                out.append((anchor, len(pcc_list), None))
                continue
            min_stab = float(np.min(w - nc[qi]))
            out.append((anchor, len(pcc_list), min_stab))
        if os.environ.get('BETA_SELFCHECK'):
            ref = []
            for pcc in self._pccs:
                pl = sorted(pcc); anc = pl[0]
                if len(pl) < 2:
                    ref.append((anc, len(pl), None)); continue
                ne, pe = [], []
                for i in range(len(pl)):
                    for j in range(i + 1, len(pl)):
                        uu, vv = pl[i], pl[j]
                        if self.G.has_edge(uu, vv):
                            d = self.G[uu][vv].get('data')
                            if d and d.label == EdgeLabel.NEGATIVE:
                                ne.append((uu, vv, d.confidence))
                            elif d and d.label == EdgeLabel.POSITIVE:
                                pe.append((uu, vv, d.confidence))
                if not ne or not pe:
                    ref.append((anc, len(pl), None)); continue
                pg2 = nx.Graph()
                for uu, vv, cc in pe:
                    pg2.add_edge(uu, vv, weight=cc)
                m2 = nx.maximum_spanning_tree(pg2, weight='weight')
                best = float('inf')
                for uu, vv, ncf in ne:
                    if not m2.has_node(uu) or not m2.has_node(vv):
                        continue
                    try:
                        path = nx.shortest_path(m2, uu, vv)
                    except nx.NetworkXNoPath:
                        continue
                    if len(path) < 2:
                        continue
                    mn = min(m2[path[k]][path[k + 1]]['weight'] for k in range(len(path) - 1))
                    best = min(best, mn - ncf)
                ref.append((anc, len(pl), best if best != float('inf') else None))
            ok = len(ref) == len(out) and all(
                a1 == b1 and c1 == d1 and ((e1 is None and f1 is None)
                                           or (e1 is not None and f1 is not None and abs(e1 - f1) < 1e-9))
                for (a1, c1, e1), (b1, d1, f1) in zip(ref, out))
            if not ok:
                logger.error("SELFCHECK FAIL per_pcc_internal_stability")
                raise AssertionError('per_pcc_internal_stability mismatch')
            logger.info(f"SELFCHECK ok: per_pcc_internal {len(out)} PCCs")
        return out

    def _intra_pcc_edges(self):
        """{pcc_id: (negatives, positives)} for intra-PCC edges, each list of
        (u, v, confidence) in ascending (u, v) order -- the same order the old
        node-pair loop produced, so MSTs built from `positives` break ties the
        same way. One vectorised pass over the edge mirror."""
        self._ensure_edge_arrays()
        res: Dict[int, Tuple[list, list]] = {}
        if self._earr_u is None or self._earr_u.size == 0:
            return res
        n = int(max(self._earr_u.max(), self._earr_v.max())) + 1
        pcc_of = np.full(n, -1, dtype=np.int64)
        for nd, pid in self._node_to_pcc.items():
            if 0 <= nd < n:
                pcc_of[nd] = pid
        pu = pcc_of[self._earr_u]; pv = pcc_of[self._earr_v]
        lab = self._earr_lab
        neg_code = _LABEL_CODE[EdgeLabel.NEGATIVE]; pos_code = _LABEL_CODE[EdgeLabel.POSITIVE]
        m = (pu >= 0) & (pu == pv) & ((lab == neg_code) | (lab == pos_code))
        sel = np.flatnonzero(m)
        if sel.size == 0:
            return res
        lo = np.minimum(self._earr_u[sel], self._earr_v[sel])
        hi = np.maximum(self._earr_u[sel], self._earr_v[sel])
        order = np.lexsort((hi, lo, pu[sel]))
        sel = sel[order]; lo = lo[order]; hi = hi[order]
        p = pu[sel]; c = self._earr_conf[sel]; l = lab[sel]
        for i in range(sel.size):
            b = res.get(int(p[i]))
            if b is None:
                b = ([], []); res[int(p[i])] = b
            b[0 if l[i] == neg_code else 1].append((int(lo[i]), int(hi[i]), float(c[i])))
        return res

    def external_stabilities(self):
        """External stability (max_neg - max_posinact) for every PCC pair that
        has a crossing negative -- 'support for keeping them apart' per
        separation decision. Same sign convention as internal stability:
        >= 0 supported, < 0 unstable. Returns a flat list of values."""
        self._ensure_pcc_cache()
        # Fourth full-graph scan, replaced by the mirror. Result is a flat list
        # consumed as an unordered distribution, so only the values matter.
        max_neg, max_pi = self._cross_pcc_pair_maxima()
        vals = [max_neg[k] - max_pi.get(k, 0.0) for k in max_neg]
        if os.environ.get('BETA_SELFCHECK'):
            rn, rp = {}, {}
            for u, v, attr in self.G.edges(data=True):
                ed = attr.get('data')
                if ed is None:
                    continue
                pu_ = self._node_to_pcc.get(u); pv_ = self._node_to_pcc.get(v)
                if pu_ is None or pv_ is None or pu_ == pv_:
                    continue
                k = (min(pu_, pv_), max(pu_, pv_))
                if ed.label == EdgeLabel.NEGATIVE:
                    if k not in rn or ed.confidence > rn[k]:
                        rn[k] = ed.confidence
                elif ed.label == EdgeLabel.POSITIVE_INACTIVE:
                    if k not in rp or ed.confidence > rp[k]:
                        rp[k] = ed.confidence
            ref = sorted(rn[k] - rp.get(k, 0.0) for k in rn)
            if len(ref) != len(vals) or any(abs(x - y) > 1e-9 for x, y in zip(ref, sorted(vals))):
                logger.error("SELFCHECK FAIL external_stabilities")
                raise AssertionError('external_stabilities mismatch')
            logger.info(f"SELFCHECK ok: external_stabilities n={len(vals)}")
        return vals

    def _cross_pcc_pair_maxima(self):
        """(max negative conf, max positive-inactive conf) per cross-PCC pair."""
        self._ensure_edge_arrays()
        if self._earr_u is None or self._earr_u.size == 0:
            return {}, {}
        n = int(max(self._earr_u.max(), self._earr_v.max())) + 1
        pcc_of = np.full(n, -1, dtype=np.int64)
        for nd, pid in self._node_to_pcc.items():
            if 0 <= nd < n:
                pcc_of[nd] = pid
        pu = pcc_of[self._earr_u]; pv = pcc_of[self._earr_v]
        cross = (pu >= 0) & (pv >= 0) & (pu != pv)
        lo = np.minimum(pu, pv); hi = np.maximum(pu, pv)
        K = int(hi.max()) + 1 if hi.size else 1
        conf = self._earr_conf

        def pair_max(mask):
            i = np.flatnonzero(mask)
            if i.size == 0:
                return {}
            k = lo[i] * K + hi[i]
            uk, iv = np.unique(k, return_inverse=True)
            mx = np.zeros(uk.size, dtype=np.float64)
            np.maximum.at(mx, iv, conf[i])
            return {(int(x // K), int(x % K)): float(y) for x, y in zip(uk, mx)}

        return (pair_max(cross & (self._earr_lab == _LABEL_CODE[EdgeLabel.NEGATIVE])),
                pair_max(cross & (self._earr_lab == _LABEL_CODE[EdgeLabel.POSITIVE_INACTIVE])))

    def compute_external_stability(self, pcc_a: int, pcc_b: int) -> Optional[float]:
        """
        Compute external stability between two PCCs.

        Per PDF: "The stability of any pair of PCCs is the maximum confidence of any
        negative edges joining the PCCs, minus the maximum confidence of any
        positive-inactive edges joining them."

        Returns None if no negative edge exists between the PCCs.
        """
        self._ensure_pcc_cache()

        if pcc_a >= len(self._pccs) or pcc_b >= len(self._pccs):
            return None

        nodes_a = self._pccs[pcc_a]
        nodes_b = self._pccs[pcc_b]

        max_neg_conf = None
        max_pos_inactive_conf = 0.0

        for u in nodes_a:
            for v in self.G.neighbors(u):
                if v not in nodes_b:
                    continue

                edge_data = self.G[u][v].get('data')
                if edge_data is None:
                    continue

                if edge_data.label == EdgeLabel.NEGATIVE:
                    if max_neg_conf is None:
                        max_neg_conf = edge_data.confidence
                    else:
                        max_neg_conf = max(max_neg_conf, edge_data.confidence)
                elif edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                    max_pos_inactive_conf = max(max_pos_inactive_conf, edge_data.confidence)

        if max_neg_conf is None:
            return None  # No negative edge = external stability not defined

        return max_neg_conf - max_pos_inactive_conf

    def find_unstable_internal_pairs(self, alpha: float = 0.0) -> List[Tuple[int, int, float, Tuple[int, int]]]:
        """
        Find all vertex pairs within PCCs with stability < alpha.

        Returns: List of (u, v, stability, min_edge_on_msp)
        """
        self._ensure_pcc_cache()
        unstable = []

        for pcc_id, pcc in enumerate(self._pccs):
            if len(pcc) < 2:
                continue

            # Collect negative edges within this PCC first
            negative_edges = []
            for u in pcc:
                for v in self.G.neighbors(u):
                    if v in pcc and u < v:
                        edge_data = self.G[u][v].get('data')
                        if edge_data and edge_data.label == EdgeLabel.NEGATIVE:
                            negative_edges.append((u, v, edge_data.confidence))

            if not negative_edges:
                continue

            # Build MST ONCE for this PCC
            pcc_graph = nx.Graph()
            for u in pcc:
                for v in self.G.neighbors(u):
                    if v in pcc and u < v:
                        edge_data = self.G[u][v].get('data')
                        if edge_data and edge_data.label == EdgeLabel.POSITIVE:
                            pcc_graph.add_edge(u, v, weight=edge_data.confidence)

            if pcc_graph.number_of_edges() == 0:
                continue

            mst = nx.maximum_spanning_tree(pcc_graph, weight='weight')

            # Check each negative edge pair using the cached MST
            for u, v, neg_conf in negative_edges:
                if not mst.has_node(u) or not mst.has_node(v):
                    continue

                try:
                    path = nx.shortest_path(mst, u, v)
                except nx.NetworkXNoPath:
                    continue

                if len(path) < 2:
                    continue

                # Find minimum edge on path (MSP strength)
                min_conf = float('inf')
                min_edge = None
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    conf = mst[a][b]['weight']
                    if conf < min_conf:
                        min_conf = conf
                        min_edge = (a, b)

                # stability = MSP - neg_conf
                stability = min_conf - neg_conf
                if stability < alpha:
                    unstable.append((u, v, stability, min_edge))

        return unstable

    def find_unstable_external_pairs(self, alpha: float = 0.0) -> List[Tuple[int, int, float, Tuple[int, int]]]:
        """
        Find all PCC pairs with external stability < alpha.

        Returns: List of (pcc_a, pcc_b, stability, highest_neg_edge)
        """
        self._ensure_pcc_cache()
        unstable = []

        for pcc_a in range(len(self._pccs)):
            for pcc_b in range(pcc_a + 1, len(self._pccs)):
                stability = self.compute_external_stability(pcc_a, pcc_b)

                if stability is not None and stability < alpha:
                    # Find highest confidence negative edge
                    max_neg_edge = None
                    max_neg_conf = -float('inf')

                    for u in self._pccs[pcc_a]:
                        for v in self.G.neighbors(u):
                            if v not in self._pccs[pcc_b]:
                                continue
                            edge_data = self.G[u][v].get('data')
                            if edge_data and edge_data.label == EdgeLabel.NEGATIVE:
                                if edge_data.confidence > max_neg_conf:
                                    max_neg_conf = edge_data.confidence
                                    max_neg_edge = (u, v)

                    if max_neg_edge:
                        unstable.append((pcc_a, pcc_b, stability, max_neg_edge))

        return unstable

    def _ensure_adj_rank(self):
        """For every mirror edge (lo < hi), the position of hi within G._adj[lo].

        `make_zero_stable` historically built each PCC's subgraph by sweeping
        `for a in pcc_set: for b in G[a]` and keeping b > a, so its edge insertion
        order -- which networkx's MST uses to break equal-weight ties -- is
        (position of a in the PCC set, position of b in a's adjacency dict). One
        pass over the adjacency records the second key for every edge; the first
        is cheap per call. Graph structure never changes in a run (edges are added
        only during construction, which drops the mirror), so this is built once.
        """
        self._ensure_edge_arrays()
        if self._earr_rank is not None or self._earr_u is None:
            return
        from array import array
        A = array('q'); B = array('q'); R = array('q')
        for a_, nbrs in self.G._adj.items():
            r = 0
            for b_ in nbrs:
                if a_ < b_:
                    A.append(a_); B.append(b_); R.append(r)
                r += 1
        A = np.frombuffer(A, dtype=np.int64); B = np.frombuffer(B, dtype=np.int64)
        R = np.frombuffer(R, dtype=np.int64)
        lo = np.minimum(self._earr_u, self._earr_v); hi = np.maximum(self._earr_u, self._earr_v)
        N = int(max(hi.max() if hi.size else 0, B.max() if B.size else 0)) + 1
        k_adj = A * N + B
        o = np.argsort(k_adj, kind='stable')
        k_sorted = k_adj[o]
        k_m = lo * N + hi
        pos = np.searchsorted(k_sorted, k_m)
        if pos.size and not np.array_equal(k_sorted[np.minimum(pos, k_sorted.size - 1)], k_m):
            raise AssertionError('adjacency rank alignment failed')
        self._earr_rank = R[o][pos]

    def _intra_edges_sweep_order(self):
        """Per PCC id: (pos_lo, pos_hi, pos_conf, neg_lo, neg_hi, neg_conf) arrays in
        the exact order the historical adjacency sweep in make_zero_stable emitted
        them: PCC by PCC, then position of lo in the PCC set, then position of hi in
        G._adj[lo]."""
        self._ensure_adj_rank()
        out = {}
        if self._earr_u is None or self._earr_u.size == 0:
            return out
        n = int(max(self._earr_u.max(), self._earr_v.max())) + 1
        pcc_of = np.full(n, -1, dtype=np.int64)
        setpos = np.zeros(n, dtype=np.int64)
        for pid, pcc in enumerate(self._pccs):
            for i, nd in enumerate(pcc):
                if 0 <= nd < n:
                    pcc_of[nd] = pid; setpos[nd] = i
        pu = pcc_of[self._earr_u]; pv = pcc_of[self._earr_v]
        lab = self._earr_lab
        pos_code = _LABEL_CODE[EdgeLabel.POSITIVE]; neg_code = _LABEL_CODE[EdgeLabel.NEGATIVE]
        sel = np.flatnonzero((pu >= 0) & (pu == pv) & ((lab == pos_code) | (lab == neg_code)))
        if sel.size == 0:
            return out
        lo = np.minimum(self._earr_u[sel], self._earr_v[sel])
        hi = np.maximum(self._earr_u[sel], self._earr_v[sel])
        order = np.lexsort((self._earr_rank[sel], setpos[lo], pu[sel]))
        sel = sel[order]; lo = lo[order]; hi = hi[order]
        pp = pu[sel]; ll = lab[sel]; cc = self._earr_conf[sel]
        starts = np.flatnonzero(np.r_[True, pp[1:] != pp[:-1]])
        ends = np.r_[starts[1:], pp.size]
        for st, en in zip(starts.tolist(), ends.tolist()):
            gl = ll[st:en]; ip = gl == pos_code; ineg = gl == neg_code
            out[int(pp[st])] = (lo[st:en][ip], hi[st:en][ip], cc[st:en][ip],
                                lo[st:en][ineg], hi[st:en][ineg], cc[st:en][ineg])
        return out

    def make_zero_stable(self, alpha: float = 0.0) -> int:
        """
        Make the graph alpha-stable by deactivating positive edges.

        Per PDF: "The initial goal is to make positive-inactive assignments that
        will make the graph 0-stable. This is done entirely without human input.
        Note that we do not need to explicitly inactivate negative edges."

        Args:
            alpha: Stability threshold. Default 0.0 for strict 0-stability.
                   Negative values (e.g., -0.1) allow small instabilities,
                   resulting in less aggressive fragmentation.

        For each within-PCC negative pair (u,v) with stability < alpha, one
        step of Step 2 inactivates the weakest positive edge on MSP(u,v). The
        PCC's maximum spanning tree is RECOMPUTED after each cut and the loop
        repeats until every internal negative pair is at least alpha-stable --
        per the paper, "once a negative starts to cause the removal of positive
        edges, it continues until v_i and v_j are in different PCCs". Recomputing
        the tree is what lets non-tree positive edges be reconsidered, so a
        densely-connected pair is actually separated instead of leaving a severed
        spanning-tree path while other edges still join it.

        Returns: Number of edges deactivated.

        Cost: no global MST forest is built. Only PCCs that contain a negative
        edge do any work, each on its own (small) positive subgraph; a PCC that
        is already 0-stable costs a single local tree build.
        """
        deactivations = 0

        self._pcc_cache_valid = False
        self._ensure_pcc_cache()

        # Every PCC's positive and negative edges, from the edge mirror, in exactly
        # the order the per-node adjacency sweep used to produce them. That sweep
        # visited ~4,000 neighbours for every node of every PCC on every call (once
        # per restabilisation, i.e. every review batch). Cuts inside one PCC only
        # relabel that PCC's own edges and nothing here refreshes the PCC cache, so
        # a single upfront grouping stays exact for the whole loop.
        _groups = self._intra_edges_sweep_order()

        for _pid, pcc in enumerate(list(self._pccs)):
            if len(pcc) < 2:
                continue
            pcc_set = pcc if isinstance(pcc, set) else set(pcc)
            _g = _groups.get(_pid)

            P = nx.Graph()
            P.add_nodes_from(pcc_set)
            if _g is not None and _g[0].size:
                P.add_edges_from((a, b, {'weight': c}) for a, b, c in
                                 zip(_g[0].tolist(), _g[1].tolist(), _g[2].tolist()))

            if os.environ.get('BETA_SELFCHECK'):
                _P0 = nx.Graph(); _P0.add_nodes_from(pcc_set); _neg0 = []
                for a in pcc_set:
                    for b, edge in self.G[a].items():
                        if b <= a or b not in pcc_set:
                            continue
                        d = edge.get('data')
                        if not d:
                            continue
                        if d.label == EdgeLabel.POSITIVE:
                            _P0.add_edge(a, b, weight=d.confidence)
                        elif d.label == EdgeLabel.NEGATIVE:
                            _neg0.append((a, b, d.confidence))
                _negN = ([] if _g is None else
                         list(zip(_g[3].tolist(), _g[4].tolist(), _g[5].tolist())))
                if (list(_P0.edges(data='weight')) != list(P.edges(data='weight'))
                        or _neg0 != _negN):
                    logger.error(f"SELFCHECK FAIL make_zero_stable subgraph, PCC {_pid}")
                    raise AssertionError('make_zero_stable subgraph/negatives order mismatch')
                self._selfcheck_mzs = getattr(self, '_selfcheck_mzs', 0) + 1

            if _g is None or _g[3].size == 0:
                continue

            pcc_deactivations = 0
            # Query arrays are built ONCE per PCC and reused by every cutting
            # round: only P's EDGES change as we cut, never its node set.
            _nodes = list(P.nodes())
            _idx = {n: i for i, n in enumerate(_nodes)}
            _inv = np.array(_nodes, dtype=np.int64)
            _nu = _g[3]; _nv = _g[4]; _nc = _g[5]
            _map = np.zeros(int(max(_inv.max(), _nu.max(), _nv.max())) + 1, dtype=np.int64)
            _map[_inv] = np.arange(len(_nodes), dtype=np.int64)
            _nu = _map[_nu]; _nv = _map[_nv]
            while P.number_of_edges() > 0:
                mst = nx.maximum_spanning_tree(P, weight='weight')
                # Step 2, batched ("assign the weakest edge on the MSP for EACH
                # unstable pair, then recompute"): cut the weakest edge of every
                # pair with stability < alpha this round, then rebuild. Batching
                # keeps the number of MST rebuilds small even when a giant
                # over-merged PCC needs thousands of cuts.
                #
                # _edges_to_cut is the vectorised form of the old
                # `_bottleneck_edges` + per-pair Python loop (kept below as the
                # reference implementation); it cuts exactly the same edges with
                # the same deactivators, verified pair-for-pair on this workload,
                # at 29.5 s -> 4.7 s per round.
                to_cut = _edges_to_cut(mst, _idx, _inv, _nu, _nv, _nc, alpha)

                if not to_cut:
                    break  # every negative pair is now >= alpha-stable

                for (a, b), deact in to_cut.items():
                    if P.has_edge(a, b):
                        self.deactivate_positive((a, b), deactivator=deact)
                        P.remove_edge(a, b)
                        deactivations += 1
                        pcc_deactivations += 1

            if pcc_deactivations > 0:
                logger.info(f"PCC (size {len(pcc)}): deactivated "
                            f"{pcc_deactivations} edges for {alpha}-stability")

        if os.environ.get('BETA_SELFCHECK'):
            logger.info(f"SELFCHECK ok: make_zero_stable subgraphs {getattr(self, '_selfcheck_mzs', 0)} PCCs (cumulative)")
        self._pcc_cache_valid = False
        return deactivations

    def _get_msp_for_pair(self, u: int, v: int, pcc_id: int) -> Optional[Tuple[float, Tuple[int, int]]]:
        """Get MSP strength and min edge for a specific pair. Returns (strength, min_edge)."""
        pcc = self._pccs[pcc_id]

        # Build MST for this PCC only
        pcc_graph = nx.Graph()
        for a in pcc:
            for b in self.G.neighbors(a):
                if b in pcc and a < b:
                    edge_data = self.G[a][b].get('data')
                    if edge_data and edge_data.label == EdgeLabel.POSITIVE:
                        pcc_graph.add_edge(a, b, weight=edge_data.confidence)

        if pcc_graph.number_of_edges() == 0:
            return None

        mst = nx.maximum_spanning_tree(pcc_graph, weight='weight')

        if not mst.has_node(u) or not mst.has_node(v):
            return None

        try:
            path = nx.shortest_path(mst, u, v)
        except nx.NetworkXNoPath:
            return None

        if len(path) < 2:
            return None

        # Find minimum edge on path
        min_conf = float('inf')
        min_edge = None
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            conf = mst[a][b]['weight']
            if conf < min_conf:
                min_conf = conf
                min_edge = (a, b)

        return (min_conf, min_edge)

    def get_review_candidates(self, alpha: float) -> List[StabilityCandidate]:
        """
        Get candidates for human review with stability < alpha.

        Per PDF algorithm steps 1-3:
        1. Compute internal stability for each pair in each PCC
        2. Compute external stability for each PCC pair with negative edges
        3. Order by increasing stability, select first k pairs

        OPTIMIZATION: Build MST forest once at start.
        """
        self._ensure_pcc_cache()

        # Build MST forest once for all internal stability calculations
        self._build_mst_forest()

        candidates = []

        # Internal candidates
        for pcc_id, pcc in enumerate(self._pccs):
            if len(pcc) < 2:
                continue

            # Collect negative edges within this PCC
            negative_edges = []
            for u in pcc:
                for v in self.G.neighbors(u):
                    if v in pcc and u < v:
                        edge_data = self.G[u][v].get('data')
                        if edge_data and edge_data.label == EdgeLabel.NEGATIVE:
                            negative_edges.append((u, v, edge_data.confidence))

            if not negative_edges:
                continue

            # Extract MST for this PCC from the forest
            mst = self._mst_forest.subgraph(pcc).copy()
            if mst.number_of_edges() == 0:
                continue

            # Precompute all paths in this PCC's MST (avoids repeated BFS)
            all_paths = dict(nx.all_pairs_shortest_path(mst))

            # Check each negative edge pair using precomputed paths
            for u, v, neg_conf in negative_edges:
                # Look up precomputed path (O(1) instead of BFS)
                path = all_paths.get(u, {}).get(v)
                if path is None:
                    continue

                if len(path) < 2:
                    continue

                # Find minimum edge on path (MSP strength)
                min_conf = float('inf')
                min_edge = None
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    conf = mst[a][b]['weight']
                    if conf < min_conf:
                        min_conf = conf
                        min_edge = (a, b)

                # stability = MSP - neg_conf
                stability = min_conf - neg_conf
                if stability < alpha:
                    candidates.append(StabilityCandidate(
                        candidate_type="INTERNAL",
                        stability=stability,
                        review_edge=(u, v),
                        node_pair=(u, v),
                        pcc_id=pcc_id
                    ))

        # External candidates - optimized: pre-collect cross-PCC negative edges
        # Build map of PCC pairs with negative edges and their max negative edge
        pcc_pair_neg_edges: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}
        pcc_pair_pos_inactive: Dict[Tuple[int, int], float] = {}

        for u in self.G.nodes():
            pcc_u = self._node_to_pcc.get(u)
            if pcc_u is None:
                continue

            for v in self.G.neighbors(u):
                pcc_v = self._node_to_pcc.get(v)
                if pcc_v is None or pcc_u == pcc_v:
                    continue

                # Normalize PCC pair order
                pcc_pair = (min(pcc_u, pcc_v), max(pcc_u, pcc_v))
                edge_data = self.G[u][v].get('data')
                if not edge_data:
                    continue

                if edge_data.label == EdgeLabel.NEGATIVE:
                    current = pcc_pair_neg_edges.get(pcc_pair)
                    if current is None or edge_data.confidence > current[0]:
                        pcc_pair_neg_edges[pcc_pair] = (edge_data.confidence, (u, v))
                elif edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                    current = pcc_pair_pos_inactive.get(pcc_pair, 0.0)
                    pcc_pair_pos_inactive[pcc_pair] = max(current, edge_data.confidence)

        # Now compute external stability only for PCC pairs with negative edges
        for pcc_pair, (max_neg_conf, max_neg_edge) in pcc_pair_neg_edges.items():
            max_pos_inactive = pcc_pair_pos_inactive.get(pcc_pair, 0.0)
            stability = max_neg_conf - max_pos_inactive

            if stability < alpha:
                candidates.append(StabilityCandidate(
                    candidate_type="EXTERNAL",
                    stability=stability,
                    review_edge=max_neg_edge,
                    pcc_pair=pcc_pair
                ))

        # Sort by stability ascending (most unstable first)
        candidates.sort(key=lambda c: c.stability)
        return candidates

    def _build_likely_fn_pos_graph(self, method: str, human_boost: float):
        """Return (nodes, node_to_idx, sparse A) for the positive subgraph
        the candidate selector operates on.

        Weight scheme for `*_weighted` methods: uses the edge's `score`
        field directly. Because `apply_human_review` saturates score to 1.0
        for human-confirmed positives, those edges naturally dominate the
        algorithm-classified ones (typical score ~0.6-0.8). No special-case
        ranker lookup needed.
        """
        import scipy.sparse as sp
        all_nodes = sorted(self.G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        N = len(all_nodes)
        rows, cols, data = [], [], []
        for u, v, attr in self.G.edges(data=True):
            ed = attr.get('data')
            if ed is None:
                continue
            if ed.label not in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE):
                continue
            i = node_to_idx.get(u); j = node_to_idx.get(v)
            if i is None or j is None or i == j:
                continue
            if method.endswith('weighted'):
                score = float(getattr(ed, 'score', 0.5) or 0.0)
                # Re-center on the "no info" baseline (0.5 in pipeline space)
                # so that human-confirmed positives (score=1.0 -> weight=0.5)
                # dominate algorithm-classified positives (score~0.65 ->
                # weight~0.15). Below-baseline scores get weight 0 (effectively
                # excluded), avoiding spurious weak-positive links.
                w = max(score - 0.5, 0.0)
                if w <= 0:
                    continue
            else:
                w = 1.0
            if i < j:
                rows.append(i); cols.append(j); data.append(w)
        if not rows:
            return all_nodes, node_to_idx, None
        all_rows = np.array(rows + cols, dtype=np.int64)
        all_cols = np.array(cols + rows, dtype=np.int64)
        all_data = np.array(data + data, dtype=np.float64)
        A = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(N, N)).tocsr()
        return all_nodes, node_to_idx, A

    def _build_likely_fn_confidence_signed_graph(self):
        """Signed adjacency keyed on the edge's existing `confidence` field.

        `confidence` is already the right quantity: ThresholdBasedClassifier
        defines it as |score - threshold| / max_range — so it accounts for
        the actual classifier threshold and the data-specific score range
        without any special-case baseline. After human review it saturates
        toward 1.0 in apply_human_review (or is set to 1.0 directly in
        apply_ground_truth_review). The label provides the sign.

        Weight rule:
            positive / positive-inactive  →  +confidence
            negative                      →  −confidence

        Verified-positive and verified-negative edges both end up at
        magnitude 1.0; algorithm-classified edges contribute proportionally
        to their distance from the classifier's decision boundary.
        """
        import scipy.sparse as sp
        all_nodes = sorted(self.G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        N = len(all_nodes)

        rows, cols, data = [], [], []
        for u, v, attr in self.G.edges(data=True):
            ed = attr.get('data')
            if ed is None:
                continue
            i = node_to_idx.get(u); j = node_to_idx.get(v)
            if i is None or j is None or i == j:
                continue
            conf = float(getattr(ed, 'confidence', 0.0) or 0.0)
            if conf <= 0:
                continue
            if ed.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE):
                w = conf
            elif ed.label == EdgeLabel.NEGATIVE:
                w = -conf
            else:
                continue
            rows.extend([i, j]); cols.extend([j, i]); data.extend([w, w])
        if not rows:
            return all_nodes, node_to_idx, None
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        return all_nodes, node_to_idx, A

    def _build_likely_fn_signed_graph(self, human_boost: float, neg_repulsion: float):
        """Signed adjacency: positives weighted by `score - 0.5` (so human-
        confirmed pull strongly, algorithm-classified pull weakly); negatives
        identified by score == 0.0 (human-confirmed negatives, saturated by
        apply_human_review) get weight `-neg_repulsion`. Algorithm-classified
        negatives (score > 0) are excluded since they're not high-confidence
        enough to use as repulsion signal."""
        import scipy.sparse as sp
        all_nodes = sorted(self.G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        N = len(all_nodes)
        rows, cols, data = [], [], []
        for u, v, attr in self.G.edges(data=True):
            ed = attr.get('data')
            if ed is None:
                continue
            i = node_to_idx.get(u); j = node_to_idx.get(v)
            if i is None or j is None or i == j:
                continue
            score = float(getattr(ed, 'score', 0.5) or 0.0)
            if ed.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE):
                w = max(score - 0.5, 0.0)
                if w <= 0:
                    continue
            elif ed.label == EdgeLabel.NEGATIVE and score == 0.0:
                # Human-confirmed negative (apply_human_review saturated score
                # to 0.0). Algorithm-classified negatives have score > 0 and
                # are excluded — they are too noisy to use as repulsion.
                w = -neg_repulsion
            else:
                continue
            rows.extend([i, j]); cols.extend([j, i]); data.extend([w, w])
        if not rows:
            return all_nodes, node_to_idx, None
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        return all_nodes, node_to_idx, A

    def _compute_node_heat_features(self, A, signed: bool, n_eigs: int, heat_t: float):
        """Return phi_weighted ([N, k]) such that K_t(i, j) = phi_w[i] · phi_w[j]."""
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        N = A.shape[0]
        if signed:
            abs_degrees = np.array(np.abs(A).sum(axis=1)).flatten()
            D_bar = sp.diags(abs_degrees)
            L = D_bar - A
        else:
            degrees = np.array(A.sum(axis=1)).flatten()
            d_inv_sqrt = np.zeros(N, dtype=np.float64)
            nz = degrees > 0
            d_inv_sqrt[nz] = 1.0 / np.sqrt(degrees[nz])
            D_inv_sqrt = sp.diags(d_inv_sqrt)
            L = sp.eye(N, format='csr') - (D_inv_sqrt @ A @ D_inv_sqrt)
        k = min(n_eigs, max(N - 1, 1))
        try:
            lambdas, phi = eigsh(L, k=k, which='SM', tol=1e-6, maxiter=N * 10)
        except Exception as e:
            logger.warning(f"Heat-kernel eigsh failed ({e}); using dense fallback")
            lambdas_full, phi_full = np.linalg.eigh(L.toarray())
            lambdas, phi = lambdas_full[:k], phi_full[:, :k]
        order = np.argsort(lambdas)
        lambdas = np.clip(lambdas[order], 0.0, None)
        phi = phi[:, order]
        weights = np.exp(-heat_t * lambdas)
        return phi * np.sqrt(np.maximum(weights, 0.0))

    def _compute_node_ppr(self, A, alpha: float):
        """Return full PPR matrix (dense, N×N)."""
        import scipy.sparse as sp
        N = A.shape[0]
        # PPR requires non-negative transition probabilities. For weighted
        # graphs A is already non-negative; we don't call this for signed graphs.
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        D_inv = sp.diags(1.0 / row_sums)
        P = D_inv @ A
        M = sp.eye(N, format='csr') - (1 - alpha) * P.T
        try:
            return alpha * np.linalg.inv(M.toarray())
        except np.linalg.LinAlgError:
            return alpha * np.linalg.pinv(M.toarray())

    def select_likely_fn_candidates(
        self,
        verified_edges: Optional[Set[Tuple[int, int]]] = None,
        max_count: int = 100,
        min_score: float = 0.0,
        exclude_edges: Optional[Set[Tuple[int, int]]] = None,
        n_eigs: int = 64,
        heat_kernel_t: float = 10.0,
        method: str = 'heat_kernel',
        human_boost: float = 20.0,
        ppr_alpha: float = 0.15,
        neg_repulsion: float = 5.0,
    ) -> List[Tuple[int, int, float]]:
        """Generate heat-kernel-ranked false-negative candidates for human review.

        For every active cross-PCC negative edge (u, v) we compute the heat
        kernel similarity over the *ever-positive* graph (active positive
        edges + positive-inactive edges Phase 0 deactivated):

            K_t(u, v) = Σ_k exp(-t · λ_k) · φ_k(u) · φ_k(v)

        where {λ_k, φ_k} are the lowest eigenvalues / eigenvectors of the
        normalized symmetric Laplacian L_sym = I - D^{-1/2} A D^{-1/2} of the
        ever-positive graph. High K_t(u, v) means u and v are in the same
        diffusion neighborhood — strong evidence they are actually the same
        individual whose connecting edges Phase 0 deactivated.

        Why ever-positive (active+inactive) and not just active: Phase 0
        deactivates edges to resolve internal contradictions, which is exactly
        the information needed to identify FNs. Using the ever-positive graph
        recovers that hidden signal. (Validated in
        live metric_tracker runs have since shown spectral/heat-kernel
        metrics collapse to ~0 FN precision after the first few batches.)

        Runs in PARALLEL with `generate_candidate_pools`; the stability
        mechanism is untouched.

        Args:
            verified_edges: edges already presented to humans; skipped.
            max_count: maximum number of candidates to return.
            min_score: only return candidates with heat-kernel strictly above this.
            exclude_edges: additional edges to skip (e.g. already in batch).
            n_eigs: number of Laplacian eigenvectors to compute (cost ~N*k).
            heat_kernel_t: diffusion time. Larger t = more global similarity.

        Returns: list of (u, v, score) tuples sorted by heat-kernel descending,
            length up to max_count. Empty list if max_count <= 0.
        """
        if max_count <= 0:
            return []
        try:
            import scipy.sparse  # noqa
        except ImportError:
            logger.error("scipy is required for graph-spectral candidate selection")
            return []

        _verified = verified_edges or set()
        _exclude = exclude_edges or set()
        self._ensure_pcc_cache()

        all_nodes = sorted(self.G.nodes())
        if len(all_nodes) < 2:
            return []

        # Build the appropriate positive (or signed) graph and a per-node
        # scoring function score_fn(i, j) -> heat/PPR/etc.
        if method in ('heat_kernel', 'heat_kernel_weighted'):
            all_nodes, node_to_idx, A = self._build_likely_fn_pos_graph(method, human_boost)
            if A is None:
                return []
            phi_w = self._compute_node_heat_features(A, signed=False,
                                                    n_eigs=n_eigs, heat_t=heat_kernel_t)
            score_fn = lambda i, j: float(np.dot(phi_w[i], phi_w[j]))
        elif method in ('heat_kernel_unnorm', 'heat_kernel_unnorm_weighted'):
            # Unnormalized Laplacian (D - A). Empirically much more
            # discriminative than the normalized version for FN detection —
            # preserves the hub/community structure that maps to individuals
            # instead of rescaling it away by sqrt(degree). At iter 0 of the
            # GZCD megadescriptor run this hit 32/50 vs the normalized 12/50.
            build_method = ('heat_kernel_weighted'
                            if method == 'heat_kernel_unnorm_weighted'
                            else 'heat_kernel')
            all_nodes, node_to_idx, A = self._build_likely_fn_pos_graph(build_method, human_boost)
            if A is None:
                return []
            phi_w = self._compute_node_heat_features(A, signed=True,
                                                    n_eigs=n_eigs, heat_t=heat_kernel_t)
            score_fn = lambda i, j: float(np.dot(phi_w[i], phi_w[j]))
        elif method in ('ppr', 'ppr_weighted'):
            # PPR needs a positive graph; reuse the same builder.
            heat_method = 'heat_kernel_weighted' if method == 'ppr_weighted' else 'heat_kernel'
            all_nodes, node_to_idx, A = self._build_likely_fn_pos_graph(heat_method, human_boost)
            if A is None:
                return []
            ppr = self._compute_node_ppr(A, alpha=ppr_alpha)
            score_fn = lambda i, j: float(0.5 * (ppr[i, j] + ppr[j, i]))
        elif method == 'human_signed':
            all_nodes, node_to_idx, A = self._build_likely_fn_signed_graph(
                human_boost=human_boost, neg_repulsion=neg_repulsion
            )
            if A is None:
                return []
            phi_w = self._compute_node_heat_features(A, signed=True,
                                                    n_eigs=n_eigs, heat_t=heat_kernel_t)
            score_fn = lambda i, j: float(np.dot(phi_w[i], phi_w[j]))
        elif method == 'confidence_signed':
            # Unnormalized signed Laplacian using confidence as the natural
            # weight: |conf| handles threshold/score-range automatically, and
            # the sign comes from the edge label. Human-reviewed edges of
            # both polarities saturate to |1.0| — both positives and
            # negatives become primary signal.
            all_nodes, node_to_idx, A = self._build_likely_fn_confidence_signed_graph()
            if A is None:
                return []
            phi_w = self._compute_node_heat_features(A, signed=True,
                                                    n_eigs=n_eigs, heat_t=heat_kernel_t)
            score_fn = lambda i, j: float(np.dot(phi_w[i], phi_w[j]))
        else:
            raise ValueError(f"Unknown method: {method!r}")

        # Score each active cross-PCC negative edge.
        scored: List[Tuple[float, int, int, float]] = []  # (-heat, u, v, raw_score)
        for u, v, attr in self.G.edges(data=True):
            ed = attr.get('data')
            if ed is None or ed.label != EdgeLabel.NEGATIVE:
                continue
            edge_key = (min(u, v), max(u, v))
            if edge_key in _verified or edge_key in _exclude:
                continue
            pcc_u = self._node_to_pcc.get(u)
            pcc_v = self._node_to_pcc.get(v)
            if pcc_u is None or pcc_v is None or pcc_u == pcc_v:
                continue
            iu = node_to_idx.get(u)
            iv = node_to_idx.get(v)
            if iu is None or iv is None:
                continue
            metric = score_fn(iu, iv)
            if metric <= min_score:
                continue
            raw_score = float(ed.score) if ed.score is not None else 0.0
            # negate so ascending sort picks largest metric first
            scored.append((-metric, u, v, raw_score))

        if not scored:
            return []

        scored.sort(key=lambda t: t[0])
        top = scored[:max_count]
        return [(u, v, s) for _, u, v, s in top]

    def generate_candidate_pools(
        self,
        alpha: float,
        verified_edges: Optional[Set[Tuple[int, int]]] = None,
        unverified_threshold: float = 0.0,
        pcc_separation_strength: Optional[Dict[int, float]] = None,
        include_weak_positive: bool = False,
        review_positive_inactive: bool = False,
    ) -> Dict[str, List["StabilityCandidate"]]:
        """Generate sorted candidate pools (no batch cap, no selection).

        The caller composes the final review batch by drawing from pools in
        whatever priority order it wants, applying its own per-PCC conflict
        rules and budget cap.

        Args:
            alpha: stability threshold (candidates with stability >= alpha
                are filtered out).
            verified_edges: edges already human-reviewed — excluded from pools.
            unverified_threshold: confidence cutoff for the unverified pool
                (0 disables it).
            pcc_separation_strength: NIS n_hat analog for scoring
                UNVERIFIED_NEG (lower = more isolated = higher priority).

        Returns:
            Dict with keys 'internal', 'external', 'unverified', each a
            sorted list of StabilityCandidate (highest priority first).
        """
        self._ensure_pcc_cache()

        # Build MST forest once for internal candidate generation
        self._build_mst_forest()

        # Per-pool accumulators. Internal (intra-PCC negative edges driving
        # instability) and unverified-pos (intra-PCC low-confidence positive
        # MST edges) come from the same per-PCC pass; external and
        # unverified-neg come from the global cross-PCC edge scan.
        internal_pool: List[StabilityCandidate] = []
        external_pool: List[StabilityCandidate] = []
        unverified_pool: List[StabilityCandidate] = []
        _verified = verified_edges or set()
        # Intra-PCC negatives for every PCC in one vectorised pass, in ascending
        # (u, v) order -- exactly the order the node-pair loop produced.
        _intra_all = self._intra_pcc_edges()

        for pcc_id, pcc in enumerate(self._pccs):
            if len(pcc) < 2:
                continue

            # Extract MST for this PCC from the forest
            mst = self._mst_forest.subgraph(pcc).copy()
            if mst.number_of_edges() == 0:
                continue

            # Root MST and compute subtree sizes for structural impact (O(N) per PCC)
            n_pcc = len(pcc)
            mst_root = next(iter(pcc))
            mst_parent = {}
            mst_order = []
            stack = [(mst_root, None)]
            while stack:
                node, par = stack.pop()
                mst_parent[node] = par
                mst_order.append(node)
                for nbr in mst.neighbors(node):
                    if nbr != par:
                        stack.append((nbr, node))
            mst_subtree_size = {node: 1 for node in mst_order}
            for node in reversed(mst_order):
                if mst_parent[node] is not None:
                    mst_subtree_size[mst_parent[node]] += mst_subtree_size[node]

            # Chuck's internal-pool fix: the stability of a pair with no
            # negative edge is just its MSP strength (the min positive edge on
            # the path). The negative-edge loop below never emits these, so a
            # component held together only by weak positives -- canonically a
            # wrong size-2 component -- is never flagged even when its weakest
            # link is below beta. Here we add the missing branch: if the PCC's
            # weakest MST edge (its bottleneck = min internal MSP) is below the
            # SAME threshold alpha (beta) already in use, emit it as an INTERNAL
            # candidate reviewing that edge. Same pool, same priority, no new
            # threshold. (Pairs that also have a negative edge are handled by
            # the loop below and, being more unstable, win the 1-per-PCC slot.)
            if include_weak_positive:
                _weak_edge = None
                _weak_conf = float('inf')
                for a, b, attr in mst.edges(data=True):
                    edge_key = (min(a, b), max(a, b))
                    if edge_key in _verified:
                        continue
                    conf = attr.get('weight', 1.0)
                    if conf < _weak_conf:
                        _weak_conf = conf
                        _weak_edge = (a, b)
                if _weak_edge is not None and _weak_conf < alpha:
                    a, b = _weak_edge
                    if mst_parent.get(b) == a:
                        child = b
                    elif mst_parent.get(a) == b:
                        child = a
                    else:
                        child = b
                    child_size = mst_subtree_size.get(child, 1)
                    impact = float(min(child_size, n_pcc - child_size))
                    internal_pool.append(StabilityCandidate(
                        candidate_type="WEAK_POSITIVE",
                        stability=_weak_conf,
                        review_edge=_weak_edge,
                        structural_impact=max(impact, 1.0),
                        node_pair=_weak_edge,
                        pcc_id=pcc_id,
                    ))

            # Collect unverified MST edges below threshold
            if unverified_threshold > 0:
                for a, b, attr in mst.edges(data=True):
                    edge_key = (min(a, b), max(a, b))
                    if edge_key in _verified:
                        continue
                    conf = attr.get('weight', 1.0)
                    if conf < unverified_threshold:
                        # Structural impact: size of smaller subtree if this edge were cut
                        if mst_parent.get(b) == a:
                            child = b
                        elif mst_parent.get(a) == b:
                            child = a
                        else:
                            child = b
                        child_size = mst_subtree_size.get(child, 1)
                        impact = float(min(child_size, n_pcc - child_size))
                        unverified_pool.append(StabilityCandidate(
                            candidate_type="UNVERIFIED",
                            stability=conf,  # use confidence as sort key (lower = higher priority)
                            review_edge=(a, b),
                            structural_impact=max(impact, 1.0),
                            node_pair=(a, b),
                            pcc_id=pcc_id
                        ))

            # Intra-PCC negatives come from the precomputed index instead of probing
            # every node pair, and each negative's tree path is rebuilt from the DFS
            # parent pointers instead of materialising ALL pairs' paths with
            # nx.all_pairs_shortest_path (O(n^2) paths per PCC, every call). A tree
            # has exactly one path between two nodes, so the walk yields the same
            # node sequence from u to v and the same first-minimum edge below.
            negative_edges = _intra_all.get(pcc_id, ([], []))[0]

            if not negative_edges:
                continue

            _depth = {mst_root: 0}
            for _nd in mst_order:
                _p = mst_parent[_nd]
                if _p is not None:
                    _depth[_nd] = _depth[_p] + 1

            for u, v, neg_conf in negative_edges:
                if u not in _depth or v not in _depth:
                    continue
                _a, _b = u, v
                _left, _right = [u], [v]
                while _depth[_a] > _depth[_b]:
                    _a = mst_parent[_a]; _left.append(_a)
                while _depth[_b] > _depth[_a]:
                    _b = mst_parent[_b]; _right.append(_b)
                while _a != _b:
                    _a = mst_parent[_a]; _left.append(_a)
                    _b = mst_parent[_b]; _right.append(_b)
                path = _left + _right[-2::-1]

                if len(path) < 2:
                    continue

                min_conf = float('inf')
                min_edge = None
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    conf = mst[a][b]['weight']
                    if conf < min_conf:
                        min_conf = conf
                        min_edge = (a, b)

                stability = min_conf - neg_conf
                if stability < alpha:
                    # Compute structural impact: size of smaller subtree if weakest edge is cut
                    impact = 1.0
                    if min_edge is not None:
                        a, b = min_edge
                        if mst_parent.get(b) == a:
                            child = b
                        elif mst_parent.get(a) == b:
                            child = a
                        else:
                            child = b
                        child_size = mst_subtree_size.get(child, 1)
                        impact = float(min(child_size, n_pcc - child_size))

                    internal_pool.append(StabilityCandidate(
                        candidate_type="INTERNAL",
                        stability=stability,
                        review_edge=(u, v),
                        structural_impact=max(impact, 1.0),
                        node_pair=(u, v),
                        pcc_id=pcc_id
                    ))

        if os.environ.get('BETA_SELFCHECK'):
            _ref_internal = []
            for _pid, _pcc in enumerate(self._pccs):
                if len(_pcc) < 2:
                    continue
                _m = self._mst_forest.subgraph(_pcc).copy()
                if _m.number_of_edges() == 0:
                    continue
                _n = len(_pcc); _root = next(iter(_pcc)); _par = {}; _ord = []; _st = [(_root, None)]
                while _st:
                    _x, _px = _st.pop(); _par[_x] = _px; _ord.append(_x)
                    for _y in _m.neighbors(_x):
                        if _y != _px:
                            _st.append((_y, _x))
                _sub = {_x: 1 for _x in _ord}
                for _x in reversed(_ord):
                    if _par[_x] is not None:
                        _sub[_par[_x]] += _sub[_x]
                if include_weak_positive:
                    _we, _wc = None, float('inf')
                    for _p1, _p2, _at in _m.edges(data=True):
                        if (min(_p1, _p2), max(_p1, _p2)) in _verified:
                            continue
                        _c = _at.get('weight', 1.0)
                        if _c < _wc:
                            _wc, _we = _c, (_p1, _p2)
                    if _we is not None and _wc < alpha:
                        _p1, _p2 = _we
                        _ch = _p2 if _par.get(_p2) == _p1 else (_p1 if _par.get(_p1) == _p2 else _p2)
                        _cs = _sub.get(_ch, 1)
                        _ref_internal.append(StabilityCandidate(
                            candidate_type="WEAK_POSITIVE", stability=_wc, review_edge=_we,
                            structural_impact=max(float(min(_cs, _n - _cs)), 1.0),
                            node_pair=_we, pcc_id=_pid))
                _pl = sorted(_pcc); _ne = []
                for _i in range(len(_pl)):
                    for _j in range(_i + 1, len(_pl)):
                        _u, _v = _pl[_i], _pl[_j]
                        if self.G.has_edge(_u, _v):
                            _d = self.G[_u][_v].get('data')
                            if _d and _d.label == EdgeLabel.NEGATIVE:
                                _ne.append((_u, _v, _d.confidence))
                if not _ne:
                    continue
                _ap = dict(nx.all_pairs_shortest_path(_m))
                for _u, _v, _nc in _ne:
                    _path = _ap.get(_u, {}).get(_v)
                    if _path is None or len(_path) < 2:
                        continue
                    _mc, _me = float('inf'), None
                    for _k in range(len(_path) - 1):
                        _c = _m[_path[_k]][_path[_k + 1]]['weight']
                        if _c < _mc:
                            _mc, _me = _c, (_path[_k], _path[_k + 1])
                    _stab = _mc - _nc
                    if _stab < alpha:
                        _imp = 1.0
                        if _me is not None:
                            _p1, _p2 = _me
                            _ch = _p2 if _par.get(_p2) == _p1 else (_p1 if _par.get(_p1) == _p2 else _p2)
                            _cs = _sub.get(_ch, 1)
                            _imp = float(min(_cs, _n - _cs))
                        _ref_internal.append(StabilityCandidate(
                            candidate_type="INTERNAL", stability=_stab, review_edge=(_u, _v),
                            structural_impact=max(_imp, 1.0), node_pair=(_u, _v), pcc_id=_pid))
            if _ref_internal != internal_pool:
                _k = next((i for i, (x, y) in enumerate(zip(_ref_internal, internal_pool)) if x != y), None)
                logger.error(f"SELFCHECK FAIL internal pool: ref={len(_ref_internal)} new={len(internal_pool)} "
                             f"first diff idx={_k}: {(_ref_internal[_k] if _k is not None else None)} vs "
                             f"{(internal_pool[_k] if _k is not None else None)}")
                raise AssertionError('internal candidate pool mismatch')
            logger.info(f"SELFCHECK ok: internal_pool {len(internal_pool)} candidates")

        # Step 2: Generate external candidates.
        # Per PDF: external_stability = max_neg - max_pos_inactive for ANY
        # PCC pair with a negative edge (not just those with pos-inactive edges).
        pcc_pair_max_neg: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}
        # track the strongest positive-inactive EDGE per pair (conf, edge), so it
        # can itself become a review target -- the dual of the INTERNAL negative.
        pcc_pair_pos_inactive: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}

        # Read the columnar edge mirror rather than walking all 14.5M edges in
        # Python (~96 s per call, once per outer iteration). Two orderings from the
        # original scan are load-bearing and are reproduced exactly:
        #   * within a PCC pair, strict `>` keeps the FIRST edge attaining the max
        #     confidence in G.edges() order -- that edge becomes the review target;
        #   * the dicts are iterated later by insertion order, i.e. pairs ordered by
        #     FIRST appearance in the scan.
        # The mirror is built by iterating G.edges(data=True), so its index order is
        # exactly that scan order.
        self._ensure_edge_arrays()
        _pmn, _ppi = {}, {}
        if self._earr_u is not None and self._earr_u.size:
            _n = int(max(self._earr_u.max(), self._earr_v.max())) + 1
            _pcc_of = np.full(_n, -1, dtype=np.int64)
            for _nd, _pid in self._node_to_pcc.items():
                if 0 <= _nd < _n:
                    _pcc_of[_nd] = _pid
            _pu = _pcc_of[self._earr_u]; _pv = _pcc_of[self._earr_v]
            _cross = (_pu >= 0) & (_pv >= 0) & (_pu != _pv)
            _lo = np.minimum(_pu, _pv); _hi = np.maximum(_pu, _pv)
            _K = int(_hi.max()) + 1 if _hi.size else 1
            _key = _lo * _K + _hi
            _conf = self._earr_conf

            def _best(_mask):
                _idx = np.flatnonzero(_mask)
                if _idx.size == 0:
                    return {}
                _k = _key[_idx]; _c = _conf[_idx]
                # max confidence, earliest scan position wins ties
                _o = np.lexsort((_idx, -_c, _k))
                _u1, _f1 = np.unique(_k[_o], return_index=True)
                _best_i = _idx[_o[_f1]]
                # first appearance of each pair, to reproduce dict insertion order
                _o2 = np.lexsort((_idx, _k))
                _u2, _f2 = np.unique(_k[_o2], return_index=True)
                _first_i = _idx[_o2[_f2]]
                _seq = np.argsort(_first_i, kind='stable')
                return {(int(_lo[i]), int(_hi[i])):
                        (float(_conf[i]), (int(self._earr_u[i]), int(self._earr_v[i])))
                        for i in _best_i[_seq]}

            _pmn = _best(_cross & (self._earr_lab == _LABEL_CODE[EdgeLabel.NEGATIVE]))
            _ppi = _best(_cross & (self._earr_lab == _LABEL_CODE[EdgeLabel.POSITIVE_INACTIVE]))
        pcc_pair_max_neg.update(_pmn)
        pcc_pair_pos_inactive.update(_ppi)
        if os.environ.get('BETA_SELFCHECK'):
            _rmn, _rpi = {}, {}
            for u, v, attr in self.G.edges(data=True):
                ed = attr.get('data')
                if ed is None:
                    continue
                pcc_u = self._node_to_pcc.get(u); pcc_v = self._node_to_pcc.get(v)
                if pcc_u is None or pcc_v is None or pcc_u == pcc_v:
                    continue
                pr = (min(pcc_u, pcc_v), max(pcc_u, pcc_v))
                if ed.label == EdgeLabel.NEGATIVE:
                    cur = _rmn.get(pr)
                    if cur is None or ed.confidence > cur[0]:
                        _rmn[pr] = (ed.confidence, (u, v))
                elif ed.label == EdgeLabel.POSITIVE_INACTIVE:
                    cur = _rpi.get(pr)
                    if cur is None or ed.confidence > cur[0]:
                        _rpi[pr] = (ed.confidence, (u, v))
            # values AND key order must match: the pair order drives candidate order
            ok = (list(_rmn) == list(_pmn) and list(_rpi) == list(_ppi)
                  and _rmn == _pmn and _rpi == _ppi)
            if not ok:
                logger.error(
                    f"SELFCHECK FAIL candidate pools: neg {len(_rmn)}/{len(_pmn)} "
                    f"order={list(_rmn)[:5]} vs {list(_pmn)[:5]}; "
                    f"pi {len(_rpi)}/{len(_ppi)}")
                raise AssertionError('generate_candidate_pools mirror mismatch')
            logger.info(f"SELFCHECK ok: pools neg={len(_pmn)} posinact={len(_ppi)}")

        # Step 3: Build external candidates.
        # external stability = max_neg_conf - max_posinact_conf; a pair is a
        # candidate when that is < alpha. The review edge is the DECISIVE side:
        # normally the strongest crossing negative, but when review_positive_inactive
        # is on and the strongest positive-inactive rivals/exceeds the negative,
        # review the positive-inactive edge instead (the dual of INTERNAL negative;
        # 'same' -> reactivate -> merge). Pairs joined by a positive-inactive with
        # no crossing negative are only reachable when the flag is on.
        # preserve the original (negative-pair) iteration order so the default-off
        # path is byte-identical; append positive-inactive-only pairs after.
        pairs = list(pcc_pair_max_neg)
        if review_positive_inactive:
            pairs += [p for p in pcc_pair_pos_inactive if p not in pcc_pair_max_neg]
        for pcc_pair in pairs:
            neg = pcc_pair_max_neg.get(pcc_pair)          # (conf, edge) or None
            pi = pcc_pair_pos_inactive.get(pcc_pair)      # (conf, edge) or None
            max_neg_conf = neg[0] if neg else 0.0
            max_pi_conf = pi[0] if pi else 0.0
            stability = max_neg_conf - max_pi_conf
            if stability >= alpha:
                continue
            # pick the review edge: positive-inactive when it is the out-of-place side
            if review_positive_inactive and pi is not None and max_pi_conf >= max_neg_conf:
                review_edge = pi[1]
            elif neg is not None:
                review_edge = neg[1]
            else:
                continue
            edge_key = (min(review_edge), max(review_edge))
            if edge_key in _verified:
                continue
            pcc_a_id, pcc_b_id = pcc_pair
            impact = float(min(len(self._pccs[pcc_a_id]), len(self._pccs[pcc_b_id])))
            external_pool.append(StabilityCandidate(
                candidate_type="EXTERNAL",
                stability=stability,
                review_edge=review_edge,
                structural_impact=max(impact, 1.0),
                pcc_pair=pcc_pair
            ))

        # Step 3a: Generate unverified negative candidates (potential merges)
        # Low-confidence negative edges between PCCs that haven't been human-reviewed
        # NIS-style: if pcc_separation_strength is provided, score by PCC isolation
        # (min separation strength of the two PCCs) rather than individual edge confidence.
        # This biases review toward edges connecting weakly-separated PCCs — the ones
        # most likely to be incorrect negatives hiding real merges.
        if unverified_threshold > 0:
            for u, v, attr in self.G.edges(data=True):
                edge_data = attr.get('data')
                if edge_data is None or edge_data.label != EdgeLabel.NEGATIVE:
                    continue
                edge_key = (min(u, v), max(u, v))
                if edge_key in _verified:
                    continue
                if edge_data.confidence >= unverified_threshold:
                    continue
                # Must be cross-PCC
                pcc_u = self._node_to_pcc.get(u)
                pcc_v = self._node_to_pcc.get(v)
                if pcc_u is None or pcc_v is None or pcc_u == pcc_v:
                    continue
                impact = float(min(len(self._pccs[pcc_u]), len(self._pccs[pcc_v])))

                # NIS-style scoring: use min PCC separation strength (lower = more isolated)
                if pcc_separation_strength is not None:
                    score_key = min(
                        pcc_separation_strength.get(pcc_u, 0.0),
                        pcc_separation_strength.get(pcc_v, 0.0)
                    )
                else:
                    score_key = edge_data.confidence

                unverified_pool.append(StabilityCandidate(
                    candidate_type="UNVERIFIED_NEG",
                    stability=score_key,  # lower = higher priority (more isolated or less confident)
                    review_edge=(u, v),
                    structural_impact=max(impact, 1.0),
                    pcc_pair=(min(pcc_u, pcc_v), max(pcc_u, pcc_v))
                ))

        # Sort each instability pool by combined stability + structural impact + confidence.
        # Lower sort key = higher priority. Internal and external are sorted with
        # the same blend (consistent semantics across pools); the caller may
        # interleave them in whatever priority order it chooses.
        def _get_review_edge_confidence(candidate):
            u, v = candidate.review_edge
            ed = self.get_edge(u, v)
            return ed.confidence if ed else 0.5

        def _sort_instability(pool: List[StabilityCandidate]):
            pool.sort(key=lambda c: c.stability)

        # Internal pool now mixes negative-driven and weak-positive candidates;
        # a single sort by stability puts the most unstable first, so the
        # 1-per-PCC selection naturally prefers a negative-driven candidate
        # over a weak-positive one when a PCC has both.
        _sort_instability(internal_pool)
        _sort_instability(external_pool)
        # Unverified candidates: weakest-confidence (or most-isolated) first.
        unverified_pool.sort(key=lambda c: c.stability)

        n_unverified_pos = sum(1 for c in unverified_pool if c.candidate_type == "UNVERIFIED")
        n_unverified_neg = sum(1 for c in unverified_pool if c.candidate_type == "UNVERIFIED_NEG")
        logger.info(
            f"Generated candidate pools: "
            f"{len(internal_pool)} internal/split, "
            f"{len(external_pool)} external/merge, "
            f"{len(unverified_pool)} unverified "
            f"({n_unverified_pos} pos/split, {n_unverified_neg} neg/merge)"
        )

        return {
            'internal': internal_pool,
            'external': external_pool,
            'unverified': unverified_pool,
        }

    def apply_human_review(self, u: int, v: int, human_agrees: bool, ch: float,
                           cap_confidence: bool = True):
        """
        Apply human review result to an edge.

        Per PDF step 4:
        (a) Agree: add ch to edge confidence
        (b) Disagree: subtract ch; if negative, flip label and confidence

        cap_confidence=True clamps the accumulated confidence at 1.0 (original
        behavior). When False, confidence accumulates unbounded so repeated
        confirmations build real evidence and a single conflicting review can't
        flip a well-established edge (Chuck's suggestion).
        """
        if not self.G.has_edge(u, v):
            return

        edge_data = self.G[u][v]['data']

        old_label = edge_data.label
        edge_data.ranker = 'human'

        if human_agrees:
            new_conf = edge_data.confidence + ch
            edge_data.confidence = min(1.0, new_conf) if cap_confidence else new_conf
            # Re-activate positive-inactive edges when human confirms they're positive
            if edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                edge_data.label = EdgeLabel.POSITIVE
        else:
            edge_data.confidence -= ch
            if edge_data.confidence < 0:
                # Flip label
                if edge_data.label == EdgeLabel.POSITIVE:
                    edge_data.label = EdgeLabel.NEGATIVE
                elif edge_data.label == EdgeLabel.NEGATIVE:
                    edge_data.label = EdgeLabel.POSITIVE
                elif edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                    edge_data.label = EdgeLabel.NEGATIVE
                edge_data.confidence = abs(edge_data.confidence)
        self._touch_edge(u, v, edge_data)

        # Update edge counts and pos-inactive tracking if label changed
        if edge_data.label != old_label:
            self._decrement_edge_count(old_label)
            self._increment_edge_count(edge_data.label)
            key = (min(u, v), max(u, v))
            if edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                self._pos_inactive_edges.add(key)
            elif old_label == EdgeLabel.POSITIVE_INACTIVE:
                self._pos_inactive_edges.discard(key)

        # Saturate the edge score to reflect the human verdict directly.
        # Algorithm-classified positives have score ~0.6-0.8; pushing human-
        # confirmed positives to 1.0 (or negatives to 0.0) lets any downstream
        # consumer of `edge_data.score` (histograms, candidate scoring,
        # spectral metrics) respect human input without special-case ranker
        # lookups.
        if edge_data.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE):
            edge_data.score = 1.0
        elif edge_data.label == EdgeLabel.NEGATIVE:
            edge_data.score = 0.0

        self._invalidate_cache()

    def apply_ground_truth_review(self, u: int, v: int, is_positive: bool):
        """
        Apply a ground-truth human review: directly set the edge label.

        Unlike apply_human_review which uses confidence arithmetic,
        this sets the label definitively with confidence 1.0.
        Used during VST verification where human reviews are authoritative.
        """
        if not self.G.has_edge(u, v):
            return

        edge_data = self.G[u][v]['data']
        old_label = edge_data.label

        new_label = EdgeLabel.POSITIVE if is_positive else EdgeLabel.NEGATIVE
        edge_data.label = new_label
        edge_data.confidence = 1.0
        edge_data.ranker = 'human'
        # Saturate score to match the authoritative GT label.
        edge_data.score = 1.0 if is_positive else 0.0
        self._touch_edge(u, v, edge_data)

        if edge_data.label != old_label:
            self._decrement_edge_count(old_label)
            self._increment_edge_count(edge_data.label)
            key = (min(u, v), max(u, v))
            if edge_data.label == EdgeLabel.POSITIVE_INACTIVE:
                self._pos_inactive_edges.add(key)
            elif old_label == EdgeLabel.POSITIVE_INACTIVE:
                self._pos_inactive_edges.discard(key)

        self._invalidate_cache()

    def get_clustering(self) -> Tuple[Dict[int, Set[int]], Dict[int, int]]:
        """Get current clustering as dictionaries."""
        self._ensure_pcc_cache()
        cluster_dict = {pcc_id: pcc for pcc_id, pcc in enumerate(self._pccs)}
        node2cid = {node: pcc_id for pcc_id, pcc in enumerate(self._pccs) for node in pcc}
        return cluster_dict, node2cid

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the current graph state.

        Optimized to avoid iterating all edges (which can be 10M+).
        - Edge counts: maintained incrementally
        - Internal stability: iterates PCC node pairs (O(N^2) per PCC, N small)
        - External stability: iterates only positive-inactive edges (O(small) vs O(10M))
        """
        self._ensure_pcc_cache()

        pcc_sizes = [len(pcc) for pcc in self._pccs]

        _intra = self._intra_pcc_edges()

        # Compute min internal stability using the per-PCC index built above
        min_internal = float('inf')
        for pcc_id, pcc in enumerate(self._pccs):
            if len(pcc) < 2:
                continue

            # Intra-PCC edges, read from the precomputed per-PCC index rather than
            # probing every node PAIR. The docstring's "N small" does not hold on a
            # graph with a 5429-node PCC: that is ~14.7M has_edge() probes per call,
            # once per outer iteration. Lists are emitted in ascending (u, v) order,
            # which is exactly the order the pair loop produced, so the MST built
            # from positive_edges resolves ties identically.
            negative_edges, positive_edges = _intra.get(pcc_id, ([], []))

            if not negative_edges:
                continue

            # Build MST from positive edges within this PCC
            pcc_graph = nx.Graph()
            for u, v, conf in positive_edges:
                pcc_graph.add_edge(u, v, weight=conf)

            if pcc_graph.number_of_edges() == 0:
                continue

            mst = nx.maximum_spanning_tree(pcc_graph, weight='weight')

            for u, v, neg_conf in negative_edges:
                if not mst.has_node(u) or not mst.has_node(v):
                    continue
                try:
                    path = nx.shortest_path(mst, u, v)
                except nx.NetworkXNoPath:
                    continue
                if len(path) < 2:
                    continue

                msp_strength = min(mst[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                stab = msp_strength - neg_conf
                if stab < min_internal:
                    min_internal = stab

        # Compute min external stability.
        # Per PDF: external_stability(A,B) = max_neg_conf - max_pos_inactive_conf
        # Even PCC pairs with NO positive-inactive edges have finite external stability
        # equal to their max negative confidence (since max_pos_inactive = 0).
        min_external = float('inf')
        pcc_pair_max_neg: Dict[Tuple[int, int], float] = {}
        pcc_pair_pos_inactive: Dict[Tuple[int, int], float] = {}

        # Cross-PCC maxima from the mirror (shared with external_stabilities()).
        pcc_pair_max_neg, pcc_pair_pos_inactive = self._cross_pcc_pair_maxima()

        for pcc_pair, max_neg in pcc_pair_max_neg.items():
            max_pi = pcc_pair_pos_inactive.get(pcc_pair, 0.0)
            stab = max_neg - max_pi
            if stab < min_external:
                min_external = stab

        if os.environ.get('BETA_SELFCHECK'):
            _rn, _rp = {}, {}
            for u, v, attr in self.G.edges(data=True):
                ed = attr.get('data')
                if ed is None:
                    continue
                a_ = self._node_to_pcc.get(u); b_ = self._node_to_pcc.get(v)
                if a_ is None or b_ is None or a_ == b_:
                    continue
                pr = (min(a_, b_), max(a_, b_))
                if ed.label == EdgeLabel.NEGATIVE:
                    _rn[pr] = max(_rn.get(pr, 0.0), ed.confidence)
                elif ed.label == EdgeLabel.POSITIVE_INACTIVE:
                    _rp[pr] = max(_rp.get(pr, 0.0), ed.confidence)
            _rext = float('inf')
            for pr, mn in _rn.items():
                _rext = min(_rext, mn - _rp.get(pr, 0.0))
            _rintra = {}
            for pid, pcc in enumerate(self._pccs):
                if len(pcc) < 2:
                    continue
                pl = sorted(pcc); ne, pe = [], []
                for i in range(len(pl)):
                    for j in range(i + 1, len(pl)):
                        uu, vv = pl[i], pl[j]
                        if self.G.has_edge(uu, vv):
                            d = self.G[uu][vv].get('data')
                            if d:
                                if d.label == EdgeLabel.NEGATIVE:
                                    ne.append((uu, vv, d.confidence))
                                elif d.label == EdgeLabel.POSITIVE:
                                    pe.append((uu, vv, d.confidence))
                if ne or pe:
                    _rintra[pid] = (ne, pe)
            _ok = (_rext == min_external
                   and all(_rintra.get(k, ([], [])) == v for k, v in _intra.items())
                   and all(_intra.get(k, ([], [])) == v for k, v in _rintra.items()))
            if not _ok:
                logger.error(f"SELFCHECK FAIL get_graph_stats: ext {_rext} vs {min_external}; "
                             f"intra pccs {len(_rintra)} vs {len(_intra)}")
                raise AssertionError('get_graph_stats mirror mismatch')
            logger.info(f"SELFCHECK ok: graph_stats ext={min_external:.6f} intra_pccs={len(_intra)}")

        return {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'edge_counts': dict(self._edge_counts),
            'num_pccs': len(self._pccs),
            'pcc_sizes': pcc_sizes,
            'min_pcc_size': min(pcc_sizes) if pcc_sizes else 0,
            'max_pcc_size': max(pcc_sizes) if pcc_sizes else 0,
            'min_internal_stability': min_internal,
            'min_external_stability': min_external,
        }

    def get_mst_cache_stats(self) -> Dict[str, Any]:
        """Get MST forest statistics (no caching - built explicitly when needed)."""
        return {
            'mst_forest_edges': self._mst_forest.number_of_edges() if self._mst_forest else 0
        }

