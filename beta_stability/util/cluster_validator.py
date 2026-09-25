"""Evaluation of a clustering against ground truth, and reporting of the result.

Three metric families are implemented and all of them stay available; which ones
are computed and reported is a configuration choice:

    metrics:
      report: [hungarian, pairwise, ceaf]   # any subset, in any order
      ceaf_sim: dice

Each family is one `Metric` entry in `METRICS`: it knows how to compute itself
from the clustering under test and the reference clustering, which keys it
contributes to the logged block, and which keys it contributes to the metrics
file. Adding a family means adding an entry, not editing the validator.

If `report` is absent only CEAF is reported, which is what the results are
measured with; the other families stay available by listing them.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import networkx as nx

import beta_stability.util.cluster_tools as ct
from beta_stability.util.tools import SetEncoder

logger = logging.getLogger("beta_stability")


@dataclass(frozen=True)
class Metric:
    """One family of clustering metrics.

    compute: (clustering, node2cid, true_clustering, true_node2cid, options) ->
             ordered mapping of logged key -> value
    row:     logged mapping -> mapping of metrics-file key -> value
    """
    name: str
    compute: Callable[..., Dict[str, float]]
    row: Callable[[Dict[str, float]], Dict[str, float]]


def _hungarian(clustering, node2cid, true_clustering, true_node2cid, options):
    h = ct.hungarian_cluster_matching(clustering, true_clustering)
    return {'Hungarian precision': h['precision'],
            'Hungarian recall': h['recall'],
            'Hungarian f1 score': h['f1']}


def _pairwise(clustering, node2cid, true_clustering, true_node2cid, options):
    frac, prec, rec, _per_size, _non_equal, f1 = ct.percent_and_PR(
        clustering, node2cid, true_clustering, true_node2cid)
    return {'frac correct': frac, 'precision': prec, 'recall': rec,
            'error_rate': 1 - frac, 'f1 score': f1}


def _ceaf(clustering, node2cid, true_clustering, true_node2cid, options):
    c = ct.ceaf_cluster_matching(clustering, true_clustering,
                                 sim=options.get('ceaf_sim', 'dice'))
    return {'CEAF precision': c['precision'],
            'CEAF recall': c['recall'],
            'CEAF f1 score': c['f1']}


# Declaration order is the order in which keys appear in the logged block.
METRICS: Dict[str, Metric] = {
    'hungarian': Metric('hungarian', _hungarian, lambda v: {
        'h_f1': v['Hungarian f1 score'],
        'h_precision': v['Hungarian precision'],
        'h_recall': v['Hungarian recall']}),
    'pairwise': Metric('pairwise', _pairwise, lambda v: {
        'pcc_f1': v['f1 score'],
        'pcc_precision': v['precision'],
        'pcc_recall': v['recall']}),
    'ceaf': Metric('ceaf', _ceaf, lambda v: {
        'ceaf_f1': v['CEAF f1 score'],
        'ceaf_precision': v['CEAF precision'],
        'ceaf_recall': v['CEAF recall']}),
}

DEFAULT_REPORT = ('ceaf',)


@dataclass
class ReportingConfig:
    """What to measure and where to put it."""
    report: List[str] = field(default_factory=lambda: list(DEFAULT_REPORT))
    options: Dict[str, object] = field(default_factory=lambda: {'ceaf_sim': 'dice'})
    metrics_file: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    trace_reachable: bool = True

    def __post_init__(self):
        unknown = [m for m in self.report if m not in METRICS]
        if unknown:
            raise ValueError(f'unknown metric(s) {unknown}; known: {sorted(METRICS)}')
        # keep declaration order so the logged block is stable regardless of
        # the order the caller listed them in
        self.report = [m for m in METRICS if m in self.report]

    @classmethod
    def from_config(cls, config, default_method=None, default_species=None, default_seed=None):
        m = (config.get('metrics') or {})
        report = m.get('report') or list(DEFAULT_REPORT)
        return cls(
            report=list(report),
            options={'ceaf_sim': m.get('ceaf_sim', 'dice')},
            metrics_file=m.get('metrics_file'),
            metadata={'method': m.get('method', default_method),
                      'species': m.get('species', default_species),
                      'seed': m.get('seed', default_seed)},
            trace_reachable=m.get('trace_reachable', True),
        )


class ClusterValidator(object):
    def __init__(self, gt_clustering, gt_node2cid, reporting=None):
        self.gt_clustering = gt_clustering
        self.gt_node2cid = gt_node2cid
        self.prev_num_human = 0
        self.gt_results = []
        self.r_results = []
        self.reporting = reporting or ReportingConfig()

        if self.reporting.metrics_file:
            metrics_dir = os.path.dirname(self.reporting.metrics_file)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            # Start each configured run with a fresh metrics file.
            open(self.reporting.metrics_file, 'w').close()

    @classmethod
    def from_config(cls, config, gt_clustering, gt_node2cid, **defaults):
        return cls(gt_clustering, gt_node2cid,
                   ReportingConfig.from_config(config, **defaults))

    def create_reachable(self, G):
        """The "reachable" ground truth: the best obtainable result given matches
        that were never found, which can leave a true cluster disconnected."""
        r_clustering = {}
        k = 0
        for cc in self.gt_clustering.values():
            H = G.subgraph(cc)
            for new_cc in nx.connected_components(H):
                r_clustering[k] = new_cc
                k += 1
        return r_clustering, ct.build_node_to_cluster_mapping(r_clustering)

    def trace_start_human(self, clustering, node2cid, G, num_human=0):
        """Start recording accuracy against the number of human decisions, both
        against ground truth and, optionally, against the reachable ground truth."""
        self.gt_results = [self._trace_gt(clustering, node2cid, num_human)]
        self.r_results = [r] if (r := self._trace_reachable(clustering, node2cid, G, num_human)) else []
        self.prev_num_human = num_human

    def trace_iter_compare_to_gt(self, clustering, node2cid, num_human, G):
        if num_human <= self.prev_num_human:
            return
        self.gt_results.append(self._trace_gt(clustering, node2cid, num_human))
        if (r := self._trace_reachable(clustering, node2cid, G, num_human)):
            self.r_results.append(r)
        self.prev_num_human = num_human

    def _trace_gt(self, clustering, node2cid, num_human):
        return self.incremental_stats(num_human, clustering, node2cid,
                                      self.gt_clustering, self.gt_node2cid, 'Basic stats')

    def _trace_reachable(self, clustering, node2cid, G, num_human):
        if not self.reporting.trace_reachable:
            return None
        r_clustering, r_node2cid = self.create_reachable(G)
        return self.incremental_stats(num_human, clustering, node2cid,
                                      r_clustering, r_node2cid, 'Reachable stats')

    def incremental_stats(self, num_human, clustering, node2cid, true_clustering,
                          true_node2cid, info_text="Incremental stats"):
        result = {'num human': num_human,
                  'num clusters': len(clustering),
                  'num true clusters': len(true_clustering)}
        for name in self.reporting.report:
            result.update(METRICS[name].compute(
                clustering, node2cid, true_clustering, true_node2cid, self.reporting.options))

        logger.info(f'{info_text}: {json.dumps(result, indent=4, cls=SetEncoder)}')
        if info_text == 'Basic stats':
            self._write_metrics_row(result)
        return result

    def _write_metrics_row(self, result):
        if not self.reporting.metrics_file:
            return
        row = {**self.reporting.metadata, 'reviews': result['num human']}
        for name in self.reporting.report:
            row.update(METRICS[name].row(result))
        with open(self.reporting.metrics_file, 'a') as f:
            f.write(json.dumps(row, cls=SetEncoder) + '\n')
