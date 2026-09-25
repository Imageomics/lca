"""
Beta Stability Algorithm

Implements the algorithm as defined in:
"LCA V2 Formulation" by Charles Stewart, January 2026

Algorithm phases:
1. Phase 0: Make graph 0-stable (no human input, only deactivate positive edges)
2. Active Review: Iteratively improve stability with human reviews until target alpha

Per PDF:
- Only positive edges can be deactivated (become positive-inactive)
- Negative edges are NEVER deactivated
- Human review: agree adds ch, disagree subtracts ch (flip if negative)
- After human review, re-run 0-stability
"""

import os
import time
import networkx as nx
import numpy as np
import logging
from typing import Dict, List, Tuple, Set, Optional, Any

from beta_stability.stability_graph import StabilityGraph, EdgeLabel, StabilityCandidate, _CODE_NEGATIVE
from beta_stability.util.tools import order_edge
from beta_stability.util.tools import resolve_checkpoints, CHECKPOINT_LABEL

logger = logging.getLogger("beta_stability")


class BetaStabilityAlgorithm:
    """
    Beta Stability stability-driven clustering algorithm.

    Implements the standard algorithm interface (step, is_finished, get_clustering).
    """

    def __init__(self, config: Dict, classifier_manager, cluster_validator=None):
        self.config = config
        self.classifier_manager = classifier_manager
        self.cluster_validator = cluster_validator

        # Core graph structure
        self.graph = StabilityGraph()

        # Algorithm state
        self.phase = "PHASE0"  # PHASE0 -> ACTIVE_REVIEW -> FINISHED
        self.target_alpha = config.get('target_alpha', 0.5)
        self.num_human_reviews = 0
        self.max_human_reviews = config.get('max_human_reviews', 1000)

        # Human review parameters
        self.ch = config.get('review_confidence', config.get('human_confidence', 0.5))  # Confidence change per review
        self.edges_per_review_batch = config.get('edges_per_review_batch', 20)  # Max edges per batch
        self.human_attempts = {}  # Track review attempts per edge: {(n0, n1): count}

        # Cross-PCC edge discovery
        self._all_edges_sorted = None
        self._sorted_edge_index = 0
        self.cross_pcc_max_edges = config.get('cross_pcc_max_edges', 0)

        # Phase 0 controls
        self.max_phase0_iterations = config.get('max_phase0_iterations', 20)
        self.phase0_alpha = config.get('phase0_alpha', 0.0)

        # Cut margin m >= 0: shift the auto-cut threshold to alpha = -m so a
        # positive edge is deactivated only when the crossing negative beats the
        # weakest positive link on its MSP by margin m. Marginal negatives (stability
        # in [-m, target_alpha)) are then deferred to human review instead of being
        # auto-cut on classifier evidence alone. Default 0.0 = current behavior.
        self.cut_margin = config.get('cut_margin', 0.0)

        # Stable-merge dual: make_zero_stable is cut-only. When enabled, after
        # each cut pass we also reactivate positive-inactive edges that join two
        # PCCs alpha-stably (the merge dual), then re-cut to rectify. This is the
        # deterministic version of what a noisy reviewer does by accident:
        # propose merges, let the cut rule keep the contradiction-free ones.

        # Merge dual (contract-on-cut, the mirror of the cut's delete-on-cycle).
        # Trigger: a human-confirmed negative that SPLITS a PCC. We then reunite
        # the two halves by flipping their strongest same-evidence crossing edge
        # to positive (a "contraction"), keep the confirmed negative inside, and
        # let the existing cut re-derive the boundary. If the re-cut deactivates
        # the flipped edge, it is restored to NEGATIVE (the human verdict stands).
        # Confidence given to the flipped edge (oscillation knob). Higher = the
        # reunion holds and the re-cut must find a different boundary.
        # Which crossing edge to contract: 'highest_score' = weakest negative =
        # most-probable same (the dual pick); 'lowest_score' = strongest negative.

        # Diagnostic: tally reviews by (error?, verdict, current-label, created-
        # instability?) so we can see which of the 4 error kinds the imperfect
        # reviewer makes and which actually flip structure (reversible merge vs
        # irreversible split). Populated in _apply_human_feedback when GT exists.
        from collections import Counter as _Counter
        self._review_tally = _Counter()

        # Stability tracking (Task 3): snapshot per-PCC internal stability each
        # active-review iteration so we can visualize convergence/oscillation.
        self.track_stability = bool(config.get('track_stability', False))
        self.stability_track_path = config.get('stability_track_path', None)
        self._stability_snapshots = []
        self._active_iter = 0
        self._phase0_prev_pcc_count = None
        self._phase0_stall_count = 0

        # Densification strategy: if True, add likely negative edges first (more aggressive)
        # Default False adds likely positive edges first (less fragmentation)

        # Explicit cap on EXTERNAL allocation per batch (None = uncapped,
        # absorbs whatever's leftover after INTERNAL).
        self.external_per_batch = config.get('external_per_batch', None)

        # WEAK_POSITIVE selector (Chuck's proposal): review the weakest positive
        # MST edge of each PCC, threshold-free, weakest-first, 1-per-PCC. Lets
        # components held together only by weak positives (no internal negative
        # to flag them) be split. Fills remaining budget after INTERNAL/EXTERNAL,
        # so the 1-per-PCC cap makes it target PCCs that got no other review.
        # Default ON: this is the current baseline (send weakest positive or negative).
        self.review_weakest_positive = bool(config.get('review_weakest_positive', True))
        # Dual of the INTERNAL negative selector: send the strongest cross-PCC
        # positive-inactive edge for review when it is the side making the pair
        # unstable (max_posinact >= max_neg); 'same' -> reactivate -> merge.
        # Default ON: part of the baseline (small but real, artifact-free gain).
        self.review_positive_inactive = bool(config.get('review_positive_inactive', True))
        # Cap accumulated review confidence at 1.0 (True = original). False lets
        # confirmations build unbounded evidence so a well-established edge resists
        # a single conflicting review (Chuck's suggestion).
        self.cap_confidence = bool(config.get('cap_confidence', True))
        # Batch seat MEMBERSHIP rule. Default (False) = legacy type-priority:
        # INTERNAL (incl. weak-positive) claims seats first, EXTERNAL fills the
        # rest. True = unified: fill the shared budget from internal ∪ external
        # ranked by stability (lowest = most unstable first), so a candidate
        # wins a seat on its β-deficit regardless of type. Removes the early
        # weak-positive flood that displaces high-value external merges.
        self.unified_candidate_ranking = bool(config.get('unified_candidate_ranking', False))

        # Runtime metric trajectory tracking. When enabled, on each active-
        # review iteration the algorithm also computes top-N candidates by a
        # set of alternative metrics and reports, against ground truth, how
        # many of each set are actual false negatives. Lets us see how each
        # metric's discriminative power evolves as human reviews accumulate.
        # Requires cluster_validator with ground truth (simulation mode).
        # Validation
        self.validation_step = config.get('validation_step', 20)
        self.validation_initialized = False
        # Exact evaluation checkpoints at requested review counts (util/tools.py);
        # 'N' = one review per annotation. Resolved once the graph's node count is known.
        self.report_at_reviews = config.get('report_at_reviews', ['N'])
        self._checkpoints = None

        # Diagnostic: map pending review edges to their candidate type
        self._pending_review_types: Dict[Tuple[int, int], str] = {}
        # Cumulative agreement/disagreement counts per candidate type
        from collections import defaultdict
        self._cumulative_agree: Dict[str, int] = defaultdict(int)
        self._cumulative_disagree: Dict[str, int] = defaultdict(int)

        # Phase 0 iteration counter for metric logging
        self.phase0_iteration = 0

        logger.info(f"Beta Stability Algorithm initialized")
        logger.info(f"  Target alpha: {self.target_alpha}")
        logger.info(f"  Phase 0 alpha: {self.phase0_alpha}")
        logger.info(f"  Cut margin: {self.cut_margin}")
        logger.info(f"  Max Phase 0 iterations: {self.max_phase0_iterations}")
        logger.info(f"  Human confidence (ch): {self.ch}")
        logger.info(f"  Max human reviews: {self.max_human_reviews}")

    def _stabilize(self, alpha: float) -> int:
        """Run the cut rule and, if enabled, its merge dual to a joint fixpoint.

        Returns the number of positive edges deactivated.
        """
        return self.graph.make_zero_stable(alpha=alpha)

    def _snapshot_stability(self, phase='active'):
        per_pcc = self.graph.per_pcc_internal_stability()
        external_vals = self.graph.external_stabilities()
        # Save the RAW per-decision stabilities so ANY stability plot (unified,
        # split, min, distribution, fraction<beta, per-PCC trajectory) is doable
        # offline with no re-run. `internal` keeps PCC identity for trajectories;
        # `external` is the flat list of separation-decision stabilities.
        # `step` is a global monotonic index across BOTH phases (phase 0 resolves
        # most instability before any review), `phase` marks which.
        self._stability_snapshots.append({
            'step': len(self._stability_snapshots),
            'phase': phase,
            'phase0_iter': self.phase0_iteration if phase == 'phase0' else None,
            'reviews': self.num_human_reviews,
            'n_pccs': len(per_pcc),
            'internal': [[a, s, (None if st is None else round(float(st), 5))]
                         for (a, s, st) in per_pcc],
            'external': [round(float(x), 5) for x in external_vals],
        })

    def _dump_stability_track(self):
        if not self.track_stability or not self._stability_snapshots:
            return
        import json
        path = self.stability_track_path or 'stability_track.json'
        try:
            with open(path, 'w') as f:
                json.dump(self._stability_snapshots, f)
            logger.info(f"Stability track: wrote {len(self._stability_snapshots)} snapshots to {path}")
        except Exception as e:
            logger.warning(f"Stability track dump failed: {e}")


    def step(self, new_edges: List[Tuple]) -> List[Tuple[int, int, float]]:
        """
        Perform one step of the algorithm.

        Args:
            new_edges: List of (n0, n1, score, verifier_name) tuples

        Returns:
            List of edges needing human review: [(n0, n1, score), ...]
        """
        # Process and add incoming edges
        self._process_and_add_edges(new_edges)

        # Count human reviews and apply feedback
        needs_restabilization = False
        # Track agreement/disagreement per candidate type for informativeness metric
        from collections import defaultdict
        type_agree = defaultdict(int)
        type_disagree = defaultdict(int)

        for edge in new_edges:
            if len(edge) > 3 and 'human' in str(edge[3]):
                n0, n1, score, verifier_name = edge[:4]

                # Track human review attempts per edge
                edge_key = (min(n0, n1), max(n0, n1))
                self.human_attempts[edge_key] = self.human_attempts.get(edge_key, 0) + 1
                self.num_human_reviews += 1

                # Determine agreement BEFORE applying feedback
                edge_data = self.graph.get_edge(n0, n1)
                if edge_data:
                    human_decision = score > 0.5
                    current_is_positive = edge_data.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE)
                    human_agrees = (human_decision == current_is_positive)
                    ctype = self._pending_review_types.get(edge_key, "UNKNOWN")
                    if human_agrees:
                        type_agree[ctype] += 1
                    else:
                        type_disagree[ctype] += 1

                if self._apply_human_feedback(n0, n1, score):
                    needs_restabilization = True
                if self.num_human_reviews in self._checkpoint_counts():
                    self._log_checkpoint(needs_restabilization)

        # Log per-type disagreement rates (higher = more informative candidates)
        all_types = set(type_agree.keys()) | set(type_disagree.keys())
        if all_types:
            # Update cumulative counts
            for ctype in all_types:
                self._cumulative_agree[ctype] += type_agree[ctype]
                self._cumulative_disagree[ctype] += type_disagree[ctype]


            # Batch stats
            parts = []
            for ctype in sorted(all_types):
                a = type_agree[ctype]
                d = type_disagree[ctype]
                total = a + d
                pct = d / total * 100 if total > 0 else 0
                parts.append(f"{ctype}: {d}/{total} disagree ({pct:.0f}%)")
            logger.info(f"Review informativeness (batch): {', '.join(parts)}")

            # Cumulative stats
            all_cumulative = set(self._cumulative_agree.keys()) | set(self._cumulative_disagree.keys())
            parts = []
            for ctype in sorted(all_cumulative):
                a = self._cumulative_agree[ctype]
                d = self._cumulative_disagree[ctype]
                total = a + d
                pct = d / total * 100 if total > 0 else 0
                parts.append(f"{ctype}: {d}/{total} disagree ({pct:.0f}%)")
            logger.info(f"Review informativeness (cumulative): {', '.join(parts)}")

        # Detect PCC splits from human reviews and queue cross-check edges
        # Per PDF step 5: "Update the labels of edges to make the graph 0-stable"
        if needs_restabilization:
            logger.info("Starting re-stabilization...")
            _t_restab = time.time()
            deactivations = self._stabilize(alpha=-self.cut_margin)
            logger.info(f"TIMER restabilize {time.time() - _t_restab:.2f}s")
            if deactivations > 0:
                logger.info(f"Re-stabilized after human reviews: {deactivations} deactivations")

        # Phase dispatch
        if self.phase == "PHASE0":
            return self._phase0_step()
        elif self.phase == "ACTIVE_REVIEW":
            return self._active_review_step()
        else:
            return []

    def _phase0_step(self) -> List[Tuple[int, int, float]]:
        """
        Phase 0: Make graph 0-stable without human input.

        Per PDF: "The initial goal is to make positive-inactive assignments that
        will make the graph 0-stable. This is done entirely without human input."
        """
        # Increment phase 0 iteration counter
        self.phase0_iteration += 1
        logger.info(f"=== Phase 0 iteration {self.phase0_iteration} ===")

        # Snapshot BEFORE this iteration's stabilization -- iteration 1 captures
        # the raw initial graph (maximal instability), later ones the progressive
        # collapse to 0-stable. This is where most instability is resolved.
        if self.track_stability:
            self._snapshot_stability(phase='phase0')

        # Init eval: the raw initial-graph clustering, before any 0-stability
        # change, for the stagewise-decomposition table (logged as "Init").
        if self.phase0_iteration == 1 and self.cluster_validator:
            # graph not needed here -- get_clustering() would copy all 14.5M edges
            _clust, _n2c = self.graph.get_clustering()
            self.cluster_validator.incremental_stats(
                0, _clust, _n2c,
                self.cluster_validator.gt_clustering,
                self.cluster_validator.gt_node2cid, 'Init')


        pccs = self.graph.get_pccs()
        logger.info(f"Found {len(pccs)} PCCs")

        # Make alpha-stable (phase0_alpha controls aggressiveness, default 0.0)
        deactivations = self._stabilize(alpha=self.phase0_alpha - self.cut_margin)
        logger.info(f"Made {deactivations} edges positive-inactive for {self.phase0_alpha}-stability")

        # Log metrics only when significant changes occur (saves ~30s per skipped iteration)
        if deactivations >= 5 or self.phase0_iteration <= 1:
            self._log_phase0_metrics(self.phase0_iteration)

        # Check for transition to ACTIVE_REVIEW
        all_edges_processed = self._all_edges_sorted and self._sorted_edge_index >= len(self._all_edges_sorted)
        no_edges_to_process = self._all_edges_sorted is None or len(self._all_edges_sorted) == 0
        converged = (deactivations == 0)

        # Detect oscillation: PCC count unchanged for consecutive iterations
        current_pcc_count = len(pccs)
        if current_pcc_count == self._phase0_prev_pcc_count:
            self._phase0_stall_count += 1
        else:
            self._phase0_stall_count = 0
        self._phase0_prev_pcc_count = current_pcc_count

        # Force convergence if: iteration limit reached OR oscillating (stalled 3+ iters)
        force_converge = (
            self.phase0_iteration >= self.max_phase0_iterations or
            (self._phase0_stall_count >= 3 and all_edges_processed)
        )

        if force_converge and not converged:
            logger.info(f"=== Phase 0 complete - forced (iteration={self.phase0_iteration}, "
                       f"stall_count={self._phase0_stall_count}, deactivations={deactivations}) ===")
            converged = True

        if converged or force_converge:
            logger.info("=== Phase 0 complete - converged ===")
            self._log_stats()

            # Validation after Phase 0 completes
            self._handle_validation()

            if self.target_alpha > 0:
                self.phase = "ACTIVE_REVIEW"
                logger.info("=== Starting Active Review Phase ===")
                return self._active_review_step()
            else:
                self.phase = "FINISHED"
                return []

        if all_edges_processed and not converged:
            logger.info(f"All edges processed but not converged (deactivations={deactivations}) - continuing")

        # Continue Phase 0 (no human review yet)
        return []

    def _compute_adaptive_quotas(self, remaining_budget: int) -> Dict[str, int]:
        """Compute per-bucket slot allocation for the next batch's selectors.

        With experimental selectors stripped, EXTERNAL is the sole non-INTERNAL
        bucket, so it takes the full remaining budget after INTERNAL (capped
        by external_per_batch if set).
        """
        if remaining_budget <= 0:
            return {'EXTERNAL': 0}
        quota = remaining_budget
        if self.external_per_batch is not None:
            quota = min(quota, int(self.external_per_batch))
        return {'EXTERNAL': quota}

    def _active_review_step(self) -> List[Tuple[int, int, float]]:
        """
        Active Review: Select candidates for human review to improve stability.

        Per PDF algorithm:
        1. Compute internal and external stability for all pairs
        2. Order by increasing stability
        3. Select first k pairs for human review
        4. (Human provides feedback - handled in step())
        5. Update labels to make graph 0-stable
        """
        # Check termination
        if self.num_human_reviews >= self.max_human_reviews:
            logger.info(f"Reached max human reviews ({self.max_human_reviews})")
            self._finalize('max_human_reviews')
            return []

        # Stability snapshot for this iteration (Task 3).
        if self.track_stability:
            self._snapshot_stability()

        # Get current stability
        _t = time.time()
        stats = self.graph.get_graph_stats()
        logger.info(f"TIMER get_graph_stats {time.time() - _t:.2f}s")
        min_internal = stats['min_internal_stability']
        min_external = stats['min_external_stability']

        # Handle inf values correctly:
        # - inf means "perfectly stable" (no conflicts), not "unknown"
        # - Single-element PCCs have inf internal stability (nothing to destabilize)
        # - PCCs with no cross-PCC negative edges have inf external stability
        if min_internal == float('inf') and min_external == float('inf'):
            # Graph is perfectly stable
            current_alpha = float('inf')
        elif min_internal == float('inf'):
            # Internal is perfect, use external only
            current_alpha = min_external
        elif min_external == float('inf'):
            # External is perfect, use internal only
            current_alpha = min_internal
        else:
            current_alpha = min(min_internal, min_external)

        logger.info(f"Current stability: internal={min_internal:.4f}, external={min_external:.4f}")

        if current_alpha >= self.target_alpha:
            logger.info(f"Reached target alpha ({self.target_alpha})")
            self._finalize('target_alpha')
            return []

        # Generate candidate pools — internal, external — without selecting
        # or capping. Caller (here) owns priority order and budget.
        logger.info("Starting candidate generation...")
        # No edge is ever retired by a review count. There is no per-edge review cap:
        # an edge stays eligible for as long as it is still a stability candidate, and
        # it leaves the pool only when review has actually settled it (accumulated
        # confidence lifts it above beta) or the graph no longer flags it. Capping tries
        # was arbitrary and, at the earlier `set(human_attempts.keys())`, retired every
        # edge after ONE review -- so a contested edge (e.g. a flipped 2-vertex PCC below
        # beta) could never be revisited. Termination is governed by target_alpha,
        # no_candidates, or the global review budget -- never by a per-edge quota.
        verified_edges = set()
        _t = time.time()
        pcc_strength = self._compute_pcc_separation_strength()
        logger.info(f"TIMER pcc_separation_strength {time.time() - _t:.2f}s")

        _t = time.time()
        pools = self.graph.generate_candidate_pools(
            alpha=self.target_alpha,
            verified_edges=verified_edges,
            unverified_threshold=0.0,
            pcc_separation_strength=pcc_strength,
            include_weak_positive=self.review_weakest_positive,
            review_positive_inactive=self.review_positive_inactive,
        )
        logger.info(f"TIMER generate_candidate_pools {time.time() - _t:.2f}s")
        logger.info("Candidate generation complete")

        # Compose the batch in strict priority order under ONE shared budget.
        # Order: INTERNAL → EXTERNAL. Internal owns intra-PCC splits, the
        # cheapest high-value reviews; EXTERNAL fills the remaining slots.
        # Total len(batch) <= edges_per_review_batch.
        budget = self.edges_per_review_batch
        batch: List[Tuple[int, int, float]] = []
        self._pending_review_types = {}
        pccs_used: Set[int] = set()
        used_keys: Set[Tuple[int, int]] = set()
        type_counts: Dict[str, int] = {}
        escalated = 0

        def _add_stability_candidate(candidate) -> bool:
            nonlocal escalated
            """Add a StabilityCandidate respecting per-PCC conflict + dedup.

            INTERNAL/EXTERNAL candidates are 1-per-PCC (the MST cut-point
            is a single decision per cluster).
            """
            if len(batch) >= budget:
                return False
            enforce_pcc_conflict = candidate.pcc_id is not None
            if enforce_pcc_conflict and candidate.pcc_id in pccs_used:
                return False
            u, v = candidate.review_edge
            key = (min(u, v), max(u, v))
            if key in used_keys:
                return False
            edge_data = self.graph.get_edge(u, v)
            score = edge_data.score if edge_data else 0.0

            # Verifier chain: a candidate goes to a human only once every
            # automatic verifier after the one that last classified it has had
            # its turn. Escalating costs no seat, so the batch still presents
            # edges_per_review_batch pairs to the reviewer. With a single
            # automatic verifier the chain is exhausted immediately and every
            # candidate goes straight to review, as before.
            current = edge_data.ranker if edge_data else None
            nxt = self.classifier_manager.get_next_classifier(current)
            if nxt != 'human':
                n0, n1, s, conf, label, ranker = self.classifier_manager.classify_edge(u, v, nxt)
                self.graph.add_edge(
                    n0, n1,
                    EdgeLabel.POSITIVE if label == "positive" else EdgeLabel.NEGATIVE,
                    conf, s, ranker)
                escalated += 1
                logger.debug(f"Escalated edge ({u}, {v}) from '{current}' to '{nxt}' "
                             f"instead of human review")
                used_keys.add(key)   # settled this round; do not re-offer it
                return False         # no seat consumed: composition keeps scanning

            batch.append((u, v, score))
            self._pending_review_types[key] = candidate.candidate_type
            used_keys.add(key)
            if enforce_pcc_conflict:
                pccs_used.add(candidate.pcc_id)
            type_counts[candidate.candidate_type] = type_counts.get(candidate.candidate_type, 0) + 1
            return True

        if self.unified_candidate_ranking:
            # Unified seat membership: internal ∪ external compete for the
            # shared budget by stability (lowest = most unstable first). A
            # weak-positive (stability >= 0) only wins a seat once the conflict
            # candidates (internal-negative / external merge, stability <= 0)
            # are exhausted, so it no longer displaces merges early. The
            # external_per_batch cap (if set) still bounds EXTERNAL seats.
            ext_cap = self.external_per_batch
            ext_added = 0
            merged = sorted(pools['internal'] + pools['external'],
                            key=lambda c: c.stability)
            for c in merged:
                if len(batch) >= budget:
                    break
                if (c.candidate_type == 'EXTERNAL' and ext_cap is not None
                        and ext_added >= int(ext_cap)):
                    continue
                if _add_stability_candidate(c) and c.candidate_type == 'EXTERNAL':
                    ext_added += 1
        else:
            # Legacy type-priority: INTERNAL claims seats first, EXTERNAL fills
            # the rest (optionally capped). 1-per-PCC enforced.
            for c in pools['internal']:
                if len(batch) >= budget:
                    break
                _add_stability_candidate(c)

            remaining_after_internal = budget - len(batch)
            quotas = self._compute_adaptive_quotas(remaining_after_internal)
            if quotas['EXTERNAL'] > 0:
                ext_quota = min(quotas['EXTERNAL'], budget - len(batch))
                ext_added = 0
                for c in pools['external']:
                    if len(batch) >= budget or ext_added >= ext_quota:
                        break
                    if _add_stability_candidate(c):
                        ext_added += 1

        logger.info(
            f"Batch composition (budget={budget}): "
            f"INTERNAL={type_counts.get('INTERNAL', 0)}, "
            f"WEAK_POSITIVE={type_counts.get('WEAK_POSITIVE', 0)}, "
            f"EXTERNAL={type_counts.get('EXTERNAL', 0)}, "
            f"total={len(batch)}, escalated={escalated}"
        )

        if not batch:
            if escalated:
                # Every candidate still had an automatic verifier left, so the
                # round did algorithmic work and the labels changed. That is
                # progress, not exhaustion: recompute and come back.
                logger.info(f"No edges for review this round: {escalated} candidate(s) "
                            f"escalated to the next automatic verifier")
                return []
            logger.info("No review candidates - graph at maximum stability")
            self._finalize('no_candidates')
            return []

        logger.info(f"Selected {len(batch)} non-conflicting edges for parallel review")

        # Validation
        logger.info("Starting validation...")
        self._handle_validation()
        logger.info("Validation complete")

        logger.info(f"Returning {len(batch)} edges for human review")
        return batch


    def _apply_human_feedback(self, n0: int, n1: int, score: float) -> bool:
        """
        Apply human feedback to an edge.

        Per PDF step 4: Apply confidence change (agree/disagree)

        Returns: True if this could create new instability requiring re-stabilization.
        """
        if not self.graph.has_edge(n0, n1):
            return False

        edge_data = self.graph.get_edge(n0, n1)
        human_decision = score > 0.5  # Human says positive if score > 0.5

        # Determine if human agrees with current label
        current_is_positive = edge_data.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE)
        human_agrees = (human_decision == current_is_positive)

        # Check if this could create instability:
        # 1. Disagree on negative -> flips to positive (new positive edge between PCCs)
        # 2. Agree on positive-inactive -> re-activates (merges PCCs, may need re-stabilization)
        # 3. Disagree on an active positive (WEAK_POSITIVE review) -> flips to
        #    negative; if the PCC does not split on that edge alone it becomes a
        #    within-PCC negative and can require re-stabilization.
        was_positive_inactive = edge_data.label == EdgeLabel.POSITIVE_INACTIVE
        was_active_positive = edge_data.label == EdgeLabel.POSITIVE
        could_create_instability = (
            ((not human_agrees) and (not current_is_positive)) or
            ((not human_agrees) and was_active_positive) or
            (human_agrees and was_positive_inactive)
        )

        # Diagnostic tally: categorize this review vs ground truth.
        validator = self.cluster_validator
        gt = getattr(validator, 'gt_node2cid', None) if validator else None
        if gt is not None:
            gu, gv = gt.get(n0), gt.get(n1)
            if gu is not None and gv is not None:
                gt_same = (gu == gv)
                verdict_same = human_decision                 # True = human says "same"
                is_error = (verdict_same != gt_same)
                verdict = 'same' if verdict_same else 'diff'
                label = 'pos' if current_is_positive else 'neg'
                # reversibility: a verdict that flips a positive->negative is a
                # SPLIT (cut-only rectifier cannot undo); negative->positive is a
                # MERGE (rectifier can cut back).
                self._review_tally[(
                    'ERR' if is_error else 'ok', verdict, label,
                    'flip' if could_create_instability else 'noflip')] += 1

        # Apply the review
        self.graph.apply_human_review(n0, n1, human_agrees, self.ch,
                                      cap_confidence=self.cap_confidence)

        logger.info(f"Human review on ({n0},{n1}): decision={'positive' if human_decision else 'negative'}, "
                   f"agrees={human_agrees}")

        return could_create_instability

    def _process_and_add_edges(self, raw_edges: List[Tuple]):
        """Process raw edges and add to graph."""
        prob_human_correct = self.config.get('prob_human_correct', 0.98)

        for edge in raw_edges:
            if len(edge) < 4:
                continue

            n0, n1, score, verifier_name = edge[:4]

            if verifier_name in {'human', 'simulated_human', 'ui_human'}:
                # Human edge - don't add directly, handled by _apply_human_feedback
                continue
            else:
                # Algorithmic classification
                classified = self.classifier_manager.classify_edge(n0, n1, verifier_name)
                _, _, score, confidence, label, ranker = classified
                edge_label = EdgeLabel.POSITIVE if label == "positive" else EdgeLabel.NEGATIVE
                self.graph.add_edge(n0, n1, edge_label, confidence, score, ranker)


    def _compute_pcc_separation_strength(self) -> Dict[int, float]:
        """Compute separation strength for each PCC (NIS-style n_hat analog).

        For each PCC, sums the max negative confidence to each other PCC it connects to.
        High sum = confidently separated from all neighbors (well-known identity).
        Low sum = weakly separated (uncertain, may need merging) = "isolated" in NIS sense.

        Returns: Dict mapping pcc_id -> total separation confidence.
        """
        self.graph._ensure_pcc_cache()
        node_to_pcc = self.graph._node_to_pcc
        from collections import defaultdict

        # Read the columnar edge mirror instead of walking every node's adjacency.
        # The Python walk visited each edge twice -- ~29M dereferences per call at
        # ~3 us each, ~94 s -- and this runs once per outer iteration.
        g = self.graph
        g._ensure_edge_arrays()
        if g._earr_u is None or g._earr_u.size == 0:
            return {}
        n_nodes = int(max(g._earr_u.max(), g._earr_v.max())) + 1
        pcc_of = np.full(n_nodes, -1, dtype=np.int64)
        for nd, pid in node_to_pcc.items():
            if 0 <= nd < n_nodes:
                pcc_of[nd] = pid
        neg = g._earr_lab == _CODE_NEGATIVE
        pu = pcc_of[g._earr_u[neg]]
        pv = pcc_of[g._earr_v[neg]]
        conf = g._earr_conf[neg]
        keep = (pu >= 0) & (pv >= 0) & (pu != pv)
        if not keep.any():
            return {}
        lo = np.minimum(pu[keep], pv[keep]); hi = np.maximum(pu[keep], pv[keep])
        conf = conf[keep]
        key = lo * (int(hi.max()) + 1) + hi
        uk, inv = np.unique(key, return_inverse=True)
        mx = np.zeros(uk.size, dtype=np.float64)
        np.maximum.at(mx, inv, conf)                 # max negative conf per PCC pair
        first = np.zeros(uk.size, dtype=np.int64); first[inv[::-1]] = np.arange(inv.size)[::-1]
        a_ids = lo[first]; b_ids = hi[first]
        pcc_strength: Dict[int, float] = defaultdict(float)
        for a_id, b_id, m in zip(a_ids.tolist(), b_ids.tolist(), mx.tolist()):
            pcc_strength[a_id] += m
            pcc_strength[b_id] += m

        result = dict(pcc_strength)
        if os.environ.get('BETA_SELFCHECK'):
            # Reference: the original per-adjacency walk, plus a full rebuild of the
            # columnar mirror compared against the incrementally maintained one.
            ref_pair = {}
            for pcc_id, pcc in enumerate(self.graph._pccs):
                for node in pcc:
                    for nbr in self.graph.G.neighbors(node):
                        nbr_pcc = node_to_pcc.get(nbr)
                        if nbr_pcc is None or nbr_pcc == pcc_id:
                            continue
                        ed = self.graph.G[node][nbr].get('data')
                        if ed and ed.label == EdgeLabel.NEGATIVE:
                            pr = (min(pcc_id, nbr_pcc), max(pcc_id, nbr_pcc))
                            ref_pair[pr] = max(ref_pair.get(pr, 0.0), ed.confidence)
            ref = defaultdict(float)
            for (x, y), c in ref_pair.items():
                ref[x] += c; ref[y] += c
            ref = dict(ref)
            same = set(ref) == set(result) and all(
                abs(ref[k] - result[k]) < 1e-9 for k in ref)
            if not same or not self.graph.verify_edge_arrays():
                logger.error(
                    f"SELFCHECK FAIL: strength_match={same} "
                    f"mirror_ok={self.graph.verify_edge_arrays()} "
                    f"n_ref={len(ref)} n_new={len(result)}")
                raise AssertionError('pcc separation strength / edge mirror mismatch')
            logger.info(f"SELFCHECK ok: {len(result)} PCCs, mirror in sync")
        return result


    def _initialize_sorted_edges(self, embeddings):
        """Initialize sorted list of all possible edges using vectorized operations."""
        import numpy as np

        graph_nodes = set(self.graph.G.nodes())
        embedding_ids = embeddings.ids

        # Find which embedding indices correspond to our graph nodes
        # Build mapping: embedding_index -> node_id (only for nodes in graph)
        valid_indices = []
        valid_node_ids = []
        for idx, node_id in enumerate(embedding_ids):
            if node_id in graph_nodes:
                valid_indices.append(idx)
                valid_node_ids.append(node_id)

        n_valid = len(valid_indices)
        logger.info(f"Computing scores for {n_valid} nodes ({n_valid * (n_valid - 1) // 2} pairs)...")

        # Get embeddings for valid nodes only
        valid_embeddings = embeddings.embeddings[valid_indices]

        # Compute all pairwise cosine similarities at once using matrix multiplication
        # Normalize embeddings
        norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = valid_embeddings / norms

        # Cosine similarity matrix = dot product of normalized vectors
        similarity_matrix = np.dot(normalized, normalized.T)

        # Convert to scores using the same formula as embeddings.get_score_from_cosine_distance
        # cosine_distance = 1 - cosine_similarity
        # score = 1 - sign_power(cosine_distance, distance_power) * 0.5
        cosine_dist = 1 - similarity_matrix
        if embeddings.distance_power != 1:
            scores_matrix = 1 - np.sign(cosine_dist) * np.power(np.abs(cosine_dist), embeddings.distance_power) * 0.5
        else:
            scores_matrix = 1 - cosine_dist * 0.5

        # Extract upper triangular indices (i < j pairs only)
        triu_i, triu_j = np.triu_indices(n_valid, k=1)
        scores = scores_matrix[triu_i, triu_j]

        # Sort by score descending
        sorted_order = np.argsort(-scores)

        # Build sorted edge list
        all_edges = [
            (valid_node_ids[triu_i[idx]], valid_node_ids[triu_j[idx]], scores[idx])
            for idx in sorted_order
        ]

        self._all_edges_sorted = all_edges
        logger.info(f"Initialized sorted edge list with {len(all_edges)} edges")

    def is_finished(self) -> bool:
        """Check if algorithm is finished."""
        if self.phase == "FINISHED":
            return True

        # Phase 0 convergence is handled inside _phase0_step() — don't recompute here
        if self.phase == "PHASE0":
            return False

        # Active review termination
        if self.num_human_reviews >= self.max_human_reviews:
            return True

        return False

    def get_clustering(self) -> Tuple[Dict, Dict, nx.Graph]:
        """Get current clustering results."""
        cluster_dict, node2cid = self.graph.get_clustering()

        # Build NetworkX graph for compatibility
        G = nx.Graph()
        G.add_nodes_from(self.graph.G.nodes())

        for u, v, data in self.graph.G.edges(data=True):
            edge_data = data.get('data')
            if edge_data:
                G.add_edge(u, v,
                          label=edge_data.label.value,
                          confidence=edge_data.confidence,
                          score=edge_data.score,
                          ranker=edge_data.ranker,
                          is_active=(edge_data.label != EdgeLabel.POSITIVE_INACTIVE))

        return cluster_dict, node2cid, G

    def _checkpoint_counts(self):
        if self._checkpoints is None:
            self._checkpoints = set(resolve_checkpoints(self.report_at_reviews,
                                                        self.graph.G.number_of_nodes()))
        return self._checkpoints

    def _log_checkpoint(self, needs_restabilization):
        """Log the clustering after exactly `num_human_reviews` reviews, mid-batch if need be.

        The end-of-batch path is: each review applied to its edge, one re-stabilization,
        then the clustering is scored. A checkpoint runs that same path on a deep copy of
        the graph holding the reviews applied so far, and scores it with the validator's
        stateless `incremental_stats`. The live graph, the validator's trace state and all
        counters are untouched, so the run continues exactly as it would without it.
        """
        if not (self.cluster_validator and self.validation_initialized):
            return
        import copy
        live = self.graph
        try:
            self.graph = copy.deepcopy(live)
            if needs_restabilization:
                self._stabilize(alpha=-self.cut_margin)
            clustering, node2cid, _ = self._clustering_for_validation()
            self.cluster_validator.incremental_stats(
                self.num_human_reviews, clustering, node2cid,
                self.cluster_validator.gt_clustering, self.cluster_validator.gt_node2cid,
                CHECKPOINT_LABEL)
        finally:
            self.graph = live

    def _clustering_for_validation(self):
        """Clustering plus a graph for reachability, WITHOUT copying the edge set.

        `get_clustering()` materialises a fresh networkx graph holding every edge
        (14,569,752 on whale shark 2023) with five attributes each -- ~80 s -- and
        validation called it at every point, which was ~74 min of a 5 h run. The
        validator only hands the graph to `create_reachable`, which uses
        `subgraph` + `connected_components`, i.e. connectivity alone. So give it a
        read-only VIEW of the live graph. The edge filter mirrors
        `get_clustering()`, which likewise skips edges carrying no data, so the
        reachable clustering is unchanged.
        """
        cluster_dict, node2cid = self.graph.get_clustering()
        _G = self.graph.G
        view = nx.subgraph_view(
            _G, filter_edge=lambda u, v: _G[u][v].get('data') is not None)
        return cluster_dict, node2cid, view

    def _handle_validation(self):
        """Handle periodic validation against ground truth."""
        if not self.cluster_validator:
            return

        if not self.validation_initialized:
            clustering, node2cid, G = self._clustering_for_validation()
            self.cluster_validator.trace_start_human(clustering, node2cid, G, self.num_human_reviews)
            self.validation_initialized = True
            return

        # Periodic validation based on validation_step
        if hasattr(self.cluster_validator, 'prev_num_human'):
            if self.num_human_reviews - self.cluster_validator.prev_num_human >= self.validation_step:
                clustering, node2cid, G = self._clustering_for_validation()
                self.cluster_validator.trace_iter_compare_to_gt(
                    clustering, node2cid, self.num_human_reviews, G
                )

    def _finalize(self, reason):
        """Terminate: record WHY, and score the final clustering.

        Periodic validation only fires every `validation_step` reviews, so the state
        after the last partial batch was previously never evaluated -- a run that spent
        its full 5000-review budget reported its 4800-review clustering. Emitting the
        stats here makes the last plotted/reported point the actual final state, and
        makes it budget-matched with the baselines. `trace_iter_compare_to_gt` is a
        no-op when no reviews have happened since the last point, so this is safe to
        call unconditionally.
        """
        self.phase = "FINISHED"
        logger.info(f"TERMINATION: reason={reason} num_human_reviews={self.num_human_reviews}")
        if self.cluster_validator and self.validation_initialized:
            clustering, node2cid, G = self._clustering_for_validation()
            self.cluster_validator.trace_iter_compare_to_gt(
                clustering, node2cid, self.num_human_reviews, G
            )

    def _log_stats(self):
        """Log algorithm statistics."""
        stats = self.graph.get_graph_stats()
        mst_stats = self.graph.get_mst_cache_stats()

        logger.info("=" * 60)
        logger.info("Beta Stability Algorithm Statistics")
        logger.info("=" * 60)
        logger.info(f"Phase: {self.phase}")
        logger.info(f"Human reviews: {self.num_human_reviews}")
        logger.info(f"Target alpha: {self.target_alpha}")
        logger.info("-" * 60)
        logger.info(f"Nodes: {stats['num_nodes']}")
        logger.info(f"Edges: {stats['num_edges']}")
        logger.info(f"  Positive: {stats['edge_counts']['positive']}")
        logger.info(f"  Positive-inactive: {stats['edge_counts']['positive_inactive']}")
        logger.info(f"  Negative: {stats['edge_counts']['negative']}")
        logger.info("-" * 60)
        logger.info(f"PCCs: {stats['num_pccs']}")
        logger.info(f"PCC sizes: {stats['min_pcc_size']} - {stats['max_pcc_size']}")
        logger.info(f"Min internal stability: {stats['min_internal_stability']:.4f}")
        logger.info(f"Min external stability: {stats['min_external_stability']:.4f}")
        logger.info("-" * 60)
        logger.info(f"MST Forest:")
        logger.info(f"  Edges: {mst_stats['mst_forest_edges']}")
        logger.info("-" * 60)
        pcc_strength = self._compute_pcc_separation_strength()
        strengths = [pcc_strength.get(i, 0.0) for i in range(stats['num_pccs'])]
        if strengths:
            sorted_s = sorted(strengths)
            n_zero = sum(1 for s in strengths if s == 0.0)
            median_s = sorted_s[len(sorted_s) // 2]
            n_isolated = sum(1 for s in strengths if s <= median_s)
            logger.info(f"PCC supergraph (separation strength):")
            logger.info(f"  Strength: min={min(strengths):.3f}, "
                        f"median={median_s:.3f}, max={max(strengths):.3f}")
            logger.info(f"  Zero strength (no negative edges): {n_zero}/{len(strengths)}")
            logger.info(f"  Isolated (<=median): {n_isolated}/{len(strengths)}")
        logger.info("=" * 60)

    def show_stats(self):
        """Public method to display algorithm statistics."""
        self._log_stats()
        self._log_review_tally()
        self._dump_stability_track()

    def _log_review_tally(self):
        """Dump the (error?, verdict, label, flip?) breakdown of all reviews."""
        t = self._review_tally
        if not t:
            return
        total = sum(t.values())
        errs = {k: v for k, v in t.items() if k[0] == 'ERR'}
        n_err = sum(errs.values())
        logger.info("=== Review tally (vs ground truth) ===")
        logger.info(f"  total reviews with GT: {total}, errors: {n_err} "
                    f"({100*n_err/total:.1f}%)")
        # The 4 error kinds = verdict x current-label, split by whether they flipped.
        logger.info("  ERROR breakdown [verdict on current-label -> effect]:")
        for (e, verdict, label, flip), v in sorted(errs.items()):
            # name the structural effect
            if verdict == 'same' and label == 'neg':
                eff = 'MERGE (reversible: rectifier can cut back)'
            elif verdict == 'diff' and label == 'pos':
                eff = 'SPLIT (IRREVERSIBLE: cut-only rectifier)'
            elif verdict == 'same' and label == 'pos':
                eff = 'entrench-merge'
            else:
                eff = 'entrench-split'
            logger.info(f"    err: say '{verdict}' on {label} edge, {flip:6s} -> {eff}: {v}")
        # summary: reversible vs irreversible flipping errors
        rev = sum(v for (e, vd, lb, fl), v in errs.items() if vd == 'same' and lb == 'neg' and fl == 'flip')
        irr = sum(v for (e, vd, lb, fl), v in errs.items() if vd == 'diff' and lb == 'pos' and fl == 'flip')
        logger.info(f"  flipping errors -> reversible MERGE: {rev} | irreversible SPLIT: {irr}")

    def _log_phase0_metrics(self, iteration):
        """Log metrics during phase 0 iterations before human review starts."""
        if not self.cluster_validator:
            return

        # graph not needed here -- get_clustering() would copy all 14.5M edges
        clustering, node2cid = self.graph.get_clustering()

        # Use cluster_validator's incremental_stats directly to avoid prev_num_human check
        gt_clustering = self.cluster_validator.gt_clustering
        gt_node2cid = self.cluster_validator.gt_node2cid

        info_text = f'Phase 0 iteration {iteration}'
        result = self.cluster_validator.incremental_stats(
            0,  # num_human = 0 during phase 0
            clustering, node2cid, gt_clustering, gt_node2cid, info_text
        )
        result['phase0_iteration'] = iteration
        return result
