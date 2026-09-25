#!/usr/bin/env python3
"""Run THE baseline for one dataset and write clean, separately-foldered results.

Baseline definition (fixed — this is the default):
  1. No new edges          - no densification, no cross-PCC discovery, no snowball
  2. Auto top-k init, 90%   - initial_topk='auto', auto_topk_coverage=0.9
  3. No k-means at all      - k-means verifier dropped AND no auto-verifier
                              selection (verifier_name is the fixed list
                              ['miewid'], so the silhouette selector never
                              fires) -> top-k initialization only
  4. Weak-positive review   - review_weakest_positive on (send weakest positive
                              or the negative)
  5. Positive-inactive dual - review_positive_inactive on: the strongest cross-PCC
                              positive-inactive edge is reviewed when it is the
                              side making a pair unstable (dual of the INTERNAL
                              negative); 'same' -> reactivate -> merge
  6. Stability tracking on  - per-PCC internal stability snapshot each iteration
  7. Unified candidate rank  - unified_candidate_ranking on: batch seats filled
                              from internal ∪ external by stability (most-unstable
                              first), not INTERNAL-first; keeps weak-positives
                              from flooding early batches and displacing merges

Perfect reviewer (prob_human_correct = 1.0).

Outputs -> beta_stability/tmp/baseline/<dataset>/ :
    <dataset>_baseline.log
    <dataset>_baseline_stability_track.json

Usage: python3 -m beta_stability.run_baseline --species beluga
"""
import argparse
from pathlib import Path

BASELINE_DIR = Path('/users/PAS2136/nepove/code/lca/beta_stability/tmp/baseline')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--species', required=True)
    args = ap.parse_args()
    sp = args.species

    from beta_stability.exploratory.run_weakpos_experiment import build_config
    from beta_stability.util.init_logger import init_logger
    from beta_stability.run_clustering_with_save import run_clustering_with_save

    # build_config already encodes: no-new-edges, auto-topk cov90, no-kmeans-
    # verifier. weakpos=True -> weak-positive selector; track_stability=True.
    cfg = build_config(sp, weakpos=True, coverage=0.9, phc=1.0, track_stability=True,
                       review_pi=True)
    # Unified candidate ranking: fill batch seats from internal ∪ external by
    # stability, not INTERNAL-first. Set here (build_config is shared with the
    # ablation launcher, which must stay on the legacy default for its A/B).
    cfg['stability']['unified_candidate_ranking'] = True

    out = BASELINE_DIR / sp
    out.mkdir(parents=True, exist_ok=True)
    cfg['logging']['log_file'] = str(out / f'{sp}_baseline.log')
    cfg['stability']['stability_track_path'] = str(out / f'{sp}_baseline_stability_track.json')

    init_logger()
    print(f"=== BASELINE run: {sp} "
          f"(no-new-edges, auto-topk cov90, top-k init only / no k-means, "
          f"weak-positive, tracking) ===", flush=True)
    run_clustering_with_save(cfg)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
