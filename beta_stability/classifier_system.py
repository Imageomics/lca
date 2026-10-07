"""
Simple classifier system for GC multi-classifier support.
Minimal implementation focused on current use cases.
"""

import logging
import numpy as np

logger = logging.getLogger("beta_stability")


class Classifier:
    """Abstract classifier that converts score to (label, confidence)."""

    def classify(self, score):
        """Return (label, confidence) from score."""
        raise NotImplementedError

    def refit(self, scores, labels) -> bool:
        """Update classifier parameters using labeled (score, label) data.

        Default: no-op (constant classifiers, weighter-based classifiers
        that don't support online refitting). Subclasses with learnable
        parameters override and return True if parameters changed.

        Args:
            scores: 1-D array of float scores
            labels: 1-D array of booleans (True = positive)

        Returns:
            True if classifier was updated, False otherwise.
        """
        return False


class WeighterBasedClassifier(Classifier):
    """Uses weighter for classification."""
    
    def __init__(self, weighter, weight_threshold=0):
        self.weighter = weighter
        self.weight_threshold = weight_threshold
    
    def classify(self, score):
        weight = self.weighter.wgt_smooth(score)
        is_positive = weight > self.weight_threshold
        max_range = abs(self.weighter.max_weight - self.weight_threshold)
        confidence = min(abs(score - self.weight_threshold) / max_range, 1)
        label = "positive" if is_positive else "negative"
        return (label, confidence)


class ThresholdBasedClassifier(Classifier):
    """Simple threshold classification.

    Learnable when the threshold was auto-fit at setup (e.g. via GMM on
    score distribution). Constant thresholds (numeric value from config)
    stay fixed.
    """

    def __init__(self, threshold=0.7, learnable=False):
        self.threshold = threshold
        self.learnable = learnable

    def classify(self, score):
        is_positive = score > self.threshold
        # Confidence is distance from threshold, normalized
        max_range = min(1 - self.threshold, self.threshold)
        confidence = min(abs(score - self.threshold) / max_range, 1)
        label = "positive" if is_positive else "negative"
        return (label, confidence)

    def refit(self, scores, labels, negative_floor=None) -> bool:
        """F1-maximising threshold from labeled scores. Only fires when
        this classifier was created as learnable (i.e. originally auto-fit).

        If ``negative_floor`` is supplied, the refitted threshold is clamped
        to be no smaller than that value. This implements the safety floor
        from Guo et al. (IEEE T-ASE 2026, eq. defining Δ_neg): the threshold
        should not drop into the bulk of the negative-pair score distribution
        regardless of what the F1 sweep prefers.
        """
        if not self.learnable:
            return False
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=bool)
        if len(scores) < 10 or labels.sum() == 0 or (~labels).sum() == 0:
            return False
        # Sweep candidate thresholds across the score range.
        lo, hi = float(scores.min()), float(scores.max())
        candidates = np.linspace(lo, hi, 200)
        best_f1, best_t = -1.0, self.threshold
        n_pos = int(labels.sum())
        for t in candidates:
            pred = scores > t
            tp = int(np.sum(pred & labels))
            if tp == 0:
                continue
            fp = int(np.sum(pred & ~labels))
            fn = n_pos - tp
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        floor_msg = ''
        if negative_floor is not None and best_t < float(negative_floor):
            floor_msg = (f" [clamped from {best_t:.4f} to negative-floor "
                         f"{float(negative_floor):.4f}]")
            best_t = float(negative_floor)
        if abs(best_t - self.threshold) < 1e-9:
            return False
        logger.info(
            f"ThresholdBasedClassifier.refit: {self.threshold:.4f} → "
            f"{best_t:.4f} (max-F1 = {best_f1:.4f} on "
            f"{len(scores)} labeled pairs, {n_pos} positives){floor_msg}"
        )
        self.threshold = best_t
        return True


class ClassifierManager:
    """
    Cycles through algorithmic classifiers for each edge.
    Does NOT handle human reviews - that's done by the run() loop.
    """
    
    def __init__(self, verifier_names, classifier_units):
        """
        Initialize classifier manager for algorithmic classifiers only.
        
        Args:
            verifier_names: List of verifier names (e.g., ['miewid', 'lightglue', 'human'])
            classifier_units: Dict mapping verifier_name -> (embeddings, classifier)
        """
        self.verifier_names = verifier_names
        self.classifier_units = classifier_units
        
        # Only algorithmic classifiers - exclude human types
        self.algo_classifiers = [name for name in verifier_names if 'human' not in name]
        
        logger.info(f"Created ClassifierManager with algorithmic classifiers: {self.algo_classifiers}")
    
    def get_next_classifier(self, current_classifier=None):
        """
        Get next algorithmic classifier for this edge, or 'human' if exhausted.
        
        Returns:
            str: Next algorithmic classifier name, or 'human' if all used
        """
        """
        Get next algorithmic classifier after current one, or 'human' if exhausted.
        
        Args:
            current_classifier: Currently used classifier (None for new edge)
            
        Returns:
            str: Next algorithmic classifier name, or 'human' if all used
        """
        if current_classifier is None:
            # New edge - start with first classifier
            if len(self.algo_classifiers) > 0:
                return self.algo_classifiers[0]
            else:
                return 'human'
        if 'human' in current_classifier:
            return 'human'
        if current_classifier not in self.algo_classifiers:
            # Current is human or unknown - go to human
            return self.algo_classifiers[0]
        
        # Find next classifier after current one
        current_index = self.algo_classifiers.index(current_classifier)
        next_index = current_index + 1
        
        if next_index < len(self.algo_classifiers):
            return self.algo_classifiers[next_index]
        else:
            return 'human'
    
    def classify_or_request_human(self, edge_classifier_pairs):
        """
        Classify edges if algorithmic classifiers available, otherwise mark for human review.
        
        Args:
            edge_classifier_pairs: List of ((n0, n1), current_classifier) tuples
            
        Returns:
            tuple: (new_classifications, edges_for_human)
                - new_classifications: List of classified edges in raw format
                - edges_for_human: List of (n0, n1) pairs needing human review
        """
        new_classifications = []
        edges_for_human = []
        
        for n0, n1, current_classifier in edge_classifier_pairs:
            next_classifier = self.get_next_classifier(current_classifier)
            if next_classifier == 'human':
                # Exhausted all algorithmic classifiers
                if current_classifier != 'human':
                    # logger.info(f"Edge ({n0}, {n1}) exhausted all algorithmic classifiers, requesting human review")
                    embeddings, classifier = self.classifier_units[current_classifier]
                    score = embeddings.get_score(n0, n1)
                else:
                    score = 1.0  # Default score for human-reviewed edges
                edges_for_human.append((n0, n1, score))
            else:
                # Use next algorithmic classifier
                edge = self.classify_edge(n0, n1, next_classifier)
                new_classifications.append(edge)
        
        return new_classifications, edges_for_human
    
    def classify_edge(self, n0, n1, classifier_name=None):
        """
        Classify edge using specified algorithmic classifier.
        
        Args:
            n0, n1: Edge nodes
            classifier_name: Algorithmic classifier to use (required)
            
        Returns:
            tuple: (n0, n1, score, confidence, label, classifier_name)
        """
        if classifier_name is None:
            classifier_name = self.verifier_names[0]

        if 'human' in classifier_name:
            raise ValueError("ClassifierManager cannot handle human reviews")
        
        if classifier_name not in self.classifier_units:
            raise ValueError(f"Unknown algorithmic classifier: {classifier_name}")
        
        embeddings, classifier = self.classifier_units[classifier_name]
        score = embeddings.get_score(n0, n1)
        label, confidence = classifier.classify(score)
        
        return (n0, n1, score, confidence, label, classifier_name)