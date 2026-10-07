import functools
import logging

import numpy as np


logger = logging.getLogger("beta_stability")


class GeometricEmbeddings(object):
    """Geometric verification wrapper for an existing embedding verifier.

    The wrapper uses the base verifier for candidate generation and Fisher-vector
    scores, then rescoring requested pairs using local ALFREID features stored in
    an HDF5 file under /annotations/<uuid>/{descriptors,keypoints}.
    """

    def __init__(self, base_embeddings, node2uuid, local_feature_file,
                 config=None, print_func=print):
        if not local_feature_file:
            raise ValueError("geometric verifier requires data.local_feature_file")

        self.base_embeddings = base_embeddings
        self.uuids = node2uuid
        self.ids = list(node2uuid.keys())
        self.local_feature_file = local_feature_file
        self.config = config or {}
        self.print_func = print_func

        self.descriptor_dataset = self.config.get('descriptor_dataset', 'descriptors')
        self.keypoint_dataset = self.config.get('keypoint_dataset', 'keypoints')
        self.max_matches = int(self.config.get('max_matches', 200))
        self.match_percentile = float(self.config.get('match_percentile', 10))
        self.ransac_max_iters = int(self.config.get('ransac_max_iters', 5000))
        self.ransac_reproj_err = float(self.config.get('ransac_reproj_err', 0.2))
        self.min_matches = int(self.config.get('min_matches', 4))
        self.fallback_to_base_score = self.config.get('fallback_to_base_score', True)
        self.geom_weight = float(self.config.get('geom_weight', 0.25))
        self.mode = self.config.get('mode', 'blend')
        self.return_score = self.config.get('return_score', 'geometric')
        self.candidate_pool_multiplier = max(
            1, int(self.config.get('candidate_pool_multiplier', 1))
        )
        self.base_score_weight = float(self.config.get('base_score_weight', 0.5))
        if not 0.0 <= self.base_score_weight <= 1.0:
            raise ValueError("geometric.base_score_weight must be between 0 and 1")
        self.geometric_score_weight = 1.0 - self.base_score_weight
        valid_modes = {'blend', 'rerank_edges'}
        if self.mode not in valid_modes:
            raise ValueError(f"geometric.mode must be one of {sorted(valid_modes)}")
        valid_return_scores = {'base', 'geometric', 'blend'}
        if self.return_score not in valid_return_scores:
            raise ValueError(
                f"geometric.return_score must be one of {sorted(valid_return_scores)}"
            )

        logger.info(
            "GeometricEmbeddings: mode=%s, return_score=%s, "
            "candidate_pool_multiplier=%d, geom_weight=%.3f, "
            "base_score_weight=%.3f, geometric_score_weight=%.3f",
            self.mode,
            self.return_score,
            self.candidate_pool_multiplier,
            self.geom_weight,
            self.base_score_weight,
            self.geometric_score_weight
        )

        self._h5 = None
        self._h5py = None
        self._cv2 = None

    def get_base(self):
        return self.base_embeddings

    def _load_dependencies(self):
        if self._h5py is None:
            try:
                import h5py
            except ImportError as exc:
                raise ImportError(
                    "GeometricEmbeddings requires h5py to read ALFREID local "
                    "feature files. Install h5py in the active environment."
                ) from exc
            self._h5py = h5py
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise ImportError(
                    "GeometricEmbeddings requires OpenCV. Install "
                    "opencv-python-headless or provide cv2 in the active "
                    "environment."
                ) from exc
            self._cv2 = cv2

    def _file(self):
        self._load_dependencies()
        if self._h5 is None:
            self._h5 = self._h5py.File(self.local_feature_file, 'r')
            if 'annotations' not in self._h5:
                raise ValueError(
                    f"Local feature file {self.local_feature_file} does not "
                    "contain an /annotations group"
                )
        return self._h5

    @functools.lru_cache(maxsize=200000)
    def _load_features(self, uuid):
        h5 = self._file()
        path = f'annotations/{uuid}'
        if path not in h5:
            return None
        group = h5[path]
        if self.descriptor_dataset not in group or self.keypoint_dataset not in group:
            return None

        descriptors = np.asarray(group[self.descriptor_dataset], dtype=np.float32)
        keypoints = np.asarray(group[self.keypoint_dataset], dtype=np.float32)
        if descriptors.ndim != 2 or keypoints.ndim != 2 or keypoints.shape[1] < 2:
            return None
        n = min(descriptors.shape[0], keypoints.shape[0])
        if n == 0:
            return None
        return descriptors[:n], keypoints[:n, :2]

    def _base_score(self, id1, id2):
        return float(self.base_embeddings.get_score(id1, id2))

    def _fallback_score(self, id1, id2):
        if self.fallback_to_base_score:
            return self._base_score(id1, id2)
        return 0.0

    def _match_descriptors(self, descriptors1, descriptors2):
        if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        desc1 = self._l2_normalize(descriptors1)
        desc2 = self._l2_normalize(descriptors2)
        distances = 1.0 - np.matmul(desc1, desc2.T)

        best_db = np.argmin(distances, axis=1)
        best_dist = distances[np.arange(distances.shape[0]), best_db]
        if best_dist.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        threshold = np.percentile(best_dist, self.match_percentile)
        query_inds = np.nonzero(best_dist <= threshold)[0]
        if query_inds.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        order = np.argsort(best_dist[query_inds])
        query_inds = query_inds[order]
        db_inds = best_db[query_inds]

        # Keep the best query match for each database descriptor to avoid
        # degenerate duplicate correspondences in homography estimation.
        if db_inds.size:
            _, unique_pos = np.unique(db_inds, return_index=True)
            unique_pos = np.sort(unique_pos)
            query_inds = query_inds[unique_pos]
            db_inds = db_inds[unique_pos]

        if query_inds.size > self.max_matches:
            query_inds = query_inds[:self.max_matches]
            db_inds = db_inds[:self.max_matches]
        return query_inds, db_inds

    @staticmethod
    def _l2_normalize(descriptors):
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms[norms <= np.finfo(np.float32).eps] = 1.0
        return descriptors / norms

    @staticmethod
    def _normalize_points(points):
        if points.size == 0:
            return points
        pts = points.astype(np.float64, copy=True)
        pts -= np.mean(pts, axis=0, keepdims=True)
        radii = np.linalg.norm(pts, axis=1)
        max_radius = np.max(radii) if radii.size else 0.0
        if max_radius > np.finfo(float).eps:
            pts /= max_radius
        return pts

    def _ransac_inliers(self, points1, points2):
        if points1.shape[0] < self.min_matches or points2.shape[0] < self.min_matches:
            return None

        self._load_dependencies()
        norm1 = self._normalize_points(points1)
        norm2 = self._normalize_points(points2)
        homography, mask = self._cv2.findHomography(
            norm1,
            norm2,
            method=self._cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_err,
            maxIters=self.ransac_max_iters
        )
        if homography is None or mask is None:
            return None
        return int(np.sum(mask))

    def _geometric_score(self, id1, id2, base_score):
        uuid1 = self.uuids[id1]
        uuid2 = self.uuids[id2]
        features1 = self._load_features(uuid1)
        features2 = self._load_features(uuid2)
        if features1 is None or features2 is None:
            return self._fallback_score(id1, id2)

        descriptors1, keypoints1 = features1
        descriptors2, keypoints2 = features2
        query_inds, db_inds = self._match_descriptors(descriptors1, descriptors2)
        if query_inds.size < self.min_matches:
            return self._fallback_score(id1, id2)

        inliers = self._ransac_inliers(keypoints1[query_inds], keypoints2[db_inds])
        if inliers is None:
            return self._fallback_score(id1, id2)

        fisher_distance = np.clip(2.0 * (1.0 - base_score), 0.0, 2.0)
        match_count = max(int(query_inds.size), 1)
        inlier_ratio = np.clip(inliers / match_count, 0.0, 1.0)
        adjusted_distance = fisher_distance * (1.0 + self.geom_weight * (1.0 - inlier_ratio))
        geometric_score = 1.0 - 0.5 * adjusted_distance
        return float(np.clip(geometric_score, 0.0, 1.0))

    def _blend_score(self, base_score, geometric_score):
        blended_score = (
            self.base_score_weight * base_score
            + self.geometric_score_weight * geometric_score
        )
        return float(np.clip(blended_score, 0.0, 1.0))

    def _score_for_return(self, id1, id2):
        if id2 < id1:
            id1, id2 = id2, id1

        base_score = self._base_score(id1, id2)
        if self.return_score == 'base' or self.mode == 'rerank_edges':
            return base_score

        geometric_score = self._geometric_score(id1, id2, base_score)
        if self.return_score == 'geometric':
            return geometric_score
        return self._blend_score(base_score, geometric_score)

    @functools.lru_cache(maxsize=1000000)
    def get_score(self, id1, id2):
        return self._score_for_return(id1, id2)

    @functools.lru_cache(maxsize=1000000)
    def _ordering_score(self, id1, id2):
        if id2 < id1:
            id1, id2 = id2, id1
        base_score = self._base_score(id1, id2)
        geometric_score = self._geometric_score(id1, id2, base_score)
        if self.return_score == 'blend':
            return self._blend_score(base_score, geometric_score)
        return geometric_score

    def _get_expanded_edges(self, topk, botk, target_edges, target_proportion,
                            uuids_filter, lower_threshold, upper_threshold):
        expanded_target_edges = target_edges
        if target_edges is not None:
            expanded_target_edges = int(target_edges * self.candidate_pool_multiplier)
        expanded_topk = int(topk * self.candidate_pool_multiplier)
        expanded_botk = int(botk * self.candidate_pool_multiplier)
        expanded_target_proportion = target_proportion
        if target_proportion is not None:
            expanded_target_proportion = min(1.0, target_proportion * self.candidate_pool_multiplier)

        return set(self.base_embeddings.get_edges(
            topk=expanded_topk,
            botk=expanded_botk,
            target_edges=expanded_target_edges,
            target_proportion=expanded_target_proportion,
            uuids_filter=uuids_filter,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold
        ))

    def get_edges(self, topk=5, botk=0, target_edges=10000, target_proportion=None,
                  uuids_filter=None, lower_threshold=None, upper_threshold=None):
        if self.mode != 'rerank_edges':
            base_edges = self.base_embeddings.get_edges(
                topk=topk,
                botk=botk,
                target_edges=target_edges,
                target_proportion=target_proportion,
                uuids_filter=uuids_filter,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold
            )
            return set((n0, n1, self.get_score(n0, n1)) for n0, n1, _ in base_edges)

        base_edges = set(self.base_embeddings.get_edges(
            topk=topk,
            botk=botk,
            target_edges=target_edges,
            target_proportion=target_proportion,
            uuids_filter=uuids_filter,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold
        ))
        if not base_edges or self.candidate_pool_multiplier <= 1:
            return set((n0, n1, base_score) for n0, n1, base_score in base_edges)

        pool_edges = self._get_expanded_edges(
            topk,
            botk,
            target_edges,
            target_proportion,
            uuids_filter,
            lower_threshold,
            upper_threshold
        )

        base_score_by_pair = {
            (min(n0, n1), max(n0, n1)): base_score
            for n0, n1, base_score in pool_edges
        }
        reranked = []
        for (n0, n1), base_score in base_score_by_pair.items():
            order_score = self._ordering_score(n0, n1)
            reranked.append((n0, n1, base_score, order_score))

        reranked.sort(key=lambda edge: edge[3], reverse=True)
        selected = reranked[:len(base_edges)]
        return set((n0, n1, base_score) for n0, n1, base_score, _ in selected)

    def get_all_scores(self):
        # Threshold estimation should not trigger all-pairs RANSAC. Configs should
        # normally reference the base threshold, but this keeps accidental auto()
        # thresholds cheap and consistent with the wrapped verifier.
        return self.base_embeddings.get_all_scores()

    def get_stats(self, df, filter_key, id_key='uuid'):
        return self.base_embeddings.get_stats(df, filter_key, id_key)

    def get_top20_matches(self, df, filter_key):
        return self.base_embeddings.get_top20_matches(df, filter_key)

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
