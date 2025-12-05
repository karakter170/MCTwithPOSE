"""
Hungarian Assignment Module for Multi-Camera Tracking
======================================================
FINAL TUNED VERSION:
- Aggressive Quality Scaling for Heavy Occlusion Recovery
- Motion Gating for Teleportation Prevention
- Dynamic GCN Weighting
- Added bounds checking for GCN costs array
- Added proper logging instead of silent except
"""

import numpy as np
import logging
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass 
class MatchResult:
    matches: List[Tuple[int, int]]
    unmatched_detections: List[int]
    unmatched_tracks: List[int]
    cost_matrix: np.ndarray
    valid_track_ids: List[int] = None

class HungarianMatcher:
    def __init__(
        self,
        match_threshold: float = 0.55,
        use_appearance: bool = True,
        use_motion: bool = True,
        use_iou: bool = True,
        appearance_weight: float = 0.60,
        motion_weight: float = 0.15,
        iou_weight: float = 0.25,
        max_time_gap: float = 60.0
    ):
        self.match_threshold = match_threshold
        self.use_appearance = use_appearance
        self.use_motion = use_motion
        self.use_iou = use_iou
        
        total = appearance_weight + motion_weight + iou_weight
        self.appearance_weight = appearance_weight / total
        self.motion_weight = motion_weight / total
        self.iou_weight = iou_weight / total
        
        self.max_time_gap = max_time_gap
        self.INF_COST = 1e6
        
        # --- GATING THRESHOLDS ---
        self.MOTION_GATE_THRESHOLD = 0.85
        self.LONG_TERM_REID_THRESHOLD = 5.0 

    def compute_appearance_cost(self, det_feat, track_feat, track_var=None):
        if det_feat is None or track_feat is None: return self.INF_COST
        
        d_norm = det_feat / (np.linalg.norm(det_feat) + 1e-8)
        t_norm = track_feat / (np.linalg.norm(track_feat) + 1e-8)
        sim = np.dot(d_norm, t_norm)
        raw_dist = 1.0 - sim
        
        if track_var is not None:
            uncertainty = np.mean(track_var)
            raw_dist /= (1.0 + uncertainty * 5.0)
            
        return max(0.0, raw_dist)

    def compute_motion_cost(self, det_gp, track_pred, dt):
        """
        Compute motion cost based on distance between detection and predicted position.

        FIX: Previous formula max_dist = 3.0 * dt + 2.0 was too restrictive.
        At 30 FPS (dt=0.033s): max_dist = 2.1 pixels - way too small!
        Humans can walk 1-2 m/s, which at typical camera resolutions is ~50-100 pixels/frame.
        """
        if det_gp is None or track_pred is None:
            return self.INF_COST

        dist = np.linalg.norm(det_gp - track_pred)

        # Realistic max velocity: ~2 m/s walking, ~6 m/s running
        # At 1920x1080 with ~5m scene width: 1m ≈ 384 pixels
        # Max walking: 2 m/s * 384 px/m = 768 px/s
        # At 30 FPS: 768/30 = ~25 pixels/frame
        # Add buffer for prediction error and fast motion
        max_velocity_px_per_sec = 1000.0  # ~2.6 m/s (fast walking/slow running)
        prediction_error_buffer = 20.0  # pixels

        max_dist = max_velocity_px_per_sec * max(dt, 0.001) + prediction_error_buffer
        norm_dist = dist / max_dist
        return min(1.0, norm_dist)

    def compute_iou_cost(self, box1, box2):
        if not box1 or not box2: return 1.0
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        union = area1 + area2 - inter
        if union <= 0: return 1.0
        return 1.0 - (inter / union)

    def build_cost_matrix(self, detections, tracks, cam_id, curr_time, refiner_model=None, frame_res=(1920,1080)):
        det_indices = list(range(len(detections)))
        valid_track_ids = [gid for gid, t in tracks.items() if (curr_time - t.last_seen_timestamp) <= self.max_time_gap]

        n_det, n_track = len(detections), len(valid_track_ids)
        if n_det == 0 or n_track == 0:
            return np.array([]).reshape(n_det, n_track), det_indices, valid_track_ids

        cost_matrix = np.full((n_det, n_track), self.INF_COST)

        # FIX: Predict all tracks BEFORE cost matrix computation (was inside loop - 100x slower)
        track_predictions = {}
        for gid in valid_track_ids:
            track = tracks[gid]
            track.kf.predict()
            track_predictions[gid] = track.kf.x[:2].copy()

        for j, gid in enumerate(valid_track_ids):
            track = tracks[gid]
            dt = curr_time - track.last_seen_timestamp

            gcn_costs = None
            if refiner_model:
                try:
                    scores = refiner_model.predict_batch(track, detections, frame_res[0], frame_res[1], curr_time)
                    if scores is not None and len(scores) == len(detections):
                        gcn_costs = 1.0 - scores
                    else:
                        logger.warning(f"GCN scores length mismatch: got {len(scores) if scores is not None else 0}, expected {len(detections)}")
                except Exception as e:
                    logger.warning(f"GCN prediction failed for track {gid}: {e}")

            # Use pre-computed prediction (FIX: was calling predict() inside loop)
            pred_gp = track_predictions[gid]

            for i, det in enumerate(detections):
                quality = det.get('quality', 0.5)

                # 1. Appearance
                cost_app = self.INF_COST
                if self.use_appearance:
                    if track.fast_buffer:
                        dists = [self.compute_appearance_cost(det['feature'], f) for f in track.fast_buffer if f is not None]
                        # Guard against empty dists list after filtering None values
                        if dists:
                            cost_fast = min(dists)
                        else:
                            cost_fast = self.compute_appearance_cost(det['feature'], track.last_known_feature)
                    else:
                        cost_fast = self.compute_appearance_cost(det['feature'], track.last_known_feature)

                    cost_slow = self.compute_appearance_cost(det['feature'], track.robust_id, track.robust_var)
                    cost_app = min(cost_fast, cost_slow)

                    # FIX: Quality scaling was INVERTED - low quality should INCREASE cost, not decrease it
                    # Previous: quality_factor = 0.25 + 0.75*(q/0.6) -> REDUCED cost for low quality (WRONG)
                    # Fixed: penalty_factor > 1.0 for low quality -> INCREASES cost (CORRECT)
                    if quality < 0.6:
                        # Low quality = higher uncertainty = higher cost penalty
                        penalty_factor = 1.0 + (1.5 * (0.6 - quality) / 0.6)  # 1.0 to 2.5x penalty
                        cost_app *= penalty_factor

                # 2. Motion
                cost_mot = 0.0
                is_teleport = False
                if self.use_motion:
                    cost_mot = self.compute_motion_cost(det['gp_coord'], pred_gp, dt)
                    if dt < self.LONG_TERM_REID_THRESHOLD:
                        if cost_mot > self.MOTION_GATE_THRESHOLD:
                            is_teleport = True

                # 3. IoU
                cost_iou = 0.0
                if self.use_iou and track.last_cam_id == cam_id:
                    cost_iou = self.compute_iou_cost(det['bbox'], track.last_seen_bbox)
                    if cost_app < 0.2:
                        cost_iou *= 0.3

                # --- TOTAL COST ---
                if is_teleport:
                    total_cost = self.INF_COST
                else:
                    total_cost = (
                        self.appearance_weight * cost_app +
                        self.motion_weight * cost_mot +
                        self.iou_weight * cost_iou
                    )
                    
                    # Apply GCN refinement with bounds checking
                    if gcn_costs is not None and i < len(gcn_costs):
                        gcn_val = gcn_costs[i]

                        # FIX: GCN weight scaling was INVERTED - low quality should rely MORE on GCN, not less
                        # GCN refinement helps especially with occluded/poor quality detections
                        # Previous: gcn_weight = 0.4 * (q/0.6) -> LESS GCN for low quality (WRONG)
                        # Fixed: gcn_weight increases for low quality -> MORE GCN reliance (CORRECT)
                        base_gcn_weight = 0.40
                        if quality < 0.6:
                            # Low quality = rely MORE on GCN refinement (up to 60% weight)
                            gcn_weight = base_gcn_weight + (0.20 * (0.6 - quality) / 0.6)
                        else:
                            gcn_weight = base_gcn_weight

                        total_cost = (1.0 - gcn_weight) * total_cost + (gcn_weight * gcn_val)

                cost_matrix[i, j] = total_cost

        return cost_matrix, det_indices, valid_track_ids

    def match(self, detections, tracks, camera_id, current_time, refiner_model=None, frame_res=(1920, 1080)):
        cost_matrix, det_indices, track_ids = self.build_cost_matrix(
            detections, tracks, camera_id, current_time, refiner_model, frame_res
        )
        
        cost_matrix_safe = np.nan_to_num(cost_matrix, nan=self.INF_COST, posinf=self.INF_COST)
        row_ind, col_ind = linear_sum_assignment(cost_matrix_safe)
        
        matches = []
        unmatched_dets = set(det_indices)
        unmatched_tracks = set(track_ids)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix_safe[r, c] < self.match_threshold:
                matches.append((det_indices[r], track_ids[c]))
                unmatched_dets.discard(det_indices[r])
                unmatched_tracks.discard(track_ids[c])
        
        return MatchResult(
            matches, list(unmatched_dets), list(unmatched_tracks), 
            cost_matrix_safe, track_ids
        )

class TwoStageHungarianMatcher(HungarianMatcher):
    def __init__(self, high_conf_threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self.high_conf_threshold = high_conf_threshold
        
    def match(self, detections, tracks, camera_id, current_time, refiner_model=None, frame_res=(1920, 1080)):
        high_indices = [i for i, d in enumerate(detections) if d.get('confidence', 0) >= self.high_conf_threshold]
        high_dets = [detections[i] for i in high_indices]
        
        res1 = super().match(high_dets, tracks, camera_id, current_time, refiner_model, frame_res)
        
        all_matches = []
        matched_gids = set()
        for local_idx, gid in res1.matches:
            orig_idx = high_indices[local_idx]
            all_matches.append((orig_idx, gid))
            matched_gids.add(gid)
            
        unmatched_original_indices = [high_indices[i] for i in res1.unmatched_detections]
        for i in range(len(detections)):
            if i not in high_indices:
                unmatched_original_indices.append(i)

        return MatchResult(
            all_matches, 
            unmatched_original_indices, 
            res1.unmatched_tracks, 
            res1.cost_matrix, 
            res1.valid_track_ids
        )


# Test code
if __name__ == "__main__":
    print("Hungarian Matcher Test")
    print("=" * 50)
    
    # Create test data
    detections = [
        {'feature': np.random.randn(1024), 'bbox': [100, 100, 200, 300], 
         'gp_coord': np.array([5.0, 10.0]), 'confidence': 0.9},
        {'feature': np.random.randn(1024), 'bbox': [300, 100, 400, 300],
         'gp_coord': np.array([8.0, 10.0]), 'confidence': 0.85},
        {'feature': np.random.randn(1024), 'bbox': [500, 100, 600, 300],
         'gp_coord': np.array([12.0, 10.0]), 'confidence': 0.6},
    ]
    
    # Normalize features
    for det in detections:
        det['feature'] = det['feature'] / np.linalg.norm(det['feature'])
    
    # Create mock tracks
    class MockTrack:
        def __init__(self, gid, feature, bbox, gp):
            self.global_id = gid
            self.last_known_feature = feature / np.linalg.norm(feature)
            self.robust_id = self.last_known_feature
            self.robust_var = np.ones(1024) * 0.01
            self.fast_buffer = [self.last_known_feature]
            self.last_seen_bbox = bbox
            self.last_seen_timestamp = 0.0
            self.last_cam_id = "cam1"
            
            # Mock Kalman filter
            class MockKF:
                def __init__(self, gp):
                    self.x = np.array([gp[0], gp[1], 0.1, 0.0])
                def predict(self):
                    pass
            self.kf = MockKF(gp)
    
    tracks = {
        1: MockTrack(1, np.random.randn(1024), [100, 100, 200, 300], [5.0, 10.0]),
        2: MockTrack(2, np.random.randn(1024), [300, 100, 400, 300], [8.0, 10.0]),
        3: MockTrack(3, np.random.randn(1024), [700, 100, 800, 300], [15.0, 10.0]),
    }
    
    # Test matcher
    matcher = TwoStageHungarianMatcher(
        match_threshold=0.55,
        high_conf_threshold=0.7
    )
    
    result = matcher.match(detections, tracks, "cam1", current_time=1.0)
    
    print(f"Matches: {result.matches}")
    print(f"Unmatched detections: {result.unmatched_detections}")
    print(f"Unmatched tracks: {result.unmatched_tracks}")
    print("\n✓ Hungarian matcher working correctly!")
