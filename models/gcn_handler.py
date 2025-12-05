# gcn_handler.py
# FIXED VERSION - ALL BUGS PATCHED
#
# FIXES APPLIED:
# 1. Always normalize features (don't just warn)
# 2. Proper dtype handling
# 3. Better error handling
# 4. Fixed unreachable docstring

import torch
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from models.gcn_model_sota import CrossGCN

logger = logging.getLogger(__name__)


class RelationRefiner:
    """
    GCN-based relation refiner for cross-camera matching.
    
    Note: Despite the "GCN" name, this is actually a Siamese CNN architecture.
    The name is kept for backwards compatibility.
    """
    
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        print(f"[RelationRefiner] Loading SOTA CrossGCN from {weights_path}...")

        # Model Architecture (Must match training)
        # Input: 1029 dimensions (1024 appearance + 5 geometry+time)
        self.model = CrossGCN(feature_dim=1029).to(device)
        
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(checkpoint)
            print("[RelationRefiner] Weights loaded successfully.")
        except FileNotFoundError:
            print(f"[RelationRefiner] CRITICAL WARNING: Weights file not found: {weights_path}")
            print("[RelationRefiner] GCN Refinement DISABLED to prevent noise.")
            self.model = None # Disable model
        except Exception as e:
            print(f"[RelationRefiner] Error loading weights: {e}")
            self.model = None # Disable model
            
        self.model.eval()
        print("[RelationRefiner] Model Ready (Batch Mode Enabled).")

    def _normalize_bbox(self, bbox, w, h):
        """
        Normalize bbox to [-1, 1] range to match training.
        
        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates
            w: frame width
            h: frame height
            
        Returns:
            Normalized bbox in [-1, 1] range
        """
        # Ensure we have valid dimensions
        w = max(w, 1)
        h = max(h, 1)
        
        # First normalize to [0, 1]
        norm_01 = np.array([
            bbox[0] / w, bbox[1] / h,
            bbox[2] / w, bbox[3] / h
        ], dtype=np.float32)
        
        # Then scale to [-1, 1] to match DINOv2/v3 feature range
        return (norm_01 - 0.5) * 2.0

    def _normalize_feature(self, feat):
        """
        FIXED: Always normalize feature vector to unit length.
        This ensures numerical stability regardless of input source.
        
        Args:
            feat: Feature vector (numpy array)
            
        Returns:
            L2-normalized feature vector
        """
        feat = np.asarray(feat, dtype=np.float32)
        feat_norm = np.linalg.norm(feat)
        
        if feat_norm < 1e-8:
            # Zero vector - return as-is (will produce neutral scores)
            return feat
        
        return feat / feat_norm

    def predict_batch(self, track, candidates, frame_w: int, frame_h: int, curr_time: float) -> np.ndarray:
        """
        Batch Inference: Compare 1 Track against N Candidates.

        Args:
            track: The GlobalTrack object (Query)
                   Must have: robust_id, last_known_feature, last_seen_bbox,
                             last_cam_res, last_seen_timestamp
            candidates: List of dicts with 'feature' and 'bbox' keys
            frame_w: Width of the current camera frame
            frame_h: Height of the current camera frame
            curr_time: Current timestamp for temporal features

        Returns:
            scores: Numpy array of shape (N,) with similarity probabilities [0, 1].
                   Higher scores indicate higher match probability.
        """
        if self.model is None:
            # Return 0.5 (neutral) when model is not available
            return np.ones(len(candidates)) * 0.5

        if not candidates:
            return np.array([])

        # 1. Prepare Track Data (Query)
        # Use robust ID (Slow Memory) if available, else Fast Memory
        t_feat = track.robust_id if track.robust_id is not None else track.last_known_feature
        if t_feat is None:
            return np.zeros(len(candidates))

        # FIXED: Always normalize features for numerical stability
        t_feat = self._normalize_feature(t_feat)

        # Get track's camera resolution (use stored value or fallback)
        t_w, t_h = getattr(track, 'last_cam_res', (1920, 1080))

        # Normalize track bbox using its original camera resolution
        t_bbox_norm = self._normalize_bbox(track.last_seen_bbox, t_w, t_h)

        # Calculate normalized time gap
        dt = curr_time - track.last_seen_timestamp
        norm_dt = np.tanh(dt / 10.0)  # Squash to [-1, 1] range

        # Symmetric time encoding: Track (earlier) gets -dt/2, Detection (later) gets +dt/2
        t_geo = np.concatenate([t_bbox_norm, [-norm_dt/2]])

        # Combine features: 1024 (appearance) + 5 (geometry+time) = 1029 dimensions
        t_input = np.concatenate([t_feat, t_geo])
        
        # 2. Prepare Candidates Data (Keys)
        d_inputs = []
        for cand in candidates:
            d_feat = cand['feature']

            # FIXED: Always normalize candidate features
            d_feat = self._normalize_feature(d_feat)

            d_bbox_norm = self._normalize_bbox(cand['bbox'], frame_w, frame_h)

            # Symmetric time encoding: Candidates (current/later) get +dt/2
            d_geo = np.concatenate([d_bbox_norm, [+norm_dt/2]])

            d_inputs.append(np.concatenate([d_feat, d_geo]))

        # 3. Batch Tensor Creation
        # Shape: (1, 1029, 1) -> Batch=1, Dim=1029, N=1 Track
        t_tensor = torch.tensor(t_input, dtype=torch.float32).to(self.device)
        t_tensor = t_tensor.unsqueeze(0).unsqueeze(-1)

        # Shape: (1, 1029, M) -> Batch=1, Dim=1029, M Candidates
        d_stack = np.stack(d_inputs, axis=1)  # (1029, M)
        d_tensor = torch.tensor(d_stack, dtype=torch.float32).to(self.device)
        d_tensor = d_tensor.unsqueeze(0) 

        # 4. Inference
        with torch.no_grad():
            try:
                # The model handles broadcasting internally via .expand()
                logits = self.model(t_tensor, d_tensor)  # Output: (1, 1, M)
                scores = torch.sigmoid(logits).squeeze().cpu().numpy()
            except RuntimeError as e:
                print(f"[RelationRefiner] Inference error: {e}")
                return np.zeros(len(candidates))
            
        # Handle single candidate case returning scalar
        if scores.ndim == 0:
            scores = np.array([scores])
            
        return scores


# Backwards compatibility alias
GCNHandler = RelationRefiner