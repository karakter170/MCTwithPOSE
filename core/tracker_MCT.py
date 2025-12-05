# tracker_MCT.py
# FIXED VERSION - ALL BUGS PATCHED + ADAPTIVE THRESHOLD INTEGRATED
#
# FIXES APPLIED:
# 1. Fixed GCNHandler import
# 2. Added FAISS lock
# 3. Added pending tracks GC
# 4. Improved thread safety
# 5. INTEGRATED: Adaptive Threshold Logic (Improvement 3)

import numpy as np
from scipy.signal import savgol_filter 
from numpy.linalg import inv, norm 
import pickle 
import time 
import faiss 
from datetime import datetime 
from utils.staff_filter import StaffFilter
import threading
from core.hungarian_matcher import TwoStageHungarianMatcher, MatchResult
from typing import List, Dict, Tuple
from utils.adaptive_threshold import AdaptiveThresholdManager

# FIX #1: Correct import - class was renamed to RelationRefiner
try:
    from models.gcn_handler import RelationRefiner as GCNHandler
    GCN_AVAILABLE = True
except ImportError:
    print("WARNING: 'models/gcn_handler.py' not found! GCN disabled.")
    GCN_AVAILABLE = False

try:
    from utils.re_ranking import re_ranking
    RERANK_AVAILABLE = True
except ImportError:
    print("WARNING: 'utils/re_ranking.py' not found! Re-Ranking disabled.")
    RERANK_AVAILABLE = False
    def re_ranking(q, g, **kwargs): return None

def get_direction(v):
    m = norm(v)
    return (v / (m + 1e-6), m) if m > 0 else (np.zeros_like(v), 0)

def calculate_iou(bbox1, bbox2):
    """Intersection over Union (IoU)"""
    xx1 = max(bbox1[0], bbox2[0]); yy1 = max(bbox1[1], bbox2[1])
    xx2 = min(bbox1[2], bbox2[2]); yy2 = min(bbox1[3], bbox2[3])
    w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
    inter_area = w * h
    area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    return inter_area / (area1 + area2 - inter_area + 1e-6)

def cosine_distance(query_vec, gallery_vecs):
    """Fast cosine distance using numpy."""
    if len(gallery_vecs) == 0:
        return np.array([])
    similarities = gallery_vecs @ query_vec.T
    distances = 1.0 - similarities.squeeze()
    if distances.ndim == 0:
        distances = np.array([distances])
    return distances

def cosine_distance_single(query, target):
    """Computes distance between 1 query and 1 target (0..2)."""
    if query is None or target is None: return 2.0
    sim = np.dot(query, target)
    return 1.0 - sim

def compute_fused_distance(query_vec, query_bbox, gallery_vecs, gallery_bboxes, 
                           same_camera_mask, alpha=0.7, beta=0.3):
    """Compute fused distance combining appearance and geometry."""
    N = len(gallery_vecs)
    if N == 0:
        return np.array([])
    
    app_dist = cosine_distance(query_vec, gallery_vecs)
    
    iou_dist = np.ones(N)
    for i, (gb, same_cam) in enumerate(zip(gallery_bboxes, same_camera_mask)):
        if same_cam and gb is not None:
            iou_val = calculate_iou(query_bbox, gb)
            iou_dist[i] = 1.0 - iou_val
    
    fused = np.where(same_camera_mask, alpha * app_dist + beta * iou_dist, app_dist)
    return fused


class OCSORTTracker:
    def __init__(self, dt, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov):
        self.dt = dt
        self.F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = process_noise_cov 
        self.R_base = measurement_noise_cov
        self.x = initial_state      
        self.P = initial_covariance 
        self.last_observation = initial_state[:2]
        self.time_since_update = 0
        self.last_update_time = time.time()
        self.history_observations = []; self.history_x = []; self.history_y = []
        self.smooth_pos = initial_state[:2]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += self.dt
        return self.x

    def update(self, z, current_bbox=None, last_bbox=None, use_orb=True, conf=None):
        adaptive_R = self.R_base.copy()
        
        if conf is not None:
            scale_factor = 1.0 / (conf + 1e-4)
            adaptive_R *= scale_factor

        if current_bbox is not None and last_bbox is not None:
            h_curr = current_bbox[3] - current_bbox[1]
            h_last = last_bbox[3] - last_bbox[1]
            if h_last > 0:
                ratio = (h_curr - h_last) / h_last
                if ratio > 0.40 or ratio < -0.25: adaptive_R *= 10.0
        
        if use_orb and self.time_since_update > (self.dt * 3.0):
            delta_pos = z - self.last_observation
            delta_time = self.time_since_update
            virtual_velocity = delta_pos / (delta_time + 1e-6)
            prev_speed = norm(self.x[2:])
            curr_speed = norm(virtual_velocity)
            if prev_speed > 0 and curr_speed > (prev_speed * 3.0): virtual_velocity *= 0.5
            self.x = np.array([z[0], z[1], virtual_velocity[0], virtual_velocity[1]])
            self.P = np.eye(4) * adaptive_R[0,0] * 5.0 

        y = z - self.H @ self.x 
        S = self.H @ self.P @ self.H.T + adaptive_R 
        K = self.P @ self.H.T @ inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        
        self.last_observation = z
        self.time_since_update = 0
        self.last_update_time = time.time()
        self.history_observations.append(z)
        if len(self.history_observations) > 10: self.history_observations.pop(0)
        
        self.history_x.append(float(self.x[0])); self.history_y.append(float(self.x[1]))
        if len(self.history_x) > 7: self.history_x.pop(0); self.history_y.pop(0)
        
        if len(self.history_x) >= 7:
            try:
                self.smooth_pos = np.array([savgol_filter(self.history_x, 7, 2)[-1], savgol_filter(self.history_y, 7, 2)[-1]])
            except (ValueError, IndexError):
                self.smooth_pos = self.x[:2]
        else:
            self.smooth_pos = self.x[:2]
        return self.x

    def interpolate(self, new_pos, n_steps):
        if n_steps <= 0: return
        start_pos = self.last_observation
        step_vec = (new_pos - start_pos) / (n_steps + 1)
        for i in range(1, n_steps + 1):
            inter_p = start_pos + (step_vec * i)
            self.history_x.append(float(inter_p[0]))
            self.history_y.append(float(inter_p[1]))
            if len(self.history_x) > 15:
                self.history_x.pop(0); self.history_y.pop(0)


def uncertainty_adjusted_distance(query, target_mean, target_var):
    """
    Computes Cosine Distance, relaxed by the target's Variance (Uncertainty).
    """
    if query is None or target_mean is None: return 2.0
    
    sim = np.dot(query, target_mean)
    raw_dist = 1.0 - sim
    
    if target_var is None:
        return raw_dist
        
    uncertainty = np.mean(target_var)
    SCALE_FACTOR = 5.0
    adjusted_dist = raw_dist / (1.0 + (uncertainty * SCALE_FACTOR))
    
    return adjusted_dist


class GlobalTrack:
    """Represents a globally tracked identity across multiple cameras."""
    def __init__(self, global_id, group_id, dt, initial_state, initial_P, Q_cov, R_cov):
        self.global_id = global_id
        self.group_id = group_id
        self.kf = OCSORTTracker(dt, initial_state, initial_P, Q_cov, R_cov)
        self.last_cam_id = None
        self.last_seen_bbox = [0, 0, 0, 0]
        self.last_seen_timestamp = time.time()
        self.is_staff = False
        self.shadow_role = None
        self.shadow_name = None
        self.last_cam_res = (1920, 1080)
        
        # Memory States (Dual-Query System)
        self.last_known_feature = None
        self.fast_buffer = []
        self.robust_id = None
        self.robust_var = None


class TrackerManagerMCT:
    def __init__(self, dt, Q_cov, R_cov, feature_dim=1024, redis_client=None, topology_manager=None):
        self.redis_client = redis_client
        self.topology = topology_manager
        self.feature_dim = feature_dim
        self.dt = dt
        self.Q_cov = Q_cov
        self.R_cov = R_cov
        
        self.reid_threshold = 0.55
        self.max_time_gap = 60.0
        self.staff_filters = {}
        self.global_tracks = {}
        self.edge_to_global_map = {}
        self.pending_tracks = {}
        self.last_gc_time = time.time()
        self.gc_interval = 60.0
        self.max_keep_alive = 300.0
        self.max_faiss_size = 20000
        self.faiss_index = faiss.IndexFlatIP(self.feature_dim)
        self.faiss_id_map = []
        
        self.use_reranking_cross_camera = True
        self.use_reranking_same_camera = False
        self.rerank_cache = {}
        self.rerank_cache_ttl = 5.0
        
        self.min_features_for_id = 3
        self.pending_track_timeout = 30.0 

        self._faiss_lock = threading.RLock()

        self.hungarian_matcher = TwoStageHungarianMatcher(
            match_threshold=self.reid_threshold,
            use_appearance=True,
            use_motion=True,
            use_iou=True,
            appearance_weight=0.6,
            motion_weight=0.2,
            iou_weight=0.2,
            high_conf_threshold=0.7
        )

        self.threshold_manager = AdaptiveThresholdManager(
            base_threshold=0.55,
            min_threshold=0.35,
            max_threshold=0.75,
            adaptation_rate=0.1
        )
        
        self.gcn_refiner = None
        if GCN_AVAILABLE:
            try:
                self.gcn_refiner = GCNHandler("models/sota_gcn_dinov3_5_2dim.pth")
                print("[Central] GCN Refiner Loaded.")
            except Exception as e:
                print(f"[Central] GCN Load Failed: {e}")

        if not self.redis_client.exists("mct:next_global_id"):
            self.redis_client.set("mct:next_global_id", 1)

    def _get_staff_filter(self, group_id):
        if group_id not in self.staff_filters:
            self.staff_filters[group_id] = StaffFilter(self.redis_client, group_id)
        return self.staff_filters[group_id]

    def _run_garbage_collection(self):
        curr_time = time.time()
        
        expired_pending = []
        for key, pt in self.pending_tracks.items():
            kf = pt.get('kf')
            if kf is not None:
                age = curr_time - getattr(kf, 'last_update_time', curr_time - 60)
            else:
                age = self.pending_track_timeout + 1
            
            if age > self.pending_track_timeout:
                expired_pending.append(key)
        
        if expired_pending:
            print(f"[GC] Removing {len(expired_pending)} expired pending tracks.")
            for key in expired_pending:
                del self.pending_tracks[key]
        
        with self._faiss_lock:
            expired = [gid for gid, t in self.global_tracks.items() 
                      if (curr_time - t.last_seen_timestamp) > self.max_keep_alive]
            if expired:
                print(f"[GC] Removing {len(expired)} expired tracks.")
                for gid in expired:
                    del self.global_tracks[gid]
                    keys_to_del = [k for k, v in self.edge_to_global_map.items() if v == gid]
                    for k in keys_to_del: 
                        del self.edge_to_global_map[k]
        
        if self.faiss_index.ntotal > self.max_faiss_size:
            print("[GC] Rebuilding FAISS Index...")
            with self._faiss_lock:
                active_gids = list(self.global_tracks.keys())
                new_vectors = []
                new_id_map = []
                
                for gid in active_gids:
                    track = self.global_tracks[gid]
                    feat = track.robust_id if track.robust_id is not None else track.last_known_feature
                    if feat is None:
                        continue
                    norm_feat = feat / (norm(feat) + 1e-6)
                    new_vectors.append(norm_feat.astype(np.float32))
                    new_id_map.append(gid)
                
                self.faiss_index.reset()
                self.faiss_id_map = []
                
                if new_vectors:
                    vectors_array = np.array(new_vectors).astype(np.float32)
                    self.faiss_index.add(vectors_array)
                    self.faiss_id_map = new_id_map
                
                print(f"[GC] FAISS rebuilt: {len(new_id_map)} vectors")
        
        expired_cache = [k for k, v in self.rerank_cache.items() 
                        if curr_time - v['time'] > self.rerank_cache_ttl]
        for k in expired_cache:
            del self.rerank_cache[k]

    def _manage_gallery_diversity(self, gid, new_vector):
        if new_vector is None: return None
        key = f"gallery_core:{gid}"
        norm_vec = new_vector / (norm(new_vector) + 1e-6)
        
        data = self.redis_client.get(key)
        core_set = pickle.loads(data) if data else []
        
        if not core_set:
            core_set.append(norm_vec)
        else:
            dists = [np.dot(v, norm_vec) for v in core_set]
            max_sim = max(dists)
            best_idx = np.argmax(dists)
            
            if max_sim > 0.95:
                core_set[best_idx] = 0.9 * core_set[best_idx] + 0.1 * norm_vec
                core_set[best_idx] /= (norm(core_set[best_idx]) + 1e-6)
            elif max_sim < 0.85 and len(core_set) < 5:
                core_set.append(norm_vec)
            elif len(core_set) >= 5:
                core_set[best_idx] = 0.7 * core_set[best_idx] + 0.3 * norm_vec
                core_set[best_idx] /= (norm(core_set[best_idx]) + 1e-6)

        self.redis_client.set(key, pickle.dumps(core_set), ex=86400)
        return norm_vec

    def _update_counters(self, cam_id, group_id, mode, gid=None, role=None):
        today = datetime.now().strftime("%Y-%m-%d")
        p = self.redis_client.pipeline()
        if role: 
            key = f"mct:shadow:{group_id}:{today}"
            p.hincrby(key, f"{role}_count", 1)
        else:
            if mode == "tripwire":
                p.hincrby(f"mct:stats:{cam_id}:{today}", "total_tripwire", 1)
            elif mode == "unique" and gid:
                p.pfadd(f"mct:unique:{cam_id}:{today}", gid)
        p.execute()

    def _calculate_ocm_cost(self, track, new_gp, dt):
        track_vel = track.kf.x[2:] 
        candidate_vec = new_gp - track.kf.last_observation
        candidate_vel = candidate_vec / (dt + 1e-6)
        t_dir, t_mag = get_direction(track_vel)
        c_dir, c_mag = get_direction(candidate_vel)
        cos_sim = np.dot(t_dir, c_dir)
        if t_mag < 0.5: return 0.0 
        return 0.5 * (1.0 - cos_sim)

    def _fast_match(self, cam_id, group_id, feature, bbox, gp, curr_time, frame_res=(1920, 1080)):
        if self.faiss_index.ntotal == 0:
            return None, 100.0
        
        # ---------------- INTEGRATED: ADAPTIVE THRESHOLD ----------------
        detections = [{'feature': feature, 'bbox': bbox}]
        adaptive_threshold = self.threshold_manager.get_threshold(
            cam_id, detections, self.global_tracks, curr_time
        )
        # ----------------------------------------------------------------
        
        q_vec = np.array([feature]).astype(np.float32)
        q_vec = q_vec / (norm(q_vec) + 1e-6)
        
        shortlist_k = min(20, self.faiss_index.ntotal)
        
        with self._faiss_lock:
            D_raw, I_raw = self.faiss_index.search(q_vec, k=shortlist_k)
            id_map_snapshot = self.faiss_id_map.copy()

        candidates = []
        for idx in I_raw[0]:
            if idx == -1 or idx >= len(id_map_snapshot): continue
            cand_gid = id_map_snapshot[idx]
            cand_track = self.global_tracks.get(cand_gid)
            if cand_track and cand_track.group_id == group_id:
                candidates.append(cand_track)
        
        if not candidates:
            return None, 100.0
        
        best_gid = None
        best_score = 100.0
        
        for track in candidates:
            # DUAL-QUERY matching
            if track.fast_buffer:
                dists = [cosine_distance_single(q_vec[0], f) for f in track.fast_buffer]
                dist_fast = min(dists)
            else:
                dist_fast = cosine_distance_single(q_vec[0], track.last_known_feature)
            
            dist_slow = uncertainty_adjusted_distance(
                q_vec[0], 
                track.robust_id, 
                track.robust_var
            )
            
            app_dist = min(dist_fast, dist_slow)
            
            if track.last_cam_id == cam_id:
                iou_val = calculate_iou(bbox, track.last_seen_bbox)
                iou_dist = 1.0 - iou_val
                fused = 0.7 * app_dist + 0.3 * iou_dist
            else:
                fused = app_dist
            
            dt = curr_time - track.last_seen_timestamp
            if self.topology and track.last_cam_id != cam_id:
                prob_geo = self.topology.get_transition_prob(group_id, track.last_cam_id, cam_id, dt)
                if prob_geo < 0.01: fused = 100.0
                elif prob_geo < 0.5: fused *= 1.2
            
            motion_cost = self._calculate_ocm_cost(track, gp, dt)
            if motion_cost > 0.7: fused = 100.0
            elif motion_cost > 0.3: fused *= 1.15
            
            for k, v in self.edge_to_global_map.items():
                if v == track.global_id and k.startswith(f"{cam_id}_"):
                    fused = 100.0
                    break
        
            # UPDATED: Use adaptive threshold instead of fixed self.reid_threshold
            if fused < adaptive_threshold and fused < best_score:
                best_score = fused
                best_gid = track.global_id
        
        # ADDED: Report match for adaptive learning
        if best_gid is not None:
            self.threshold_manager.report_match(
                camera_id=cam_id,
                match_score=best_score,
                is_cross_camera=False
            )
        
        return best_gid, best_score

    def _cross_camera_match(self, cam_id, group_id, feature, bbox, gp, curr_time, frame_res):
        """Cross-camera matching with batch GCN refinement + Adaptive Threshold."""
        if self.faiss_index.ntotal == 0:
            return None, 100.0
        
        frame_w, frame_h = frame_res

        # ---------------- INTEGRATED: ADAPTIVE THRESHOLD ----------------
        # Get adaptive threshold (stricter for cross-camera)
        detections = [{'feature': feature, 'bbox': bbox}]
        base_threshold = self.threshold_manager.get_threshold(
            cam_id, detections, self.global_tracks, curr_time
        )
        cross_cam_threshold = base_threshold * 0.95  # 5% stricter
        # ----------------------------------------------------------------
        
        q_vec = np.array([feature]).astype(np.float32)
        q_vec = q_vec / (norm(q_vec) + 1e-6)

        shortlist_k = min(100, self.faiss_index.ntotal)
        
        with self._faiss_lock:
            D_raw, I_raw = self.faiss_index.search(q_vec, k=shortlist_k)
            id_map_snapshot = self.faiss_id_map.copy()

        batch_candidates = []
        # UPDATED: Use dynamic cutoff instead of fixed 0.7 if desired, 
        # or keep 0.7 as a loose pre-filter and use cross_cam_threshold for final.
        VISUAL_CUTOFF = max(0.7, cross_cam_threshold + 0.1)
        
        for idx in I_raw[0]:
            if idx == -1 or idx >= len(id_map_snapshot): continue
            
            cand_gid = id_map_snapshot[idx]
            track = self.global_tracks.get(cand_gid)
            
            if not track or track.group_id != group_id: continue
            if track.last_cam_id == cam_id: continue
            
            dt = curr_time - track.last_seen_timestamp
            if dt > self.max_time_gap: continue

            dist_fast = cosine_distance_single(q_vec[0], track.last_known_feature)
            dist_slow = uncertainty_adjusted_distance(q_vec[0], track.robust_id, track.robust_var)
            visual_score = min(dist_fast, dist_slow)
            
            geo_penalty = 1.0
            if self.topology:
                prob_geo = self.topology.get_transition_prob(group_id, track.last_cam_id, cam_id, dt)
                if prob_geo < 0.01: continue 
                elif prob_geo < 0.5: geo_penalty = 1.2
            
            motion_cost = self._calculate_ocm_cost(track, gp, dt)
            if motion_cost > 0.7: continue
            elif motion_cost > 0.3: geo_penalty *= 1.15
            
            final_visual_score = visual_score * geo_penalty

            if final_visual_score < VISUAL_CUTOFF:
                batch_candidates.append({
                    'gid': cand_gid,
                    'track': track,
                    'visual_score': final_visual_score
                })

        best_gid = None
        best_score = 100.0

        if self.gcn_refiner and batch_candidates:
            try:
                norm_q_bbox = [
                    bbox[0] / frame_w, bbox[1] / frame_h,
                    bbox[2] / frame_w, bbox[3] / frame_h
                ]

                class DummyTrack:
                    def __init__(self, feat, box, timestamp):
                        self.robust_id = feat
                        self.last_known_feature = feat
                        self.last_seen_bbox = box
                        self.last_cam_res = (1.0, 1.0)
                        self.last_seen_timestamp = timestamp

                dummy_query = DummyTrack(feature, norm_q_bbox, curr_time)
                
                refiner_candidates = []
                for item in batch_candidates:
                    t = item['track']
                    feat = t.robust_id if t.robust_id is not None else t.last_known_feature
                    
                    t_w, t_h = getattr(t, 'last_cam_res', (1920, 1080))
                    t_bbox = t.last_seen_bbox
                    norm_t_bbox = [
                        t_bbox[0] / t_w, t_bbox[1] / t_h,
                        t_bbox[2] / t_w, t_bbox[3] / t_h
                    ]
                    
                    refiner_candidates.append({
                        'feature': feat,
                        'bbox': norm_t_bbox
                    })
                
                refine_scores = self.gcn_refiner.predict_batch(
                    dummy_query, 
                    refiner_candidates, 
                    frame_w=1.0, 
                    frame_h=1.0,
                    curr_time=curr_time
                )
                
                for i, item in enumerate(batch_candidates):
                    visual = item['visual_score']
                    gcn_conf = refine_scores[i]
                    gcn_dist = 1.0 - gcn_conf
                    
                    fused_score = visual * 0.6 + gcn_dist * 0.4
                    
                    # UPDATED: Use adaptive cross_cam_threshold
                    if fused_score < cross_cam_threshold and fused_score < best_score:
                        best_score = fused_score
                        best_gid = item['gid']
                        
            except Exception as e:
                print(f"[Tracker] Refinement Error: {e}")
                for item in batch_candidates:
                    # UPDATED: Fallback uses adaptive threshold too
                    if item['visual_score'] < cross_cam_threshold and item['visual_score'] < best_score:
                        best_score = item['visual_score']
                        best_gid = item['gid']
        else:
            for item in batch_candidates:
                # UPDATED: Standard match uses adaptive threshold
                if item['visual_score'] < cross_cam_threshold and item['visual_score'] < best_score:
                    best_score = item['visual_score']
                    best_gid = item['gid']

        # ADDED: Report match for adaptive learning
        if best_gid is not None:
            self.threshold_manager.report_match(
                camera_id=cam_id,
                match_score=best_score,
                is_cross_camera=True
            )

        return best_gid, best_score

    def batch_match_detections(
        self, 
        cam_id: str, 
        group_id: str, 
        detections: List[Dict],
        current_time: float
    ) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Match multiple detections to tracks using Hungarian algorithm + GCN Refinement.
        """
        # 1. Filter tracks by group
        group_tracks = {
            gid: track for gid, track in self.global_tracks.items()
            if track.group_id == group_id
        }
        
        if not group_tracks:
            return [], list(range(len(detections))), []
        
        # 2. INTEGRATED: ADAPTIVE THRESHOLD
        # Sahne yoğunluğuna göre dinamik eşik değerini al
        adaptive_threshold = self.threshold_manager.get_threshold(
            cam_id, detections, self.global_tracks, current_time
        )
        self.hungarian_matcher.match_threshold = adaptive_threshold

        # 3. GCN MODELİNİ HAZIRLA (Refiner Model)
        # Eğer GCN yüklü ise (self.gcn_refiner), bunu Hungarian Matcher'a ileteceğiz.
        refiner_model = None
        if hasattr(self, 'gcn_refiner') and self.gcn_refiner is not None:
            refiner_model = self.gcn_refiner
            
        # 4. Frame Çözünürlüğünü Tespit Et
        # Edge tarafından gönderilen veride 'frame_res' alanı varsa kullan, yoksa varsayılan.
        frame_res = (1920, 1080)
        if detections and 'original_event' in detections[0]:
             orig = detections[0].get('original_event', {})
             if 'frame_res' in orig:
                 frame_res = tuple(orig['frame_res'])

        # 5. Run Hungarian matching (GCN Destekli)
        result = self.hungarian_matcher.match(
            detections=detections,
            tracks=group_tracks,
            camera_id=cam_id,
            current_time=current_time,
            refiner_model=refiner_model, # <--- GCN Modeli Burada Gönderiliyor
            frame_res=frame_res          # <--- Çözünürlük Burada Gönderiliyor
        )
        
        # 6. Eşleşme Sonuçlarını Raporla (Adaptive Threshold Feedback)
        # Başarılı eşleşmeleri sisteme bildir ki threshold kendini eğitsin.
        for det_idx, global_id in result.matches:
            # Match score'u maliyet matrisinden çekmeye çalışıyoruz
            score = 0.5 # Varsayılan güvenli skor
            
            if hasattr(result, 'cost_matrix') and result.cost_matrix is not None:
                try:
                    # FIX: HungarianMatcher'dan dönen valid_track_ids listesini kullan
                    # Bu liste matrisin sütunlarının hangi Global ID'ye ait olduğunu söyler.
                    valid_ids = getattr(result, 'valid_track_ids', None)
                    
                    if valid_ids is not None:
                        if global_id in valid_ids:
                            col_idx = valid_ids.index(global_id)
                            score = result.cost_matrix[det_idx, col_idx]
                    else:
                        # Fallback (Eski usül, riskli ama hiç yoktan iyidir)
                        # Eğer valid_ids yoksa, group_tracks.keys() sırasına güvenmeye çalışırız
                        track_keys = list(group_tracks.keys())
                        if global_id in track_keys:
                            track_idx = track_keys.index(global_id)
                            score = result.cost_matrix[det_idx, track_idx]
                            
                except (ValueError, IndexError):
                    # Herhangi bir indeks hatasında varsayılan skoru (0.5) kullan
                    pass
            
            # Cross-camera olup olmadığını kontrol et
            is_cross = False
            if global_id in group_tracks:
                track = group_tracks[global_id]
                if track.last_cam_id != cam_id:
                    is_cross = True

            self.threshold_manager.report_match(
                camera_id=cam_id,
                match_score=score,
                is_cross_camera=is_cross
            )
        
        # Eşleşmeyen (Yeni) detectionları da raporla
        for det_idx in result.unmatched_detections:
            quality = detections[det_idx].get('quality', 0.5)
            self.threshold_manager.report_new_id(cam_id, quality)

        return result.matches, result.unmatched_detections, result.unmatched_tracks
    
    def update_edge_track_position(self, cam_id, group_id, edge_id, gp, conf, bbox):
        map_key = f"{cam_id}_{edge_id}"
        gid = self.edge_to_global_map.get(map_key)
        
        if gid and gid in self.global_tracks:
            track = self.global_tracks[gid]
            track.kf.predict()
            track.kf.update(gp, current_bbox=bbox, last_bbox=track.last_seen_bbox, use_orb=False, conf=conf)
            track.last_seen_bbox = list(bbox) if not isinstance(bbox, list) else bbox
            track.last_seen_timestamp = time.time()
            track.last_cam_id = cam_id
            
            sf = self._get_staff_filter(group_id)
            is_staff, role, name = sf.identify_staff(vector=None, global_id=gid)
            if is_staff:
                track.is_staff = True; track.shadow_role = role; track.shadow_name = name
            return track
        elif map_key in self.pending_tracks:
            self.pending_tracks[map_key]["kf"].predict()
            self.pending_tracks[map_key]["kf"].update(gp, current_bbox=bbox, last_bbox=self.pending_tracks[map_key]["last_bbox"], use_orb=False, conf=conf)
            self.pending_tracks[map_key]["last_bbox"] = list(bbox) if not isinstance(bbox, list) else bbox
        return None

    def update_edge_track_feature(self, cam_id, group_id, edge_id, gp, conf, bbox, feature, quality_score, frame_res=(1920, 1080)):
        if time.time() - self.last_gc_time > self.gc_interval:
            self._run_garbage_collection()
            self.last_gc_time = time.time()

        map_key = f"{cam_id}_{edge_id}"
        curr_time = time.time()
        bbox = list(bbox) if not isinstance(bbox, list) else bbox
        
        existing_gid = self.edge_to_global_map.get(map_key)
        
        if existing_gid and existing_gid in self.global_tracks:
            best_gid = existing_gid
        else:
            best_gid, best_score = self._fast_match(cam_id, group_id, feature, bbox, gp, curr_time, frame_res)
            if best_gid is None:
                best_gid, best_score = self._cross_camera_match(cam_id, group_id, feature, bbox, gp, curr_time, frame_res)

        if best_gid:
            gt = self.global_tracks[best_gid]
            sf = self._get_staff_filter(group_id)
            is_staff, role, name = sf.identify_staff(vector=feature, global_id=best_gid)
            if is_staff:
                gt.is_staff = True; gt.shadow_role = role; gt.shadow_name = name

            if self.topology and gt.last_cam_id and gt.last_cam_id != cam_id:
                self.topology.update_topology(group_id, gt.last_cam_id, cam_id, curr_time - gt.last_seen_timestamp)
            
            if gt.last_cam_id == cam_id:
                time_gap = curr_time - gt.last_seen_timestamp
                if 0.2 < time_gap < 2.0:
                    steps = int(time_gap * 20) 
                    if steps > 0: gt.kf.interpolate(gp, steps)

            gt.kf.update(gp, current_bbox=bbox, last_bbox=gt.last_seen_bbox, use_orb=True, conf=conf)
            
            is_bad_quality_box = False
            if gt.last_seen_bbox is not None:
                h_curr = bbox[3] - bbox[1]; h_last = gt.last_seen_bbox[3] - gt.last_seen_bbox[1]
                if h_last > 0:
                    ratio = (h_curr - h_last) / h_last
                    if ratio < -0.25 or ratio > 0.40: is_bad_quality_box = True
            
            gt.fast_buffer.append(feature)
            if len(gt.fast_buffer) > 5:
                gt.fast_buffer.pop(0)
            
            gt.last_seen_bbox = bbox
            gt.last_seen_timestamp = curr_time
            gt.last_cam_id = cam_id
            gt.last_known_feature = feature
            gt.last_cam_res = frame_res
            
            self.edge_to_global_map[map_key] = best_gid
            if map_key in self.pending_tracks: del self.pending_tracks[map_key]
            
            if not is_bad_quality_box:
                self._manage_gallery_diversity(best_gid, feature)
            
            if not gt.is_staff: self._update_counters(cam_id, group_id, "unique", best_gid)
            return gt
        
        else:
            # Report new ID creation to threshold manager (feedback that no match was found)
            self.threshold_manager.report_new_id(cam_id, quality_score)
            
            if map_key in self.pending_tracks:
                pt = self.pending_tracks[map_key]
                pt["kf"].update(gp, current_bbox=bbox, last_bbox=pt["last_bbox"], use_orb=False, conf=conf)
                pt["features_count"] += 1
                pt["last_bbox"] = bbox
                
                if "first_feature" not in pt:
                    pt["first_feature"] = feature
                pt["last_feature"] = feature
                
                if pt["features_count"] >= self.min_features_for_id:
                    pt_data = self.pending_tracks.pop(map_key)
                    new_gid = int(self.redis_client.incr("mct:next_global_id"))
                    
                    new_gt = GlobalTrack(
                        new_gid, group_id, self.dt, 
                        pt_data["kf"].x, 
                        np.eye(4)*100, 
                        self.Q_cov, 
                        self.R_cov
                    )
                    
                    new_gt.last_seen_bbox = pt_data["last_bbox"]
                    new_gt.last_seen_timestamp = curr_time
                    new_gt.last_cam_id = cam_id
                    new_gt.last_known_feature = pt_data.get("last_feature", feature)
                    
                    sf = self._get_staff_filter(group_id)
                    is_staff, role, name = sf.identify_staff(vector=feature, global_id=new_gid)
                    if is_staff:
                        new_gt.is_staff = True
                        new_gt.shadow_role = role
                        new_gt.shadow_name = name
                        self._update_counters(cam_id, group_id, "shadow", role=role)
                    else:
                        self._update_counters(cam_id, group_id, "unique", new_gid)
                    
                    self.global_tracks[new_gid] = new_gt
                    self.edge_to_global_map[map_key] = new_gid
                    
                    norm_vec = self._manage_gallery_diversity(new_gid, feature)
                    
                    with self._faiss_lock:
                        if self.faiss_index is not None and norm_vec is not None:
                            self.faiss_index.add(np.array([norm_vec]).astype(np.float32))
                            self.faiss_id_map.append(new_gid)
                    
                    print(f"[Tracker] NEW ID assigned: G{new_gid} for {map_key}")
                    return new_gt
            else:
                kf = OCSORTTracker(
                    self.dt, 
                    np.array([gp[0], gp[1], 0, 0]), 
                    np.eye(4)*100, 
                    self.Q_cov, 
                    self.R_cov
                )
                kf.update(gp, current_bbox=bbox, last_bbox=bbox, use_orb=False, conf=conf)
                
                self.pending_tracks[map_key] = {
                    "kf": kf, 
                    "last_bbox": bbox, 
                    "features_count": 1,
                    "first_feature": feature,
                    "last_feature": feature,
                    "group_id": group_id
                }
            
            return None

    def register_new_edge_track(self, cam_id, group_id, edge_id, gp, conf, bbox):
        map_key = f"{cam_id}_{edge_id}"
        if map_key in self.edge_to_global_map or map_key in self.pending_tracks:
            return
        
        bbox = list(bbox) if not isinstance(bbox, list) else bbox
        
        kf = OCSORTTracker(
            self.dt, 
            np.array([gp[0], gp[1], 0, 0]), 
            np.eye(4)*100, 
            self.Q_cov, 
            self.R_cov
        )
        kf.update(gp, current_bbox=bbox, last_bbox=bbox, use_orb=False, conf=conf)
        
        self.pending_tracks[map_key] = {
            "kf": kf, 
            "last_bbox": bbox, 
            "features_count": 0,
            "group_id": group_id
        }

    def lost_edge_track(self, cam_id, edge_id):
        map_key = f"{cam_id}_{edge_id}"
        self.edge_to_global_map.pop(map_key, None)
        self.pending_tracks.pop(map_key, None)

    def get_global_id_for_edge_track(self, cam_id, edge_id):
        return self.edge_to_global_map.get(f"{cam_id}_{edge_id}", "Unknown")
    
    def get_viz_data_for_camera(self, cam_id):
        viz = []
        for k, gid in self.edge_to_global_map.items():
            if k.startswith(f"{cam_id}_"):
                gt = self.global_tracks.get(gid)
                if gt:
                    viz.append({
                        "edge_track_id": int(k.split('_')[-1]),
                        "global_id": gid,
                        "bbox": gt.last_seen_bbox,
                        "smooth_gp": gt.kf.smooth_pos.tolist(),
                        "is_staff": gt.is_staff,
                        "role": gt.shadow_role,
                        "name": getattr(gt, 'shadow_name', 'Unknown')
                    })
        return viz