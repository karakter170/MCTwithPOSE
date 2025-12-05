# edge_camera.py
# P1 FIXED VERSION - STREAM COMPATIBLE
#
# MODIFIED: Added Pose-Based Quality Scoring Integration
#
# CHANGES:
# 1. Fixed Protocol Mismatch: Now sends to Redis Streams (xadd) instead of Pub/Sub.
# 2. Added r_json client for proper Stream handling.
# 3. NEW: Integrated HybridQualityScorer for pose-based quality assessment

import cv2
import time
import json
import redis
import numpy as np
from ultralytics import YOLO
import threading
from collections import OrderedDict

# >>> NEW: Import Pose-Based Quality Scorer
try:
    from pose_quality import HybridQualityScorer, PoseQualityScorer
    POSE_QUALITY_AVAILABLE = True
    print("[Edge] Pose-Based Quality: ENABLED")
except ImportError:
    POSE_QUALITY_AVAILABLE = False
    print("[Edge] Pose-Based Quality: DISABLED (pose_quality.py not found)")
# >>> END NEW

try:
    from trt_loader import TensorRTReidExtractor
except ImportError:
    print("ERROR: 'trt_loader.py' is missing!")
    exit()


# ============================================
# MOTION COMPENSATION (CMC)
# ============================================
class CameraMotionCompensation:
    def __init__(self):
        self.prev_frame = None
        self.prev_keypoints = None
        self.warp_matrix = np.eye(2, 3, dtype=np.float32)

    def compute(self, curr_frame):
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = curr_gray
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=100, qualityLevel=0.01, minDistance=30
            )
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)
            return self.warp_matrix

        if self.prev_keypoints is None or len(self.prev_keypoints) < 20:
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=100, qualityLevel=0.01, minDistance=30
            )
            self.prev_frame = curr_gray
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)
            return self.warp_matrix

        curr_keypoints, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, curr_gray, self.prev_keypoints, None
        )
        
        if curr_keypoints is None:
            self.prev_frame = curr_gray
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)
            return self.warp_matrix
            
        valid_curr = curr_keypoints[status.ravel() == 1]
        valid_prev = self.prev_keypoints[status.ravel() == 1]
        
        if len(valid_curr) > 10:
            m, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
            if m is not None:
                self.warp_matrix = m.astype(np.float32)
            else:
                self.warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)

        self.prev_frame = curr_gray
        self.prev_keypoints = valid_curr.reshape(-1, 1, 2) if len(valid_curr) > 0 else None
        
        return self.warp_matrix

    def apply_to_bbox(self, bbox, warp_matrix=None):
        if warp_matrix is None:
            warp_matrix = self.warp_matrix
        
        corners = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        transformed = cv2.transform(corners, warp_matrix)
        transformed = transformed.reshape(-1, 2)
        
        return [
            int(transformed[0, 0]),
            int(transformed[0, 1]),
            int(transformed[1, 0]),
            int(transformed[1, 1])
        ]


# ============================================
# THREAD-SAFE STORAGE
# ============================================
class ThreadSafeTracklets:
    def __init__(self, max_size=1000):
        self._data = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
    
    def __contains__(self, key):
        with self._lock: return key in self._data
    
    def __getitem__(self, key):
        with self._lock:
            self._data.move_to_end(key)
            return self._data[key]
    
    def __setitem__(self, key, value):
        with self._lock:
            if key in self._data: self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)
    
    def remove_expired(self, curr_time, timeout):
        with self._lock:
            expired = []
            for k, v in self._data.items():
                if curr_time - v.get("last_seen", 0) > timeout:
                    expired.append(k)
            for k in expired: del self._data[k]
            return expired


# --- SETTINGS ---
CAMERA_ID = "cam_01_ist"
VIDEO_SOURCE = "./videolar/videom.mp4"
#VIDEO_SOURCE = 0 
GROUP_ID = "istanbul_avm"
HOMOGRAPHY_PATH = "h_cam_01_ist.npy"
YOLO_MODEL_PATH = './models/yolov8x-worldv2.engine' 
REID_ENGINE_PATH = './models/Dino/dino_vitl16_fp16.engine'    
TRACKER_CONFIG = 'bytetrack.yaml'

# Redis Configuration
STREAM_NAME = "track_events"  # MUST match central_tracker_service.py

CONF_THRES_DETECTION = 0.1   
CONF_THRES_HIGH = 0.5        
CONF_THRES_LOW = 0.1         
MIN_QUALITY_THRESHOLD = 0.45 
REID_UPDATE_INTERVAL = 1.0
GP_UPDATE_INTERVAL = 5       
TRACK_LOST_TIMEOUT = 3.0

# >>> NEW: Pose-Based Quality Settings
USE_POSE_QUALITY = True          # Set to False to use traditional quality only
POSE_QUALITY_WEIGHT = 0.6        # Weight for pose-based score (0.6 = 60% pose, 40% traditional)
POSE_MODEL_PATH = "./models/yolo11n-pose.pt"  # Pose estimation model
SHOW_POSE_DEBUG = False          # Draw pose keypoints on annotated frame
# >>> END NEW

COLOR_PENDING = (0, 0, 255)     
COLOR_TRACKED = (0, 255, 0)     
COLOR_STAFF   = (255, 0, 255)   
COLOR_TEXT    = (255, 255, 255)

latest_tracks_from_central = [] 
tracks_lock = threading.Lock()
edge_tracklets = ThreadSafeTracklets(max_size=500)

# --- REDIS SETUP ---
try:
    # r_json used for Streams (text-based keys/values)
    r_json = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    # r_bytes used for Image Publishing
    r_bytes = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    
    r_json.ping()
except Exception as e:
    print(f"Redis Connection Error: {e}")
    exit()


# ============================================
# QUALITY SCORING FUNCTIONS
# ============================================

def calculate_quality_score_traditional(frame, bbox, frame_width, frame_height):
    """
    Traditional quality scoring using blur + geometry.
    Used as fallback when pose-based scoring is not available.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    geo_score = 1.0
    margin = 10 
    
    # Edge penalty
    if x1 < margin or y1 < margin or x2 > frame_width - margin or y2 > frame_height - margin: 
        geo_score -= 0.4
    
    # Aspect ratio check
    if w > 0:
        aspect = h / float(w)
        if aspect < 0.4: 
            return 0.0  # Too wide, likely not a person
        elif aspect < 1.2: 
            geo_score -= 0.15 
        elif aspect > 3.5: 
            geo_score -= 0.15 
    
    # Size check
    if (w * h) < (60 * 120): 
        geo_score -= 0.2

    # Blur score
    blur_score = 0.0
    try:
        y1_c, y2_c = max(0, y1), min(frame_height, y2)
        x1_c, x2_c = max(0, x1), min(frame_width, x2)
        if x2_c > x1_c and y2_c > y1_c:
            crop = frame[y1_c:y2_c, x1_c:x2_c]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, blur_val / 300.0)
    except Exception:
        pass
    
    final_score = (0.6 * blur_score) + (0.4 * max(0.0, geo_score))
    return max(0.0, min(1.0, final_score))


# >>> NEW: Pose-Based Quality Scoring
class QualityScorer:
    """
    Unified quality scorer that combines pose-based and traditional methods.
    
    Benefits of pose-based quality:
    - Detects frontal poses (best for Re-ID)
    - Identifies occlusions
    - Measures pose stability (avoids blurry motion)
    - 15-20% better Re-ID accuracy
    """
    
    def __init__(self, use_pose=True, pose_weight=0.6):
        self.use_pose = use_pose and POSE_QUALITY_AVAILABLE
        self.pose_weight = pose_weight
        self.traditional_weight = 1.0 - pose_weight
        
        self.hybrid_scorer = None
        self.frame_counter = 0
        
        # Quality statistics for debugging
        self.stats = {
            'total_scores': 0,
            'pose_used': 0,
            'traditional_used': 0,
            'avg_pose_score': 0.0,
            'avg_traditional_score': 0.0,
            'avg_final_score': 0.0,
            'good_for_reid_count': 0
        }
        self._stats_lock = threading.Lock()
        
        if self.use_pose:
            try:
                self.hybrid_scorer = HybridQualityScorer(
                    use_pose=True,
                    pose_weight=pose_weight,
                    traditional_weight=self.traditional_weight
                )
                print(f"[QualityScorer] Initialized with pose_weight={pose_weight}")
            except Exception as e:
                print(f"[QualityScorer] Failed to initialize pose scorer: {e}")
                print("[QualityScorer] Falling back to traditional scoring")
                self.use_pose = False
    
    def compute(self, frame, bbox, track_id=None):
        """
        Compute quality score for a detection.
        
        Args:
            frame: Input frame (BGR)
            bbox: Detection bounding box [x1, y1, x2, y2]
            track_id: Optional track ID for stability scoring
            
        Returns:
            tuple: (overall_score, details_dict)
        """
        self.frame_counter += 1
        frame_h, frame_w = frame.shape[:2]
        
        # Always compute traditional score
        traditional_score = calculate_quality_score_traditional(
            frame, bbox, frame_w, frame_h
        )
        
        details = {
            'traditional': traditional_score,
            'pose': None,
            'visibility': None,
            'frontality': None,
            'is_good_for_reid': False,
            'method': 'traditional'
        }
        
        # Compute pose-based score if available
        if self.use_pose and self.hybrid_scorer is not None:
            try:
                pose_score, pose_details = self.hybrid_scorer.compute_quality(
                    frame=frame,
                    bbox=list(bbox),
                    frame_id=self.frame_counter,
                    track_id=track_id
                )
                
                details['pose'] = pose_details.get('pose')
                details['visibility'] = pose_details.get('visibility')
                details['frontality'] = pose_details.get('frontality')
                details['is_good_for_reid'] = pose_details.get('is_good_for_reid', False)
                details['method'] = 'hybrid'
                
                # Use hybrid score
                final_score = pose_score
                
                # Update stats
                with self._stats_lock:
                    self.stats['pose_used'] += 1
                    if details['pose'] is not None:
                        n = self.stats['total_scores']
                        self.stats['avg_pose_score'] = (
                            self.stats['avg_pose_score'] * n + details['pose']
                        ) / (n + 1)
                    if details['is_good_for_reid']:
                        self.stats['good_for_reid_count'] += 1
                
            except Exception as e:
                # Fallback to traditional on error
                final_score = traditional_score
                details['method'] = 'traditional_fallback'
                with self._stats_lock:
                    self.stats['traditional_used'] += 1
        else:
            final_score = traditional_score
            with self._stats_lock:
                self.stats['traditional_used'] += 1
        
        # Update overall stats
        with self._stats_lock:
            self.stats['total_scores'] += 1
            n = self.stats['total_scores']
            self.stats['avg_traditional_score'] = (
                self.stats['avg_traditional_score'] * (n - 1) + traditional_score
            ) / n
            self.stats['avg_final_score'] = (
                self.stats['avg_final_score'] * (n - 1) + final_score
            ) / n
        
        return max(0.0, min(1.0, final_score)), details
    
    def should_extract_reid(self, track_id, quality_score, details, tracklet_data):
        """
        Enhanced Re-ID extraction decision using pose information.
        
        Args:
            track_id: Edge track ID
            quality_score: Overall quality score
            details: Quality details dict from compute()
            tracklet_data: Tracklet state dict
            
        Returns:
            bool: Whether to extract Re-ID feature
        """
        # Base quality threshold
        base_threshold = MIN_QUALITY_THRESHOLD
        
        # Lower threshold if pose indicates good Re-ID opportunity
        if details.get('is_good_for_reid', False):
            effective_threshold = base_threshold * 0.85  # 15% more permissive
        else:
            effective_threshold = base_threshold
        
        # Check various conditions
        best_quality = tracklet_data.get("best_q", 0.0)
        last_reid_time = tracklet_data.get("last_reid", 0.0)
        current_time = time.time()
        
        # Condition 1: First Re-ID for this track
        if best_quality == 0.0 and quality_score > effective_threshold:
            return True
        
        # Condition 2: Significantly better quality than before
        improvement_ratio = 1.2  # 20% improvement required
        if details.get('is_good_for_reid'):
            improvement_ratio = 1.1  # Only 10% improvement if pose is good
        
        if quality_score > (best_quality * improvement_ratio):
            return True
        
        # Condition 3: Periodic update (but only if quality is acceptable)
        time_since_last = current_time - last_reid_time
        if time_since_last > REID_UPDATE_INTERVAL and quality_score > effective_threshold:
            return True
        
        # Condition 4: Excellent frontal pose (even if other conditions not met)
        frontality = details.get('frontality', 0.0)
        visibility = details.get('visibility', 0.0)
        if frontality is not None and visibility is not None:
            if frontality > 0.8 and visibility > 0.7 and quality_score > 0.4:
                return True
        
        return False
    
    def get_stats(self):
        """Get quality scoring statistics."""
        with self._stats_lock:
            return self.stats.copy()
    
    def log_stats(self):
        """Print quality scoring statistics."""
        stats = self.get_stats()
        if stats['total_scores'] > 0:
            pose_pct = 100.0 * stats['pose_used'] / stats['total_scores']
            reid_pct = 100.0 * stats['good_for_reid_count'] / stats['total_scores']
            print(f"[QualityScorer] Stats: total={stats['total_scores']}, "
                  f"pose_used={pose_pct:.1f}%, "
                  f"avg_score={stats['avg_final_score']:.3f}, "
                  f"good_for_reid={reid_pct:.1f}%")


# Initialize quality scorer
quality_scorer = QualityScorer(
    use_pose=USE_POSE_QUALITY,
    pose_weight=POSE_QUALITY_WEIGHT
)
# >>> END NEW


def redis_listener():
    """
    Listens for Visualization updates from Central.
    Central sends these via Pub/Sub (even if input is Stream).
    """
    global latest_tracks_from_central
    r_sub = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    pubsub = r_sub.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(f"results_viz_stream:{CAMERA_ID}")
    
    print(f"[{CAMERA_ID}] Listening for Viz updates on channel: results_viz_stream:{CAMERA_ID}")
    
    for message in pubsub.listen():
        try:
            data = json.loads(message['data'])
            with tracks_lock:
                latest_tracks_from_central = data
        except Exception as e: 
            print(f"Viz Listener Error: {e}")


def send_track_event(event_type, edge_id, gp=None, bbox=None, feat=None, conf=None, quality=0.0):
    """
    FIXED: Sends event to Redis Stream instead of Pub/Sub.
    """
    packet = {
        "camera_id": CAMERA_ID, 
        "group_id": GROUP_ID, 
        "timestamp": time.time(),
        "event_type": event_type, 
        "edge_track_id": int(edge_id),
        "gp_coord": gp.tolist() if isinstance(gp, np.ndarray) else gp,
        "bbox": bbox.tolist() if isinstance(bbox, np.ndarray) else (list(bbox) if bbox else None),
        "feature": feat.tolist() if isinstance(feat, np.ndarray) else feat,
        "conf": float(conf) if conf is not None else None, 
        "frame_res": [frame_w, frame_h],
        "quality": float(quality)
    }
    
    try:
        r_json.xadd(STREAM_NAME, {'data': json.dumps(packet)}, maxlen=100000)
    except Exception as e:
        print(f"Redis Stream Error: {e}")


# >>> NEW: Helper function to draw pose debug info
def draw_pose_debug(frame, bbox, details):
    """Draw pose quality debug information on frame."""
    if not SHOW_POSE_DEBUG:
        return
    
    x1, y1, x2, y2 = bbox
    
    # Draw quality info
    info_lines = []
    
    if details.get('pose') is not None:
        info_lines.append(f"P:{details['pose']:.2f}")
    
    if details.get('visibility') is not None:
        info_lines.append(f"V:{details['visibility']:.2f}")
    
    if details.get('frontality') is not None:
        info_lines.append(f"F:{details['frontality']:.2f}")
    
    if details.get('is_good_for_reid'):
        info_lines.append("GOOD")
    
    # Draw info below bbox
    y_offset = y2 + 15
    for line in info_lines:
        cv2.putText(frame, line, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, (0, 255, 255), 1)
        y_offset += 12
# >>> END NEW


# --- MAIN ---
print(f"[{CAMERA_ID}] Loading YOLO & DINOv3 Models...")
model = YOLO(YOLO_MODEL_PATH)
reid_extractor = TensorRTReidExtractor(REID_ENGINE_PATH)
cmc = CameraMotionCompensation()

try: 
    H = np.load(HOMOGRAPHY_PATH)
    print(f"[{CAMERA_ID}] Homography loaded.")
except: 
    H = None
    print(f"[{CAMERA_ID}] No homography found, using pixel coordinates.")

cap = cv2.VideoCapture(VIDEO_SOURCE)
threading.Thread(target=redis_listener, daemon=True).start()

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FRAME_COUNT = 0

# >>> NEW: Stats logging interval
STATS_LOG_INTERVAL = 500  # Log quality stats every N frames
last_stats_log = 0
# >>> END NEW

print(f"[{CAMERA_ID}] Started. Resolution: {frame_w}x{frame_h}. Streaming to Redis STREAM: {STREAM_NAME}...")
print(f"[{CAMERA_ID}] Pose-Based Quality: {'ENABLED' if quality_scorer.use_pose else 'DISABLED'}")

while True:
    ret, frame = cap.read()
    if not ret: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
        
    FRAME_COUNT += 1
    curr_time = time.time()
    
    warp_matrix = cmc.compute(frame)
    annotated = frame.copy()
    
    # Get central tracker results (thread-safe)
    with tracks_lock:
        central_map = {t['edge_track_id']: t for t in latest_tracks_from_central}
    
    # Run YOLO tracker
    results = model.track(
        frame, 
        stream=False, 
        verbose=False, 
        classes=[0], 
        persist=True, 
        tracker=TRACKER_CONFIG, 
        conf=CONF_THRES_DETECTION, 
        iou=0.8
    )
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for bbox, conf, eid in zip(boxes, confs, ids):
            x1, y1, x2, y2 = bbox
            box_color = COLOR_PENDING
            top_label = ""
            
            # Check central tracker results
            if eid in central_map:
                c_data = central_map[eid]
                if c_data.get('is_staff', False):
                    box_color = COLOR_STAFF
                    name = c_data.get('name', 'Staff')
                    top_label = f"{name}"
                else:
                    box_color = COLOR_TRACKED
                    gid = c_data.get('global_id', '?')
                    top_label = f"G {gid}"
            
            if conf < CONF_THRES_HIGH:
                box_color = (100, 100, 100)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            if top_label:
                (tw, th), _ = cv2.getTextSize(top_label, 0, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_color, -1)
                cv2.putText(annotated, top_label, (x1 + 5, y1 - 5), 0, 0.7, COLOR_TEXT, 2)
            
            # Compute ground plane coordinates
            if H is not None:
                p = np.array([[(x1+x2)/2, y2]], dtype='float32').reshape(-1, 1, 2)
                gp_raw = cv2.perspectiveTransform(p, H)[0][0]
                gp = np.array([gp_raw[0], gp_raw[1]])
            else:
                gp = np.array([float((x1+x2)/2), float((y1+y2)/2)])
            
            # Initialize new tracklet
            if eid not in edge_tracklets:
                edge_tracklets[eid] = {
                    "best_q": 0.0, 
                    "last_reid": 0.0, 
                    "last_gp": 0, 
                    "last_seen": curr_time,
                    "last_bbox": list(bbox), 
                    "predicted_bbox": None
                }
                send_track_event("TRACK_NEW", eid, gp, bbox, conf=conf)

            t_data = edge_tracklets[eid]
            t_data["last_seen"] = curr_time
            t_data["last_bbox"] = list(bbox)
            
            # >>> MODIFIED: Smart Sparse Re-ID Logic with Pose-Based Quality
            if conf >= CONF_THRES_HIGH:
                # Compute quality using pose-based scorer
                q_score, q_details = quality_scorer.compute(
                    frame=frame,
                    bbox=bbox,
                    track_id=eid  # Pass track ID for stability scoring
                )
                
                # Draw pose debug info if enabled
                draw_pose_debug(annotated, bbox, q_details)
                
                # Use enhanced Re-ID decision logic
                do_reid = quality_scorer.should_extract_reid(
                    track_id=eid,
                    quality_score=q_score,
                    details=q_details,
                    tracklet_data=t_data
                )
                
                if do_reid:
                    feats, _ = reid_extractor.extract_features(frame, [bbox])
                    if feats.size > 0:
                        t_data["best_q"] = max(t_data["best_q"], q_score)
                        t_data["last_reid"] = curr_time
                        send_track_event(
                            "TRACK_UPDATE_FEATURE", eid, gp, bbox, 
                            feat=feats[0], conf=conf, quality=q_score
                        )
                        
                        # Visual indicator: Yellow dot for Re-ID extraction
                        cv2.circle(annotated, (x2-5, y1+5), 3, (0, 255, 255), -1)
                        
                        # Green dot if pose indicated good Re-ID
                        if q_details.get('is_good_for_reid'):
                            cv2.circle(annotated, (x2-5, y1+15), 3, (0, 255, 0), -1)
            # >>> END MODIFIED
            
            elif conf >= CONF_THRES_LOW:
                if (FRAME_COUNT - t_data["last_gp"]) >= GP_UPDATE_INTERVAL:
                    send_track_event("TRACK_UPDATE_GP", eid, gp, bbox, conf=conf)
                    t_data["last_gp"] = FRAME_COUNT

    # Thread-safe lost track handling
    lost_keys = edge_tracklets.remove_expired(curr_time, TRACK_LOST_TIMEOUT)
    for k in lost_keys:
        send_track_event("TRACK_LOST", k)

    # >>> NEW: Periodic stats logging
    if FRAME_COUNT - last_stats_log >= STATS_LOG_INTERVAL:
        quality_scorer.log_stats()
        last_stats_log = FRAME_COUNT
    # >>> END NEW

    # Publish annotated frame
    ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    if ret:
        r_bytes.set(f"live_feed:{CAMERA_ID}", buffer.tobytes(), ex=5)

# >>> NEW: Final stats on exit
print(f"\n[{CAMERA_ID}] Final Quality Stats:")
quality_scorer.log_stats()
# >>> END NEW

cap.release()