# central_tracker_service_v2.py
# P2: ENHANCED NESTED LEARNING INTEGRATED VERSION
#
# MODIFIED: Added Hungarian Matching for optimal detection-to-track assignment
#
# Uses ContinuumStateV2 with:
# - Quality-weighted learning
# - Temporal decay
# - Multi-modal appearance support
# - Improved breakout mechanism
# - Confidence scoring
#
# NEW: Hungarian Matching for batch detection processing

import redis
import json
import time
import numpy as np
import csv
import threading
import queue
import signal
import sys
import logging
from typing import Optional
from dataclasses import dataclass
import os
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

from core.tracker_MCT import TrackerManagerMCT
from topology_manager import TopologyManager
from core.continuum_memory import ContinuumStateV2, ContinuumConfig

# >>> NEW: Import Hungarian matcher
try:
    from core.hungarian_matcher import TwoStageHungarianMatcher
    HUNGARIAN_AVAILABLE = True
    print("[Central] Hungarian Matcher: ENABLED")
except ImportError:
    HUNGARIAN_AVAILABLE = False
    print("[Central] Hungarian Matcher: DISABLED (core/hungarian_matcher.py not found)")

# >>> NEW: Import Adaptive Threshold (optional)
try:
    from utils.adaptive_threshold import AdaptiveThresholdManager
    ADAPTIVE_THRESHOLD_AVAILABLE = True
    print("[Central] Adaptive Threshold: ENABLED")
except ImportError:
    ADAPTIVE_THRESHOLD_AVAILABLE = False
    print("[Central] Adaptive Threshold: DISABLED")
# >>> END NEW

# ============================================
# Service Configuration
# ============================================
@dataclass
class ServiceConfig:
    """Centralized configuration for the tracking service."""
    # CSV Logging
    public_csv: str = 'MCT_Public_Log.csv'
    secret_csv: str = 'MCT_Shadow_Log_SECRET.csv'
    csv_batch_size: int = 50

    # Redis Streams
    stream_name: str = "track_events"
    consumer_group: str = "mct_processors"
    dead_letter_stream: str = "track_events_dlq"
    batch_size: int = 100
    block_ms: int = 1000
    max_retries: int = 3

    # Timeouts (milliseconds)
    message_timeout_ms: int = 30000  # 30 seconds
    retry_cleanup_interval: int = 600  # 10 minutes

    # Cache Settings
    local_cache_ttl: float = 5.0  # seconds
    redis_cache_ttl: int = 3600  # 1 hour
    cache_cleanup_interval: float = 60.0  # 1 minute

    # Metrics
    metrics_log_interval: int = 100  # Log every N updates

    # >>> NEW: Hungarian Matching Settings
    hungarian_enabled: bool = True  # Enable/disable Hungarian matching
    hungarian_max_wait_ms: float = 50.0  # Max wait time for batch (50ms ≈ 1-2 frames)
    hungarian_max_batch_size: int = 50  # Max detections per batch
    # >>> END NEW

    def get_consumer_name(self) -> str:
        """Generate unique consumer name."""
        return f"processor_{int(time.time())}"

SERVICE_CONFIG = ServiceConfig()

# ============================================
# Nested Learning Configuration
# ============================================
CONTINUUM_CONFIG = ContinuumConfig(
    buffer_size=int(os.environ.get('CMS_BUFFER_SIZE', 7)),
    use_learned_gating=os.environ.get('CMS_USE_LEARNED_GATING', 'true').lower() == 'true',
    gating_model_path=os.environ.get('CMS_GATING_MODEL_PATH', './models/gating_network_msmt172.pt'),

    alpha_slow_base=0.05,
    alpha_slow_min=0.02,
    alpha_slow_max=0.20,
    stability_thresh=0.65,
    breakout_limit=30,
    breakout_confirmation=10,
    max_modes=3,
    temporal_decay_half_life=30.0,
    bootstrap_frames=15,
    maturity_frames=100,
    min_quality_for_update=0.3
)
# ============================================


class FrameBatcher:
    """
    Collects detection events and batches them by camera + frame.
    
    This enables Hungarian matching by grouping all detections
    from the same frame together before processing.
    """
    
    def __init__(self, max_wait_time: float = 0.05, max_batch_size: int = 50):
        """
        Args:
            max_wait_time: Maximum seconds to wait for more detections (50ms default)
            max_batch_size: Maximum detections before forcing batch processing
        """
        self.max_wait_time = max_wait_time
        self.max_batch_size = max_batch_size
        
        # Batch storage: {(camera_id, frame_id): {'events': [...], 'first_time': ...}}
        self.batches: Dict[tuple, Dict] = {}
        self.lock = Lock()
    
    def add_event(self, event: Dict) -> List[Tuple[tuple, List[Dict]]]:
        """
        Add an event to the appropriate batch.
        
        Returns:
            List of (batch_key, events) tuples for ready batches
        """
        camera_id = event.get('camera_id')
        # Use frame_id if available, otherwise use timestamp bucketed to 50ms windows
        frame_id = event.get('frame_id')
        if frame_id is None:
            # Bucket timestamp to 50ms windows for grouping
            ts = event.get('timestamp', time.time())
            frame_id = int(ts * 20)  # 20 buckets per second = 50ms windows
        
        batch_key = (camera_id, frame_id)
        
        with self.lock:
            current_time = time.time()
            
            # Check if any existing batches are ready (timed out)
            ready_batches = []
            for key, batch in list(self.batches.items()):
                age = current_time - batch['first_time']
                if age > self.max_wait_time or len(batch['events']) >= self.max_batch_size:
                    ready_batches.append((key, batch['events']))
                    del self.batches[key]
            
            # Add current event to its batch
            if batch_key not in self.batches:
                self.batches[batch_key] = {
                    'events': [],
                    'first_time': current_time
                }
            
            self.batches[batch_key]['events'].append(event)
            
            # Check if current batch is ready (hit max size)
            current_batch = self.batches[batch_key]
            if len(current_batch['events']) >= self.max_batch_size:
                ready_batches.append((batch_key, current_batch['events']))
                del self.batches[batch_key]
        
        return ready_batches
    
    def flush_all(self) -> List[Tuple[tuple, List[Dict]]]:
        """Force flush all pending batches."""
        with self.lock:
            ready = [(key, batch['events']) for key, batch in self.batches.items()]
            self.batches.clear()
            return ready
    
    def get_pending_count(self) -> int:
        """Get number of pending events across all batches."""
        with self.lock:
            return sum(len(b['events']) for b in self.batches.values())


class BatchCSVWriter:
    """Thread-safe CSV writer with batching for performance."""

    def __init__(self, filename, headers, batch_size=None):
        self.filename = filename
        self.headers = headers
        self.batch_size = batch_size or SERVICE_CONFIG.csv_batch_size
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=False)
        self.file = None
        self.writer = None
        self._running = True
        
    def start(self):
        self.file = open(self.filename, mode='w', newline='', buffering=1) 
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.headers)
        self.thread.start()

    def write(self, row):
        self.queue.put(row)

    def stop(self):
        self._running = False
        self.queue.put(None) 
        self.thread.join(timeout=5)
        if self.file:
            self.file.close()

    def _worker(self):
        batch = []
        while self._running:
            try:
                item = self.queue.get(timeout=1.0)
                if item is None:
                    break
                batch.append(item)
                if len(batch) >= self.batch_size or self.queue.empty():
                    if self.writer:
                        self.writer.writerows(batch)
                        self.file.flush()
                    batch = []
            except queue.Empty:
                if batch and self.writer:
                    self.writer.writerows(batch)
                    self.file.flush()
                    batch = []
            except (IOError, OSError) as e:
                print(f"[CSVWriter] I/O Error: {e}")
            except Exception as e:
                print(f"[CSVWriter] Unexpected error: {e}")
                import traceback
                traceback.print_exc()


class NestedLearningManager:
    """
    Manages ContinuumStateV2 instances for all tracked identities.
    Handles Redis persistence and provides statistics.

    THREAD-SAFE: All cache and stats operations are protected by locks.
    """

    def __init__(self, redis_client, config: ContinuumConfig = None):
        self.redis = redis_client
        self.config = config or CONTINUUM_CONFIG

        self.local_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = SERVICE_CONFIG.local_cache_ttl
        self._cache_lock = threading.RLock()

        self.stats = {
            'updates': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'breakouts': 0,
            'new_modes': 0,
            'errors': 0
        }
        self._stats_lock = threading.Lock()
    
    def _get_key(self, global_id: int) -> str:
        return f"mct:continuum:v2:{global_id}"
    
    def _load_state(self, global_id: int) -> ContinuumStateV2:
        current_time = time.time()

        with self._cache_lock:
            if global_id in self.local_cache:
                cache_age = current_time - self.cache_timestamps.get(global_id, 0)
                if cache_age < self.cache_ttl:
                    with self._stats_lock:
                        self.stats['cache_hits'] += 1
                    return self.local_cache[global_id]

            with self._stats_lock:
                self.stats['cache_misses'] += 1

            key = self._get_key(global_id)
            raw_data = self.redis.get(key)

            if raw_data:
                try:
                    data = json.loads(raw_data)
                    cms = ContinuumStateV2(data=data, config=self.config)
                except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                    print(f"[NestedLearning] Error loading state for {global_id}: {e}")
                    cms = ContinuumStateV2(config=self.config)
                    with self._stats_lock:
                        self.stats['errors'] += 1
            else:
                cms = ContinuumStateV2(config=self.config)

            self.local_cache[global_id] = cms
            self.cache_timestamps[global_id] = current_time

            return cms
    
    def _save_state(self, global_id: int, cms: ContinuumStateV2):
        key = self._get_key(global_id)
        data = cms.to_dict()
        self.redis.set(key, json.dumps(data), ex=SERVICE_CONFIG.redis_cache_ttl)

        with self._cache_lock:
            self.local_cache[global_id] = cms
            self.cache_timestamps[global_id] = time.time()
    
    def update(self, global_id: int, feature: np.ndarray, quality: float = 1.0) -> dict:
        quality = max(0.0, min(1.0, quality))
        cms = self._load_state(global_id)

        modes_before = len(cms.modes)
        divergence_before = cms.divergence_counter

        learn_result = cms.learn(feature, quality=quality)

        with self._stats_lock:
            self.stats['updates'] += 1
            if len(cms.modes) > modes_before:
                self.stats['new_modes'] += 1
            if divergence_before > cms.config.breakout_limit and cms.divergence_counter == 0:
                self.stats['breakouts'] += 1

        self._save_state(global_id, cms)
        identity = cms.get_identity()

        return {
            'learn_result': learn_result,
            'identity': identity,
            'all_modes': cms.get_all_modes(),
            'confidence': cms.get_confidence(),
            'statistics': cms.get_statistics()
        }
    
    def get_identity(self, global_id: int) -> Optional[tuple]:
        cms = self._load_state(global_id)
        return cms.get_identity()
    
    def get_match_score(self, global_id: int, query_vector: np.ndarray) -> float:
        cms = self._load_state(global_id)
        return cms.match_score(query_vector)
    
    def get_confidence(self, global_id: int) -> float:
        cms = self._load_state(global_id)
        return cms.get_confidence()
    
    def get_statistics(self, global_id: int) -> dict:
        cms = self._load_state(global_id)
        return cms.get_statistics()
    
    def get_manager_stats(self) -> dict:
        with self._stats_lock:
            stats_copy = self.stats.copy()
        with self._cache_lock:
            stats_copy['cached_identities'] = len(self.local_cache)
        return stats_copy

    def cleanup_cache(self, max_age: float = None):
        if max_age is None:
            max_age = SERVICE_CONFIG.cache_cleanup_interval

        current_time = time.time()

        with self._cache_lock:
            expired = [
                gid for gid, ts in self.cache_timestamps.items()
                if current_time - ts > max_age
            ]
            for gid in expired:
                self.local_cache.pop(gid, None)
                self.cache_timestamps.pop(gid, None)

        if expired:
            print(f"[NestedLearning] Cleaned up {len(expired)} expired cache entries")


# --- SETUP ---
r_json = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
r_bytes = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

topo_manager = TopologyManager(r_bytes)

print("[Central] Tracker Manager Starting (DINOv2 + Enhanced Nested Learning)...")
tracker = TrackerManagerMCT(
    dt=1.0, 
    Q_cov=np.eye(4)*0.5, 
    R_cov=np.eye(2)*50, 
    feature_dim=1024,
    redis_client=r_bytes, 
    topology_manager=topo_manager
)

# Initialize Enhanced Nested Learning Manager
nested_learning = NestedLearningManager(r_json, CONTINUUM_CONFIG)
print(f"[Central] Nested Learning Config: {CONTINUUM_CONFIG}")

# >>> NEW: Initialize Hungarian Matcher and Frame Batcher
frame_batcher = FrameBatcher(
    max_wait_time=SERVICE_CONFIG.hungarian_max_wait_ms / 1000.0,
    max_batch_size=SERVICE_CONFIG.hungarian_max_batch_size
)

hungarian_matcher = None
if HUNGARIAN_AVAILABLE and SERVICE_CONFIG.hungarian_enabled:
    hungarian_matcher = TwoStageHungarianMatcher(
        match_threshold=tracker.reid_threshold,
        use_appearance=True,
        use_motion=True,
        use_iou=True,
        appearance_weight=0.6,
        motion_weight=0.2,
        iou_weight=0.2,
        high_conf_threshold=0.7
    )
    print(f"[Central] Hungarian Matcher initialized (threshold={tracker.reid_threshold})")

# >>> NEW: Initialize Adaptive Threshold Manager (optional)
threshold_manager = None
if ADAPTIVE_THRESHOLD_AVAILABLE:
    threshold_manager = AdaptiveThresholdManager(
        base_threshold=tracker.reid_threshold,
        min_threshold=0.35,
        max_threshold=0.75,
        adaptation_rate=0.1
    )
    print("[Central] Adaptive Threshold Manager initialized")
# >>> END NEW

public_log = BatchCSVWriter(
    SERVICE_CONFIG.public_csv,
    ['Time', 'Group', 'Cam', 'GID', 'Event', 'X', 'Y']
)
secret_log = BatchCSVWriter(
    SERVICE_CONFIG.secret_csv,
    ['Time', 'Group', 'Cam', 'GID', 'Event', 'X', 'Y', 'Role', 'Name']
)
public_log.start()
secret_log.start()

# >>> NEW: Statistics for Hungarian matching
hungarian_stats = {
    'batches_processed': 0,
    'detections_matched': 0,
    'detections_new': 0,
    'total_detections': 0
}
hungarian_stats_lock = Lock()
# >>> END NEW


# ============================================
# Stream Setup Functions
# ============================================
def setup_stream():
    """Initialize Redis Stream consumer group."""
    try:
        r_json.xgroup_create(
            name=SERVICE_CONFIG.stream_name,
            groupname=SERVICE_CONFIG.consumer_group,
            id='0',
            mkstream=True
        )
        print(f"[Stream] Created consumer group '{SERVICE_CONFIG.consumer_group}'")
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            print(f"[Stream] Consumer group '{SERVICE_CONFIG.consumer_group}' already exists")
        else:
            raise

def migrate_from_pubsub():
    """Bridge from legacy pub/sub to Redis Streams."""
    def bridge_worker():
        pubsub = r_json.pubsub()
        pubsub.subscribe("track_event_stream")
        print("[Migration] Bridge running: pub/sub -> stream")
        for msg in pubsub.listen():
            if msg['type'] == 'message':
                try:
                    r_json.xadd(
                        SERVICE_CONFIG.stream_name,
                        {'data': msg['data']},
                        maxlen=100000
                    )
                except (redis.RedisError, redis.ConnectionError) as e:
                    print(f"[Migration] Bridge error: {e}")
    thread = threading.Thread(target=bridge_worker, daemon=True)
    thread.start()
    return thread


# >>> NEW: Batch processing function for Hungarian matching
def process_detection_batch(camera_id: str, events: List[Dict], current_time: float):
    """
    Process a batch of detections using Hungarian matching.
    
    This is the KEY function that enables optimal assignment.
    All detections from a frame are matched together.
    """
    global hungarian_stats
    
    if not events:
        return
    
    group_id = events[0].get('group_id', 'default')
    
    # Convert events to detection format
    detections = []
    for evt in events:
        feat = np.array(evt['feature']) if evt.get('feature') else None
        if feat is None:
            continue
            
        det = {
            'feature': feat,
            'bbox': evt.get('bbox', [0, 0, 0, 0]),
            'gp_coord': np.array(evt['gp_coord']) if evt.get('gp_coord') else np.array([0, 0]),
            'confidence': evt.get('conf', 0.5),
            'quality': max(0.0, min(1.0, evt.get('quality', 0.5))),
            'edge_track_id': evt.get('edge_track_id'),
            'timestamp': evt.get('timestamp', current_time),
            'original_event': evt  # Keep reference for logging
        }
        detections.append(det)
    
    if not detections:
        return
    
    # Get tracks for this group
    group_tracks = {
        gid: track for gid, track in tracker.global_tracks.items()
        if track.group_id == group_id
    }
    
    # >>> HUNGARIAN MATCHING <<<
    if hungarian_matcher is not None and len(detections) > 0:
        # Get adaptive threshold if available
        if threshold_manager is not None:
            adaptive_threshold = threshold_manager.get_threshold(
                camera_id, detections, group_tracks, current_time
            )
            hungarian_matcher.match_threshold = adaptive_threshold
        
        # Run Hungarian matching
        result = hungarian_matcher.match(
            detections=detections,
            tracks=group_tracks,
            camera_id=camera_id,
            current_time=current_time
        )
        
        matches = result.matches
        unmatched_det_indices = result.unmatched_detections
        
        # Update statistics
        with hungarian_stats_lock:
            hungarian_stats['batches_processed'] += 1
            hungarian_stats['total_detections'] += len(detections)
            hungarian_stats['detections_matched'] += len(matches)
            hungarian_stats['detections_new'] += len(unmatched_det_indices)
    else:
        # Fallback: treat all as unmatched (original behavior)
        matches = []
        unmatched_det_indices = list(range(len(detections)))
    
    # Process matched detections
    for det_idx, global_id in matches:
        det = detections[det_idx]
        evt = det['original_event']
        
        # Update track with detection
        track = tracker.update_edge_track_feature(
            cam_id=camera_id,
            group_id=group_id,
            edge_id=det['edge_track_id'],
            gp=det['gp_coord'],
            conf=det['confidence'],
            bbox=det['bbox'],
            feature=det['feature'],
            quality_score=det['quality']
        )
        
        # Update nested learning
        if track and det['feature'] is not None:
            _update_nested_learning(track, det['feature'], det['quality'])
        
        # Logging
        _log_track_update(track, evt, camera_id, group_id)
        
        # Report to threshold manager with proper index handling
        if threshold_manager is not None:
            match_score = 0.5  # Default score
            if hasattr(result, 'cost_matrix') and result.cost_matrix.size > 0 and result.valid_track_ids:
                try:
                    # Use valid_track_ids for correct indexing (not group_tracks.keys())
                    if global_id in result.valid_track_ids:
                        track_idx = result.valid_track_ids.index(global_id)
                        if det_idx < result.cost_matrix.shape[0] and track_idx < result.cost_matrix.shape[1]:
                            match_score = result.cost_matrix[det_idx, track_idx]
                except (ValueError, IndexError) as e:
                    logger.warning(f"Cost matrix index error: {e}")

            threshold_manager.report_match(
                camera_id=camera_id,
                match_score=match_score,
                is_cross_camera=(track.last_cam_id != camera_id if track else False)
            )
    
    # Process unmatched detections (potential new tracks)
    for det_idx in unmatched_det_indices:
        det = detections[det_idx]
        evt = det['original_event']
        
        # This will create pending track or assign new ID
        track = tracker.update_edge_track_feature(
            cam_id=camera_id,
            group_id=group_id,
            edge_id=det['edge_track_id'],
            gp=det['gp_coord'],
            conf=det['confidence'],
            bbox=det['bbox'],
            feature=det['feature'],
            quality_score=det['quality']
        )
        
        # Update nested learning for new tracks
        if track and det['feature'] is not None:
            _update_nested_learning(track, det['feature'], det['quality'])
        
        # Logging
        _log_track_update(track, evt, camera_id, group_id)
        
        # Report new ID to threshold manager
        if threshold_manager is not None:
            threshold_manager.report_new_id(camera_id, det['quality'])
    
    # Publish visualization
    viz = tracker.get_viz_data_for_camera(camera_id)
    if viz:
        r_json.publish(f"results_viz_stream:{camera_id}", json.dumps(viz))


def _update_nested_learning(track, feature: np.ndarray, quality: float):
    """Helper to update nested learning for a track."""
    try:
        gid = track.global_id
        result = nested_learning.update(gid, feature, quality=quality)
        
        identity = result['identity']
        if identity is not None:
            robust_mean, robust_var = identity
            track.robust_id = robust_mean
            track.robust_var = robust_var
        
        # Periodic stats logging
        stats = nested_learning.get_manager_stats()
        if stats['updates'] % SERVICE_CONFIG.metrics_log_interval == 0:
            print(f"[NestedLearning] Stats: {stats}")
            
    except (ValueError, RuntimeError, AttributeError, TypeError) as e:
        print(f"[NestedLearning] Error updating track: {e}")


def _log_track_update(track, event_data: Dict, camera_id: str, group_id: str):
    """Helper to log track updates."""
    if track is None:
        return
    
    gp = event_data.get('gp_coord')
    if gp is None:
        return
    
    gx, gy = track.kf.smooth_pos
    gid = track.global_id
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event_data.get('timestamp', time.time())))
    evt = event_data.get('event_type', 'TRACK_UPDATE_FEATURE')
    
    if track.is_staff:
        secret_log.write([ts, group_id, camera_id, gid, evt, gx, gy, track.shadow_role, "Unknown"])
    else:
        public_log.write([ts, group_id, camera_id, gid, evt, gx, gy])
# >>> END NEW


def handle_event(data, message_id=None):
    """
    Process a single track event.
    
    MODIFIED: TRACK_UPDATE_FEATURE events are now routed through
    the batch processor for Hungarian matching.
    """
    try:
        cam = data['camera_id']
        group = data.get('group_id', 'default')
        evt = data['event_type']
        eid = data['edge_track_id']
        gp = np.array(data['gp_coord']) if data['gp_coord'] else None
        feat = np.array(data['feature']) if data.get('feature') else None

        quality = data.get('quality', 1.0)
        quality = max(0.0, min(1.0, quality))
        
        track = None
        
        if evt == "TRACK_NEW": 
            tracker.register_new_edge_track(cam, group, eid, gp, data['conf'], data['bbox'])
            
        elif evt == "TRACK_UPDATE_GP": 
            track = tracker.update_edge_track_position(cam, group, eid, gp, data['conf'], data['bbox'])
            
        elif evt == "TRACK_UPDATE_FEATURE":
            # >>> MODIFIED: Route through batcher for Hungarian matching
            if hungarian_matcher is not None and SERVICE_CONFIG.hungarian_enabled:
                # Add to batch - will be processed when batch is ready
                ready_batches = frame_batcher.add_event(data)
                
                # Process any ready batches
                current_time = time.time()
                for (batch_cam_id, batch_frame_id), batch_events in ready_batches:
                    process_detection_batch(batch_cam_id, batch_events, current_time)
                
                # Return True - event handled (batched for later processing)
                return True
            else:
                # >>> FALLBACK: Original behavior if Hungarian not available
                track = tracker.update_edge_track_feature(
                    cam, group, eid, gp, data['conf'], 
                    data['bbox'], feat, quality
                )
                
                if track and feat is not None:
                    _update_nested_learning(track, feat, quality)
            # >>> END MODIFIED

        elif evt == "TRACK_LOST":
            tracker.lost_edge_track(cam, eid)

        # Logging & Visualization (for non-batched events)
        if gp is not None and track:
            gx, gy = track.kf.smooth_pos
            gid = track.global_id
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['timestamp']))

            if track.is_staff:
                secret_log.write([ts, group, cam, gid, evt, gx, gy, track.shadow_role, "Unknown"])
            else:
                public_log.write([ts, group, cam, gid, evt, gx, gy])

        # Publish visualization update
        viz = tracker.get_viz_data_for_camera(cam)
        if viz:
            r_json.publish(f"results_viz_stream:{cam}", json.dumps(viz))

        return True

    except KeyError as e:
        print(f"[Handler] Missing required field in message {message_id}: {e}")
        return False
    except (ValueError, TypeError) as e:
        print(f"[Handler] Invalid data in message {message_id}: {e}")
        return False
    except Exception as e:
        print(f"[Handler] Unexpected error processing {message_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================
# Dead Letter Queue Processor
# ============================================
def process_dlq():
    """Process dead letter queue messages (monitoring thread)."""
    print("[DLQ] Processor starting...")
    dlq_count = 0

    while not shutdown_flag.is_set():
        try:
            messages = r_json.xread(
                {SERVICE_CONFIG.dead_letter_stream: '0'},
                count=10,
                block=5000
            )

            if messages:
                for stream_name, msgs in messages:
                    for msg_id, data in msgs:
                        dlq_count += 1
                        print(f"[DLQ] Failed message #{dlq_count}: {data}")
        except redis.ConnectionError:
            time.sleep(5)
        except Exception as e:
            print(f"[DLQ] Error: {e}")

        time.sleep(60)


# ============================================
# Stream Consumer Loop
# ============================================
def process_stream():
    """Main processing loop using Redis Streams."""
    consumer_name = SERVICE_CONFIG.get_consumer_name()
    print(f"[Stream] Consumer '{consumer_name}' starting...")

    retry_counts = {}
    retry_timestamps = {}
    last_cache_cleanup = time.time()
    last_retry_cleanup = time.time()
    last_batch_flush = time.time()  # >>> NEW: Track last batch flush
    last_stats_log = time.time()     # >>> NEW: Track last stats log
    
    while True:
        try:
            current_time = time.time()

            # Periodic cache cleanup
            if current_time - last_cache_cleanup > SERVICE_CONFIG.cache_cleanup_interval:
                nested_learning.cleanup_cache()
                last_cache_cleanup = current_time

            # Periodic retry_counts cleanup
            if current_time - last_retry_cleanup > SERVICE_CONFIG.retry_cleanup_interval:
                old_retries = {
                    k: v for k, v in retry_counts.items()
                    if current_time - retry_timestamps.get(k, current_time) < 3600
                }
                removed = len(retry_counts) - len(old_retries)
                retry_counts = old_retries
                retry_timestamps = {k: v for k, v in retry_timestamps.items() if k in old_retries}
                if removed > 0:
                    print(f"[Stream] Cleaned up {removed} old retry entries")
                last_retry_cleanup = current_time

            # >>> NEW: Periodic batch flush (ensure batches don't wait forever)
            if current_time - last_batch_flush > 0.1:  # Every 100ms
                ready_batches = frame_batcher.flush_all()
                for (batch_cam_id, batch_frame_id), batch_events in ready_batches:
                    process_detection_batch(batch_cam_id, batch_events, current_time)
                last_batch_flush = current_time
            
            # >>> NEW: Periodic Hungarian stats logging
            if current_time - last_stats_log > 60.0:  # Every minute
                with hungarian_stats_lock:
                    if hungarian_stats['batches_processed'] > 0:
                        match_rate = hungarian_stats['detections_matched'] / max(1, hungarian_stats['total_detections'])
                        print(f"[Hungarian] Stats: batches={hungarian_stats['batches_processed']}, "
                              f"matched={hungarian_stats['detections_matched']}, "
                              f"new={hungarian_stats['detections_new']}, "
                              f"match_rate={match_rate:.1%}")
                last_stats_log = current_time
            # >>> END NEW

            # Read messages
            messages = r_json.xreadgroup(
                groupname=SERVICE_CONFIG.consumer_group,
                consumername=consumer_name,
                streams={SERVICE_CONFIG.stream_name: '>'},
                count=SERVICE_CONFIG.batch_size,
                block=SERVICE_CONFIG.block_ms
            )
            
            if not messages:
                # >>> NEW: Flush pending batches on idle
                ready_batches = frame_batcher.flush_all()
                for (batch_cam_id, batch_frame_id), batch_events in ready_batches:
                    process_detection_batch(batch_cam_id, batch_events, current_time)
                # >>> END NEW
                
                # Handle pending/failed messages
                pending = r_json.xpending_range(
                    SERVICE_CONFIG.stream_name,
                    SERVICE_CONFIG.consumer_group,
                    min='-', max='+', count=10
                )
                for p in pending:
                    msg_id = p['message_id']
                    if p['time_since_delivered'] > SERVICE_CONFIG.message_timeout_ms:
                        claimed = r_json.xclaim(
                            SERVICE_CONFIG.stream_name,
                            SERVICE_CONFIG.consumer_group,
                            consumer_name,
                            min_idle_time=SERVICE_CONFIG.message_timeout_ms,
                            message_ids=[msg_id]
                        )
                        for cid, cdata in claimed:
                            retry_counts[cid] = retry_counts.get(cid, 0) + 1
                            retry_timestamps[cid] = current_time

                            if retry_counts[cid] > SERVICE_CONFIG.max_retries:
                                r_json.xadd(SERVICE_CONFIG.dead_letter_stream, {
                                    'original_id': cid,
                                    'data': cdata.get('data', '{}'),
                                    'error': 'max_retries',
                                    'retry_count': retry_counts[cid]
                                })
                                r_json.xack(SERVICE_CONFIG.stream_name, SERVICE_CONFIG.consumer_group, cid)
                                retry_counts.pop(cid, None)
                                retry_timestamps.pop(cid, None)
                            else:
                                try:
                                    data = json.loads(cdata.get('data', '{}'))
                                    if handle_event(data, cid):
                                        r_json.xack(SERVICE_CONFIG.stream_name, SERVICE_CONFIG.consumer_group, cid)
                                        retry_counts.pop(cid, None)
                                        retry_timestamps.pop(cid, None)
                                except json.JSONDecodeError:
                                    r_json.xack(SERVICE_CONFIG.stream_name, SERVICE_CONFIG.consumer_group, cid)
                                    retry_counts.pop(cid, None)
                                    retry_timestamps.pop(cid, None)
                continue
            
            # Process new messages
            for stream_name, stream_messages in messages:
                for message_id, message_data in stream_messages:
                    try:
                        data_str = message_data.get('data', '{}')
                        data = json.loads(data_str)

                        if handle_event(data, message_id):
                            r_json.xack(SERVICE_CONFIG.stream_name, SERVICE_CONFIG.consumer_group, message_id)
                    except json.JSONDecodeError as e:
                        print(f"[Stream] Invalid JSON in message {message_id}: {e}")
                        r_json.xack(SERVICE_CONFIG.stream_name, SERVICE_CONFIG.consumer_group, message_id)

        except redis.ConnectionError as e:
            print(f"[Stream] Redis connection error: {e}")
            time.sleep(5)
        except redis.RedisError as e:
            print(f"[Stream] Redis error: {e}")
            time.sleep(2)
        except KeyboardInterrupt:
            print("[Stream] Shutdown requested")
            break
        except Exception as e:
            print(f"[Stream] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

def process_pubsub_legacy():
    """Legacy pub/sub mode (for backward compatibility)."""
    print("[Legacy] Using Pub/Sub mode")
    pubsub = r_json.pubsub()
    pubsub.subscribe("track_event_stream")

    for msg in pubsub.listen():
        if msg['type'] == 'message':
            try:
                data = json.loads(msg['data'])
                handle_event(data)
            except json.JSONDecodeError as e:
                print(f"[Legacy] Invalid JSON: {e}")
            except Exception as e:
                print(f"[Legacy] Error: {e}")

# ============================================
# Graceful Shutdown
# ============================================
shutdown_flag = threading.Event()

def signal_handler(signum, frame):
    print("\n[Central] Shutdown signal received...")
    shutdown_flag.set()
    
    # >>> NEW: Flush remaining batches
    print("[Central] Flushing remaining batches...")
    ready_batches = frame_batcher.flush_all()
    for (batch_cam_id, batch_frame_id), batch_events in ready_batches:
        process_detection_batch(batch_cam_id, batch_events, time.time())
    # >>> END NEW
    
    # Print final statistics
    print(f"[NestedLearning] Final Stats: {nested_learning.get_manager_stats()}")
    
    # >>> NEW: Print Hungarian stats
    with hungarian_stats_lock:
        if hungarian_stats['batches_processed'] > 0:
            print(f"[Hungarian] Final Stats: {hungarian_stats}")
    # >>> END NEW
    
    public_log.stop()
    secret_log.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MCT Central Tracker Service (Enhanced & Optimized)')
    parser.add_argument('--mode', choices=['stream', 'pubsub', 'bridge'],
                       default='stream', help='Processing mode')
    parser.add_argument('--enable-dlq', action='store_true',
                       help='Enable Dead Letter Queue processor')
    # >>> NEW: Add Hungarian matching arguments
    parser.add_argument('--disable-hungarian', action='store_true',
                       help='Disable Hungarian matching (use greedy)')
    parser.add_argument('--hungarian-wait-ms', type=float, default=50.0,
                       help='Max wait time for batch collection (ms)')
    # >>> END NEW
    args = parser.parse_args()

    # >>> NEW: Apply command line arguments
    if args.disable_hungarian:
        SERVICE_CONFIG.hungarian_enabled = False
        print("[Central] Hungarian matching DISABLED via command line")
    
    SERVICE_CONFIG.hungarian_max_wait_ms = args.hungarian_wait_ms
    # >>> END NEW

    print("=" * 60)
    print("MCT Central Tracker Service - ENHANCED & OPTIMIZED")
    print(f"Mode: {args.mode}")
    print(f"Nested Learning: ContinuumStateV2 (Multi-Modal)")
    print(f"Hungarian Matching: {'ENABLED' if SERVICE_CONFIG.hungarian_enabled and HUNGARIAN_AVAILABLE else 'DISABLED'}")
    print(f"Adaptive Threshold: {'ENABLED' if ADAPTIVE_THRESHOLD_AVAILABLE else 'DISABLED'}")
    print(f"Thread Safety: ENABLED")
    print(f"Memory Leak Prevention: ENABLED")
    print("=" * 60)

    # Start DLQ processor if enabled
    dlq_thread = None
    if args.enable_dlq and args.mode in ['stream', 'bridge']:
        dlq_thread = threading.Thread(target=process_dlq, daemon=True, name="DLQ-Processor")
        dlq_thread.start()
        print("[DLQ] Processor thread started")

    if args.mode == 'stream':
        setup_stream()
        print("[Central] Service Running with Redis Streams...")
        process_stream()

    elif args.mode == 'pubsub':
        print("[Central] Service Running with Pub/Sub (legacy)...")
        process_pubsub_legacy()

    elif args.mode == 'bridge':
        setup_stream()
        migrate_from_pubsub()
        print("[Central] Bridge mode: forwarding pub/sub to stream...")
        process_stream()