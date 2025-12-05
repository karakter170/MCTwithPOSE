# stream_utils.py
# P1-4: Redis Streams utility for edge cameras
#
# This module provides a unified interface for publishing events
# that works with both Redis Pub/Sub and Streams.

import redis
import json
import time


class EventPublisher:
    """
    Unified event publisher that supports both Pub/Sub and Streams.
    
    Usage:
        publisher = EventPublisher(redis_client, mode='stream')
        publisher.publish_event(event_data)
    """
    
    def __init__(self, redis_client, mode='stream', stream_name='track_events', 
                 channel_name='track_event_stream', max_stream_len=100000):
        """
        Initialize event publisher.
        
        Args:
            redis_client: Redis client instance
            mode: 'stream' or 'pubsub'
            stream_name: Name of Redis Stream (if mode='stream')
            channel_name: Name of Pub/Sub channel (if mode='pubsub')
            max_stream_len: Maximum stream length before trimming
        """
        self.redis = redis_client
        self.mode = mode
        self.stream_name = stream_name
        self.channel_name = channel_name
        self.max_stream_len = max_stream_len
        
        # Statistics
        self.published_count = 0
        self.error_count = 0
        self.last_error = None
        
    def publish_event(self, event_data):
        """
        Publish an event to Redis.
        
        Args:
            event_data: Dict containing event data
            
        Returns:
            bool: True if successful
        """
        try:
            json_data = json.dumps(event_data)
            
            if self.mode == 'stream':
                # Add to Redis Stream
                self.redis.xadd(
                    self.stream_name,
                    {'data': json_data},
                    maxlen=self.max_stream_len,
                    approximate=True  # Faster, ~100 items tolerance
                )
            else:
                # Publish to Pub/Sub channel
                self.redis.publish(self.channel_name, json_data)
            
            self.published_count += 1
            return True
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            print(f"[Publisher] Error: {e}")
            return False
    
    def get_stats(self):
        """Get publishing statistics."""
        return {
            'published': self.published_count,
            'errors': self.error_count,
            'last_error': self.last_error,
            'mode': self.mode
        }


class TrackEventBuilder:
    """
    Builder class for constructing track events.
    Ensures consistent event format across all edge cameras.
    """
    
    def __init__(self, camera_id, group_id):
        self.camera_id = camera_id
        self.group_id = group_id
    
    def build_new_track(self, edge_id, gp, bbox, conf):
        """Build TRACK_NEW event."""
        return self._build_base(
            event_type="TRACK_NEW",
            edge_id=edge_id,
            gp=gp,
            bbox=bbox,
            conf=conf
        )
    
    def build_update_gp(self, edge_id, gp, bbox, conf):
        """Build TRACK_UPDATE_GP event."""
        return self._build_base(
            event_type="TRACK_UPDATE_GP",
            edge_id=edge_id,
            gp=gp,
            bbox=bbox,
            conf=conf
        )
    
    def build_update_feature(self, edge_id, gp, bbox, feature, conf, quality):
        """Build TRACK_UPDATE_FEATURE event."""
        event = self._build_base(
            event_type="TRACK_UPDATE_FEATURE",
            edge_id=edge_id,
            gp=gp,
            bbox=bbox,
            conf=conf
        )
        event['feature'] = feature.tolist() if hasattr(feature, 'tolist') else feature
        event['quality'] = float(quality)
        return event
    
    def build_lost(self, edge_id):
        """Build TRACK_LOST event."""
        return {
            "camera_id": self.camera_id,
            "group_id": self.group_id,
            "timestamp": time.time(),
            "event_type": "TRACK_LOST",
            "edge_track_id": int(edge_id),
            "gp_coord": None,
            "bbox": None,
            "feature": None,
            "conf": None,
            "quality": 0.0
        }
    
    def _build_base(self, event_type, edge_id, gp, bbox, conf):
        """Build base event structure."""
        import numpy as np
        
        return {
            "camera_id": self.camera_id,
            "group_id": self.group_id,
            "timestamp": time.time(),
            "event_type": event_type,
            "edge_track_id": int(edge_id),
            "gp_coord": gp.tolist() if isinstance(gp, np.ndarray) else gp,
            "bbox": bbox.tolist() if isinstance(bbox, np.ndarray) else (list(bbox) if bbox else None),
            "feature": None,
            "conf": float(conf) if conf is not None else None,
            "quality": 0.0
        }


# ============================================
# Example Usage
# ============================================
if __name__ == "__main__":
    # Example: How to use in edge_camera.py
    
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    # Create publisher (use 'stream' for durability, 'pubsub' for legacy)
    publisher = EventPublisher(r, mode='stream')
    
    # Create event builder
    builder = TrackEventBuilder(camera_id="cam_01", group_id="mall_1")
    
    # Publish events
    import numpy as np
    
    event = builder.build_new_track(
        edge_id=42,
        gp=np.array([100.0, 200.0]),
        bbox=[10, 20, 50, 100],
        conf=0.85
    )
    publisher.publish_event(event)
    
    event = builder.build_update_feature(
        edge_id=42,
        gp=np.array([105.0, 202.0]),
        bbox=[12, 22, 52, 102],
        feature=np.random.randn(1024),
        conf=0.9,
        quality=0.75
    )
    publisher.publish_event(event)
    
    print(f"Stats: {publisher.get_stats()}")