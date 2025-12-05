"""
Distributed processing services for the MCT system.

This module contains the central tracker service for Redis-based coordination,
edge camera processor for detection, and stream utilities for event handling.
"""

from services.stream_utils import EventPublisher, TrackEventBuilder

__all__ = [
    'EventPublisher',
    'TrackEventBuilder',
]
