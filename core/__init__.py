"""
Core tracking algorithms for Multi-Camera Tracking (MCT) system.

This module contains the main tracking engine, Hungarian matching algorithm,
and continuum memory for identity management.
"""

from core.tracker_MCT import TrackerManagerMCT
from core.hungarian_matcher import TwoStageHungarianMatcher, MatchResult
from core.continuum_memory import ContinuumStateV2, ContinuumConfig

__all__ = [
    'TrackerManagerMCT',
    'TwoStageHungarianMatcher',
    'MatchResult',
    'ContinuumStateV2',
    'ContinuumConfig',
]
