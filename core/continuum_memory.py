# continuum_memory.py
# ENHANCED NESTED LEARNING MODULE (MODULAR VERSION - FIXED & OPTIMIZED)
#
# Integration:
# This module imports 'LearnedGating' and 'extract_gating_context' 
# from 'learned_gating.py' instead of hardcoding them.

import numpy as np
from numpy.linalg import norm
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum

# =============================================================================
# IMPORTS FROM LOCAL MODULES
# =============================================================================
try:
    from learned_gating import LearnedGating, extract_gating_context
    LEARNED_GATING_AVAILABLE = True
except ImportError:
    print("[ContinuumV2] WARNING: 'learned_gating.py' not found. System will force Sigmoid fallback.")
    LEARNED_GATING_AVAILABLE = False


# =============================================================================
# GLOBAL CACHE FOR GATING HANDLER (Performance Optimization)
# =============================================================================
# Bu cache sayesinde model her track güncellemesinde diskten tekrar okunmaz.
# FIXED: Added thread-safe lock for singleton pattern
_SHARED_GATING_HANDLER = None
_SHARED_HANDLER_PATH = None
_GATING_HANDLER_LOCK = threading.Lock()


# =============================================================================
# CONFIGURATION
# =============================================================================

class LearningPhase(Enum):
    BOOTSTRAP = "bootstrap"
    STABILIZING = "stabilizing"
    MATURE = "mature"
    ADAPTING = "adapting"


@dataclass
class ContinuumConfig:
    """Configuration for Nested Learning system."""
    # Buffer settings
    buffer_size: int = 7
    
    # NEW: Learned Gating Settings
    use_learned_gating: bool = True
    gating_model_path: str = "models/gating_network_msmt172.pt"
    
    # Learning rates
    alpha_slow_base: float = 0.05
    alpha_slow_min: float = 0.02
    alpha_slow_max: float = 0.20
    
    # Fallback Gating thresholds (used if model missing or disabled)
    stability_thresh: float = 0.65
    sigmoid_scale: float = 10.0
    
    # Breakout mechanism
    breakout_limit: int = 30
    breakout_confirmation: int = 10
    breakout_similarity_thresh: float = 0.75
    
    # Bootstrap phase
    bootstrap_frames: int = 15
    bootstrap_alpha_multiplier: float = 2.0
    
    # Temporal decay
    temporal_decay_half_life: float = 30.0
    
    # Multi-modal settings
    max_modes: int = 3
    mode_creation_thresh: float = 0.65
    mode_merge_thresh: float = 0.90
    mode_min_weight: float = 0.15
    
    # Maturity settings
    maturity_frames: int = 100
    
    # Quality settings
    min_quality_for_update: float = 0.3
    quality_alpha_scale: float = 0.5


@dataclass
class BufferEntry:
    feature: np.ndarray
    quality: float
    timestamp: float
    
    def age(self, current_time: float) -> float:
        return current_time - self.timestamp


@dataclass 
class AppearanceMode:
    mean: np.ndarray
    variance: np.ndarray
    weight: float
    count: int
    last_update: float
    
    def __post_init__(self):
        self.mean = np.array(self.mean, dtype=np.float32)
        self.variance = np.array(self.variance, dtype=np.float32)


@dataclass
class BreakoutCandidate:
    feature: np.ndarray
    confirmation_count: int
    first_seen: float
    last_seen: float


# =============================================================================
# MAIN CLASS
# =============================================================================

class ContinuumStateV2:
    """
    Enhanced Nested Learning Module (Modular Integration).
    """
    
    def __init__(self, feature_dim: int = 1024, config: ContinuumConfig = None, data: dict = None):
        self.feature_dim = feature_dim
        self.config = config or ContinuumConfig()
        
        # Initialize Learned Gating Module (With Singleton/Cache pattern)
        self.gating_module = self._get_shared_gating_module()
        
        if data:
            self._load_from_dict(data)
        else:
            self._initialize_fresh()
            
    def _get_shared_gating_module(self):
        """
        Retrieves or creates the shared gating handler to avoid reloading
        the model from disk for every single track object.
        FIXED: Thread-safe singleton pattern with lock.
        """
        global _SHARED_GATING_HANDLER, _SHARED_HANDLER_PATH

        if not self.config.use_learned_gating or not LEARNED_GATING_AVAILABLE:
            return None

        path = self.config.gating_model_path

        # Thread-safe singleton check
        with _GATING_HANDLER_LOCK:
            # Double-check pattern: verify again inside lock
            if _SHARED_GATING_HANDLER is not None and _SHARED_HANDLER_PATH == path:
                return _SHARED_GATING_HANDLER

            # Load model (only happens once per process)
            try:
                handler = LearnedGating(
                    model_path=path,
                    fallback_threshold=self.config.stability_thresh
                )
                _SHARED_GATING_HANDLER = handler
                _SHARED_HANDLER_PATH = path
                return handler
            except Exception as e:
                print(f"[ContinuumV2] Failed to init LearnedGating: {e}")
                return None
    
    def _initialize_fresh(self):
        self.fast_buffer: List[BufferEntry] = []
        self.modes: List[AppearanceMode] = []
        self.primary_mean: Optional[np.ndarray] = None
        self.primary_variance: Optional[np.ndarray] = None
        
        self.count = 0
        self.total_quality_sum = 0.0
        self.created_at = time.time()
        self.last_update = time.time()
        
        self.divergence_counter = 0
        self.breakout_candidate: Optional[BreakoutCandidate] = None
        
        self.consistency_history: List[float] = []
        self.consistency_ema = 0.5
        self.quality_history: List[float] = []
        
        self.phase = LearningPhase.BOOTSTRAP

    def _load_from_dict(self, data: dict):
        self.fast_buffer = []
        for entry in data.get('fast_buffer', []):
            self.fast_buffer.append(BufferEntry(
                feature=np.array(entry['feature'], dtype=np.float32),
                quality=entry['quality'],
                timestamp=entry['timestamp']
            ))
        
        self.modes = []
        for mode_data in data.get('modes', []):
            self.modes.append(AppearanceMode(
                mean=np.array(mode_data['mean'], dtype=np.float32),
                variance=np.array(mode_data['variance'], dtype=np.float32),
                weight=mode_data['weight'],
                count=mode_data['count'],
                last_update=mode_data['last_update']
            ))
        
        if 'primary_mean' in data and data['primary_mean'] is not None:
            self.primary_mean = np.array(data['primary_mean'], dtype=np.float32)
            self.primary_variance = np.array(data['primary_variance'], dtype=np.float32)
            
        self.count = data.get('count', 0)
        self.created_at = data.get('created_at', time.time())
        self.last_update = data.get('last_update', time.time())
        self.divergence_counter = data.get('divergence_counter', 0)
        self.consistency_ema = data.get('consistency_ema', 0.5)
        
        # --- FIX: Ensure these lists are initialized ---
        self.consistency_history = data.get('consistency_history', [])
        self.quality_history = data.get('quality_history', [])
        
        # --- FIX: Initialize missing attributes to prevent AttributeError ---
        self.breakout_candidate = None
        self.total_quality_sum = data.get('total_quality_sum', 0.0)
        # -------------------------------------------------------------------
        
        phase_str = data.get('phase', 'bootstrap')
        try: self.phase = LearningPhase(phase_str)
        except: self.phase = LearningPhase.BOOTSTRAP

    def to_dict(self) -> dict:
        return {
            'fast_buffer': [{'feature': e.feature.tolist(), 'quality': e.quality, 'timestamp': e.timestamp} for e in self.fast_buffer],
            'modes': [{'mean': m.mean.tolist(), 'variance': m.variance.tolist(), 'weight': m.weight, 'count': m.count, 'last_update': m.last_update} for m in self.modes],
            'primary_mean': self.primary_mean.tolist() if self.primary_mean is not None else None,
            'primary_variance': self.primary_variance.tolist() if self.primary_variance is not None else None,
            'count': self.count,
            'created_at': self.created_at,
            'last_update': self.last_update,
            'divergence_counter': self.divergence_counter,
            'consistency_ema': self.consistency_ema,
            'consistency_history': self.consistency_history[-50:], # Save last 50
            'quality_history': self.quality_history[-50:], # Save last 50
            'phase': self.phase.value,
            'total_quality_sum': self.total_quality_sum
        }

    # =========================================================================
    # CORE UTILITIES
    # =========================================================================
    
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = norm(v)
        return v / (n + 1e-8) if n > 0 else v
    
    def _get_maturity_factor(self) -> float:
        return min(1.0, self.count / self.config.maturity_frames)

    def _update_phase(self):
        if self.count < self.config.bootstrap_frames:
            self.phase = LearningPhase.BOOTSTRAP
        elif self.divergence_counter > self.config.breakout_limit // 2:
            self.phase = LearningPhase.ADAPTING
        elif self._get_maturity_factor() < 0.5:
            self.phase = LearningPhase.STABILIZING
        else:
            self.phase = LearningPhase.MATURE

    # =========================================================================
    # BUFFER & MEMORY MANAGEMENT
    # =========================================================================

    def _add_to_buffer(self, feature: np.ndarray, quality: float, timestamp: float):
        entry = BufferEntry(feature.copy(), quality, timestamp)
        self.fast_buffer.append(entry)
        while len(self.fast_buffer) > self.config.buffer_size:
            self.fast_buffer.pop(0)

    def _compute_weighted_centroid(self, current_time: float) -> Tuple[np.ndarray, float]:
        if not self.fast_buffer: return None, 0.0
        
        weights = []
        features = []
        for entry in self.fast_buffer:
            age = entry.age(current_time)
            time_weight = np.exp(-age / self.config.temporal_decay_half_life)
            combined_weight = entry.quality * time_weight
            weights.append(combined_weight)
            features.append(entry.feature)
            
        weights = np.array(weights)
        total_weight = weights.sum()
        if total_weight < 1e-8: return None, 0.0
        
        centroid = np.average(np.array(features), axis=0, weights=weights/total_weight)
        return self._normalize(centroid), total_weight

    def _find_best_mode(self, feature: np.ndarray) -> Tuple[int, float]:
        if not self.modes: return -1, 0.0
        best_idx, best_sim = -1, -1.0
        for i, mode in enumerate(self.modes):
            sim = np.dot(feature, mode.mean)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx, best_sim

    def _create_mode(self, feature: np.ndarray, current_time: float, initial_weight: float = 0.3):
        new_mode = AppearanceMode(
            mean=feature.copy(),
            variance=np.ones(self.feature_dim, dtype=np.float32) * 0.05,
            weight=initial_weight, count=1, last_update=current_time
        )
        self.modes.append(new_mode)
        if len(self.modes) == 1:
            self.primary_mean = feature.copy()
            self.primary_variance = new_mode.variance.copy()

    def _update_mode(self, mode_idx: int, feature: np.ndarray, alpha: float, current_time: float):
        mode = self.modes[mode_idx]
        diff = feature - mode.mean
        mode.mean = (1 - alpha) * mode.mean + alpha * feature
        mode.mean = self._normalize(mode.mean)
        mode.variance = (1 - alpha) * mode.variance + alpha * (diff ** 2)
        
        mode.count += 1
        mode.last_update = current_time
        mode.weight = min(1.0, mode.weight + 0.02)
        
        if mode_idx == 0 or mode.weight >= self.modes[0].weight:
            self.primary_mean = mode.mean.copy()
            self.primary_variance = mode.variance.copy()

    def _decay_other_modes(self, active_idx: int):
        for i, mode in enumerate(self.modes):
            if i != active_idx: mode.weight *= 0.995
    
    def _prune_weak_modes(self):
        if len(self.modes) <= 1: return
        self.modes = [m for m in self.modes if m.weight >= self.config.mode_min_weight]

        # FIXED: Safe recovery when all modes are pruned
        if not self.modes:
            if self.primary_mean is not None:
                # Restore from primary mean
                self._create_mode(self.primary_mean, time.time(), 1.0)
            elif self.fast_buffer:
                # Fallback: recreate from fast buffer centroid
                centroid, _ = self._compute_weighted_centroid(time.time())
                if centroid is not None:
                    self._create_mode(centroid, time.time(), 1.0)
                    print("[ContinuumV2] WARNING: All modes pruned, recreated from buffer")
    
    def _normalize_mode_weights(self):
        if not self.modes: return
        total = sum(m.weight for m in self.modes)
        if total > 0:
            for mode in self.modes: mode.weight /= total

    # =========================================================================
    # BREAKOUT LOGIC
    # =========================================================================
    
    def _handle_divergence(self, fast_centroid: np.ndarray, consistency: float, current_time: float):
        if consistency >= self.config.stability_thresh:
            self.divergence_counter = max(0, self.divergence_counter - 1)
            if consistency > self.config.stability_thresh + 0.1: 
                self.breakout_candidate = None
            return False
        
        self.divergence_counter += 1
        if self.divergence_counter <= self.config.breakout_limit: 
            return False
        
        # Candidate logic
        if self.breakout_candidate is None:
            self.breakout_candidate = BreakoutCandidate(fast_centroid.copy(), 1, current_time, current_time)
            return False
        
        sim = np.dot(fast_centroid, self.breakout_candidate.feature)
        if sim > self.config.breakout_similarity_thresh:
            self.breakout_candidate.confirmation_count += 1
            # EMA Update of candidate
            self.breakout_candidate.feature = 0.7 * self.breakout_candidate.feature + 0.3 * fast_centroid
            self.breakout_candidate.feature = self._normalize(self.breakout_candidate.feature)
            
            if self.breakout_candidate.confirmation_count >= self.config.breakout_confirmation:
                return True
        else:
            self.breakout_candidate = BreakoutCandidate(fast_centroid.copy(), 1, current_time, current_time)
        return False

    def _execute_breakout(self, current_time: float):
        if not self.breakout_candidate: return
        new_feat = self.breakout_candidate.feature

        best_idx, best_sim = self._find_best_mode(new_feat)

        if best_sim > self.config.mode_creation_thresh:
             self._update_mode(best_idx, new_feat, 0.5, current_time)
        elif len(self.modes) < self.config.max_modes:
             self._create_mode(new_feat, current_time, 0.4)
        else:
             # FIXED: Normalize feature and use named arguments for clarity
             weakest = min(range(len(self.modes)), key=lambda i: self.modes[i].weight)
             self.modes[weakest] = AppearanceMode(
                 mean=self._normalize(new_feat),
                 variance=np.ones(self.feature_dim, dtype=np.float32) * 0.05,
                 weight=0.4,
                 count=1,
                 last_update=current_time
             )
        
        self.breakout_candidate = None
        self.divergence_counter = 0

    # =========================================================================
    # MAIN LEARNING FUNCTION
    # =========================================================================

    def learn(self, vector: np.ndarray, quality: float = 1.0) -> dict:
        current_time = time.time()
        feat = self._normalize(np.array(vector, dtype=np.float32))
        
        if quality < self.config.min_quality_for_update:
            return {'skipped': True, 'reason': 'quality_too_low'}
            
        # 1. INITIALIZATION
        if self.count == 0:
            self._add_to_buffer(feat, quality, current_time)
            self._create_mode(feat, current_time, 1.0)
            self.count = 1
            self.quality_history.append(quality)
            self._update_phase()
            return {'initialized': True}

        # 2. UPDATE SHORT-TERM MEMORY
        self._add_to_buffer(feat, quality, current_time)
        self.quality_history.append(quality)
        if len(self.quality_history) > 50: self.quality_history.pop(0)

        fast_centroid, _ = self._compute_weighted_centroid(current_time)
        if fast_centroid is None: return {'skipped': True}

        # Find best matching mode
        best_mode_idx, best_mode_sim = self._find_best_mode(fast_centroid)
        target_mode = self.modes[best_mode_idx] if best_mode_idx >= 0 else None
        
        # Consistency
        consistency = best_mode_sim if best_mode_idx >= 0 else 0.0
        
        # --- FIX: Ensure list exists before append ---
        if not hasattr(self, 'consistency_history') or self.consistency_history is None:
            self.consistency_history = []
            
        self.consistency_history.append(consistency)
        if len(self.consistency_history) > 50: self.consistency_history.pop(0)
        self.consistency_ema = 0.9 * self.consistency_ema + 0.1 * consistency

        # ---------------------------------------------------------
        # 3. GATING DECISION (Using imported module)
        # ---------------------------------------------------------
        modulation = 0.0
        gating_source = "sigmoid_fallback"

        if self.gating_module and target_mode and LEARNED_GATING_AVAILABLE:
            # Prepare fast buffer list for context
            buffer_list = [e.feature for e in self.fast_buffer]

            # Use imported extractor function
            try:
                context = extract_gating_context(
                    current_feature=fast_centroid,
                    slow_memory=target_mode.mean,
                    slow_variance=target_mode.variance,
                    fast_buffer=buffer_list,
                    quality=quality,
                    track_age=time.time() - self.created_at,
                    time_since_update=time.time() - self.last_update,
                    observation_count=self.count,
                    consistency_ema=self.consistency_ema,
                    divergence_counter=self.divergence_counter,
                    quality_history=self.quality_history,
                    max_age=300.0,
                    maturity_frames=self.config.maturity_frames,
                    breakout_limit=self.config.breakout_limit
                )

                # Predict using the module
                modulation = self.gating_module.compute_update_weight(context)
                gating_source = "learned_network"

            except (ValueError, RuntimeError, AttributeError, TypeError) as e:
                # FIXED: Catch specific exceptions, log periodically to avoid spam
                if self.count % 100 == 0:
                    print(f"[ContinuumV2] Gating network failed (count={self.count}): {e}, using sigmoid fallback")
                modulation = 1.0 / (1.0 + np.exp(-self.config.sigmoid_scale * (consistency - self.config.stability_thresh)))
        else:
            # Fallback if module not loaded
            modulation = 1.0 / (1.0 + np.exp(-self.config.sigmoid_scale * (consistency - self.config.stability_thresh)))
        
        # Bootstrap Override
        if self.phase == LearningPhase.BOOTSTRAP:
            modulation = max(modulation, 0.7)
            gating_source += "+bootstrap"

        # ---------------------------------------------------------
        # 4. EXECUTE UPDATES
        # ---------------------------------------------------------
        should_breakout = self._handle_divergence(fast_centroid, consistency, current_time)
        if should_breakout:
            self._execute_breakout(current_time)
            best_mode_idx, best_mode_sim = self._find_best_mode(fast_centroid) 
        
        # Adaptive Alpha
        base_alpha = self.config.alpha_slow_base
        final_alpha = base_alpha * modulation * (0.5 + 0.5 * quality)
        final_alpha = np.clip(final_alpha, self.config.alpha_slow_min, self.config.alpha_slow_max)
        
        # Apply Update
        if best_mode_idx >= 0 and best_mode_sim > self.config.mode_creation_thresh:
            self._update_mode(best_mode_idx, fast_centroid, final_alpha, current_time)
            self._decay_other_modes(best_mode_idx)
        elif len(self.modes) < self.config.max_modes:
            self._create_mode(fast_centroid, current_time, 0.3)
        
        # Maintenance
        self._normalize_mode_weights()
        self._prune_weak_modes()
        self.modes.sort(key=lambda m: m.weight, reverse=True)
        
        if self.modes:
            self.primary_mean = self.modes[0].mean.copy()
            self.primary_variance = self.modes[0].variance.copy()

        self.count += 1
        self.last_update = current_time
        self._update_phase()
        
        return {
            'phase': self.phase.value,
            'consistency': consistency,
            'modulation': modulation,
            'alpha': final_alpha,
            'gating_source': gating_source,
            'num_modes': len(self.modes)
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    def get_identity(self):
        """Get primary identity (mean, variance)."""
        if self.primary_mean is None: return None
        return (self.primary_mean.copy(), self.primary_variance.copy())

    def get_confidence(self) -> float:
        """Get confidence score."""
        if self.count == 0: return 0.0
        mat = self._get_maturity_factor()
        cons = self.consistency_ema
        qual = np.mean(self.quality_history) if self.quality_history else 0.0
        return (0.3 * mat + 0.4 * cons + 0.3 * qual)

    def get_all_modes(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Get all appearance modes.
        Returns: List of (mean, variance, weight) tuples
        """
        return [(m.mean.copy(), m.variance.copy(), m.weight) for m in self.modes]

    def match_score(self, query_vector: np.ndarray, use_uncertainty: bool = True) -> float:
        """
        Compute match score against this identity.
        Uses weighted mixture of all modes.
        """
        if not self.modes:
            return 0.0
        
        query = self._normalize(np.array(query_vector, dtype=np.float32))
        total_score = 0.0
        
        for mode in self.modes:
            sim = np.dot(query, mode.mean)
            
            if use_uncertainty:
                mean_var = np.mean(mode.variance)
                confidence = 1.0 / (1.0 + mean_var * 5.0)
                adjusted_sim = sim * confidence
            else:
                adjusted_sim = sim
            
            total_score += mode.weight * adjusted_sim
        
        return float(total_score)

    def get_statistics(self) -> dict:
        """Get detailed statistics about this identity."""
        return {
            'count': self.count,
            'age_seconds': time.time() - self.created_at,
            'maturity': self._get_maturity_factor(),
            'phase': self.phase.value,
            'num_modes': len(self.modes),
            'consistency_ema': self.consistency_ema,
            'divergence_counter': self.divergence_counter,
            'confidence': self.get_confidence(),
            'avg_quality': np.mean(self.quality_history) if self.quality_history else 0.0,
            'buffer_size': len(self.fast_buffer),
            'has_breakout_candidate': self.breakout_candidate is not None
        }


# =============================================================================
# TEST BLOCK
# =============================================================================
if __name__ == "__main__":
    print("Testing Modular Continuum Memory (Fixed)...")
    
    try:
        cfg = ContinuumConfig(use_learned_gating=True)
        cms = ContinuumStateV2(config=cfg)
        
        vec = np.random.randn(1024).astype(np.float32)
        vec /= norm(vec)
        
        print("\nInit:")
        print(cms.learn(vec, quality=0.9))
        
        print("\nTesting Helpers:")
        print(f"Modes count: {len(cms.get_all_modes())}")
        print(f"Match Score: {cms.match_score(vec):.4f}")
        
        # Test serialization/deserialization with missing fields
        print("\nTesting Legacy Load:")
        data = cms.to_dict()
        del data['consistency_history'] # Simulate old data
        
        cms_loaded = ContinuumStateV2(config=cfg, data=data)
        print(f"Loaded successfully. History len: {len(cms_loaded.consistency_history)}")
        print(cms_loaded.learn(vec, quality=0.85)) # Should not crash
        
        print(f"\nSuccess.")
        
    except ImportError:
        print("Please ensure 'learned_gating.py' is in the same directory.")