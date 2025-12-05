# staff_filter.py
# P2: NESTED LEARNING ARCHITECTURE
# Implements Continuum Memory System (CMS) for robust Identity Management.

import numpy as np
from numpy.linalg import norm
import json
import redis
import time

class ContinuumState:
    """
    Helper class representing the internal state of a Neural Learning Module.
    Manages the Fast (Synaptic) and Slow (System) consolidation loops.
    """
    def __init__(self, feature_dim=1024, data=None):
        # Hyperparameters from NL Paper (Approximate)
        self.ALPHA_FAST = 0.30   # High plasticity for short-term adaptation
        self.ALPHA_SLOW = 0.05   # Low plasticity for long-term stability
        self.STABILITY_THRESH = 0.65
        
        if data:
            self.fast = np.array(data['fast'], dtype=np.float32)
            self.slow = np.array(data['slow'], dtype=np.float32)
            self.count = data.get('count', 0)
        else:
            self.fast = np.zeros(feature_dim, dtype=np.float32)
            self.slow = np.zeros(feature_dim, dtype=np.float32)
            self.count = 0

    def _normalize(self, v):
        return v / (norm(v) + 1e-6)

    def learn(self, vector):
        """
        The 'Deep Optimizer' step.
        Updates the internal memory states based on new observation.
        """
        feat = self._normalize(np.array(vector, dtype=np.float32))
        
        if self.count == 0:
            self.fast = feat
            self.slow = feat
            self.count = 1
            return

        # 1. Update Fast Memory (High Frequency Loop)
        # "Online consolidation... occurs immediately"
        self.fast = (1 - self.ALPHA_FAST) * self.fast + (self.ALPHA_FAST * feat)
        self.fast = self._normalize(self.fast)
        
        # 2. Conditional Consolidation (The "Gated" Logic)
        # Only update Long-Term memory if the Short-Term memory is stable.
        consistency = np.dot(self.fast, self.slow)
        
        if consistency > self.STABILITY_THRESH:
            # Update Slow Memory (Low Frequency Loop)
            # "Strengthens and reorganizes the memory"
            self.slow = (1 - self.ALPHA_SLOW) * self.slow + (self.ALPHA_SLOW * self.fast)
            self.slow = self._normalize(self.slow)
            self.count += 1

    def get_identity(self):
        """Returns the robust Slow Memory for identification."""
        return self.slow if self.count > 0 else None

    def to_dict(self):
        return {
            "fast": self.fast.tolist(),
            "slow": self.slow.tolist(),
            "count": self.count
        }


class StaffFilter:
    def __init__(self, redis_client, group_id):
        self.r = redis_client
        self.group_id = group_id
        
        # Redis Keys
        self.key_data = f"staff_data:{group_id}"         # Hash: ID -> Full JSON State
        self.key_explicit = f"staff_explicit:{group_id}" # Set: Whitelist IDs
        self.key_centroid = f"staff_centroid:{group_id}" # String: Group Mean (Level 3)
        
        # Cached State
        self.centroid = None
        self.SIMILARITY_THRESHOLD = 0.60 # Slightly lower because Slow Memory is robust
        
        # Local RAM Cache
        self.cached_staff_list = []
        self.cached_explicit_ids = set()
        self.last_cache_time = 0
        self.CACHE_DURATION = 1.0 

        self._recalculate_group_centroid()

    def _recalculate_group_centroid(self):
        """
        Calculates the 'Level 3' Memory: The average of all Staff Slow Memories.
        Used for fast rejection of non-staff.
        """
        all_items = self.r.hgetall(self.key_data)
        if not all_items:
            self.centroid = None
            self.r.delete(self.key_centroid)
            return

        vectors = []
        for _, raw_json in all_items.items():
            try:
                data = json.loads(raw_json)
                # We use the SLOW memory for the centroid
                if 'memory' in data:
                    vectors.append(data['memory']['slow'])
            except:
                continue
            
        if vectors:
            mat = np.array(vectors)
            mean_vec = np.mean(mat, axis=0)
            self.centroid = mean_vec / (norm(mean_vec) + 1e-6)
            self.r.set(self.key_centroid, json.dumps(self.centroid.tolist()))

    def _update_local_cache(self):
        """Syncs local RAM with Redis (Read-Only)."""
        if time.time() - self.last_cache_time > self.CACHE_DURATION:
            all_items = self.r.hgetall(self.key_data)
            self.cached_staff_list = []
            
            for sid_bytes, raw_json in all_items.items():
                sid = sid_bytes.decode('utf-8') if isinstance(sid_bytes, bytes) else sid_bytes
                data = json.loads(raw_json)
                self.cached_staff_list.append({
                    "id": sid,
                    "name": data['name'],
                    "role": data.get('role', 'Staff'),
                    "memory": data.get('memory', None) # Contains fast/slow
                })
            
            explicit_members = self.r.smembers(self.key_explicit)
            self.cached_explicit_ids = {int(mid) for mid in explicit_members}
            self.last_cache_time = time.time()

    def add_staff_member(self, staff_id, name, role, initial_vector):
        """
        Initializes a new Neural Learning Module for this staff member.
        """
        # Create new Continuum State
        cms = ContinuumState(len(initial_vector))
        cms.learn(initial_vector) # Initial learning step
        
        payload = {
            "name": name,
            "role": role, 
            "added_at": time.time(),
            "memory": cms.to_dict() # Stores both Fast & Slow vectors
        }
        
        p = self.r.pipeline()
        p.hset(self.key_data, staff_id, json.dumps(payload))
        p.sadd(self.key_explicit, staff_id)
        p.execute()
        
        self._recalculate_group_centroid()
        self.last_cache_time = 0 
        print(f"[{self.group_id}] Added Staff (Continuum): {name} ({staff_id})")

    def update_staff_memory(self, staff_id, new_vector):
        """
        The 'Training' step. Called when we are sure this is the staff member.
        This allows the system to adapt to appearance changes (e.g. taking off a jacket).
        """
        raw_json = self.r.hget(self.key_data, staff_id)
        if not raw_json: return False
        
        data = json.loads(raw_json)
        
        # Load state, Learn, Save state
        cms = ContinuumState(data=data['memory'])
        cms.learn(new_vector)
        data['memory'] = cms.to_dict()
        
        self.r.hset(self.key_data, staff_id, json.dumps(data))
        # We don't recalc centroid every update to save perf
        return True

    def remove_staff_member(self, staff_id):
        p = self.r.pipeline()
        p.hdel(self.key_data, staff_id)
        p.srem(self.key_explicit, staff_id)
        res = p.execute()
        
        if res[0] > 0:
            self._recalculate_group_centroid()
            self.last_cache_time = 0
            return True
        return False

    def delete_all_staff(self):
        self.r.delete(self.key_data)
        self.r.delete(self.key_centroid)
        self.r.delete(self.key_explicit)
        self.centroid = None
        self.cached_staff_list = []
        return True

    def list_all_staff(self):
        """Returns simplified list for UI"""
        self._update_local_cache()
        results = []
        for item in self.cached_staff_list:
            results.append({
                "id": item['id'],
                "name": item['name'],
                "role": item['role']
            })
        return results

    def identify_staff(self, vector=None, global_id=None):
        """
        Nested Learning Identification Process:
        1. Check Explicit ID (Context Layer).
        2. Check Slow Memory Similarity (Consolidated Layer).
        """
        self._update_local_cache()

        # CHECK 1: EXPLICIT ID (Fastest)
        # If the tracker has already assigned a Global ID that is in our whitelist,
        # we trust the tracker's spatial continuity.
        if global_id is not None and int(global_id) in self.cached_explicit_ids:
            for item in self.cached_staff_list:
                if str(item['id']) == str(global_id):
                    # OPPORTUNISTIC LEARNING:
                    # If we have a vector, update the memory model!
                    if vector is not None:
                        # We launch this async or fire-and-forget to not block read path
                        # For simplicity here, we assume external caller handles update
                        # or we do it here:
                        self.update_staff_memory(str(global_id), vector)
                    return True, item['role'], item['name']

        # CHECK 2: VECTOR MATH (Slow Memory Match)
        if vector is not None:
            vec_norm = vector / (norm(vector) + 1e-6)
            
            # A. Group Centroid Rejection (Level 3)
            if self.centroid is not None:
                if np.dot(vec_norm, self.centroid) < (self.SIMILARITY_THRESHOLD - 0.15):
                    return False, None, None

            # B. Individual Slow Memory Check (Level 2)
            best_score = 0.0
            best_info = None

            for item in self.cached_staff_list:
                if not item['memory']: continue
                
                # Compare against SLOW memory (Robust Identity)
                slow_mem = np.array(item['memory']['slow'])
                score = np.dot(vec_norm, slow_mem)
                
                if score > best_score:
                    best_score = score
                    best_info = item

            if best_score > self.SIMILARITY_THRESHOLD:
                # If match found, return info
                # Note: We do NOT auto-learn here to prevent 'drifting' onto intruders.
                # Learning should happen only when ID is confirmed via spatial tracking.
                return True, best_info['role'], best_info['name']
            
        return False, None, None