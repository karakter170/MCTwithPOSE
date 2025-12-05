"""
MCT Architecture Integration Test
==================================

Tests the complete flow from detection batching to global tracking,
including:
1. TrackerManagerMCT initialization
2. Hungarian Matching with GCN Integration (Mocked)
3. Adaptive Thresholding
4. Continuum Memory Updates
5. Cross-Camera Re-ID logic

Usage:
    python test_architecture_full.py
"""

import unittest
import numpy as np
import time
import json
from unittest.mock import MagicMock, patch
from typing import List, Dict

# Import project modules (assumes proper PYTHONPATH setup)
try:
    from core.tracker_MCT import TrackerManagerMCT, GlobalTrack
    from core.hungarian_matcher import MatchResult
    from utils.adaptive_threshold import AdaptiveThresholdManager
    from core.continuum_memory import ContinuumStateV2
except ImportError as e:
    raise ImportError(f"Project files not found. Please set PYTHONPATH. Error: {e}")


# --- MOCK SINIFLAR ---

class MockRedis:
    """Redis servisini taklit eder (In-memory dict)."""
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value
        
    def delete(self, key):
        if key in self.store:
            del self.store[key]
            return 1
        return 0

    def exists(self, key):
        return key in self.store

    def incr(self, key):
        val = int(self.store.get(key, 0)) + 1
        self.store[key] = str(val)
        return val

    # --- HASH COMMANDS (StaffFilter için gerekli) ---
    def hgetall(self, key):
        # Redis hgetall boşsa {} döner, yoksa dict döner
        val = self.store.get(key, {})
        if not isinstance(val, dict):
            return {}
        return val

    def hset(self, key, field, value):
        if key not in self.store or not isinstance(self.store[key], dict):
            self.store[key] = {}
        self.store[key][field] = value
        return 1
    
    def hget(self, key, field):
        data = self.store.get(key, {})
        if isinstance(data, dict):
            return data.get(field)
        return None

    def hincrby(self, key, field, amount=1):
        if key not in self.store:
            self.store[key] = {}
        # Basitçe dictionary içinde tut
        current = int(self.store[key].get(field, 0))
        self.store[key][field] = str(current + amount)
        return current + amount

    def hdel(self, key, field):
        if key in self.store and isinstance(self.store[key], dict):
            if field in self.store[key]:
                del self.store[key][field]
                return 1
        return 0

    # --- SET COMMANDS (StaffFilter whitelist için gerekli) ---
    def sadd(self, key, member):
        if key not in self.store or not isinstance(self.store[key], set):
            self.store[key] = set()
        self.store[key].add(str(member))
        return 1

    def smembers(self, key):
        val = self.store.get(key, set())
        if not isinstance(val, set):
            return set()
        return val
        
    def srem(self, key, member):
        if key in self.store and isinstance(self.store[key], set):
            if str(member) in self.store[key]:
                self.store[key].remove(str(member))
                return 1
        return 0

    # --- HYPERLOGLOG & PIPELINE ---
    def pfadd(self, key, *elements):
        return 1

    def pipeline(self):
        return self

    def execute(self):
        return []


class MockGCNHandler:
    """GCN modelini taklit eder."""
    def __init__(self):
        print("[TEST] Mock GCN Initialized")

    def predict_batch(self, track, candidates, frame_w, frame_h, curr_time):
        """
        Basit bir mantıkla skor üretir:
        Eğer track.robust_id ile candidate['feature'] çok benziyorsa yüksek skor (düşük cost) ver.
        """
        scores = []
        track_feat = track.robust_id if track.robust_id is not None else track.last_known_feature
        
        for cand in candidates:
            cand_feat = cand['feature']
            # Basit Cosine Similarity
            sim = np.dot(track_feat, cand_feat) / (np.linalg.norm(track_feat) * np.linalg.norm(cand_feat) + 1e-6)
            
            # GCN 'ilişki' skoru (0-1 arası, 1=Eşleşme)
            # Simülasyon: Benzerlik > 0.8 ise GCN skoru 0.9, değilse 0.1
            gcn_score = 0.95 if sim > 0.8 else 0.05
            scores.append(gcn_score)
        
        return np.array(scores)


class TestMCTArchitecture(unittest.TestCase):
    
    def setUp(self):
        print("\n" + "="*60)
        print("SETTING UP TEST ENVIRONMENT")
        
        # 1. Mock Redis
        self.mock_redis = MockRedis()
        
        # 2. Initialize Tracker
        # Topology manager'ı None geçiyoruz, temel mantığı test ediyoruz
        self.tracker = TrackerManagerMCT(
            dt=1.0, 
            Q_cov=np.eye(4)*0.5, 
            R_cov=np.eye(2)*50, 
            feature_dim=128, # Test için küçük boyut
            redis_client=self.mock_redis, 
            topology_manager=None
        )
        
        # 3. Inject Mock GCN
        self.tracker.gcn_refiner = MockGCNHandler()
        
        # 4. Özellik vektörlerini oluştur (Sabit random seed ile)
        np.random.seed(42)
        self.feat_person_A = np.random.randn(128).astype(np.float32)
        self.feat_person_A /= np.linalg.norm(self.feat_person_A)
        
        self.feat_person_B = np.random.randn(128).astype(np.float32)
        self.feat_person_B /= np.linalg.norm(self.feat_person_B)
        
        # Person C, A'ya biraz benzesin ama farklı olsun (Hard Negative)
        self.feat_person_C = self.feat_person_A + np.random.normal(0, 0.2, 128).astype(np.float32)
        self.feat_person_C /= np.linalg.norm(self.feat_person_C)

    def test_01_initialization(self):
        """Test: Tracker ve bileşenlerin doğru başlatılması."""
        print("[TEST 1] Initialization Check")
        self.assertIsNotNone(self.tracker.hungarian_matcher, "Hungarian Matcher yüklenmedi")
        self.assertIsNotNone(self.tracker.threshold_manager, "Adaptive Threshold yüklenmedi")
        self.assertIsNotNone(self.tracker.gcn_refiner, "GCN Refiner yüklenmedi")
        print("✓ Bileşenler hazır.")

    def test_02_single_camera_tracking(self):
        """Test: Tek kamerada yeni ID oluşturma ve takip etme."""
        print("[TEST 2] Single Camera Tracking (Creation & Update)")
        
        cam_id = "cam_01"
        group_id = "group_A"
        curr_time = 1000.0
        
        # --- Adım 1: İlk Tespit (Person A) ---
        det_A1 = {
            'feature': self.feat_person_A,
            'bbox': [100, 100, 200, 300],
            'gp_coord': np.array([10.0, 10.0]),
            'confidence': 0.9,
            'quality': 0.8,
            'edge_track_id': 101,
            'original_event': {'frame_res': [1920, 1080]}
        }
        
        # Batch match çağır (Henüz kimse yok, boş dönmeli)
        matches, unmatched_dets, unmatched_tracks = self.tracker.batch_match_detections(
            cam_id, group_id, [det_A1], curr_time
        )
        
        self.assertEqual(len(matches), 0, "İlk tespitte eşleşme olmamalı")
        self.assertEqual(len(unmatched_dets), 1, "Tespit eşleşmedi olarak dönmeli")
        
        # Yeni track oluştur (Central servisin yaptığı gibi)
        # Önce pending track oluşur
        self.tracker.update_edge_track_feature(
            cam_id, group_id, det_A1['edge_track_id'], det_A1['gp_coord'], 
            det_A1['confidence'], det_A1['bbox'], det_A1['feature'], det_A1['quality']
        )
        
        # Pending durumunu global track'e çevirmek için birkaç update gerekir (min_features_for_id=3)
        # Biz testi hızlandırmak için pending track'i manuel manipüle edelim veya 3 kere gönderelim
        for i in range(3):
             self.tracker.update_edge_track_feature(
                cam_id, group_id, det_A1['edge_track_id'], det_A1['gp_coord'], 
                det_A1['confidence'], det_A1['bbox'], det_A1['feature'], det_A1['quality']
            )
             
        # Şimdi Global ID oluşmuş olmalı
        global_id_A = self.tracker.get_global_id_for_edge_track(cam_id, 101)
        self.assertNotEqual(global_id_A, "Unknown", "Global ID atanmalıydı")
        print(f"✓ Person A -> Global ID: {global_id_A}")
        
        # --- Adım 2: İkinci Tespit (Person A hareket etti) ---
        curr_time += 0.1
        det_A2 = det_A1.copy()
        det_A2['bbox'] = [110, 105, 210, 305] # Biraz hareket
        det_A2['gp_coord'] = np.array([11.0, 11.0])
        
        # Batch match çağır
        matches, unmatched_dets, unmatched_tracks = self.tracker.batch_match_detections(
            cam_id, group_id, [det_A2], curr_time
        )
        
        self.assertEqual(len(matches), 1, "Hareket eden kişi eşleşmeli")
        self.assertEqual(matches[0][1], global_id_A, "Doğru Global ID ile eşleşmeli")
        print("✓ Hareket takibi başarılı (Hungarian Match).")

    def test_03_cross_camera_reid(self):
        """Test: Farklı kamerada Re-ID ve GCN etkisi."""
        print("[TEST 3] Cross-Camera Re-ID & GCN Logic")
        
        # Önce Person A'yı sisteme kaydet (Manuel ekleyelim)
        gid_A = 1
        track_A = GlobalTrack(gid_A, "group_A", 1.0, np.array([10,10,0,0]), np.eye(4), np.eye(4), np.eye(2))
        track_A.robust_id = self.feat_person_A # Memory oturdu
        track_A.last_known_feature = self.feat_person_A
        track_A.last_cam_id = "cam_01"
        track_A.last_seen_timestamp = 1000.0
        self.tracker.global_tracks[gid_A] = track_A
        
        # --- Senaryo: Cam 2'de Person A beliriyor ---
        cam_id = "cam_02"
        group_id = "group_A"
        curr_time = 1005.0 # 5 saniye sonra
        
        det_A_cam2 = {
            'feature': self.feat_person_A, # Aynı özellikler (İdeal durum)
            'bbox': [500, 500, 600, 700],
            'gp_coord': np.array([50.0, 50.0]),
            'confidence': 0.9,
            'quality': 0.85,
            'edge_track_id': 201,
            'original_event': {'frame_res': [1920, 1080]}
        }
        
        # Hungarian Matcher içinde GCN çağrılacak mı?
        # Mock GCN, feat_A ile feat_A'yı karşılaştırınca yüksek skor verecek.
        
        matches, unmatched_dets, unmatched_tracks = self.tracker.batch_match_detections(
            cam_id, group_id, [det_A_cam2], curr_time
        )
        
        self.assertEqual(len(matches), 1, "Cross-camera eşleşme olmalı")
        self.assertEqual(matches[0][1], gid_A, "ID korunmalı")
        print("✓ Cross-camera Re-ID başarılı.")

    def test_04_hungarian_assignment(self):
        """Test: İki kişi aynı anda belirdiğinde doğru atama."""
        print("[TEST 4] Multi-Object Hungarian Assignment")
        
        # Sisteme A ve B kişilerini ekle
        gid_A = 1
        track_A = GlobalTrack(gid_A, "group_A", 1.0, np.array([10,10,0,0]), np.eye(4), np.eye(4), np.eye(2))
        track_A.robust_id = self.feat_person_A
        track_A.last_cam_id = "cam_01"
        track_A.last_seen_timestamp = 1000.0
        self.tracker.global_tracks[gid_A] = track_A
        
        gid_B = 2
        track_B = GlobalTrack(gid_B, "group_A", 1.0, np.array([50,50,0,0]), np.eye(4), np.eye(4), np.eye(2))
        track_B.robust_id = self.feat_person_B
        track_B.last_cam_id = "cam_01"
        track_B.last_seen_timestamp = 1000.0
        self.tracker.global_tracks[gid_B] = track_B
        
        # --- Senaryo: Cam 2'de ikisi de beliriyor ---
        curr_time = 1002.0
        cam_id = "cam_02"
        group_id = "group_A"
        
        # Detection listesi (Sıra karışık olabilir)
        detections = [
            { # Bu Person B olmalı
                'feature': self.feat_person_B, 
                'bbox': [520, 520, 620, 720],
                'gp_coord': np.array([52.0, 52.0]), 
                'edge_track_id': 202,
                'confidence': 0.9, 'quality': 0.9, 'original_event': {}
            },
            { # Bu Person A olmalı
                'feature': self.feat_person_A, 
                'bbox': [120, 120, 220, 320],
                'gp_coord': np.array([12.0, 12.0]), 
                'edge_track_id': 201,
                'confidence': 0.9, 'quality': 0.9, 'original_event': {}
            }
        ]
        
        matches, unmatched_dets, unmatched_tracks = self.tracker.batch_match_detections(
            cam_id, group_id, detections, curr_time
        )
        
        self.assertEqual(len(matches), 2, "İki kişi de eşleşmeli")
        
        # Doğru eşleştiler mi kontrol et
        # matches listesi [(det_idx, global_id), ...] formatında
        match_dict = {det_idx: gid for det_idx, gid in matches}
        
        self.assertEqual(match_dict[0], gid_B, "Det 0 (Person B) -> ID 2 olmalı")
        self.assertEqual(match_dict[1], gid_A, "Det 1 (Person A) -> ID 1 olmalı")
        
        print("✓ Hungarian doğru atama yaptı (A->A, B->B).")

    def test_05_adaptive_threshold(self):
        """Test: Eşik değerinin adaptasyonu."""
        print("[TEST 5] Adaptive Threshold Logic")
        
        # Başlangıç eşiği
        initial_threshold = self.tracker.threshold_manager.get_statistics("cam_01")['current_threshold']
        print(f"  Başlangıç Eşiği: {initial_threshold:.3f}")
        
        # Simüle et: Çok kalabalık ve herkes birbirine benziyor
        # AdaptiveThresholdManager'a sahte raporlar gönder
        cam_id = "cam_01"
        
        # 1. Kalabalık (Density) simülasyonu
        # 50 tane birbirine çok benzeyen detection gönderelim
        dense_detections = []
        for _ in range(30):
            # feat_person_C (Hard negative) kullanarak benzerlik yarat
            dense_detections.append({'feature': self.feat_person_C, 'bbox': [0,0,10,10]})
            
        tracks = {i: GlobalTrack(i, "g", 1, np.zeros(4), np.eye(4), np.eye(4), np.eye(2)) for i in range(30)}
        
        # Eşik değerini iste (Update tetiklenmeli)
        new_threshold = self.tracker.threshold_manager.get_threshold(
            cam_id, dense_detections, tracks, current_time=2000.0
        )
        
        print(f"  Kalabalık Sonrası Eşik: {new_threshold:.3f}")
        
        # Beklenti: Kalabalık ve benzerlik arttığı için eşik yükselmeli (Stricter)
        self.assertGreater(new_threshold, initial_threshold, "Kalabalıkta eşik değeri artmalı (Daha katı olmalı)")
        print("✓ Adaptive Threshold doğru tepki verdi.")

    def test_06_matrix_index_bug_verification(self):
        """
        Test: 'Matrix Index Mismatch' bug'ının düzeltildiğini doğrula.
        Eski trackler filtrelendiğinde matris indeksi kaymamalı.
        """
        print("[TEST 6] Matrix Index Bug Fix Verification")
        
        # A kişisi ÇOK ESKİ (Zaman aşımına uğramış)
        gid_A = 1
        track_A = GlobalTrack(gid_A, "group_A", 1.0, np.zeros(4), np.eye(4), np.eye(4), np.eye(2))
        track_A.last_seen_timestamp = 100.0 # Çok eski
        self.tracker.global_tracks[gid_A] = track_A
        
        # B kişisi YENİ
        gid_B = 2
        track_B = GlobalTrack(gid_B, "group_A", 1.0, np.zeros(4), np.eye(4), np.eye(4), np.eye(2))
        track_B.last_seen_timestamp = 2000.0 # Güncel (Current time 2000.0)
        track_B.robust_id = self.feat_person_B
        self.tracker.global_tracks[gid_B] = track_B
        
        curr_time = 2000.0
        
        # Detection sadece B kişisi için geliyor
        detections = [{
            'feature': self.feat_person_B, 
            'bbox': [0,0,10,10], 
            'gp_coord': np.array([0,0]),
            'confidence': 0.9, 'quality': 0.9, 'original_event': {}
        }]
        
        # Fonksiyonu çalıştır
        try:
            matches, _, _ = self.tracker.batch_match_detections("cam_01", "group_A", detections, curr_time)
            
            # Eşleşme B ile olmalı
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0][1], gid_B)
            
            print("✓ Hata almadan çalıştı ve doğru eşledi.")
            
        except IndexError:
            self.fail("IndexError yakalandı! Matris indeksi hatası devam ediyor.")
        except Exception as e:
            self.fail(f"Beklenmeyen hata: {e}")

if __name__ == '__main__':
    # exit=False parametresi, testler bittikten sonra programın çökmesini engeller
    unittest.main(exit=False)