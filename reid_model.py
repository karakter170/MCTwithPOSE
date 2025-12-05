# reid_model.py
import torch
import torchreid
import numpy as np
from numpy.linalg import norm
import cv2

class ReidFeatureExtractor:
    """
    Torchreid'in resmi FeatureExtractor'ını kullanan sınıf.
    DÜZENLEME: Artık model_name ve feature_dim parametrelerini dışarıdan alıyor.
    """
    
    # DÜZENLEME: __init__ imzası değişti! Artık 3 parametre alıyor.
    def __init__(self, model_name: str, model_weights_path: str, feature_dim: int):
        
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Re-ID Modeli için kullanılan cihaz: {self.device_str}")
        
        try:
            # DÜZENLEME: Parametreleri 'model_name' ve 'model_weights_path'den al
            self.extractor = torchreid.utils.FeatureExtractor(
                model_name=model_name,
                model_path=model_weights_path, 
                device=self.device_str 
            )
            
            # DÜZENLEME: Özellik boyutunu parametreden al
            self.feature_dim = feature_dim 
            
            print(f"TorchReID FeatureExtractor başarıyla yüklendi.")
            print(f"  Model Adı: {model_name}")
            print(f"  Ağırlıklar: {model_weights_path}")
            print(f"  Özellik Boyutu: {self.feature_dim}")
            
        except Exception as e:
            print(f"Hata: TorchReID FeatureExtractor yüklenemedi. Ağırlık dosyası ({model_weights_path}) ve model adı ({model_name}) uyumlu mu? Hata: {e}")
            raise

    @torch.no_grad()
    def extract_features(self, frame, bboxes):
        """
        Verilen bbounding box'lardan kırpılan görüntülerin özelliklerini çıkarır.
        (Bu fonksiyonun içi zaten numpy/cv2 için doğru ayarlanmıştı)
        """
        if not bboxes:
            return np.array([]), []

        image_batch = []
        valid_indices = []

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2] 
            
            if crop.size == 0 or (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
                
            try:
                img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) 
                image_batch.append(img_rgb) 
                valid_indices.append(i)
            except Exception as e:
                print(f"Re-ID crop hatası (RGB çevirme): {e} - Bbox: {bbox}")
        
        if not image_batch:
            return np.array([]), []

        try:
            features_tensor = self.extractor(image_batch) 
            features_np = features_tensor.cpu().numpy()
            
            norm_features_list = []
            for feature_vector in features_np:
                norm_feat = feature_vector / (norm(feature_vector) + 1e-6)
                norm_features_list.append(norm_feat)
                
            return np.array(norm_features_list), valid_indices
        
        except Exception as e:
            print(f"Re-ID özellik çıkarma hatası (Extractor çağrısı): {e}") 
            return np.array([]), []