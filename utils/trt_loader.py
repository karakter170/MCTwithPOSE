import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

class TensorRTReidExtractor:
    def __init__(self, engine_path, max_batch_size=32):
        """
        DINOv3 TensorRT Yükleyicisi (Batch Inference Optimized).
        
        Args:
            engine_path (str): .engine dosyasının yolu.
            max_batch_size (int): Tek seferde GPU'ya gönderilecek maksimum obje sayısı.
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.max_batch_size = max_batch_size
        
        print(f"[TensorRT] DINOv3 Engine Yükleniyor (Max Batch: {self.max_batch_size})...")
        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            print("[TensorRT] Motor başarıyla çalıştırıldı! (Batch Mode Ready)")
        except Exception as e:
            print(f"[TensorRT] KRİTİK HATA: Engine yüklenemedi. {e}")
            raise

        self.stream = cuda.Stream()
        
        # --- INPUT / OUTPUT BOYUTLARI ---
        self.input_h, self.input_w = 256, 128
        self.out_dim = 1024  # ViT-Large/16
        
        # --- HAFIZA YÖNETİMİ (Allocation) ---
        # GPU belleğini EN BÜYÜK senaryoya (max_batch_size) göre bir kez ayırıyoruz.
        
        # 1. Input Buffer (Float32): (MaxBatch, 3, 256, 128)
        # 4 bytes per float
        self.input_vol = 3 * self.input_h * self.input_w
        self.input_byte_size = self.max_batch_size * self.input_vol * 4
        self.d_input = cuda.mem_alloc(self.input_byte_size)
        
        # 2. Output Buffer (Float32): (MaxBatch, 1024)
        self.output_vol = self.out_dim
        self.output_byte_size = self.max_batch_size * self.output_vol * 4
        self.d_output = cuda.mem_alloc(self.output_byte_size)
        
        # --- TENSOR ADRESLERİNİ BAĞLA ---
        # TensorRT 8.5+ / 10.x API uyumlu adres bağlama
        self.tensor_name_input = "input_image" # ONNX export ismine dikkat
        self.tensor_name_output = None
        
        # Input adresini bağla
        self.context.set_tensor_address(self.tensor_name_input, int(self.d_input))
        
        # Output ismini bul ve adresini bağla
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.tensor_name_output = name
                self.context.set_tensor_address(name, int(self.d_output))
                break
        
        if self.tensor_name_output is None:
            # Fallback: Genelde outputlar inputtan sonra gelir, manuel atama denenebilir
            # Ancak modern TRT'de isimle çalışmak en sağlıklısıdır.
            print("[TensorRT] UYARI: Output tensor ismi bulunamadı! Varsayılan 'feature_vector' denenecek.")
            self.tensor_name_output = "feature_vector" # Varsayım
            self.context.set_tensor_address(self.tensor_name_output, int(self.d_output))

    def preprocess_batch(self, frames_list):
        """
        Birden fazla crop görüntüsünü tek bir numpy bloğunda (Batch, 3, H, W) hazırlar.
        """
        batch_data = []
        for img in frames_list:
            # 1. Resize
            resized = cv2.resize(img, (self.input_w, self.input_h))
            
            # 2. BGR -> RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # 3. Listeye ekle (Henüz normalizasyon yapmadık, toplu yapacağız)
            batch_data.append(rgb)
            
        # Numpy Stack: (B, 256, 128, 3)
        batch_np = np.array(batch_data, dtype=np.float32)
        
        # 4. Normalize (Vectorized - Faster than Loop)
        # Mean/Std ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        batch_np /= 255.0
        batch_np -= mean
        batch_np /= std
        
        # 5. Transpose to CHW: (B, 3, 256, 128)
        batch_np = batch_np.transpose(0, 3, 1, 2)
        
        # Contiguous array şart (Memory copy için)
        return np.ascontiguousarray(batch_np, dtype=np.float32)

    def extract_features(self, frame, bboxes):
        """
        Batch Inference Pipeline.
        Girdi: Tüm Frame ve BBox Listesi
        Çıktı: (N, 1024) Feature Matrix
        """
        if not bboxes:
            return np.array([]), []

        # 1. Geçerli Crop'ları Topla
        crops = []
        valid_indices = []
        
        frame_h, frame_w = frame.shape[:2]
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            
            # Küçük/Hatalı kutuları ele
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
                
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            crops.append(crop)
            valid_indices.append(i)

        num_crops = len(crops)
        if num_crops == 0:
            return np.array([]), []

        # Sonuçları tutacak ana liste
        all_features = []

        # 2. Batch Processing Loop (Chunking)
        # Eğer tespit sayısı > max_batch_size ise parçalara bölerek işle.
        for start_idx in range(0, num_crops, self.max_batch_size):
            end_idx = min(start_idx + self.max_batch_size, num_crops)
            current_batch_crops = crops[start_idx:end_idx]
            current_batch_size = len(current_batch_crops)
            
            # A. Preprocess (CPU)
            input_tensor = self.preprocess_batch(current_batch_crops)
            
            # B. Input Shape Ayarla (DINOv3 Dynamic Batch için KRİTİK)
            # TensorRT'ye şu anki batch boyutunu bildiriyoruz.
            self.context.set_input_shape(self.tensor_name_input, (current_batch_size, 3, self.input_h, self.input_w))
            
            # C. Host -> Device Copy (Async)
            # Sadece geçerli veri boyutu kadar kopyala
            cuda.memcpy_htod_async(self.d_input, input_tensor, self.stream)
            
            # D. Inference (Async)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            # E. Device -> Host Copy (Async)
            # Çıktı bufferını CPU'da hazırla: (Batch, 1024)
            output_cpu = np.empty((current_batch_size, self.out_dim), dtype=np.float32)
            
            # GPU'daki output buffer'ın başından itibaren current_batch_size kadar veriyi çek
            # Not: d_output buffer'ı büyük olsa da, sadece başındaki ilgili kısmı okuyoruz.
            # Pointer aritmetiği yerine pycuda slice veya size limiti kullanamadığımız için
            # Memcpy fonksiyonuna byte size veriyoruz.
            copy_size = current_batch_size * self.out_dim * 4
            cuda.memcpy_dtoh_async(output_cpu[:current_batch_size], 
                       int(self.d_output), 
                       self.stream)
            
            # F. Stream Senkronizasyonu (Bu batch bitsin)
            self.stream.synchronize()
            
            all_features.append(output_cpu)

        # 3. Sonuçları Birleştir
        if not all_features:
            return np.array([]), []
            
        final_features = np.vstack(all_features) # (Total_N, 1024)
        
        # 4. L2 Normalization (Vectorized - CPU Optimized)
        # features = features / ||features||
        # axis=1 satır bazlı norm alır, keepdims=True boyut korur (N, 1)
        norms = np.linalg.norm(final_features, axis=1, keepdims=True) + 1e-6
        final_features = final_features / norms
        
        return final_features, valid_indices

    def __del__(self):
        # Kaynakları temizle
        try:
            if hasattr(self, 'd_input'): self.d_input.free()
            if hasattr(self, 'd_output'): self.d_output.free()
        except:
            pass