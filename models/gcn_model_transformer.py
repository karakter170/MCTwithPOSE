# gcn_model_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationTransformer(nn.Module):
    def __init__(self, feature_dim=1024, geo_dim=5, hidden_dim=256, nhead=4):
        super(RelationTransformer, self).__init__()
        
        # 1. Feature Stream (DINO)
        self.app_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Geometry Stream (Bbox)
        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim) # Boyutu eşitle
        )
        
        # 3. Fusion Norm
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # 4. Cross-Attention
        # Tracks = Query, Detections = Key/Value
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        
        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, tracks, detections):
        """
        tracks: (Batch, 1028, N)
        detections: (Batch, 1028, M)
        """
        # (B, Dim, N) -> (B, N, Dim)
        tracks = tracks.transpose(1, 2)
        detections = detections.transpose(1, 2)
        
        # --- Split Features & Geometry ---
        # Giriş 1028 dim: İlk 1024 DINO, Son 4 Geo
        t_app, t_geo = tracks[:, :, :1024], tracks[:, :, 1024:]
        d_app, d_geo = detections[:, :, :1024], detections[:, :, 1024:]
        
        # --- Encode Separately ---
        # Tracks Embedding
        t_emb = self.app_encoder(t_app) + self.geo_encoder(t_geo)
        
        # Detections Embedding
        d_emb = self.app_encoder(d_app) + self.geo_encoder(d_geo)
        
        # Normalize
        t_emb = self.fusion_norm(t_emb) # (B, N, 256)
        d_emb = self.fusion_norm(d_emb) # (B, M, 256)
        
        # --- Cross Attention ---
        # Query: Tracks, Key/Value: Detections
        # attn_output: (B, N, 256)
        attn_out, _ = self.cross_attn(query=t_emb, key=d_emb, value=d_emb)
        
        # --- Relation Classification ---
        B, N, C = t_emb.shape
        M = d_emb.shape[1]
        
        # --- DÜZELTME BURADA ---
        # expand() kullanırken hedef boyutları açıkça belirtmeliyiz.
        # -1 kullanımı sadece boyutu değiştirmek istemediğimizde geçerlidir.
        
        # t_rep: Her track'i M defa çoğalt (Detection sayısı kadar)
        # (B, N, 1, C) -> (B, N, M, C)
        t_rep = t_emb.unsqueeze(2).expand(B, N, M, C)
        
        # d_rep: Her detection'ı N defa çoğalt (Track sayısı kadar)
        # (B, 1, M, C) -> (B, N, M, C)
        d_rep = d_emb.unsqueeze(1).expand(B, N, M, C)
        
        # Pair Features: (B, N, M, 512)
        pair_feat = torch.cat([t_rep, d_rep], dim=-1)
        
        # Predict: (B, N, M)
        logits = self.classifier(pair_feat).squeeze(-1)
        
        return logits