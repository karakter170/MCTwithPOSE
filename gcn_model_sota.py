import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossGCN(nn.Module):
    def __init__(self, feature_dim=1029, hidden_dim=512):
        super(CrossGCN, self).__init__()
        
        # 1. Feature Encoder
        # Input: (Batch, 1029, N) -> Output: (Batch, 512, N)
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. Relation Learner
        # Input: (Batch, 1024, N, M) -> Output: (Batch, 1, N, M)
        self.relation = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1) # Son katman (Logit üretir)
        )
        
        # --- CRITICAL: BIAS INITIALIZATION ---
        # Modelin eğitimin başında "Eşleşme Yok (0)" demeye meyilli olmasını sağlar.
        # Bu, Focal Loss'un stabil çalışması için zorunludur.
        # Başlangıç ihtimali (prior_prob) = 0.01 (%1)
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        
        # Son katmanın bias değerini ayarlıyoruz
        self.relation[-1].bias.data.fill_(bias_value)

    def forward(self, tracks, detections):
        # tracks: (B, 1029, N)
        # detections: (B, 1029, M)
        
        # A. Encode
        t_emb = self.encoder(tracks)
        d_emb = self.encoder(detections)
        
        N = t_emb.size(2)
        M = d_emb.size(2)
        
        # B. Broadcast & Concat
        t_rep = t_emb.unsqueeze(3).repeat(1, 1, 1, M) # (B, 512, N, M)
        d_rep = d_emb.unsqueeze(2).repeat(1, 1, N, 1) # (B, 512, N, M)
        
        combined = torch.cat([t_rep, d_rep], dim=1)   # (B, 1024, N, M)
        
        # C. Predict Logits
        logits = self.relation(combined) # (B, 1, N, M)
        
        # DİKKAT: Sigmoid BURADA YOK. Ham puan (Logit) dönüyoruz.
        return logits.squeeze(1)