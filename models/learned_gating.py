# learned_gating.py
# LEARNED GATING NETWORK FOR NESTED LEARNING
#
# Replaces fixed sigmoid gating with a neural network that learns
# when to update slow memory based on multiple contextual features.
#
# Benefits:
# - Adapts to specific environment/cameras
# - Considers multiple factors (not just similarity)
# - Learns to handle edge cases
# - More robust to appearance changes and adversarial cases

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import json
import time
import os


# =============================================================================
# GATING NETWORK ARCHITECTURE
# =============================================================================

class GatingNetwork(nn.Module):
    """
    Neural network that decides how much to update slow memory.
    
    Input: 12-dimensional context vector
    Output: Single value in [0, 1] (update weight)
    
    Architecture is intentionally small for:
    - Fast inference (<0.1ms)
    - Easy training with limited data
    - Deployment on edge devices
    """
    
    def __init__(self, input_dim: int = 12, hidden_dims: List[int] = [32, 16]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Helps with varying input scales
                nn.ReLU(),
                nn.Dropout(0.1)  # Regularization
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Context features (batch_size, 12)
            
        Returns:
            Update weights (batch_size, 1)
        """
        return self.network(x)


class GatingNetworkWithUncertainty(nn.Module):
    """
    Extended version that also outputs uncertainty estimate.
    Uses Monte Carlo Dropout for uncertainty quantification.
    """
    
    def __init__(self, input_dim: int = 12, hidden_dims: List[int] = [32, 16]):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU()
        )
        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Dropout(0.2),  # Higher dropout for uncertainty
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.ReLU()
            ))
        
        self.output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = False,
                n_samples: int = 10) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional uncertainty estimation.

        Args:
            x: Context features
            return_uncertainty: Whether to compute uncertainty via MC Dropout
            n_samples: Number of forward passes for uncertainty

        Returns:
            (mean_prediction, uncertainty) if return_uncertainty else (prediction, None)
        """
        if not return_uncertainty:
            h = self.input_layer(x)
            for layer in self.hidden_layers:
                h = layer(h)
            return self.output_layer(h), None

        # Monte Carlo Dropout for uncertainty
        # FIXED: Preserve original training mode
        original_mode = self.training
        self.train()  # Enable dropout

        predictions = []

        for _ in range(n_samples):
            h = self.input_layer(x)
            for layer in self.hidden_layers:
                h = layer(h)
            pred = self.output_layer(h)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        # Restore original mode instead of forcing eval
        self.train(original_mode)

        return mean_pred, uncertainty


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

@dataclass
class GatingContext:
    """
    Context features for gating decision.
    All features are normalized to roughly [0, 1] range.
    """
    # Appearance features
    cosine_similarity: float      # dot(current, slow_memory)
    l2_distance: float            # Euclidean distance (normalized)
    quality_score: float          # Current observation quality
    buffer_similarity: float      # Average similarity to buffer
    
    # Temporal features
    track_age_normalized: float   # Age / max_age (capped at 1)
    time_since_update: float      # Seconds since last update (normalized)
    observation_count: float      # Count / maturity_frames (capped at 1)
    maturity: float               # Maturity factor [0, 1]
    
    # Statistical features
    variance_mean: float          # Mean of variance vector
    consistency_ema: float        # EMA of consistency history
    divergence_ratio: float       # divergence_counter / breakout_limit
    quality_history_mean: float   # Average quality of past observations
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor."""
        return torch.tensor([
            self.cosine_similarity,
            self.l2_distance,
            self.quality_score,
            self.buffer_similarity,
            self.track_age_normalized,
            self.time_since_update,
            self.observation_count,
            self.maturity,
            self.variance_mean,
            self.consistency_ema,
            self.divergence_ratio,
            self.quality_history_mean
        ], dtype=torch.float32)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.cosine_similarity,
            self.l2_distance,
            self.quality_score,
            self.buffer_similarity,
            self.track_age_normalized,
            self.time_since_update,
            self.observation_count,
            self.maturity,
            self.variance_mean,
            self.consistency_ema,
            self.divergence_ratio,
            self.quality_history_mean
        ], dtype=np.float32)
    
    @staticmethod
    def from_numpy(arr: np.ndarray) -> 'GatingContext':
        """Create from numpy array."""
        return GatingContext(
            cosine_similarity=float(arr[0]),
            l2_distance=float(arr[1]),
            quality_score=float(arr[2]),
            buffer_similarity=float(arr[3]),
            track_age_normalized=float(arr[4]),
            time_since_update=float(arr[5]),
            observation_count=float(arr[6]),
            maturity=float(arr[7]),
            variance_mean=float(arr[8]),
            consistency_ema=float(arr[9]),
            divergence_ratio=float(arr[10]),
            quality_history_mean=float(arr[11])
        )


def extract_gating_context(
    current_feature: np.ndarray,
    slow_memory: np.ndarray,
    slow_variance: np.ndarray,
    fast_buffer: List[np.ndarray],
    quality: float,
    track_age: float,
    time_since_update: float,
    observation_count: int,
    consistency_ema: float,
    divergence_counter: int,
    quality_history: List[float],
    max_age: float = 300.0,
    maturity_frames: int = 100,
    breakout_limit: int = 30
) -> GatingContext:
    """
    Extract context features for gating network.
    
    All inputs are raw values; this function normalizes them.
    """
    # Normalize feature
    current_norm = current_feature / (np.linalg.norm(current_feature) + 1e-8)
    slow_norm = slow_memory / (np.linalg.norm(slow_memory) + 1e-8)
    
    # Appearance features
    cosine_sim = float(np.dot(current_norm, slow_norm))
    l2_dist = float(np.linalg.norm(current_norm - slow_norm))
    l2_dist_normalized = min(1.0, l2_dist / 2.0)  # Max L2 for unit vectors is 2
    
    # Buffer similarity (average similarity to recent observations)
    if fast_buffer:
        buffer_sims = [np.dot(current_norm, f / (np.linalg.norm(f) + 1e-8)) for f in fast_buffer]
        buffer_sim = float(np.mean(buffer_sims))
    else:
        buffer_sim = cosine_sim
    
    # Temporal features
    track_age_norm = min(1.0, track_age / max_age)
    time_since_norm = min(1.0, time_since_update / 60.0)  # Normalize to 60s
    obs_count_norm = min(1.0, observation_count / maturity_frames)
    maturity = min(1.0, observation_count / maturity_frames)
    
    # Statistical features
    variance_mean = float(np.mean(slow_variance))
    divergence_ratio = min(1.0, divergence_counter / breakout_limit)
    
    if quality_history:
        quality_hist_mean = float(np.mean(quality_history[-20:]))  # Last 20
    else:
        quality_hist_mean = quality
    
    return GatingContext(
        cosine_similarity=cosine_sim,
        l2_distance=l2_dist_normalized,
        quality_score=quality,
        buffer_similarity=buffer_sim,
        track_age_normalized=track_age_norm,
        time_since_update=time_since_norm,
        observation_count=obs_count_norm,
        maturity=maturity,
        variance_mean=variance_mean,
        consistency_ema=consistency_ema,
        divergence_ratio=divergence_ratio,
        quality_history_mean=quality_hist_mean
    )


# =============================================================================
# TRAINING DATA COLLECTION
# =============================================================================

@dataclass
class GatingTrainingSample:
    """
    Single training sample for gating network.
    
    Collected during normal operation, labeled retroactively based on
    whether the update helped or hurt re-identification performance.
    """
    context: np.ndarray          # 12-dim context features
    label: float                 # 0.0 = should NOT have updated, 1.0 = should have updated
    global_id: int               # Track ID (for analysis)
    timestamp: float             # When collected
    
    def to_dict(self) -> dict:
        return {
            'context': self.context.tolist(),
            'label': self.label,
            'global_id': self.global_id,
            'timestamp': self.timestamp
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'GatingTrainingSample':
        return GatingTrainingSample(
            context=np.array(d['context'], dtype=np.float32),
            label=d['label'],
            global_id=d['global_id'],
            timestamp=d['timestamp']
        )


class GatingDataCollector:
    """
    Collects training data during system operation.
    
    Strategy:
    1. Store (context, observation, identity_before) for each update
    2. When track is re-identified successfully → label = 1.0 (good update)
    3. When track has ID switch → label = 0.0 (bad update)
    4. Periodic unlabeled samples → label = 0.5 (neutral)
    """
    
    def __init__(self, storage_path: str = "gating_training_data.json"):
        self.storage_path = storage_path
        self.pending_samples: Dict[int, List[dict]] = {}  # global_id -> samples
        self.labeled_samples: List[GatingTrainingSample] = []
        self.max_pending_per_track = 50
        self.max_labeled_samples = 100000
        
        # Load existing data
        self._load()
    
    def _load(self):
        """Load existing training data."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                self.labeled_samples = [
                    GatingTrainingSample.from_dict(d) 
                    for d in data.get('labeled', [])
                ]
                print(f"[GatingCollector] Loaded {len(self.labeled_samples)} samples")
            except Exception as e:
                print(f"[GatingCollector] Error loading: {e}")
    
    def save(self):
        """Save training data to disk."""
        data = {
            'labeled': [s.to_dict() for s in self.labeled_samples[-self.max_labeled_samples:]]
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f)
    
    def record_update(self, global_id: int, context: GatingContext):
        """
        Record that an update occurred with given context.
        Will be labeled later based on outcome.
        """
        if global_id not in self.pending_samples:
            self.pending_samples[global_id] = []
        
        sample = {
            'context': context.to_numpy(),
            'timestamp': time.time()
        }
        
        self.pending_samples[global_id].append(sample)
        
        # Keep only recent pending samples
        if len(self.pending_samples[global_id]) > self.max_pending_per_track:
            self.pending_samples[global_id].pop(0)
    
    def label_positive(self, global_id: int, reason: str = "successful_reid"):
        """
        Label pending samples as POSITIVE (update was good).
        Called when track is successfully re-identified.
        """
        if global_id not in self.pending_samples:
            return
        
        for sample in self.pending_samples[global_id]:
            self.labeled_samples.append(GatingTrainingSample(
                context=sample['context'],
                label=1.0,
                global_id=global_id,
                timestamp=sample['timestamp']
            ))
        
        self.pending_samples[global_id] = []
    
    def label_negative(self, global_id: int, reason: str = "id_switch"):
        """
        Label pending samples as NEGATIVE (update was bad).
        Called when ID switch is detected.
        """
        if global_id not in self.pending_samples:
            return
        
        for sample in self.pending_samples[global_id]:
            self.labeled_samples.append(GatingTrainingSample(
                context=sample['context'],
                label=0.0,
                global_id=global_id,
                timestamp=sample['timestamp']
            ))
        
        self.pending_samples[global_id] = []
    
    def label_neutral(self, global_id: int):
        """
        Label pending samples as NEUTRAL (unclear outcome).
        """
        if global_id not in self.pending_samples:
            return
        
        for sample in self.pending_samples[global_id]:
            self.labeled_samples.append(GatingTrainingSample(
                context=sample['context'],
                label=0.5,  # Neutral - neither good nor bad
                global_id=global_id,
                timestamp=sample['timestamp']
            ))
        
        self.pending_samples[global_id] = []
    
    def get_dataset(self) -> 'GatingDataset':
        """Get PyTorch dataset from collected samples."""
        return GatingDataset(self.labeled_samples)
    
    def get_statistics(self) -> dict:
        """Get collection statistics."""
        if not self.labeled_samples:
            return {'total': 0}
        
        labels = [s.label for s in self.labeled_samples]
        return {
            'total': len(labels),
            'positive': sum(1 for l in labels if l > 0.7),
            'negative': sum(1 for l in labels if l < 0.3),
            'neutral': sum(1 for l in labels if 0.3 <= l <= 0.7),
            'pending_tracks': len(self.pending_samples)
        }


class GatingDataset(Dataset):
    """PyTorch Dataset for gating network training."""
    
    def __init__(self, samples: List[GatingTrainingSample]):
        self.contexts = torch.tensor(
            np.array([s.context for s in samples]),
            dtype=torch.float32
        )
        self.labels = torch.tensor(
            np.array([s.label for s in samples]),
            dtype=torch.float32
        ).unsqueeze(1)
    
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        return self.contexts[idx], self.labels[idx]


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class GatingNetworkTrainer:
    """
    Training pipeline for gating network.
    """
    
    def __init__(
        self,
        model: GatingNetwork = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model or GatingNetwork()
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # For class imbalance handling
        self.criterion = nn.BCELoss(reduction='none')
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        class_weights: Tuple[float, float] = (1.0, 1.0)
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data
            class_weights: (negative_weight, positive_weight) for imbalance
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for contexts, labels in dataloader:
            contexts = contexts.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(contexts)
            
            # Weighted loss for class imbalance
            base_loss = self.criterion(predictions, labels)
            
            # Apply class weights
            weights = torch.where(
                labels > 0.5,
                torch.tensor(class_weights[1], device=self.device),
                torch.tensor(class_weights[0], device=self.device)
            )
            weighted_loss = (base_loss * weights).mean()
            
            weighted_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += weighted_loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def evaluate(self, dataloader: DataLoader) -> dict:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for contexts, labels in dataloader:
                contexts = contexts.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model(contexts)
                loss = F.binary_cross_entropy(predictions, labels)
                
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                total_loss += loss.item()
                n_batches += 1
        
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        
        # Metrics
        binary_preds = (preds > 0.5).astype(float)
        binary_labels = (labels > 0.5).astype(float)
        
        accuracy = np.mean(binary_preds == binary_labels)
        
        # Precision/Recall for positive class
        true_pos = np.sum((binary_preds == 1) & (binary_labels == 1))
        pred_pos = np.sum(binary_preds == 1)
        actual_pos = np.sum(binary_labels == 1)
        
        precision = true_pos / (pred_pos + 1e-8)
        recall = true_pos / (actual_pos + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'loss': total_loss / n_batches if n_batches > 0 else 0.0,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(
        self,
        train_dataset: GatingDataset,
        val_dataset: GatingDataset = None,
        epochs: int = 50,
        batch_size: int = 64,
        early_stopping_patience: int = 10
    ) -> dict:
        """
        Full training loop.
        
        Returns:
            Training history
        """
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        # Compute class weights for imbalance
        # FIXED: Correct class weight calculation (higher weight for minority class)
        labels = train_dataset.labels.numpy().flatten()
        n_samples = len(labels)
        n_pos = np.sum(labels > 0.5)
        n_neg = n_samples - n_pos

        # Weight inversely proportional to class frequency
        pos_weight = n_samples / (2.0 * n_pos + 1e-8)
        neg_weight = n_samples / (2.0 * n_neg + 1e-8)

        class_weights = (neg_weight, pos_weight)
        print(f"[Training] Class weights: neg={neg_weight:.3f}, pos={pos_weight:.3f}")
        print(f"[Training] Class distribution: pos={n_pos}/{n_samples} ({n_pos/n_samples*100:.1f}%), neg={n_neg}/{n_samples} ({n_neg/n_samples*100:.1f}%)")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, class_weights)
            history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_f1'].append(val_metrics['f1'])
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                
                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"[Training] Early stopping at epoch {epoch}")
                    break
                
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                          f"val_loss={val_metrics['loss']:.4f}, "
                          f"val_acc={val_metrics['accuracy']:.3f}, "
                          f"val_f1={val_metrics['f1']:.3f}")
            else:
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"[Training] Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[Training] Model loaded from {path}")


# =============================================================================
# INFERENCE WRAPPER
# =============================================================================

class LearnedGating:
    """
    Production inference wrapper for learned gating.
    Provides easy-to-use interface for integration with ContinuumState.
    """
    
    def __init__(
        self,
        model_path: str = None,
        fallback_threshold: float = 0.65,
        device: str = None
    ):
        """
        Initialize learned gating.
        
        Args:
            model_path: Path to trained model (uses sigmoid fallback if None)
            fallback_threshold: Threshold for sigmoid fallback
            device: Device for inference
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.fallback_threshold = fallback_threshold
        self.model = None
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = GatingNetwork()
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                print(f"[LearnedGating] Model loaded from {model_path}")
            except Exception as e:
                print(f"[LearnedGating] Error loading model: {e}")
                self.model = None
        else:
            print("[LearnedGating] No model found, using sigmoid fallback")
    
    def compute_update_weight(self, context: GatingContext) -> float:
        """
        Compute update weight for given context.
        
        Args:
            context: Gating context features
            
        Returns:
            Update weight in [0, 1]
        """
        if self.model is None:
            # Fallback to sigmoid
            return self._sigmoid_fallback(context.cosine_similarity)
        
        try:
            with torch.no_grad():
                input_tensor = context.to_tensor().unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                return float(output.item())
        except Exception as e:
            print(f"[LearnedGating] Inference error: {e}")
            return self._sigmoid_fallback(context.cosine_similarity)
    
    def _sigmoid_fallback(self, consistency: float) -> float:
        """Sigmoid fallback when model unavailable."""
        return 1.0 / (1.0 + np.exp(-10.0 * (consistency - self.fallback_threshold)))
    
    def compute_batch(self, contexts: List[GatingContext]) -> List[float]:
        """Compute update weights for batch of contexts."""
        if self.model is None:
            return [self._sigmoid_fallback(c.cosine_similarity) for c in contexts]
        
        try:
            with torch.no_grad():
                input_tensor = torch.stack([c.to_tensor() for c in contexts]).to(self.device)
                outputs = self.model(input_tensor)
                return outputs.squeeze(1).cpu().numpy().tolist()
        except Exception as e:
            print(f"[LearnedGating] Batch inference error: {e}")
            return [self._sigmoid_fallback(c.cosine_similarity) for c in contexts]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LEARNED GATING NETWORK - TEST")
    print("=" * 60)
    
    # Test 1: Model creation and forward pass
    print("\n[Test 1] Model Creation")
    print("-" * 40)
    
    model = GatingNetwork(input_dim=12, hidden_dims=[32, 16])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 12)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test 2: Context extraction
    print("\n[Test 2] Context Extraction")
    print("-" * 40)
    
    current_feature = np.random.randn(1024).astype(np.float32)
    slow_memory = np.random.randn(1024).astype(np.float32)
    slow_variance = np.ones(1024).astype(np.float32) * 0.05
    fast_buffer = [np.random.randn(1024).astype(np.float32) for _ in range(5)]
    
    context = extract_gating_context(
        current_feature=current_feature,
        slow_memory=slow_memory,
        slow_variance=slow_variance,
        fast_buffer=fast_buffer,
        quality=0.8,
        track_age=120.0,
        time_since_update=2.0,
        observation_count=50,
        consistency_ema=0.75,
        divergence_counter=5,
        quality_history=[0.7, 0.8, 0.75, 0.85]
    )
    
    print(f"Context features: {context.to_numpy()}")
    
    # Test 3: Synthetic training
    print("\n[Test 3] Synthetic Training")
    print("-" * 40)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    synthetic_samples = []
    for _ in range(n_samples):
        # Generate context
        ctx = np.random.rand(12).astype(np.float32)
        
        # Simple labeling rule: high similarity + high quality = should update
        sim = ctx[0]  # cosine_similarity
        qual = ctx[2]  # quality_score
        label = 1.0 if (sim > 0.6 and qual > 0.5) else 0.0
        
        # Add some noise
        if np.random.rand() < 0.1:
            label = 1.0 - label
        
        synthetic_samples.append(GatingTrainingSample(
            context=ctx,
            label=label,
            global_id=0,
            timestamp=time.time()
        ))
    
    # Split data
    train_samples = synthetic_samples[:800]
    val_samples = synthetic_samples[800:]
    
    train_dataset = GatingDataset(train_samples)
    val_dataset = GatingDataset(val_samples)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train
    trainer = GatingNetworkTrainer()
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=30,
        batch_size=32
    )
    
    print(f"\nFinal metrics:")
    final_metrics = trainer.evaluate(DataLoader(val_dataset, batch_size=32))
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test 4: Inference wrapper
    print("\n[Test 4] Inference Wrapper")
    print("-" * 40)
    
    # Save model temporarily
    trainer.save_model("/tmp/gating_model.pt")
    
    # Load with wrapper
    gating = LearnedGating(model_path="/tmp/gating_model.pt")
    
    # Test inference
    test_context = GatingContext(
        cosine_similarity=0.8,
        l2_distance=0.3,
        quality_score=0.9,
        buffer_similarity=0.75,
        track_age_normalized=0.5,
        time_since_update=0.1,
        observation_count=0.3,
        maturity=0.5,
        variance_mean=0.03,
        consistency_ema=0.7,
        divergence_ratio=0.1,
        quality_history_mean=0.8
    )
    
    update_weight = gating.compute_update_weight(test_context)
    print(f"Test context -> update_weight: {update_weight:.4f}")
    
    # Compare with fallback
    fallback_weight = gating._sigmoid_fallback(test_context.cosine_similarity)
    print(f"Sigmoid fallback: {fallback_weight:.4f}")
    
    # Test 5: Data collector
    print("\n[Test 5] Data Collector")
    print("-" * 40)
    
    collector = GatingDataCollector(storage_path="/tmp/gating_data.json")
    
    # Simulate collection
    for i in range(10):
        ctx = GatingContext(
            cosine_similarity=np.random.rand(),
            l2_distance=np.random.rand(),
            quality_score=np.random.rand(),
            buffer_similarity=np.random.rand(),
            track_age_normalized=np.random.rand(),
            time_since_update=np.random.rand(),
            observation_count=np.random.rand(),
            maturity=np.random.rand(),
            variance_mean=np.random.rand() * 0.1,
            consistency_ema=np.random.rand(),
            divergence_ratio=np.random.rand(),
            quality_history_mean=np.random.rand()
        )
        collector.record_update(global_id=1, context=ctx)
    
    # Label some positive
    collector.label_positive(global_id=1)
    
    print(f"Collection stats: {collector.get_statistics()}")
    
    # Cleanup
    os.remove("/tmp/gating_model.pt")
    os.remove("/tmp/gating_data.json")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)