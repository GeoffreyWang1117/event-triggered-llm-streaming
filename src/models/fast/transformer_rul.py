"""
Dual-Attention Transformer for RUL Prediction.
Based on SOTA methods: DKAMFormer, STAR Framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
import math

from .base import FastModel
from ..base import ModelOutput


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SensorAttention(nn.Module):
    """
    Sensor-wise attention to weight feature importance.
    Inspired by STAR framework's sensor attention.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.query = nn.Linear(n_features, d_model)
        self.key = nn.Linear(n_features, d_model)
        self.value = nn.Linear(n_features, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, seq_len, d_model)
        """
        # Transpose for sensor-wise attention
        x_t = x.transpose(1, 2)  # (batch, n_features, seq_len)

        Q = self.query(x)  # (batch, seq_len, d_model)
        K = self.key(x)
        V = self.value(x)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)

        # Apply attention
        out = torch.matmul(attn, V)
        return out


class MultiscaleTemporalBlock(nn.Module):
    """
    Multiscale temporal feature extraction.
    Inspired by DKAMFormer's MTSGSA.
    """

    def __init__(self, d_model: int, n_heads: int = 4, scales: list = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads // len(scales), batch_first=True)
            for _ in scales
        ])
        self.fusion = nn.Linear(d_model * len(scales), d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        outputs = []
        for scale, attn in zip(self.scales, self.attention_heads):
            if scale > 1:
                # Downsample for different time scales
                x_scaled = F.avg_pool1d(
                    x.transpose(1, 2), kernel_size=scale, stride=scale
                ).transpose(1, 2)
            else:
                x_scaled = x

            out, _ = attn(x_scaled, x_scaled, x_scaled)

            if scale > 1:
                # Upsample back
                out = F.interpolate(
                    out.transpose(1, 2), size=x.size(1), mode='linear', align_corners=False
                ).transpose(1, 2)

            outputs.append(out)

        # Fuse multiscale features
        fused = torch.cat(outputs, dim=-1)
        return self.norm(x + self.fusion(fused))


class DualAttentionTransformer(nn.Module):
    """
    Dual-Attention Transformer for RUL prediction.

    Combines:
    1. Sensor attention (feature-wise importance)
    2. Multiscale temporal attention (long-range dependencies)
    3. Hierarchical encoder structure
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Input projection with sensor attention
        self.sensor_attention = SensorAttention(n_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Multiscale temporal blocks
        self.temporal_blocks = nn.ModuleList([
            MultiscaleTemporalBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        # Standard transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output heads
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Uncertainty head (for triggering)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            Dictionary with 'rul', 'uncertainty', optionally 'attention'
        """
        # Sensor attention
        x = self.sensor_attention(x)
        x = self.input_norm(x)

        # Positional encoding
        x = self.pos_encoder(x)

        # Multiscale temporal processing
        for block in self.temporal_blocks:
            x = block(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Global pooling
        x_pooled = x.mean(dim=1)  # (batch, d_model)

        # Predictions
        rul = self.rul_head(x_pooled).squeeze(-1)
        uncertainty = self.uncertainty_head(x_pooled).squeeze(-1)

        output = {
            'rul': rul,
            'uncertainty': uncertainty,
        }

        return output


class DualAttentionRULModel(FastModel):
    """
    Streaming wrapper for Dual-Attention Transformer.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        seq_length: int = 30,
        device: str = 'cuda',
    ):
        super().__init__(n_features)
        self.seq_length = seq_length
        self.device = device

        self.network = DualAttentionTransformer(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        ).to(device)

        # Streaming buffer
        self.buffer = []
        self.optimizer = None

    def step(self, x: np.ndarray) -> ModelOutput:
        """Process single timestep."""
        self.buffer.append(x)
        if len(self.buffer) > self.seq_length:
            self.buffer.pop(0)

        if len(self.buffer) < self.seq_length:
            # Not enough data yet
            return ModelOutput(
                prediction=np.array([0.0]),
                uncertainty=1.0,
                anomaly_score=0.0,
            )

        # Prepare input
        x_seq = np.array(self.buffer, dtype=np.float32)
        x_tensor = torch.from_numpy(x_seq).unsqueeze(0).to(self.device)

        # Forward pass
        self.network.eval()
        with torch.no_grad():
            output = self.network(x_tensor)

        rul = output['rul'].cpu().numpy()[0]
        uncertainty = output['uncertainty'].cpu().numpy()[0]

        # Compute anomaly score based on RUL prediction change
        if hasattr(self, '_last_rul'):
            anomaly_score = abs(rul - self._last_rul)
        else:
            anomaly_score = 0.0
        self._last_rul = rul

        return ModelOutput(
            prediction=np.array([rul]),
            uncertainty=float(uncertainty),
            anomaly_score=float(anomaly_score),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        val_split: float = 0.1,
        patience: int = 10,
    ) -> 'DualAttentionRULModel':
        """
        Train the model.

        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            y: RUL labels (n_samples,)
        """
        # Split data
        n_val = int(len(X) * val_split)
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Convert to tensors
        X_train = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        y_train = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
        X_val = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
        y_val = torch.from_numpy(y_val.astype(np.float32)).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.network.train()
            train_loss = 0.0

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                self.optimizer.zero_grad()
                output = self.network(batch_X)

                # MSE loss for RUL
                loss = F.mse_loss(output['rul'], batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()

                train_loss += loss.item()

            scheduler.step()

            # Validation
            self.network.eval()
            with torch.no_grad():
                val_output = self.network(X_val)
                val_loss = F.mse_loss(val_output['rul'], y_val).item()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: train_loss={train_loss/len(X_train)*batch_size:.4f}, "
                      f"val_loss={val_loss:.4f}")

        self.is_fitted = True
        return self

    def reset_state(self):
        """Reset streaming buffer."""
        self.buffer = []
        if hasattr(self, '_last_rul'):
            del self._last_rul

    def get_complexity(self) -> Dict[str, Any]:
        """Return model complexity metrics."""
        n_params = sum(p.numel() for p in self.network.parameters())
        return {
            'n_parameters': n_params,
            'memory_bytes': n_params * 4,
            'architecture': 'DualAttentionTransformer',
        }

    def predict(self, X: np.ndarray) -> ModelOutput:
        """Predict on a sequence."""
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        self.network.eval()
        with torch.no_grad():
            output = self.network(X_tensor)

        rul = output['rul'].cpu().numpy()
        uncertainty = output['uncertainty'].cpu().numpy()

        return ModelOutput(
            prediction=rul,
            uncertainty=float(uncertainty.mean()),
            anomaly_score=0.0,
        )

    def update(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Online update (not implemented for transformer)."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            'model_state_dict': self.network.state_dict(),
            'n_features': self.n_features,
            'd_model': self.network.d_model,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state."""
        self.network.load_state_dict(state['model_state_dict'])
        self.is_fitted = True
