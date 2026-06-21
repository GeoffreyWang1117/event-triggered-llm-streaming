"""
Small Transformer model for streaming prediction.
Lightweight version suitable for edge deployment.
"""
import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import math
from .base import FastModel
from ..base import ModelOutput


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerNet(nn.Module):
    """Small Transformer for time series prediction."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        output_size: int = 1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads
        self.fc_mean = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size),
        )
        self.fc_var = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_size),
            nn.Softplus(),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, n_features)
            mask: Causal mask (seq_len, seq_len)

        Returns:
            mean: Predictions (batch, output_size)
            variance: Prediction variance (batch, output_size)
        """
        # Project and add positional encoding
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Generate causal mask if not provided
        if mask is None:
            seq_len = x.size(1)
            mask = self._generate_causal_mask(seq_len, x.device)

        # Transformer
        x = self.transformer(x, mask=mask)

        # Use last position for output
        x = x[:, -1, :]

        mean = self.fc_mean(x)
        variance = self.fc_var(x)

        return mean, variance

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        return mask


class TransformerModel(FastModel):
    """
    Small Transformer for streaming prediction.

    Designed for edge deployment with limited parameters.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        output_size: int = 1,
        max_len: int = 512,
        device: str = 'auto',
        **kwargs
    ):
        super().__init__(n_features)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.output_size = output_size
        self.max_len = max_len

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.network = TransformerNet(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            output_size=output_size,
            max_len=max_len,
        ).to(self.device)

        self.optimizer = None
        self._buffer = None
        self._buffer_len = 0

        # Statistics
        self._running_mean = None
        self._running_var = None

    def reset_state(self) -> None:
        """Reset buffer for new sequence."""
        self._buffer = None
        self._buffer_len = 0

    def step(self, x: np.ndarray) -> ModelOutput:
        """
        Process single timestep.

        Args:
            x: Single observation (n_features,)

        Returns:
            ModelOutput with prediction
        """
        # Append to buffer
        if self._buffer is None:
            self._buffer = x.reshape(1, -1)
        else:
            self._buffer = np.vstack([self._buffer, x])
            # Limit buffer size
            if len(self._buffer) > self.max_len:
                self._buffer = self._buffer[-self.max_len:]

        self._buffer_len = len(self._buffer)

        # Predict
        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(
                self._buffer, dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            mean, variance = self.network(x_tensor)

            prediction = mean.cpu().numpy().flatten()
            uncertainty = variance.cpu().numpy().flatten()[0]

        # Anomaly score
        anomaly_score = self._compute_anomaly_score(x)

        return ModelOutput(
            prediction=prediction,
            uncertainty=float(np.sqrt(uncertainty)),
            anomaly_score=anomaly_score,
        )

    def _compute_anomaly_score(self, x: np.ndarray) -> float:
        """Compute anomaly score."""
        if self._running_mean is None:
            return 0.0

        deviation = x - self._running_mean
        if self._running_var is not None and np.all(self._running_var > 0):
            return float(np.sqrt(np.sum(deviation ** 2 / self._running_var)))
        return float(np.sqrt(np.sum(deviation ** 2)))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        val_split: float = 0.1,
        patience: int = 10,
    ) -> 'TransformerModel':
        """Train the model."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)

        # Split
        n_val = int(len(X) * val_split)
        indices = torch.randperm(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
        X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')
        patience_counter = 0

        self.network.train()
        for epoch in range(epochs):
            perm = torch.randperm(len(X_train))
            epoch_loss = 0.0

            for i in range(0, len(X_train), batch_size):
                batch_idx = perm[i:i + batch_size]
                X_batch = X_train[batch_idx].to(self.device)
                y_batch = y_train[batch_idx].to(self.device)

                self.optimizer.zero_grad()
                mean, variance = self.network(X_batch)

                loss = torch.mean(
                    0.5 * torch.log(variance + 1e-6) +
                    0.5 * (y_batch - mean) ** 2 / (variance + 1e-6)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            # Validation
            self.network.eval()
            with torch.no_grad():
                X_val_dev = X_val.to(self.device)
                y_val_dev = y_val.to(self.device)
                val_mean, _ = self.network(X_val_dev)
                val_loss = nn.MSELoss()(val_mean, y_val_dev).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            self.network.train()

        # Statistics
        X_flat = X.reshape(-1, self.n_features)
        self._running_mean = np.mean(X_flat, axis=0)
        self._running_var = np.var(X_flat, axis=0)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> ModelOutput:
        """Predict on sequences."""
        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            mean, variance = self.network(X_tensor)

            predictions = mean.cpu().numpy()
            uncertainties = variance.cpu().numpy()

        return ModelOutput(
            prediction=predictions.flatten(),
            uncertainty=float(np.mean(np.sqrt(uncertainties))),
        )

    def update(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Online update."""
        if y is None:
            return

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)

        self.network.train()
        self.optimizer.zero_grad()
        mean, variance = self.network(X_tensor)

        loss = torch.mean(
            0.5 * torch.log(variance + 1e-6) +
            0.5 * (y_tensor - mean) ** 2 / (variance + 1e-6)
        )
        loss.backward()
        self.optimizer.step()
        self.network.eval()

    def get_state(self) -> Dict[str, Any]:
        """Get model state."""
        return {
            'n_features': self.n_features,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'output_size': self.output_size,
            'network_state': self.network.state_dict(),
            'running_mean': self._running_mean.tolist() if self._running_mean is not None else None,
            'running_var': self._running_var.tolist() if self._running_var is not None else None,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state."""
        self.network.load_state_dict(state['network_state'])
        if state['running_mean'] is not None:
            self._running_mean = np.array(state['running_mean'])
            self._running_var = np.array(state['running_var'])
        self.is_fitted = True

    def get_complexity(self) -> Dict[str, Any]:
        """Return model complexity metrics."""
        n_params = sum(p.numel() for p in self.network.parameters())
        return {
            'n_parameters': n_params,
            'memory_bytes': n_params * 4,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
        }
