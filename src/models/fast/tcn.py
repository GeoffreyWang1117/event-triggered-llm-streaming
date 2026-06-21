"""
Temporal Convolutional Network (TCN) for streaming prediction.
Efficient 1D convolution-based model with causal padding.
"""
import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from .base import FastModel
from ..base import ModelOutput


class CausalConv1d(nn.Module):
    """Causal convolution layer with proper padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # Remove future padding to maintain causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return self.dropout(out)


class TemporalBlock(nn.Module):
    """Residual block with dilated causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation, dropout
        )
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation, dropout
        )

        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        residual = self.downsample(x)

        out = self.conv1(x)
        out = out.transpose(1, 2)  # (batch, seq_len, channels) for LayerNorm
        out = self.norm1(out)
        out = out.transpose(1, 2)  # Back to (batch, channels, seq_len)
        out = self.relu(out)

        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)

        return self.relu(out + residual)


class TCNNet(nn.Module):
    """Temporal Convolutional Network."""

    def __init__(
        self,
        n_features: int,
        n_channels: List[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_size: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, n_channels[0])

        # Build TCN layers with exponentially increasing dilation
        layers = []
        for i in range(len(n_channels)):
            dilation = 2 ** i
            in_ch = n_channels[0] if i == 0 else n_channels[i - 1]
            out_ch = n_channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.tcn = nn.Sequential(*layers)

        # Output heads
        self.fc_mean = nn.Linear(n_channels[-1], output_size)
        self.fc_var = nn.Sequential(
            nn.Linear(n_channels[-1], output_size),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, n_features)

        Returns:
            mean: Predictions (batch, output_size)
            variance: Prediction variance (batch, output_size)
        """
        # Project input features
        x = self.input_proj(x)  # (batch, seq_len, n_channels[0])

        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)

        # Use last timestep
        x = x[:, -1, :]

        mean = self.fc_mean(x)
        variance = self.fc_var(x)

        return mean, variance


class TCNModel(FastModel):
    """
    TCN model for streaming prediction.

    Advantages over RNNs:
    - Parallelizable training
    - Flexible receptive field
    - Stable gradients
    """

    def __init__(
        self,
        n_features: int,
        n_channels: List[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_size: int = 1,
        device: str = 'auto',
        **kwargs
    ):
        super().__init__(n_features)
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.output_size = output_size

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.network = TCNNet(
            n_features=n_features,
            n_channels=n_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            output_size=output_size,
        ).to(self.device)

        self.optimizer = None
        self._buffer = None
        self._buffer_size = self._compute_receptive_field()

        # Statistics for anomaly detection
        self._running_mean = None
        self._running_var = None

    def _compute_receptive_field(self) -> int:
        """Compute the receptive field of the TCN."""
        # For each layer with dilation d and kernel k: receptive field += (k-1) * d
        rf = 1
        for i in range(len(self.n_channels)):
            dilation = 2 ** i
            rf += (self.kernel_size - 1) * dilation
        return rf

    def reset_state(self) -> None:
        """Reset buffer for new sequence."""
        self._buffer = None

    def step(self, x: np.ndarray) -> ModelOutput:
        """
        Process single timestep using sliding window.

        Args:
            x: Single observation (n_features,)

        Returns:
            ModelOutput with prediction
        """
        # Maintain buffer for receptive field
        if self._buffer is None:
            self._buffer = np.zeros((self._buffer_size, self.n_features))

        # Shift buffer and add new observation
        self._buffer = np.roll(self._buffer, -1, axis=0)
        self._buffer[-1] = x

        # Predict using buffer
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
        """Compute anomaly score based on deviation from statistics."""
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
        learning_rate: float = 1e-3,
        val_split: float = 0.1,
        patience: int = 10,
    ) -> 'TCNModel':
        """Train the model."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)

        # Split validation
        n_val = int(len(X) * val_split)
        indices = torch.randperm(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
        X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            self.network.train()

        # Compute statistics
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
            'n_channels': self.n_channels,
            'kernel_size': self.kernel_size,
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
            'receptive_field': self._buffer_size,
            'n_channels': self.n_channels,
        }
