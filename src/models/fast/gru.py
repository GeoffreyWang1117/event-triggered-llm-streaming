"""
GRU-based model for streaming prediction.
Supports RUL prediction and anomaly detection.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from .base import FastModel
from ..base import ModelOutput


class GRUNet(nn.Module):
    """PyTorch GRU network for streaming inference."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        fc_input_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

        # For uncertainty estimation
        self.fc_var = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softplus(),  # Ensure positive variance
        )

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, n_features)
            h: Hidden state (n_layers * n_directions, batch, hidden_size)

        Returns:
            output: Predictions (batch, output_size)
            variance: Prediction variance (batch, output_size)
            hidden: New hidden state
        """
        gru_out, hidden = self.gru(x, h)

        # Use last timestep output
        last_output = gru_out[:, -1, :]

        output = self.fc(last_output)
        variance = self.fc_var(last_output)

        return output, variance, hidden


class GRUModel(FastModel):
    """
    GRU model for streaming RUL prediction and anomaly detection.

    Features:
    - Streaming inference with hidden state
    - Uncertainty estimation via variance head
    - Efficient edge deployment
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
        device: str = 'auto',
        **kwargs
    ):
        """
        Initialize GRU model.

        Args:
            n_features: Number of input features
            hidden_size: GRU hidden dimension
            n_layers: Number of GRU layers
            dropout: Dropout probability
            output_size: Output dimension
            device: 'cpu', 'cuda', or 'auto'
        """
        super().__init__(n_features)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size

        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Build network
        self.network = GRUNet(
            n_features=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            output_size=output_size,
        ).to(self.device)

        # Hidden state for streaming
        self._h = None

        # Training state
        self.optimizer = None
        self.loss_fn = nn.MSELoss()

        # Running statistics for anomaly detection
        self._running_mean = None
        self._running_var = None
        self._n_samples = 0

    def reset_state(self) -> None:
        """Reset hidden state for new sequence."""
        self._h = None

    def step(self, x: np.ndarray) -> ModelOutput:
        """
        Process single timestep.

        Args:
            x: Single observation (n_features,)

        Returns:
            ModelOutput with prediction and uncertainty
        """
        self.network.eval()
        with torch.no_grad():
            # Prepare input: (1, 1, n_features)
            x_tensor = torch.tensor(x, dtype=torch.float32).view(1, 1, -1).to(self.device)

            # Forward pass
            output, variance, self._h = self.network(x_tensor, self._h)

            prediction = output.cpu().numpy().flatten()
            uncertainty = variance.cpu().numpy().flatten()[0]

            # Compute anomaly score based on prediction residual
            anomaly_score = self._compute_anomaly_score(x, prediction)

        return ModelOutput(
            prediction=prediction,
            uncertainty=float(np.sqrt(uncertainty)),  # Return std instead of var
            hidden_state=self._h.cpu().numpy() if self._h is not None else None,
            anomaly_score=anomaly_score,
        )

    def _compute_anomaly_score(self, x: np.ndarray, prediction: np.ndarray) -> float:
        """Compute anomaly score based on prediction and running statistics."""
        if self._running_mean is None:
            return 0.0

        # Mahalanobis-like distance
        deviation = x - self._running_mean
        if self._running_var is not None and np.all(self._running_var > 0):
            score = np.sqrt(np.sum(deviation ** 2 / self._running_var))
        else:
            score = np.sqrt(np.sum(deviation ** 2))

        return float(score)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        val_split: float = 0.1,
        patience: int = 10,
    ) -> 'GRUModel':
        """
        Train the model on sequences.

        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            y: Target values (n_samples,) or (n_samples, output_size)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Validation split ratio
            patience: Early stopping patience

        Returns:
            self
        """
        # Prepare data
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

        # Setup training
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        self.network.train()
        for epoch in range(epochs):
            # Shuffle and batch
            perm = torch.randperm(len(X_train))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                batch_idx = perm[i:i + batch_size]
                X_batch = X_train[batch_idx].to(self.device)
                y_batch = y_train[batch_idx].to(self.device)

                self.optimizer.zero_grad()
                output, variance, _ = self.network(X_batch)

                # Negative log likelihood loss with uncertainty
                loss = torch.mean(
                    0.5 * torch.log(variance + 1e-6) +
                    0.5 * (y_batch - output) ** 2 / (variance + 1e-6)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self.network.eval()
            with torch.no_grad():
                X_val_dev = X_val.to(self.device)
                y_val_dev = y_val.to(self.device)
                val_output, _, _ = self.network(X_val_dev)
                val_loss = self.loss_fn(val_output, y_val_dev).item()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            self.network.train()

        # Compute running statistics
        self._compute_statistics(X)
        self.is_fitted = True
        return self

    def _compute_statistics(self, X: np.ndarray):
        """Compute running statistics for anomaly detection."""
        # Flatten sequences
        X_flat = X.reshape(-1, self.n_features)
        self._running_mean = np.mean(X_flat, axis=0)
        self._running_var = np.var(X_flat, axis=0)
        self._n_samples = len(X_flat)

    def predict(self, X: np.ndarray) -> ModelOutput:
        """
        Predict on sequences.

        Args:
            X: Sequences (n_samples, seq_len, n_features)

        Returns:
            ModelOutput with predictions
        """
        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            output, variance, _ = self.network(X_tensor)

            predictions = output.cpu().numpy()
            uncertainties = variance.cpu().numpy()

        return ModelOutput(
            prediction=predictions.flatten(),
            uncertainty=float(np.mean(np.sqrt(uncertainties))),
        )

    def update(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Online update with new data.

        Args:
            X: New sequences (n_samples, seq_len, n_features)
            y: Target values (n_samples,)
        """
        if y is None:
            return

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)

        # Single gradient step
        self.network.train()
        self.optimizer.zero_grad()
        output, variance, _ = self.network(X_tensor)

        loss = torch.mean(
            0.5 * torch.log(variance + 1e-6) +
            0.5 * (y_tensor - output) ** 2 / (variance + 1e-6)
        )
        loss.backward()
        self.optimizer.step()
        self.network.eval()

        # Update statistics
        X_flat = X.reshape(-1, self.n_features)
        if self._running_mean is not None:
            alpha = len(X_flat) / (self._n_samples + len(X_flat))
            self._running_mean = (1 - alpha) * self._running_mean + alpha * np.mean(X_flat, axis=0)
            self._running_var = (1 - alpha) * self._running_var + alpha * np.var(X_flat, axis=0)
            self._n_samples += len(X_flat)

    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            'n_features': self.n_features,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'output_size': self.output_size,
            'network_state': self.network.state_dict(),
            'running_mean': self._running_mean.tolist() if self._running_mean is not None else None,
            'running_var': self._running_var.tolist() if self._running_var is not None else None,
            'n_samples': self._n_samples,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state."""
        self.network.load_state_dict(state['network_state'])
        if state['running_mean'] is not None:
            self._running_mean = np.array(state['running_mean'])
            self._running_var = np.array(state['running_var'])
            self._n_samples = state['n_samples']
        self.is_fitted = True

    def get_complexity(self) -> Dict[str, Any]:
        """Return model complexity metrics."""
        n_params = sum(p.numel() for p in self.network.parameters())
        return {
            'n_parameters': n_params,
            'memory_bytes': n_params * 4,  # float32
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
        }

    def to_onnx(self, filepath: str, seq_length: int = 30):
        """Export model to ONNX format for edge deployment."""
        dummy_input = torch.randn(1, seq_length, self.n_features).to(self.device)
        torch.onnx.export(
            self.network,
            dummy_input,
            filepath,
            input_names=['input'],
            output_names=['output', 'variance', 'hidden'],
            dynamic_axes={
                'input': {0: 'batch', 1: 'seq_len'},
                'output': {0: 'batch'},
                'variance': {0: 'batch'},
            },
        )
