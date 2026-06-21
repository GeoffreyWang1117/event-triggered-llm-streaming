"""
Improved GRU model with proper uncertainty estimation.

Fixes:
1. MC Dropout for uncertainty estimation (replaces variance head)
2. Error prediction head trained on actual validation errors
3. Better calibration of uncertainty to prediction error
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import FastModel
from ..base import ModelOutput


class MCDropoutGRUNet(nn.Module):
    """
    GRU network with MC Dropout for uncertainty estimation.

    Key improvements:
    - Dropout stays active during inference for MC sampling
    - Error prediction head trained to predict actual errors
    - Ensemble-like uncertainty from multiple forward passes
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,  # Higher dropout for better MC estimates
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_rate = dropout

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Prediction head with dropout that stays active
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

        # Error prediction head - trained to predict |y - y_hat|
        self.error_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus(),  # Ensure positive error prediction
        )

        # MC Dropout layers that stay active during inference
        self.mc_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            output: Predictions (batch, output_size)
            predicted_error: Expected absolute error (batch, 1)
            hidden: New hidden state
            features: Hidden features for MC dropout
        """
        gru_out, hidden = self.gru(x, h)
        last_output = gru_out[:, -1, :]

        # Apply MC dropout (stays active in eval mode if called with enable_dropout)
        features = self.mc_dropout(last_output)

        output = self.fc(features)
        predicted_error = self.error_predictor(features)

        return output, predicted_error, hidden, features

    def mc_forward(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo forward pass for uncertainty estimation.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            mean_pred: Mean prediction across samples
            std_pred: Standard deviation (uncertainty)
            predicted_error: Error head prediction
        """
        self.train()  # Enable dropout

        predictions = []
        error_preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                output, pred_error, _, _ = self.forward(x)
                predictions.append(output)
                error_preds.append(pred_error)

        predictions = torch.stack(predictions, dim=0)  # (n_samples, batch, 1)
        error_preds = torch.stack(error_preds, dim=0)

        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        mean_error_pred = error_preds.mean(dim=0)

        return mean_pred, std_pred, mean_error_pred


class ImprovedGRUModel(FastModel):
    """
    Improved GRU model with MC Dropout uncertainty estimation.

    Key improvements over original:
    1. MC Dropout gives uncertainty correlated with actual errors
    2. Error prediction head trained on real validation errors
    3. Combined uncertainty score from MC variance + error predictor
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        mc_samples: int = 10,
        device: str = 'auto',
        **kwargs
    ):
        super().__init__(n_features)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.mc_samples = mc_samples
        self.dropout = dropout

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.network = MCDropoutGRUNet(
            n_features=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            output_size=output_size,
        ).to(self.device)

        self._h = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        self.error_loss_fn = nn.L1Loss()

        # Calibration parameters learned from validation
        self._uncertainty_scale = 1.0
        self._uncertainty_bias = 0.0

        # Running statistics
        self._running_mean = None
        self._running_var = None
        self._n_samples = 0

    def reset_state(self) -> None:
        self._h = None

    def step(self, x: np.ndarray) -> ModelOutput:
        """Process single timestep with MC Dropout uncertainty."""
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).view(1, 1, -1).to(self.device)

            # MC sampling for uncertainty
            mean_pred, std_pred, pred_error = self.network.mc_forward(x_tensor, self.mc_samples)

            prediction = mean_pred.cpu().numpy().flatten()
            mc_uncertainty = std_pred.cpu().numpy().flatten()[0]
            error_prediction = pred_error.cpu().numpy().flatten()[0]

            # Combined uncertainty: calibrated combination of MC std and error predictor
            combined_uncertainty = (
                self._uncertainty_scale * (0.5 * mc_uncertainty + 0.5 * error_prediction)
                + self._uncertainty_bias
            )

            anomaly_score = self._compute_anomaly_score(x, prediction)

        return ModelOutput(
            prediction=prediction,
            uncertainty=float(combined_uncertainty),
            hidden_state=None,
            anomaly_score=anomaly_score,
            metadata={
                'mc_uncertainty': float(mc_uncertainty),
                'error_prediction': float(error_prediction),
            }
        )

    def _compute_anomaly_score(self, x: np.ndarray, prediction: np.ndarray) -> float:
        if self._running_mean is None:
            return 0.0
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
        val_split: float = 0.15,
        patience: int = 15,
    ) -> 'ImprovedGRUModel':
        """
        Train with error prediction head.

        Two-phase training:
        1. Train prediction head
        2. Train error prediction head on actual errors
        """
        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)

        # Split validation (larger for error head training)
        n_val = int(len(X) * val_split)
        indices = torch.randperm(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
        X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

        # Phase 1: Train prediction head
        print("Phase 1: Training prediction head...")
        self._train_prediction_head(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, patience)

        # Phase 2: Train error prediction head
        print("Phase 2: Training error prediction head...")
        self._train_error_head(X_train, y_train, X_val, y_val, epochs // 2, batch_size, learning_rate * 0.5)

        # Phase 3: Calibrate uncertainty
        print("Phase 3: Calibrating uncertainty...")
        self._calibrate_uncertainty(X_val, y_val)

        # Compute statistics
        self._compute_statistics(X)
        self.is_fitted = True
        return self

    def _train_prediction_head(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
        patience: int,
    ):
        """Train main prediction head."""
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.network.train()
            perm = torch.randperm(len(X_train))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                batch_idx = perm[i:i + batch_size]
                X_batch = X_train[batch_idx].to(self.device)
                y_batch = y_train[batch_idx].to(self.device)

                self.optimizer.zero_grad()
                output, _, _, _ = self.network(X_batch)
                loss = self.loss_fn(output, y_batch)
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
                val_output, _, _, _ = self.network(X_val_dev)
                val_loss = self.loss_fn(val_output, y_val_dev).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    def _train_error_head(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
    ):
        """Train error prediction head on actual errors."""
        # Freeze prediction head, only train error predictor
        for param in self.network.gru.parameters():
            param.requires_grad = False
        for param in self.network.fc.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(self.network.error_predictor.parameters(), lr=lr)

        # Compute actual errors on training data
        self.network.eval()
        with torch.no_grad():
            X_train_dev = X_train.to(self.device)
            y_train_dev = y_train.to(self.device)
            predictions, _, _, _ = self.network(X_train_dev)
            actual_errors = torch.abs(predictions - y_train_dev)

        for epoch in range(epochs):
            self.network.train()
            perm = torch.randperm(len(X_train))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                batch_idx = perm[i:i + batch_size]
                X_batch = X_train[batch_idx].to(self.device)
                error_batch = actual_errors[batch_idx]

                optimizer.zero_grad()
                _, pred_error, _, _ = self.network(X_batch)
                loss = self.error_loss_fn(pred_error, error_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Error head epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}")

        # Unfreeze
        for param in self.network.gru.parameters():
            param.requires_grad = True
        for param in self.network.fc.parameters():
            param.requires_grad = True

    def _calibrate_uncertainty(self, X_val: torch.Tensor, y_val: torch.Tensor):
        """Calibrate uncertainty to correlate with actual errors."""
        self.network.eval()

        with torch.no_grad():
            X_val_dev = X_val.to(self.device)
            y_val_dev = y_val.to(self.device)

            # Get MC uncertainty and error predictions
            mean_pred, std_pred, pred_error = self.network.mc_forward(X_val_dev, self.mc_samples)

            # Actual errors
            actual_errors = torch.abs(mean_pred - y_val_dev).cpu().numpy().flatten()

            # Raw uncertainty
            raw_uncertainty = (0.5 * std_pred + 0.5 * pred_error).cpu().numpy().flatten()

            # Linear calibration: find scale and bias to match error distribution
            from scipy import stats
            if len(actual_errors) > 10:
                slope, intercept, r_value, _, _ = stats.linregress(raw_uncertainty, actual_errors)
                self._uncertainty_scale = max(0.1, slope)
                self._uncertainty_bias = max(0, intercept)
                print(f"  Calibration: scale={self._uncertainty_scale:.3f}, bias={self._uncertainty_bias:.3f}, r={r_value:.3f}")

    def _compute_statistics(self, X: np.ndarray):
        X_flat = X.reshape(-1, self.n_features)
        self._running_mean = np.mean(X_flat, axis=0)
        self._running_var = np.var(X_flat, axis=0)
        self._n_samples = len(X_flat)

    def predict(self, X: np.ndarray) -> ModelOutput:
        """Predict with MC Dropout uncertainty."""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        mean_pred, std_pred, pred_error = self.network.mc_forward(X_tensor, self.mc_samples)

        predictions = mean_pred.cpu().numpy()
        uncertainties = std_pred.cpu().numpy()
        error_preds = pred_error.cpu().numpy()

        # Combined calibrated uncertainty
        combined = self._uncertainty_scale * (0.5 * uncertainties + 0.5 * error_preds) + self._uncertainty_bias

        return ModelOutput(
            prediction=predictions.flatten(),
            uncertainty=float(np.mean(combined)),
            metadata={
                'mc_uncertainties': uncertainties.flatten().tolist(),
                'error_predictions': error_preds.flatten().tolist(),
            }
        )

    def get_calibrated_uncertainties(self, X: np.ndarray) -> np.ndarray:
        """Get per-sample calibrated uncertainties."""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        mean_pred, std_pred, pred_error = self.network.mc_forward(X_tensor, self.mc_samples)

        uncertainties = std_pred.cpu().numpy().flatten()
        error_preds = pred_error.cpu().numpy().flatten()

        combined = self._uncertainty_scale * (0.5 * uncertainties + 0.5 * error_preds) + self._uncertainty_bias
        return combined

    def update(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        if y is None:
            return

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)

        self.network.train()
        self.optimizer.zero_grad()
        output, _, _, _ = self.network(X_tensor)
        loss = self.loss_fn(output, y_tensor)
        loss.backward()
        self.optimizer.step()
        self.network.eval()

    def get_state(self) -> Dict[str, Any]:
        return {
            'n_features': self.n_features,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'mc_samples': self.mc_samples,
            'network_state': self.network.state_dict(),
            'uncertainty_scale': self._uncertainty_scale,
            'uncertainty_bias': self._uncertainty_bias,
            'running_mean': self._running_mean.tolist() if self._running_mean is not None else None,
            'running_var': self._running_var.tolist() if self._running_var is not None else None,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.network.load_state_dict(state['network_state'])
        self._uncertainty_scale = state.get('uncertainty_scale', 1.0)
        self._uncertainty_bias = state.get('uncertainty_bias', 0.0)
        if state['running_mean'] is not None:
            self._running_mean = np.array(state['running_mean'])
            self._running_var = np.array(state['running_var'])
        self.is_fitted = True

    def get_complexity(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.network.parameters())
        return {
            'n_parameters': n_params,
            'memory_bytes': n_params * 4,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'mc_samples': self.mc_samples,
        }
