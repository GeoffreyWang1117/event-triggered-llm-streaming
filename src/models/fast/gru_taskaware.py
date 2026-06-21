"""
Task-Aware GRU Model for RUL Prediction.

Key insight: Instead of trying to calibrate model uncertainty to correlate with error,
we use a residual-based error estimator trained on validation data.

This model learns to predict WHEN it's likely to make errors, not just make predictions.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import FastModel
from ..base import ModelOutput


class ResidualErrorEstimator(nn.Module):
    """
    Learns to predict the model's own errors from input features.

    Trained on validation set where we know the actual errors.
    """

    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features + 1, hidden_size),  # +1 for prediction
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus(),  # Ensure positive error prediction
        )

    def forward(self, features: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Predict expected absolute error.

        Args:
            features: Input features (batch, n_features)
            prediction: Model's RUL prediction (batch, 1)

        Returns:
            Estimated absolute error (batch, 1)
        """
        x = torch.cat([features, prediction], dim=-1)
        return self.net(x)


class TaskAwareGRUNet(nn.Module):
    """GRU network with task-aware components."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Main prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Critical zone classifier: predicts if RUL < threshold
        self.critical_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns dict with:
            - prediction: RUL prediction
            - critical_prob: Probability of being in critical zone
            - hidden: Hidden state
            - features: Last hidden features (for error estimator)
        """
        gru_out, hidden = self.gru(x, h)
        last_output = gru_out[:, -1, :]

        prediction = self.predictor(last_output)
        critical_prob = self.critical_classifier(last_output)

        return {
            'prediction': prediction,
            'critical_prob': critical_prob,
            'hidden': hidden,
            'features': last_output,
        }


class TaskAwareGRUModel(FastModel):
    """
    Task-Aware GRU model with:
    1. Main RUL predictor
    2. Critical zone classifier (auxiliary task)
    3. Residual-based error estimator (trained on validation)

    The error estimator learns from the model's actual mistakes, providing
    a reliable signal for when LLM assistance is needed.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        critical_threshold: float = 30.0,
        device: str = 'auto',
        **kwargs
    ):
        super().__init__(n_features)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.critical_threshold = critical_threshold

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Main network
        self.network = TaskAwareGRUNet(
            n_features=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self.device)

        # Error estimator (trained separately on validation)
        self.error_estimator = ResidualErrorEstimator(
            n_features=n_features,
            hidden_size=32,
        ).to(self.device)

        self._h = None
        self.optimizer = None
        self._error_estimator_fitted = False

        # Thresholds learned from validation
        self._error_threshold_for_trigger = None
        self._critical_prob_threshold = 0.5

        # Statistics
        self._running_mean = None
        self._running_var = None

    def reset_state(self) -> None:
        self._h = None

    def step(self, x: np.ndarray) -> ModelOutput:
        """Process single timestep with task-aware outputs."""
        self.network.eval()
        self.error_estimator.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).view(1, 1, -1).to(self.device)

            output = self.network(x_tensor, self._h)
            self._h = output['hidden']

            prediction = output['prediction'].cpu().numpy().flatten()[0]
            critical_prob = output['critical_prob'].cpu().numpy().flatten()[0]

            # Estimate expected error
            if self._error_estimator_fitted:
                features = x_tensor[:, -1, :]  # Last timestep features
                pred_tensor = output['prediction']
                estimated_error = self.error_estimator(features, pred_tensor)
                estimated_error = estimated_error.cpu().numpy().flatten()[0]
            else:
                estimated_error = 0.0

            # Compute trigger score based on task-aware signals
            trigger_score = self._compute_trigger_score(
                prediction, critical_prob, estimated_error
            )

        return ModelOutput(
            prediction=np.array([prediction]),
            uncertainty=float(estimated_error),  # Use estimated error as "uncertainty"
            anomaly_score=float(critical_prob),
            metadata={
                'critical_prob': float(critical_prob),
                'estimated_error': float(estimated_error),
                'trigger_score': float(trigger_score),
                'in_critical_zone': prediction < self.critical_threshold,
            }
        )

    def _compute_trigger_score(
        self,
        prediction: float,
        critical_prob: float,
        estimated_error: float,
    ) -> float:
        """
        Compute trigger score based on multiple task-aware signals.

        Higher score = more likely to need LLM assistance.
        """
        scores = []

        # 1. Critical zone: high score if prediction is in or near critical zone
        if prediction < self.critical_threshold:
            critical_score = 1.0
        elif prediction < self.critical_threshold * 2:
            critical_score = 1.0 - (prediction - self.critical_threshold) / self.critical_threshold
        else:
            critical_score = 0.0
        scores.append(('critical_zone', critical_score, 0.4))

        # 2. Critical probability from classifier
        scores.append(('critical_classifier', critical_prob, 0.3))

        # 3. Estimated error (normalized)
        if self._error_threshold_for_trigger is not None:
            error_score = min(1.0, estimated_error / (self._error_threshold_for_trigger * 2))
        else:
            error_score = 0.0
        scores.append(('estimated_error', error_score, 0.3))

        # Weighted combination
        total_score = sum(score * weight for _, score, weight in scores)
        return total_score

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        val_split: float = 0.2,
        patience: int = 15,
    ) -> 'TaskAwareGRUModel':
        """
        Three-stage training:
        1. Train main predictor + critical classifier
        2. Train error estimator on validation errors
        3. Calibrate trigger thresholds
        """
        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        # Create critical zone labels
        critical_labels = (y < self.critical_threshold).astype(np.float32)
        critical_tensor = torch.tensor(critical_labels, dtype=torch.float32).unsqueeze(-1)

        # Split: train / val for error estimator
        n_val = int(len(X) * val_split)
        indices = torch.randperm(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
        X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]
        critical_train = critical_tensor[train_idx]

        # Stage 1: Train main network
        print("Stage 1: Training main predictor + critical classifier...")
        self._train_main_network(X_train, y_train, critical_train, epochs, batch_size, learning_rate, patience)

        # Stage 2: Get predictions on validation and train error estimator
        print("Stage 2: Training error estimator on validation errors...")
        self._train_error_estimator(X_val, y_val, epochs // 2, batch_size, learning_rate)

        # Stage 3: Calibrate trigger thresholds
        print("Stage 3: Calibrating trigger thresholds...")
        self._calibrate_thresholds(X_val, y_val)

        # Compute statistics
        self._compute_statistics(X)
        self.is_fitted = True
        return self

    def _train_main_network(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        critical_train: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
        patience: int,
    ):
        """Train main predictor and critical classifier jointly."""
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        best_loss = float('inf')
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
                critical_batch = critical_train[batch_idx].to(self.device)

                self.optimizer.zero_grad()
                output = self.network(X_batch)

                # Combined loss: prediction + critical classification
                pred_loss = mse_loss(output['prediction'], y_batch)
                critical_loss = bce_loss(output['critical_prob'], critical_batch)

                loss = pred_loss + 0.5 * critical_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    def _train_error_estimator(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
    ):
        """Train error estimator on validation set errors."""
        # First, get predictions on validation set
        self.network.eval()
        with torch.no_grad():
            X_val_dev = X_val.to(self.device)
            output = self.network(X_val_dev)
            predictions = output['prediction']
            features = X_val_dev[:, -1, :]  # Last timestep features

        # Compute actual errors
        y_val_dev = y_val.to(self.device)
        actual_errors = torch.abs(predictions - y_val_dev)

        # Train error estimator
        optimizer = torch.optim.Adam(self.error_estimator.parameters(), lr=lr)
        loss_fn = nn.L1Loss()

        for epoch in range(epochs):
            self.error_estimator.train()
            perm = torch.randperm(len(X_val))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_val), batch_size):
                batch_idx = perm[i:i + batch_size]
                feat_batch = features[batch_idx]
                pred_batch = predictions[batch_idx]
                error_batch = actual_errors[batch_idx]

                optimizer.zero_grad()
                estimated_error = self.error_estimator(feat_batch, pred_batch)
                loss = loss_fn(estimated_error, error_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Error estimator epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}")

        self._error_estimator_fitted = True

    def _calibrate_thresholds(self, X_val: torch.Tensor, y_val: torch.Tensor):
        """Calibrate trigger thresholds based on validation performance."""
        self.network.eval()
        self.error_estimator.eval()

        with torch.no_grad():
            X_val_dev = X_val.to(self.device)
            y_val_dev = y_val.to(self.device)

            output = self.network(X_val_dev)
            predictions = output['prediction']
            features = X_val_dev[:, -1, :]

            estimated_errors = self.error_estimator(features, predictions)
            actual_errors = torch.abs(predictions - y_val_dev)

            # Find error threshold that captures high-error samples
            estimated_errors_np = estimated_errors.cpu().numpy().flatten()
            actual_errors_np = actual_errors.cpu().numpy().flatten()

            # Set threshold at median estimated error
            self._error_threshold_for_trigger = float(np.percentile(estimated_errors_np, 70))

            # Evaluate correlation
            from scipy import stats
            corr, p_val = stats.pearsonr(estimated_errors_np, actual_errors_np)
            print(f"  Error estimator correlation: r={corr:.4f} (p={p_val:.4e})")
            print(f"  Error trigger threshold: {self._error_threshold_for_trigger:.2f}")

    def _compute_statistics(self, X: np.ndarray):
        X_flat = X.reshape(-1, self.n_features)
        self._running_mean = np.mean(X_flat, axis=0)
        self._running_var = np.var(X_flat, axis=0)

    def predict(self, X: np.ndarray) -> ModelOutput:
        """Batch prediction with task-aware outputs."""
        self.network.eval()
        self.error_estimator.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            output = self.network(X_tensor)

            predictions = output['prediction'].cpu().numpy().flatten()
            critical_probs = output['critical_prob'].cpu().numpy().flatten()

            if self._error_estimator_fitted:
                features = X_tensor[:, -1, :]
                estimated_errors = self.error_estimator(features, output['prediction'])
                estimated_errors = estimated_errors.cpu().numpy().flatten()
            else:
                estimated_errors = np.zeros_like(predictions)

        return ModelOutput(
            prediction=predictions,
            uncertainty=float(np.mean(estimated_errors)),
            metadata={
                'critical_probs': critical_probs.tolist(),
                'estimated_errors': estimated_errors.tolist(),
            }
        )

    def get_trigger_decisions(
        self,
        X: np.ndarray,
        trigger_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Get trigger decisions for batch of samples.

        Returns:
            predictions: Model predictions
            should_trigger: Boolean array of trigger decisions
            details: Dictionary with detailed trigger information
        """
        self.network.eval()
        self.error_estimator.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            output = self.network(X_tensor)

            predictions = output['prediction'].cpu().numpy().flatten()
            critical_probs = output['critical_prob'].cpu().numpy().flatten()

            if self._error_estimator_fitted:
                features = X_tensor[:, -1, :]
                estimated_errors = self.error_estimator(features, output['prediction'])
                estimated_errors = estimated_errors.cpu().numpy().flatten()
            else:
                estimated_errors = np.zeros_like(predictions)

        # Compute trigger scores
        trigger_scores = np.zeros(len(predictions))
        for i in range(len(predictions)):
            trigger_scores[i] = self._compute_trigger_score(
                predictions[i], critical_probs[i], estimated_errors[i]
            )

        should_trigger = trigger_scores > trigger_threshold

        details = {
            'trigger_scores': trigger_scores,
            'critical_probs': critical_probs,
            'estimated_errors': estimated_errors,
            'in_critical_zone': predictions < self.critical_threshold,
        }

        return predictions, should_trigger, details

    def update(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        if y is None:
            return

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)
        critical_tensor = torch.tensor(
            (y < self.critical_threshold).astype(np.float32),
            dtype=torch.float32
        ).unsqueeze(-1).to(self.device)

        self.network.train()
        self.optimizer.zero_grad()

        output = self.network(X_tensor)
        pred_loss = F.mse_loss(output['prediction'], y_tensor)
        critical_loss = F.binary_cross_entropy(output['critical_prob'], critical_tensor)
        loss = pred_loss + 0.5 * critical_loss

        loss.backward()
        self.optimizer.step()
        self.network.eval()

    def get_state(self) -> Dict[str, Any]:
        return {
            'n_features': self.n_features,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'critical_threshold': self.critical_threshold,
            'network_state': self.network.state_dict(),
            'error_estimator_state': self.error_estimator.state_dict(),
            'error_threshold_for_trigger': self._error_threshold_for_trigger,
            'error_estimator_fitted': self._error_estimator_fitted,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.network.load_state_dict(state['network_state'])
        self.error_estimator.load_state_dict(state['error_estimator_state'])
        self._error_threshold_for_trigger = state.get('error_threshold_for_trigger')
        self._error_estimator_fitted = state.get('error_estimator_fitted', False)
        self.is_fitted = True

    def get_complexity(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.network.parameters())
        n_params += sum(p.numel() for p in self.error_estimator.parameters())
        return {
            'n_parameters': n_params,
            'memory_bytes': n_params * 4,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'has_error_estimator': True,
            'has_critical_classifier': True,
        }
