"""
Hidden Markov Model for streaming diagnosis.
Lightweight model suitable for edge deployment.
"""
import numpy as np
from typing import Dict, Any, Optional
from .base import FastModel
from ..base import ModelOutput


class HMMModel(FastModel):
    """
    Gaussian HMM for time series anomaly detection and state estimation.

    This model is designed for:
    - Health state estimation (healthy, degrading, critical)
    - Transition detection
    - Anomaly scoring based on emission probability
    """

    def __init__(
        self,
        n_features: int,
        n_states: int = 3,
        covariance_type: str = 'diag',
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize HMM model.

        Args:
            n_features: Number of input features
            n_states: Number of hidden states
            covariance_type: Covariance type ('full', 'diag', 'spherical')
            n_iter: Maximum EM iterations
            random_state: Random seed
        """
        super().__init__(n_features)
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        # Initialize parameters
        self._init_params()

    def _init_params(self):
        """Initialize model parameters."""
        rng = np.random.default_rng(self.random_state)

        # Initial state distribution (uniform)
        self.pi = np.ones(self.n_states) / self.n_states

        # Transition matrix (encourage staying in same state)
        self.A = np.eye(self.n_states) * 0.9
        self.A += (1 - self.A.sum(axis=1, keepdims=True)) / (self.n_states - 1)
        np.fill_diagonal(self.A, 0.9)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        # Emission parameters (Gaussian)
        self.means = rng.standard_normal((self.n_states, self.n_features))
        if self.covariance_type == 'diag':
            self.covars = np.ones((self.n_states, self.n_features))
        else:
            self.covars = np.stack([np.eye(self.n_features)] * self.n_states)

        # Forward algorithm state
        self._alpha = None

    def _emission_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Compute emission probability P(x|state) for all states.

        Args:
            x: Observation (n_features,)

        Returns:
            Probabilities (n_states,)
        """
        probs = np.zeros(self.n_states)

        for s in range(self.n_states):
            diff = x - self.means[s]
            if self.covariance_type == 'diag':
                var = self.covars[s]
                log_prob = -0.5 * (np.sum(diff**2 / var) +
                                   np.sum(np.log(var)) +
                                   self.n_features * np.log(2 * np.pi))
            else:
                # Full covariance
                cov = self.covars[s]
                cov_inv = np.linalg.inv(cov)
                log_prob = -0.5 * (diff @ cov_inv @ diff +
                                   np.log(np.linalg.det(cov)) +
                                   self.n_features * np.log(2 * np.pi))

            probs[s] = np.exp(np.clip(log_prob, -700, 700))

        # Normalize to prevent underflow
        probs = probs / (probs.sum() + 1e-10)
        return probs

    def step(self, x: np.ndarray) -> ModelOutput:
        """
        Process single timestep using forward algorithm.

        Args:
            x: Single observation (n_features,)

        Returns:
            ModelOutput with state probabilities and anomaly score
        """
        # Compute emission probabilities
        emission = self._emission_prob(x)

        if self._alpha is None:
            # First step: use initial distribution
            self._alpha = self.pi * emission
        else:
            # Forward step: alpha_t = P(x_t|s_t) * sum_s_{t-1}(P(s_t|s_{t-1}) * alpha_{t-1})
            self._alpha = emission * (self.A.T @ self._alpha)

        # Normalize
        alpha_sum = self._alpha.sum()
        if alpha_sum > 0:
            self._alpha = self._alpha / alpha_sum
        else:
            self._alpha = np.ones(self.n_states) / self.n_states

        # Most likely state
        predicted_state = np.argmax(self._alpha)

        # Anomaly score: negative log likelihood
        anomaly_score = -np.log(alpha_sum + 1e-10)

        # Uncertainty: entropy of state distribution
        entropy = -np.sum(self._alpha * np.log(self._alpha + 1e-10))
        uncertainty = entropy / np.log(self.n_states)  # Normalize to [0, 1]

        return ModelOutput(
            prediction=np.array([predicted_state]),
            uncertainty=uncertainty,
            hidden_state=self._alpha.copy(),
            anomaly_score=anomaly_score,
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'HMMModel':
        """
        Fit HMM using Baum-Welch algorithm (EM).

        Args:
            X: Training sequences (n_samples, seq_len, n_features)
               or single sequence (seq_len, n_features)
            y: Ignored (unsupervised)

        Returns:
            self
        """
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        # Simplified EM for streaming context
        # In practice, use hmmlearn for full implementation
        for iteration in range(self.n_iter):
            # E-step: compute expected sufficient statistics
            gamma_sum = np.zeros((self.n_states, self.n_features))
            gamma_count = np.zeros(self.n_states)

            for seq in X:
                self.reset_state()
                for t, x in enumerate(seq):
                    self.step(x)
                    gamma_sum += np.outer(self._alpha, x)
                    gamma_count += self._alpha

            # M-step: update parameters
            for s in range(self.n_states):
                if gamma_count[s] > 0:
                    self.means[s] = gamma_sum[s] / gamma_count[s]

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> ModelOutput:
        """
        Predict on sequence.

        Args:
            X: Sequence (seq_len, n_features)

        Returns:
            ModelOutput with predictions for entire sequence
        """
        outputs = self.predict_stream(X)
        predictions = np.array([o.prediction[0] for o in outputs])
        avg_uncertainty = np.mean([o.uncertainty for o in outputs])
        avg_anomaly = np.mean([o.anomaly_score for o in outputs])

        return ModelOutput(
            prediction=predictions,
            uncertainty=avg_uncertainty,
            anomaly_score=avg_anomaly,
        )

    def update(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Online update of emission parameters.
        Simple exponential moving average update.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        learning_rate = 0.01
        for x in X:
            # Get current state estimate
            output = self.step(x)
            state_probs = output.hidden_state

            # Update means
            for s in range(self.n_states):
                self.means[s] = (1 - learning_rate * state_probs[s]) * self.means[s] + \
                                learning_rate * state_probs[s] * x

    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            'n_features': self.n_features,
            'n_states': self.n_states,
            'pi': self.pi.tolist(),
            'A': self.A.tolist(),
            'means': self.means.tolist(),
            'covars': self.covars.tolist(),
            'covariance_type': self.covariance_type,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state."""
        self.pi = np.array(state['pi'])
        self.A = np.array(state['A'])
        self.means = np.array(state['means'])
        self.covars = np.array(state['covars'])
        self.is_fitted = True

    def get_complexity(self) -> Dict[str, Any]:
        """Return model complexity metrics."""
        n_params = (
            self.n_states +  # pi
            self.n_states ** 2 +  # A
            self.n_states * self.n_features +  # means
            self.n_states * self.n_features  # covars (diag)
        )
        return {
            'n_parameters': n_params,
            'memory_bytes': n_params * 8,  # float64
            'flops_per_step': self.n_states ** 2 + self.n_states * self.n_features,
        }
