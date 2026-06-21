"""
Base model classes.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelOutput:
    """Standard output from fast models."""
    prediction: np.ndarray  # Main prediction (RUL, anomaly score, etc.)
    uncertainty: float  # Prediction uncertainty
    hidden_state: Optional[np.ndarray] = None  # Internal state
    anomaly_score: float = 0.0  # Deviation from normal
    feature_importance: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata


class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseModel':
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelOutput:
        """Make predictions."""
        pass

    @abstractmethod
    def update(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Online update with new data."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current model state for serialization."""
        pass

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state."""
        pass

    def compute_anomaly_score(self, X: np.ndarray, prediction: np.ndarray) -> float:
        """
        Compute anomaly score based on prediction residual.
        Override in subclasses for custom scoring.
        """
        return 0.0
