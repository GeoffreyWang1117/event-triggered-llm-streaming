"""
Base class for fast models (edge-deployable).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from ..base import BaseModel, ModelOutput


class FastModel(BaseModel, ABC):
    """
    Base class for fast models that run on edge devices.

    These models are designed to be:
    - Low latency (ms-level inference)
    - Memory efficient (MB-level parameters)
    - Streaming-capable (can process data sample by sample)
    """

    def __init__(self, n_features: int, **kwargs):
        self.n_features = n_features
        self.is_fitted = False
        self._hidden_state = None

    @property
    def hidden_state(self) -> Optional[np.ndarray]:
        """Current hidden state for streaming inference."""
        return self._hidden_state

    def reset_state(self) -> None:
        """Reset hidden state for new sequence."""
        self._hidden_state = None

    @abstractmethod
    def step(self, x: np.ndarray) -> ModelOutput:
        """
        Process a single timestep in streaming mode.
        This is the primary interface for real-time inference.

        Args:
            x: Single observation (n_features,)

        Returns:
            ModelOutput with prediction and metadata
        """
        pass

    def predict_stream(self, X: np.ndarray) -> list:
        """
        Process a sequence of observations.

        Args:
            X: Sequence (seq_len, n_features)

        Returns:
            List of ModelOutput for each timestep
        """
        self.reset_state()
        outputs = []
        for t in range(X.shape[0]):
            output = self.step(X[t])
            outputs.append(output)
        return outputs

    @abstractmethod
    def get_complexity(self) -> Dict[str, Any]:
        """
        Return model complexity metrics.
        Used for edge deployment planning.
        """
        pass
