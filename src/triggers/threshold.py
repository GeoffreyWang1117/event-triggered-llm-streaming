"""
Simple threshold-based event trigger.
Basic but interpretable trigger mechanism.
"""
import numpy as np
from typing import Dict, Any

from .base import EventTrigger, TriggerResult, TriggerReason
from ..models.base import ModelOutput
from ..data.base import StreamSample


class ThresholdTrigger(EventTrigger):
    """
    Threshold-based trigger for anomaly score and uncertainty.

    Triggers when:
    - Anomaly score exceeds threshold, OR
    - Model uncertainty exceeds threshold
    """

    def __init__(
        self,
        anomaly_threshold: float = 2.0,
        uncertainty_threshold: float = 0.5,
        cooldown_steps: int = 10,
        min_evidence_window: int = 5,
    ):
        """
        Initialize threshold trigger.

        Args:
            anomaly_threshold: Threshold for anomaly score
            uncertainty_threshold: Threshold for prediction uncertainty
            cooldown_steps: Minimum steps between triggers
            min_evidence_window: Minimum evidence window size
        """
        super().__init__(cooldown_steps, min_evidence_window)
        self.anomaly_threshold = anomaly_threshold
        self.uncertainty_threshold = uncertainty_threshold

        # Adaptive threshold tracking
        self._anomaly_history = []
        self._uncertainty_history = []
        self._window_size = 100

    def evaluate(
        self,
        sample: StreamSample,
        model_output: ModelOutput,
    ) -> TriggerResult:
        """Evaluate threshold conditions."""
        # Update history
        self._anomaly_history.append(model_output.anomaly_score)
        self._uncertainty_history.append(model_output.uncertainty)

        if len(self._anomaly_history) > self._window_size:
            self._anomaly_history = self._anomaly_history[-self._window_size:]
            self._uncertainty_history = self._uncertainty_history[-self._window_size:]

        # Check conditions
        anomaly_exceeded = model_output.anomaly_score > self.anomaly_threshold
        uncertainty_exceeded = model_output.uncertainty > self.uncertainty_threshold

        should_trigger = anomaly_exceeded or uncertainty_exceeded

        if anomaly_exceeded:
            reason = TriggerReason.ANOMALY_SCORE
            confidence = min(1.0, model_output.anomaly_score / self.anomaly_threshold)
        elif uncertainty_exceeded:
            reason = TriggerReason.UNCERTAINTY
            confidence = min(1.0, model_output.uncertainty / self.uncertainty_threshold)
        else:
            reason = TriggerReason.NONE
            confidence = 0.0

        return TriggerResult(
            should_trigger=should_trigger,
            reason=reason,
            confidence=confidence,
            statistics={
                'anomaly_score': model_output.anomaly_score,
                'anomaly_threshold': self.anomaly_threshold,
                'uncertainty': model_output.uncertainty,
                'uncertainty_threshold': self.uncertainty_threshold,
                'anomaly_mean': np.mean(self._anomaly_history),
                'anomaly_std': np.std(self._anomaly_history),
            },
        )

    def reset(self) -> None:
        """Reset trigger state."""
        self._steps_since_trigger = self.cooldown_steps
        self._evidence_buffer = []
        self._anomaly_history = []
        self._uncertainty_history = []

    def get_state(self) -> Dict[str, Any]:
        """Get trigger state."""
        return {
            'anomaly_threshold': self.anomaly_threshold,
            'uncertainty_threshold': self.uncertainty_threshold,
            'anomaly_history': self._anomaly_history,
            'uncertainty_history': self._uncertainty_history,
        }

    def adapt_thresholds(self, false_positive_rate: float = 0.05):
        """
        Adapt thresholds based on observed statistics.

        Args:
            false_positive_rate: Target false positive rate
        """
        if len(self._anomaly_history) < 50:
            return

        # Set thresholds at percentile
        percentile = (1 - false_positive_rate) * 100
        self.anomaly_threshold = np.percentile(self._anomaly_history, percentile)
        self.uncertainty_threshold = np.percentile(self._uncertainty_history, percentile)
