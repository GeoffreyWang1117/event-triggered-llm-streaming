"""
CUSUM (Cumulative Sum) based trigger for change detection.
Sequential change detection algorithm.
"""
import numpy as np
from typing import Dict, Any, Optional

from .base import EventTrigger, TriggerResult, TriggerReason
from ..models.base import ModelOutput
from ..data.base import StreamSample


class CUSUMTrigger(EventTrigger):
    """
    CUSUM-based trigger for detecting changes in the data distribution.

    Implements the Cumulative Sum control chart for detecting
    small shifts in the process mean.

    Theory:
    - S_t^+ = max(0, S_{t-1}^+ + (x_t - μ_0 - k))  [upward shift]
    - S_t^- = max(0, S_{t-1}^- - (x_t - μ_0 + k))  [downward shift]
    - Alarm when S_t^+ > h or S_t^- > h
    """

    def __init__(
        self,
        slack: float = 0.5,  # k parameter (allowable slack)
        threshold: float = 5.0,  # h parameter (decision threshold)
        warmup_samples: int = 50,
        cooldown_steps: int = 20,
        min_evidence_window: int = 10,
    ):
        """
        Initialize CUSUM trigger.

        Args:
            slack: Allowable slack parameter (k)
            threshold: Decision threshold (h)
            warmup_samples: Samples for estimating baseline
            cooldown_steps: Minimum steps between triggers
            min_evidence_window: Minimum evidence window
        """
        super().__init__(cooldown_steps, min_evidence_window)
        self.slack = slack
        self.threshold = threshold
        self.warmup_samples = warmup_samples

        # CUSUM state
        self._s_plus = 0.0  # Upper CUSUM
        self._s_minus = 0.0  # Lower CUSUM

        # Baseline estimation
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
        self._warmup_data = []

    def evaluate(
        self,
        sample: StreamSample,
        model_output: ModelOutput,
    ) -> TriggerResult:
        """Evaluate CUSUM change detection."""
        # Use anomaly score as the monitored statistic
        x = model_output.anomaly_score

        # Warmup phase: collect baseline statistics
        if self._baseline_mean is None:
            self._warmup_data.append(x)
            if len(self._warmup_data) >= self.warmup_samples:
                self._baseline_mean = np.mean(self._warmup_data)
                self._baseline_std = np.std(self._warmup_data) + 1e-6
                self._warmup_data = []

            return TriggerResult(
                should_trigger=False,
                reason=TriggerReason.NONE,
                confidence=0.0,
                statistics={'phase': 'warmup', 'samples': len(self._warmup_data)},
            )

        # Standardize observation
        z = (x - self._baseline_mean) / self._baseline_std

        # Update CUSUM statistics
        self._s_plus = max(0, self._s_plus + z - self.slack)
        self._s_minus = max(0, self._s_minus - z - self.slack)

        # Check alarm condition
        alarm_plus = self._s_plus > self.threshold
        alarm_minus = self._s_minus > self.threshold
        should_trigger = alarm_plus or alarm_minus

        if should_trigger:
            # Reset after alarm
            s_max = max(self._s_plus, self._s_minus)
            confidence = min(1.0, s_max / self.threshold)
            self._s_plus = 0.0
            self._s_minus = 0.0
        else:
            confidence = max(self._s_plus, self._s_minus) / self.threshold

        return TriggerResult(
            should_trigger=should_trigger,
            reason=TriggerReason.CUSUM_ALARM if should_trigger else TriggerReason.NONE,
            confidence=confidence,
            statistics={
                's_plus': self._s_plus,
                's_minus': self._s_minus,
                'threshold': self.threshold,
                'baseline_mean': self._baseline_mean,
                'baseline_std': self._baseline_std,
                'current_z': z,
            },
        )

    def reset(self) -> None:
        """Reset trigger state."""
        super().reset()
        self._s_plus = 0.0
        self._s_minus = 0.0
        self._baseline_mean = None
        self._baseline_std = None
        self._warmup_data = []

    def get_state(self) -> Dict[str, Any]:
        """Get trigger state."""
        return {
            's_plus': self._s_plus,
            's_minus': self._s_minus,
            'baseline_mean': self._baseline_mean,
            'baseline_std': self._baseline_std,
            'slack': self.slack,
            'threshold': self.threshold,
        }

    def set_baseline(self, mean: float, std: float):
        """Manually set baseline statistics."""
        self._baseline_mean = mean
        self._baseline_std = std
        self._warmup_data = []
