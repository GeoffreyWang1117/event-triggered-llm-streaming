"""
Information-theoretic trigger based on information gain.
Triggers when semantic diagnosis would provide significant information.
"""
import numpy as np
from typing import Dict, Any, List
from collections import deque

from .base import EventTrigger, TriggerResult, TriggerReason
from ..models.base import ModelOutput
from ..data.base import StreamSample


class InformationGainTrigger(EventTrigger):
    """
    Information gain based trigger.

    Triggers LLM when the expected information gain from
    semantic diagnosis exceeds a threshold.

    Metrics:
    - Prediction entropy
    - KL divergence from baseline
    - Fisher information change
    """

    def __init__(
        self,
        entropy_threshold: float = 0.8,
        kl_threshold: float = 1.0,
        window_size: int = 50,
        cooldown_steps: int = 15,
        min_evidence_window: int = 10,
    ):
        """
        Initialize information gain trigger.

        Args:
            entropy_threshold: Normalized entropy threshold [0, 1]
            kl_threshold: KL divergence threshold
            window_size: Window for statistics estimation
            cooldown_steps: Minimum steps between triggers
            min_evidence_window: Minimum evidence window
        """
        super().__init__(cooldown_steps, min_evidence_window)
        self.entropy_threshold = entropy_threshold
        self.kl_threshold = kl_threshold
        self.window_size = window_size

        # Statistics tracking
        self._recent_scores: deque = deque(maxlen=window_size)
        self._baseline_distribution: Dict[str, float] = {}
        self._current_distribution: Dict[str, float] = {}

    def _estimate_entropy(self, uncertainty: float, n_bins: int = 10) -> float:
        """
        Estimate entropy from prediction uncertainty.

        For regression: use uncertainty as proxy for entropy
        Normalized to [0, 1]
        """
        # Simple normalization assuming uncertainty in reasonable range
        normalized = np.clip(uncertainty, 0, 1)
        return normalized

    def _estimate_kl_divergence(self) -> float:
        """
        Estimate KL divergence between current and baseline distributions.

        KL(P||Q) = sum_x P(x) log(P(x)/Q(x))
        """
        if len(self._recent_scores) < 20:
            return 0.0

        scores = np.array(self._recent_scores)

        # Split into baseline (older) and current (recent)
        split = len(scores) // 2
        baseline = scores[:split]
        current = scores[split:]

        # Estimate distributions via histogram
        n_bins = 10
        min_val = min(scores.min(), 0)
        max_val = max(scores.max(), 1)
        bins = np.linspace(min_val, max_val, n_bins + 1)

        p_baseline, _ = np.histogram(baseline, bins=bins, density=True)
        p_current, _ = np.histogram(current, bins=bins, density=True)

        # Add small epsilon for numerical stability
        eps = 1e-10
        p_baseline = p_baseline + eps
        p_current = p_current + eps

        # Normalize
        p_baseline = p_baseline / p_baseline.sum()
        p_current = p_current / p_current.sum()

        # KL divergence
        kl = np.sum(p_current * np.log(p_current / p_baseline))

        return float(kl)

    def _compute_gradient_magnitude(self) -> float:
        """
        Compute gradient magnitude of anomaly scores.
        High gradient indicates rapid change.
        """
        if len(self._recent_scores) < 5:
            return 0.0

        scores = np.array(list(self._recent_scores)[-10:])
        gradient = np.gradient(scores)
        return float(np.abs(gradient).mean())

    def evaluate(
        self,
        sample: StreamSample,
        model_output: ModelOutput,
    ) -> TriggerResult:
        """Evaluate information-theoretic criteria."""
        # Track scores
        self._recent_scores.append(model_output.anomaly_score)

        # Compute metrics
        entropy = self._estimate_entropy(model_output.uncertainty)
        kl_div = self._estimate_kl_divergence()
        gradient = self._compute_gradient_magnitude()

        # Combined information gain estimate
        # Weighted combination of entropy, KL divergence, and gradient
        info_gain = (
            0.4 * entropy / self.entropy_threshold +
            0.4 * kl_div / self.kl_threshold +
            0.2 * gradient
        )

        # Trigger conditions
        entropy_high = entropy > self.entropy_threshold
        kl_high = kl_div > self.kl_threshold

        should_trigger = entropy_high or kl_high

        if should_trigger:
            reason = TriggerReason.INFORMATION_GAIN
            confidence = min(1.0, info_gain)
        else:
            reason = TriggerReason.NONE
            confidence = info_gain

        return TriggerResult(
            should_trigger=should_trigger,
            reason=reason,
            confidence=confidence,
            statistics={
                'entropy': entropy,
                'entropy_threshold': self.entropy_threshold,
                'kl_divergence': kl_div,
                'kl_threshold': self.kl_threshold,
                'gradient': gradient,
                'info_gain': info_gain,
            },
        )

    def reset(self) -> None:
        """Reset trigger state."""
        super().reset()
        self._recent_scores.clear()

    def get_state(self) -> Dict[str, Any]:
        """Get trigger state."""
        return {
            'entropy_threshold': self.entropy_threshold,
            'kl_threshold': self.kl_threshold,
            'recent_scores': list(self._recent_scores),
        }
