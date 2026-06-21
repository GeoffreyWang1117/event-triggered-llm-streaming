"""
Adaptive threshold trigger with online learning.

Implements two adaptive methods:
- OGD (Online Gradient Descent): Updates threshold based on trigger usefulness feedback
- LinUCB (Contextual Bandits): Uses context features to adapt threshold
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from .base import EventTrigger, TriggerResult, TriggerReason
from ..models.base import ModelOutput
from ..data.base import StreamSample


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive threshold trigger."""
    initial_threshold: float = 0.5
    learning_rate: float = 0.01
    method: str = 'ogd'  # 'ogd' | 'linucb' | 'percentile'
    context_dim: int = 8
    projection_min: float = 0.01
    projection_max: float = 5.0
    # LinUCB specific
    linucb_alpha: float = 1.0
    # Percentile specific
    percentile_window: int = 200
    percentile_target: float = 95.0


class AdaptiveThresholdTrigger(EventTrigger):
    """
    Adaptive threshold trigger that learns optimal thresholds online.

    Supports three adaptation methods:
    - OGD: theta_{t+1} = proj[theta_t - eta_t * g_t]
    - LinUCB: Contextual bandits with UCB exploration
    - Percentile: Sliding-window percentile adaptation
    """

    def __init__(
        self,
        config: Optional[AdaptiveConfig] = None,
        cooldown_steps: int = 10,
        min_evidence_window: int = 5,
    ):
        super().__init__(cooldown_steps, min_evidence_window)
        self.config = config or AdaptiveConfig()
        self.threshold = self.config.initial_threshold
        self.method = self.config.method

        # Common state
        self._step_count = 0
        self._cumulative_loss = 0.0
        self._oracle_loss = 0.0
        self._trigger_history = []
        self._score_history = deque(maxlen=self.config.percentile_window)

        # OGD state
        self._ogd_sum_gradients_sq = 1e-8  # For AdaGrad-style learning rate

        # LinUCB state
        self._A = np.eye(self.config.context_dim)  # d x d
        self._b = np.zeros(self.config.context_dim)  # d
        self._A_inv = np.eye(self.config.context_dim)

        # Last context for deferred update
        self._last_context = None
        self._last_triggered = False

    def _build_context(self, sample: StreamSample, model_output: ModelOutput) -> np.ndarray:
        """Build context vector from sample and model output."""
        scores = list(self._score_history)
        recent_mean = np.mean(scores[-20:]) if len(scores) >= 20 else 0.0
        recent_std = np.std(scores[-20:]) if len(scores) >= 20 else 1.0
        trigger_rate = np.mean(self._trigger_history[-100:]) if self._trigger_history else 0.0
        time_since_last = self._steps_since_trigger / max(self.cooldown_steps, 1)

        trend = 0.0
        if len(scores) >= 10:
            recent = np.array(scores[-10:])
            trend = np.polyfit(range(len(recent)), recent, 1)[0]

        context = np.array([
            model_output.uncertainty,
            model_output.anomaly_score,
            trigger_rate,
            min(time_since_last, 10.0),
            trend,
            recent_mean,
            recent_std,
            self.threshold,
        ])

        # Pad or truncate to context_dim
        if len(context) < self.config.context_dim:
            context = np.pad(context, (0, self.config.context_dim - len(context)))
        else:
            context = context[:self.config.context_dim]

        return context

    def evaluate(
        self,
        sample: StreamSample,
        model_output: ModelOutput,
    ) -> TriggerResult:
        """Evaluate adaptive threshold condition."""
        self._step_count += 1
        score = model_output.anomaly_score + model_output.uncertainty
        self._score_history.append(score)

        context = self._build_context(sample, model_output)

        if self.method == 'linucb':
            threshold = self._linucb_predict(context)
        elif self.method == 'percentile':
            threshold = self._percentile_threshold()
        else:
            threshold = self.threshold

        should_trigger = score > threshold

        self._last_context = context
        self._last_triggered = should_trigger
        self._trigger_history.append(1.0 if should_trigger else 0.0)

        confidence = min(1.0, score / max(threshold, 1e-6)) if should_trigger else score / max(threshold, 1e-6)

        return TriggerResult(
            should_trigger=should_trigger,
            reason=TriggerReason.ANOMALY_SCORE if should_trigger else TriggerReason.NONE,
            confidence=float(np.clip(confidence, 0, 1)),
            statistics={
                'adaptive_threshold': float(threshold),
                'score': float(score),
                'method': self.method,
                'step_count': self._step_count,
                'regret': self.get_regret(),
            },
        )

    def update_threshold(self, was_useful: bool, context: Optional[np.ndarray] = None):
        """
        Update threshold based on whether the trigger decision was useful.

        Args:
            was_useful: Whether the trigger (or non-trigger) was correct
            context: Optional context override (uses last context if None)
        """
        if context is None:
            context = self._last_context

        # Compute loss: 1 if decision was wrong, 0 if correct
        loss = 0.0 if was_useful else 1.0
        self._cumulative_loss += loss

        if self.method == 'ogd':
            self._ogd_update(was_useful)
        elif self.method == 'linucb':
            self._linucb_update(was_useful, context)
        # percentile doesn't need explicit update

    def _ogd_update(self, was_useful: bool):
        """Online Gradient Descent threshold update."""
        # Gradient: if we triggered and it wasn't useful, increase threshold
        # If we didn't trigger and should have, decrease threshold
        if self._last_triggered and not was_useful:
            gradient = 1.0  # increase threshold (too sensitive)
        elif not self._last_triggered and not was_useful:
            gradient = -1.0  # decrease threshold (too conservative)
        else:
            gradient = 0.0  # correct decision

        # AdaGrad-style learning rate
        self._ogd_sum_gradients_sq += gradient ** 2
        eta_t = self.config.learning_rate / np.sqrt(self._ogd_sum_gradients_sq)

        # Update with projection
        self.threshold = np.clip(
            self.threshold - eta_t * gradient,
            self.config.projection_min,
            self.config.projection_max,
        )

    def _linucb_predict(self, context: np.ndarray) -> float:
        """LinUCB prediction: threshold = x^T theta + alpha * sqrt(x^T A^{-1} x)."""
        theta = self._A_inv @ self._b
        pred = context @ theta
        exploration = self.config.linucb_alpha * np.sqrt(context @ self._A_inv @ context)
        # Use UCB as threshold: higher UCB = more conservative (higher threshold)
        return float(np.clip(pred + exploration, self.config.projection_min, self.config.projection_max))

    def _linucb_update(self, was_useful: bool, context: Optional[np.ndarray] = None):
        """LinUCB update with reward feedback."""
        if context is None:
            return

        reward = 1.0 if was_useful else 0.0
        self._A += np.outer(context, context)
        self._b += reward * context
        # Update inverse using Sherman-Morrison
        ctx = context.reshape(-1, 1)
        self._A_inv -= (self._A_inv @ ctx @ ctx.T @ self._A_inv) / (1 + ctx.T @ self._A_inv @ ctx)

    def _percentile_threshold(self) -> float:
        """Compute percentile-based threshold from recent scores."""
        if len(self._score_history) < 20:
            return self.config.initial_threshold
        return float(np.percentile(list(self._score_history), self.config.percentile_target))

    def get_regret(self) -> float:
        """Get cumulative regret vs oracle."""
        return self._cumulative_loss - self._oracle_loss

    def reset(self) -> None:
        """Reset trigger state."""
        self._steps_since_trigger = self.cooldown_steps
        self._evidence_buffer = []
        self.threshold = self.config.initial_threshold
        self._step_count = 0
        self._cumulative_loss = 0.0
        self._oracle_loss = 0.0
        self._trigger_history = []
        self._score_history.clear()
        self._ogd_sum_gradients_sq = 1e-8
        self._A = np.eye(self.config.context_dim)
        self._b = np.zeros(self.config.context_dim)
        self._A_inv = np.eye(self.config.context_dim)

    def get_state(self) -> Dict[str, Any]:
        """Get trigger state."""
        return {
            'threshold': self.threshold,
            'method': self.method,
            'step_count': self._step_count,
            'cumulative_loss': self._cumulative_loss,
            'regret': self.get_regret(),
            'trigger_history': self._trigger_history[-100:],
        }
