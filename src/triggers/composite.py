"""
Composite trigger combining multiple trigger mechanisms.
Implements ensemble triggering with configurable logic.
"""
import numpy as np
from typing import Dict, Any, List, Optional

from .base import EventTrigger, TriggerResult, TriggerReason
from ..models.base import ModelOutput
from ..data.base import StreamSample


class CompositeTrigger(EventTrigger):
    """
    Composite trigger that combines multiple trigger mechanisms.

    Supports different combination modes:
    - 'any': Trigger if ANY sub-trigger fires (OR)
    - 'all': Trigger only if ALL sub-triggers fire (AND)
    - 'majority': Trigger if majority of sub-triggers fire
    - 'weighted': Weighted voting based on confidence scores
    """

    def __init__(
        self,
        triggers: List[EventTrigger],
        mode: str = 'weighted',
        weights: Optional[List[float]] = None,
        confidence_threshold: float = 0.6,
        cooldown_steps: int = 15,
        min_evidence_window: int = 10,
    ):
        """
        Initialize composite trigger.

        Args:
            triggers: List of sub-triggers
            mode: Combination mode ('any', 'all', 'majority', 'weighted')
            weights: Weights for each trigger (for weighted mode)
            confidence_threshold: Threshold for weighted mode
            cooldown_steps: Minimum steps between triggers
            min_evidence_window: Minimum evidence window
        """
        super().__init__(cooldown_steps, min_evidence_window)
        self.triggers = triggers
        self.mode = mode
        self.confidence_threshold = confidence_threshold

        if weights is None:
            self.weights = [1.0 / len(triggers)] * len(triggers)
        else:
            assert len(weights) == len(triggers)
            total = sum(weights)
            self.weights = [w / total for w in weights]

        # Track individual trigger results
        self._last_results: List[TriggerResult] = []

    def evaluate(
        self,
        sample: StreamSample,
        model_output: ModelOutput,
    ) -> TriggerResult:
        """Evaluate composite trigger condition."""
        # Evaluate all sub-triggers
        results = []
        for trigger in self.triggers:
            result = trigger.evaluate(sample, model_output)
            results.append(result)

        self._last_results = results

        # Combine based on mode
        if self.mode == 'any':
            should_trigger, confidence = self._combine_any(results)
        elif self.mode == 'all':
            should_trigger, confidence = self._combine_all(results)
        elif self.mode == 'majority':
            should_trigger, confidence = self._combine_majority(results)
        elif self.mode == 'weighted':
            should_trigger, confidence = self._combine_weighted(results)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Determine dominant reason
        if should_trigger:
            # Find the trigger with highest confidence
            max_conf_idx = np.argmax([r.confidence for r in results])
            reason = results[max_conf_idx].reason
        else:
            reason = TriggerReason.NONE

        # Merge statistics from all triggers
        merged_stats = {'mode': self.mode}
        for i, result in enumerate(results):
            trigger_name = type(self.triggers[i]).__name__
            merged_stats[f'{trigger_name}_triggered'] = result.should_trigger
            merged_stats[f'{trigger_name}_confidence'] = result.confidence

        return TriggerResult(
            should_trigger=should_trigger,
            reason=reason,
            confidence=confidence,
            statistics=merged_stats,
        )

    def _combine_any(self, results: List[TriggerResult]) -> tuple:
        """OR logic: trigger if any sub-trigger fires."""
        should_trigger = any(r.should_trigger for r in results)
        if should_trigger:
            confidence = max(r.confidence for r in results)
        else:
            confidence = max(r.confidence for r in results)
        return should_trigger, confidence

    def _combine_all(self, results: List[TriggerResult]) -> tuple:
        """AND logic: trigger only if all sub-triggers fire."""
        should_trigger = all(r.should_trigger for r in results)
        if should_trigger:
            confidence = min(r.confidence for r in results)
        else:
            confidence = np.mean([r.confidence for r in results])
        return should_trigger, confidence

    def _combine_majority(self, results: List[TriggerResult]) -> tuple:
        """Majority voting: trigger if more than half fire."""
        n_triggered = sum(1 for r in results if r.should_trigger)
        should_trigger = n_triggered > len(results) / 2
        confidence = n_triggered / len(results)
        return should_trigger, confidence

    def _combine_weighted(self, results: List[TriggerResult]) -> tuple:
        """Weighted voting based on confidence scores."""
        weighted_confidence = sum(
            w * r.confidence for w, r in zip(self.weights, results)
        )
        should_trigger = weighted_confidence > self.confidence_threshold
        return should_trigger, weighted_confidence

    def reset(self) -> None:
        """Reset all sub-triggers."""
        super().reset()
        for trigger in self.triggers:
            trigger.reset()
        self._last_results = []

    def get_state(self) -> Dict[str, Any]:
        """Get state of all sub-triggers."""
        return {
            'mode': self.mode,
            'weights': self.weights,
            'confidence_threshold': self.confidence_threshold,
            'sub_triggers': [
                {'type': type(t).__name__, 'state': t.get_state()}
                for t in self.triggers
            ],
        }

    def get_trigger_by_type(self, trigger_type: type) -> Optional[EventTrigger]:
        """Get a specific trigger by type."""
        for trigger in self.triggers:
            if isinstance(trigger, trigger_type):
                return trigger
        return None
