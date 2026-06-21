"""
Base classes for event-triggered mechanisms.

This module implements the theoretical framework for deciding
when to invoke the LLM for semantic diagnosis.

Key Theory:
- Event-Triggered Control: LLM invocation when state deviation exceeds threshold
- Optimal Stopping: Balance between observation cost and intervention benefit
- Sequential Hypothesis Testing: H0 (fast model sufficient) vs H1 (need LLM)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np

from ..models.base import ModelOutput
from ..data.base import EventPacket, StreamSample


class TriggerReason(Enum):
    """Enumeration of trigger reasons."""
    NONE = "none"
    ANOMALY_SCORE = "anomaly_score_exceeded"
    UNCERTAINTY = "uncertainty_exceeded"
    STATE_TRANSITION = "state_transition_detected"
    CUSUM_ALARM = "cusum_change_detected"
    SPRT_REJECT = "sprt_h0_rejected"
    INFORMATION_GAIN = "information_gain_high"
    OPTIMAL_STOP = "optimal_stopping_criterion"
    PERIODIC = "periodic_check"
    MANUAL = "manual_trigger"


@dataclass
class TriggerResult:
    """Result from trigger evaluation."""
    should_trigger: bool
    reason: TriggerReason
    confidence: float  # Confidence in trigger decision [0, 1]
    statistics: Dict[str, float] = field(default_factory=dict)
    evidence: Optional[np.ndarray] = None  # Evidence window


class EventTrigger(ABC):
    """
    Abstract base class for event-triggered LLM invocation.

    The trigger decides when the fast model's output is insufficient
    and semantic diagnosis via LLM is necessary.
    """

    def __init__(
        self,
        cooldown_steps: int = 10,
        min_evidence_window: int = 5,
    ):
        """
        Initialize trigger.

        Args:
            cooldown_steps: Minimum steps between consecutive triggers
            min_evidence_window: Minimum evidence window size for LLM
        """
        self.cooldown_steps = cooldown_steps
        self.min_evidence_window = min_evidence_window
        self._steps_since_trigger = cooldown_steps  # Allow immediate first trigger
        self._evidence_buffer: List[np.ndarray] = []

    @abstractmethod
    def evaluate(
        self,
        sample: StreamSample,
        model_output: ModelOutput,
    ) -> TriggerResult:
        """
        Evaluate whether to trigger LLM invocation.

        Args:
            sample: Current stream sample
            model_output: Output from fast model

        Returns:
            TriggerResult indicating whether to trigger
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset trigger state."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get trigger state for serialization."""
        pass

    def step(
        self,
        sample: StreamSample,
        model_output: ModelOutput,
    ) -> TriggerResult:
        """
        Process one step and decide on triggering.
        Handles cooldown and evidence buffering.

        Args:
            sample: Current stream sample
            model_output: Output from fast model

        Returns:
            TriggerResult
        """
        # Update evidence buffer
        self._evidence_buffer.append(sample.features)
        if len(self._evidence_buffer) > self.min_evidence_window * 2:
            self._evidence_buffer = self._evidence_buffer[-self.min_evidence_window * 2:]

        self._steps_since_trigger += 1

        # Check cooldown
        if self._steps_since_trigger < self.cooldown_steps:
            return TriggerResult(
                should_trigger=False,
                reason=TriggerReason.NONE,
                confidence=0.0,
                statistics={'cooldown_remaining': self.cooldown_steps - self._steps_since_trigger},
            )

        # Evaluate trigger condition
        result = self.evaluate(sample, model_output)

        if result.should_trigger:
            # Add evidence window to result
            result.evidence = np.array(self._evidence_buffer[-self.min_evidence_window:])
            self._steps_since_trigger = 0

        return result

    def create_event_packet(
        self,
        sample: StreamSample,
        model_output: ModelOutput,
        trigger_result: TriggerResult,
    ) -> EventPacket:
        """
        Create event packet for LLM diagnosis.

        Args:
            sample: Current sample
            model_output: Fast model output
            trigger_result: Trigger evaluation result

        Returns:
            EventPacket for LLM
        """
        return EventPacket(
            trigger_reason=trigger_result.reason.value,
            anomaly_score=model_output.anomaly_score,
            evidence_window=trigger_result.evidence if trigger_result.evidence is not None
                           else np.array(self._evidence_buffer[-self.min_evidence_window:]),
            system_context={
                'trigger_statistics': trigger_result.statistics,
                'trigger_confidence': trigger_result.confidence,
                'sample_metadata': sample.metadata,
            },
            uncertainty_estimate=model_output.uncertainty,
            trigger_timestamp=sample.timestamp,
            fast_model_prediction=model_output.prediction,
            feature_importance=model_output.feature_importance,
        )
