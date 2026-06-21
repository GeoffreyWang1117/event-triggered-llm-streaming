"""
Optimal stopping based trigger.
Balances cost of LLM invocation vs. risk of delayed diagnosis.
"""
import numpy as np
from typing import Dict, Any, Optional
from collections import deque

from .base import EventTrigger, TriggerResult, TriggerReason
from ..models.base import ModelOutput
from ..data.base import StreamSample


class OptimalStoppingTrigger(EventTrigger):
    """
    Optimal stopping trigger based on Bellman equation.

    At each step, decides whether to:
    - Continue observing (accumulate cost, risk grows)
    - Stop and invoke LLM (pay fixed cost, reduce risk)

    The optimal policy minimizes:
        E[C_LLM * I(τ ≤ T) + C_risk * R(τ)]

    where:
    - C_LLM: Cost of LLM invocation
    - R(τ): Cumulative risk if stopping at time τ
    - T: Time horizon

    This is approximated using a threshold policy on cumulative risk.
    """

    def __init__(
        self,
        llm_cost: float = 1.0,
        risk_weight: float = 0.1,
        discount_factor: float = 0.99,
        risk_threshold: float = 2.0,
        window_size: int = 50,
        cooldown_steps: int = 20,
        min_evidence_window: int = 10,
    ):
        """
        Initialize optimal stopping trigger.

        Args:
            llm_cost: Fixed cost of LLM invocation (normalized)
            risk_weight: Weight for cumulative risk
            discount_factor: Discount factor for future risks (γ)
            risk_threshold: Threshold for triggering (derived from cost tradeoff)
            window_size: Window for value estimation
            cooldown_steps: Minimum steps between triggers
            min_evidence_window: Minimum evidence window
        """
        super().__init__(cooldown_steps, min_evidence_window)
        self.llm_cost = llm_cost
        self.risk_weight = risk_weight
        self.discount_factor = discount_factor
        self.risk_threshold = risk_threshold
        self.window_size = window_size

        # Cumulative risk tracking
        self._cumulative_risk = 0.0
        self._risk_history: deque = deque(maxlen=window_size)

        # Value function estimation
        self._continue_value = 0.0  # Estimated value of continuing
        self._stop_value = 0.0      # Estimated value of stopping

    def _instantaneous_risk(self, model_output: ModelOutput) -> float:
        """
        Compute instantaneous risk from fast model output.

        Risk is a combination of:
        - Anomaly score (how anomalous is current state)
        - Uncertainty (how unreliable is the prediction)
        """
        anomaly_component = model_output.anomaly_score * self.risk_weight
        uncertainty_component = model_output.uncertainty * self.risk_weight

        return anomaly_component + uncertainty_component

    def _estimate_value_functions(self) -> tuple:
        """
        Estimate value functions for continue vs. stop decisions.

        V_continue = γ * E[V_{t+1}] + instantaneous_cost
        V_stop = -C_LLM (LLM resolves uncertainty)

        Returns:
            (continue_value, stop_value)
        """
        if len(self._risk_history) < 10:
            return 0.0, -self.llm_cost

        risks = np.array(self._risk_history)

        # Expected future risk (simple average)
        expected_future_risk = np.mean(risks) * self.discount_factor

        # Trend: is risk increasing?
        if len(risks) > 5:
            recent_trend = np.polyfit(range(len(risks[-10:])), risks[-10:], 1)[0]
        else:
            recent_trend = 0

        # Value of continuing: expected future cost + trend penalty
        continue_value = -(expected_future_risk + max(0, recent_trend) * 2)

        # Value of stopping: fixed LLM cost, but risk reset
        stop_value = -self.llm_cost

        return float(continue_value), float(stop_value)

    def evaluate(
        self,
        sample: StreamSample,
        model_output: ModelOutput,
    ) -> TriggerResult:
        """Evaluate optimal stopping criterion."""
        # Compute instantaneous risk
        risk = self._instantaneous_risk(model_output)
        self._risk_history.append(risk)

        # Update cumulative risk with discounting
        self._cumulative_risk = (
            self.discount_factor * self._cumulative_risk + risk
        )

        # Estimate value functions
        continue_value, stop_value = self._estimate_value_functions()
        self._continue_value = continue_value
        self._stop_value = stop_value

        # Optimal stopping: stop if V_stop > V_continue
        # Or equivalently, if cumulative risk exceeds threshold
        value_based_stop = stop_value > continue_value
        threshold_stop = self._cumulative_risk > self.risk_threshold

        should_trigger = value_based_stop or threshold_stop

        if should_trigger:
            reason = TriggerReason.OPTIMAL_STOP
            confidence = min(1.0, self._cumulative_risk / self.risk_threshold)
            # Reset cumulative risk after triggering
            self._cumulative_risk = 0.0
        else:
            reason = TriggerReason.NONE
            confidence = self._cumulative_risk / self.risk_threshold

        return TriggerResult(
            should_trigger=should_trigger,
            reason=reason,
            confidence=confidence,
            statistics={
                'cumulative_risk': self._cumulative_risk,
                'risk_threshold': self.risk_threshold,
                'instantaneous_risk': risk,
                'continue_value': continue_value,
                'stop_value': stop_value,
                'llm_cost': self.llm_cost,
                'decision': 'stop' if should_trigger else 'continue',
            },
        )

    def reset(self) -> None:
        """Reset trigger state."""
        super().reset()
        self._cumulative_risk = 0.0
        self._risk_history.clear()
        self._continue_value = 0.0
        self._stop_value = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Get trigger state."""
        return {
            'cumulative_risk': self._cumulative_risk,
            'risk_history': list(self._risk_history),
            'llm_cost': self.llm_cost,
            'risk_weight': self.risk_weight,
            'risk_threshold': self.risk_threshold,
        }

    def set_llm_cost(self, cost: float):
        """Update LLM cost (e.g., based on current load)."""
        self.llm_cost = cost
        # Adjust threshold accordingly
        self.risk_threshold = cost * 2  # Simple heuristic
