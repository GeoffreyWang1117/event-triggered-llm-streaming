"""
Improved Trigger Mechanisms for Event-Triggered LLM Invocation.

Fixes the fundamental issue: triggers should correlate with when LLM help is actually needed.

Trigger Types:
1. ErrorCorrelatedTrigger - Triggers based on estimated prediction error
2. PredictionChangeTrigger - Triggers when predictions change rapidly
3. CriticalZoneTrigger - Triggers when predictions enter critical regions
4. EnsembleDisagreementTrigger - Triggers when ensemble models disagree
5. CompositeTrigger - Combines multiple triggers
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
from abc import ABC, abstractmethod


@dataclass
class TriggerResult:
    """Result of trigger evaluation."""
    should_trigger: bool
    trigger_reason: str
    trigger_score: float  # 0-1, higher = more urgent
    metadata: Dict[str, Any]


class BaseTrigger(ABC):
    """Base class for triggers."""

    def __init__(self, name: str):
        self.name = name
        self._call_count = 0
        self._trigger_count = 0

    @abstractmethod
    def evaluate(
        self,
        prediction: float,
        uncertainty: Optional[float] = None,
        features: Optional[np.ndarray] = None,
        **kwargs
    ) -> TriggerResult:
        pass

    def reset(self):
        """Reset trigger state."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'total_calls': self._call_count,
            'total_triggers': self._trigger_count,
            'trigger_rate': self._trigger_count / max(1, self._call_count),
        }


class ErrorCorrelatedTrigger(BaseTrigger):
    """
    Trigger based on estimated prediction error.

    Uses a simple error estimator trained on validation data to predict
    when the model is likely to make large errors.
    """

    def __init__(
        self,
        error_threshold: float = 0.5,
        name: str = "error_correlated"
    ):
        super().__init__(name)
        self.error_threshold = error_threshold

        # Error estimator parameters (learned from validation)
        self._error_mean = None
        self._error_std = None
        self._feature_weights = None
        self._fitted = False

    def fit(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        true_values: np.ndarray,
    ):
        """
        Fit error estimator on validation data.

        Args:
            features: Feature matrix (n_samples, n_features)
            predictions: Model predictions
            true_values: Ground truth values
        """
        errors = np.abs(predictions.flatten() - true_values.flatten())

        self._error_mean = np.mean(errors)
        self._error_std = np.std(errors)

        # Learn which features correlate with high error
        # Simple approach: correlation between feature values and errors
        n_features = features.shape[1] if features.ndim > 1 else 1
        self._feature_weights = np.zeros(n_features)

        for i in range(n_features):
            feat_vals = features[:, i] if features.ndim > 1 else features
            corr = np.corrcoef(feat_vals, errors)[0, 1]
            if not np.isnan(corr):
                self._feature_weights[i] = abs(corr)

        # Normalize weights
        weight_sum = np.sum(self._feature_weights)
        if weight_sum > 0:
            self._feature_weights /= weight_sum

        self._fitted = True
        print(f"ErrorCorrelatedTrigger fitted: mean_error={self._error_mean:.3f}, std={self._error_std:.3f}")

    def evaluate(
        self,
        prediction: float,
        uncertainty: Optional[float] = None,
        features: Optional[np.ndarray] = None,
        **kwargs
    ) -> TriggerResult:
        self._call_count += 1

        if not self._fitted or features is None:
            return TriggerResult(
                should_trigger=False,
                trigger_reason="not_fitted",
                trigger_score=0.0,
                metadata={}
            )

        # Estimate error based on feature values
        if features.ndim == 1:
            feature_scores = np.abs(features) * self._feature_weights
        else:
            feature_scores = np.abs(features[-1]) * self._feature_weights

        estimated_error = np.sum(feature_scores) * self._error_std + self._error_mean * 0.5

        # Normalize to 0-1 score
        error_score = min(1.0, estimated_error / (self._error_mean + 2 * self._error_std))

        # Include uncertainty if available
        if uncertainty is not None:
            combined_score = 0.6 * error_score + 0.4 * min(1.0, uncertainty / (self._error_std * 2))
        else:
            combined_score = error_score

        should_trigger = combined_score > self.error_threshold

        if should_trigger:
            self._trigger_count += 1

        return TriggerResult(
            should_trigger=should_trigger,
            trigger_reason="estimated_high_error" if should_trigger else "normal",
            trigger_score=combined_score,
            metadata={
                'estimated_error': float(estimated_error),
                'error_score': float(error_score),
                'combined_score': float(combined_score),
            }
        )


class PredictionChangeTrigger(BaseTrigger):
    """
    Trigger when predictions change rapidly.

    Large changes in predictions often indicate model uncertainty
    or regime changes that need LLM verification.
    """

    def __init__(
        self,
        window_size: int = 5,
        change_threshold: float = 0.3,
        name: str = "prediction_change"
    ):
        super().__init__(name)
        self.window_size = window_size
        self.change_threshold = change_threshold

        self._prediction_history = deque(maxlen=window_size)
        self._change_mean = None
        self._change_std = None

    def fit(self, predictions: np.ndarray):
        """Fit on historical predictions to learn normal change rates."""
        changes = np.abs(np.diff(predictions.flatten()))
        self._change_mean = np.mean(changes)
        self._change_std = np.std(changes)
        print(f"PredictionChangeTrigger fitted: mean_change={self._change_mean:.3f}, std={self._change_std:.3f}")

    def evaluate(
        self,
        prediction: float,
        uncertainty: Optional[float] = None,
        features: Optional[np.ndarray] = None,
        **kwargs
    ) -> TriggerResult:
        self._call_count += 1

        self._prediction_history.append(prediction)

        if len(self._prediction_history) < 2:
            return TriggerResult(
                should_trigger=False,
                trigger_reason="insufficient_history",
                trigger_score=0.0,
                metadata={}
            )

        # Calculate change metrics
        recent_predictions = np.array(self._prediction_history)
        immediate_change = abs(recent_predictions[-1] - recent_predictions[-2])

        # Calculate trend change (second derivative)
        if len(recent_predictions) >= 3:
            first_diff = np.diff(recent_predictions)
            second_diff = np.diff(first_diff)
            trend_change = abs(second_diff[-1]) if len(second_diff) > 0 else 0
        else:
            trend_change = 0

        # Normalize scores
        if self._change_mean is not None and self._change_std > 0:
            change_score = min(1.0, immediate_change / (self._change_mean + 2 * self._change_std))
            trend_score = min(1.0, trend_change / (self._change_mean + self._change_std))
        else:
            change_score = min(1.0, immediate_change / (np.std(recent_predictions) + 1e-6))
            trend_score = 0.0

        combined_score = 0.7 * change_score + 0.3 * trend_score

        should_trigger = combined_score > self.change_threshold

        if should_trigger:
            self._trigger_count += 1

        return TriggerResult(
            should_trigger=should_trigger,
            trigger_reason="rapid_change" if should_trigger else "stable",
            trigger_score=combined_score,
            metadata={
                'immediate_change': float(immediate_change),
                'trend_change': float(trend_change),
                'change_score': float(change_score),
            }
        )

    def reset(self):
        self._prediction_history.clear()


class CriticalZoneTrigger(BaseTrigger):
    """
    Trigger when predictions enter critical zones.

    For RUL: trigger when predicted RUL is low
    For classification: trigger when confidence is low or predicted as attack
    """

    def __init__(
        self,
        critical_threshold: float = 30.0,  # For RUL
        low_confidence_threshold: float = 0.7,  # For classification
        mode: str = 'regression',  # 'regression' or 'classification'
        name: str = "critical_zone"
    ):
        super().__init__(name)
        self.critical_threshold = critical_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.mode = mode

    def evaluate(
        self,
        prediction: float,
        uncertainty: Optional[float] = None,
        features: Optional[np.ndarray] = None,
        confidence: Optional[float] = None,
        **kwargs
    ) -> TriggerResult:
        self._call_count += 1

        if self.mode == 'regression':
            # Trigger when prediction approaches critical threshold
            distance_to_critical = prediction - self.critical_threshold

            if distance_to_critical <= 0:
                trigger_score = 1.0
                trigger_reason = "in_critical_zone"
            elif distance_to_critical < self.critical_threshold:
                # Approaching critical zone
                trigger_score = 1.0 - (distance_to_critical / self.critical_threshold)
                trigger_reason = "approaching_critical"
            else:
                trigger_score = 0.0
                trigger_reason = "safe_zone"

            should_trigger = trigger_score > 0.5

        else:  # classification
            confidence = confidence or (1 - uncertainty if uncertainty else 0.5)

            if prediction == 1:  # Predicted as attack
                trigger_score = 0.8
                trigger_reason = "predicted_attack"
                should_trigger = True
            elif confidence < self.low_confidence_threshold:
                trigger_score = 1.0 - confidence
                trigger_reason = "low_confidence"
                should_trigger = True
            else:
                trigger_score = 0.0
                trigger_reason = "confident_normal"
                should_trigger = False

        if should_trigger:
            self._trigger_count += 1

        return TriggerResult(
            should_trigger=should_trigger,
            trigger_reason=trigger_reason,
            trigger_score=trigger_score,
            metadata={
                'prediction': float(prediction),
                'confidence': float(confidence) if confidence else None,
            }
        )


class EnsembleDisagreementTrigger(BaseTrigger):
    """
    Trigger when ensemble predictions disagree.

    Uses multiple model predictions (e.g., from MC Dropout) to detect
    when the model is uncertain about its prediction.
    """

    def __init__(
        self,
        disagreement_threshold: float = 0.3,
        name: str = "ensemble_disagreement"
    ):
        super().__init__(name)
        self.disagreement_threshold = disagreement_threshold

    def evaluate(
        self,
        prediction: float,
        uncertainty: Optional[float] = None,
        features: Optional[np.ndarray] = None,
        ensemble_predictions: Optional[np.ndarray] = None,
        **kwargs
    ) -> TriggerResult:
        self._call_count += 1

        if ensemble_predictions is None:
            # Use uncertainty as proxy for disagreement
            if uncertainty is not None:
                disagreement_score = min(1.0, uncertainty)
            else:
                return TriggerResult(
                    should_trigger=False,
                    trigger_reason="no_ensemble_data",
                    trigger_score=0.0,
                    metadata={}
                )
        else:
            # Calculate actual ensemble disagreement
            ensemble_std = np.std(ensemble_predictions)
            ensemble_range = np.max(ensemble_predictions) - np.min(ensemble_predictions)

            # Normalize by mean prediction
            mean_pred = np.mean(ensemble_predictions)
            if abs(mean_pred) > 1e-6:
                cv = ensemble_std / abs(mean_pred)  # Coefficient of variation
            else:
                cv = ensemble_std

            disagreement_score = min(1.0, cv)

        should_trigger = disagreement_score > self.disagreement_threshold

        if should_trigger:
            self._trigger_count += 1

        return TriggerResult(
            should_trigger=should_trigger,
            trigger_reason="high_disagreement" if should_trigger else "consensus",
            trigger_score=disagreement_score,
            metadata={
                'disagreement_score': float(disagreement_score),
            }
        )


class CompositeTrigger(BaseTrigger):
    """
    Combines multiple triggers with configurable logic.

    Modes:
    - 'any': Trigger if any sub-trigger fires
    - 'all': Trigger only if all sub-triggers fire
    - 'weighted': Weighted combination of trigger scores
    - 'priority': First trigger that fires wins
    """

    def __init__(
        self,
        triggers: List[BaseTrigger],
        mode: str = 'weighted',
        weights: Optional[List[float]] = None,
        threshold: float = 0.5,
        name: str = "composite"
    ):
        super().__init__(name)
        self.triggers = triggers
        self.mode = mode
        self.threshold = threshold

        if weights is None:
            self.weights = [1.0 / len(triggers)] * len(triggers)
        else:
            self.weights = weights

    def evaluate(
        self,
        prediction: float,
        uncertainty: Optional[float] = None,
        features: Optional[np.ndarray] = None,
        **kwargs
    ) -> TriggerResult:
        self._call_count += 1

        results = []
        for trigger in self.triggers:
            result = trigger.evaluate(prediction, uncertainty, features, **kwargs)
            results.append(result)

        if self.mode == 'any':
            should_trigger = any(r.should_trigger for r in results)
            triggered_reasons = [r.trigger_reason for r in results if r.should_trigger]
            trigger_reason = "; ".join(triggered_reasons) if triggered_reasons else "none"
            trigger_score = max(r.trigger_score for r in results)

        elif self.mode == 'all':
            should_trigger = all(r.should_trigger for r in results)
            trigger_reason = "all_triggers" if should_trigger else "not_all"
            trigger_score = min(r.trigger_score for r in results)

        elif self.mode == 'weighted':
            trigger_score = sum(w * r.trigger_score for w, r in zip(self.weights, results))
            should_trigger = trigger_score > self.threshold
            trigger_reason = f"weighted_score_{trigger_score:.3f}"

        elif self.mode == 'priority':
            should_trigger = False
            trigger_reason = "none"
            trigger_score = 0.0
            for result in results:
                if result.should_trigger:
                    should_trigger = True
                    trigger_reason = result.trigger_reason
                    trigger_score = result.trigger_score
                    break

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if should_trigger:
            self._trigger_count += 1

        return TriggerResult(
            should_trigger=should_trigger,
            trigger_reason=trigger_reason,
            trigger_score=trigger_score,
            metadata={
                'sub_results': [
                    {'name': t.name, 'score': r.trigger_score, 'triggered': r.should_trigger}
                    for t, r in zip(self.triggers, results)
                ]
            }
        )

    def reset(self):
        for trigger in self.triggers:
            trigger.reset()

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats['sub_trigger_stats'] = [t.get_stats() for t in self.triggers]
        return stats


def create_rul_trigger(
    critical_rul: float = 30.0,
    error_threshold: float = 0.4,
    change_threshold: float = 0.3,
) -> CompositeTrigger:
    """
    Factory function to create optimal trigger for RUL prediction.
    """
    triggers = [
        CriticalZoneTrigger(critical_threshold=critical_rul, mode='regression'),
        PredictionChangeTrigger(change_threshold=change_threshold),
        EnsembleDisagreementTrigger(disagreement_threshold=0.3),
    ]

    # Critical zone gets highest weight
    weights = [0.5, 0.3, 0.2]

    return CompositeTrigger(
        triggers=triggers,
        mode='weighted',
        weights=weights,
        threshold=0.4,
        name='rul_composite'
    )


def create_ids_trigger(
    confidence_threshold: float = 0.7,
) -> CompositeTrigger:
    """
    Factory function to create optimal trigger for intrusion detection.
    """
    triggers = [
        CriticalZoneTrigger(
            low_confidence_threshold=confidence_threshold,
            mode='classification'
        ),
        EnsembleDisagreementTrigger(disagreement_threshold=0.2),
    ]

    # Attack detection gets highest weight
    weights = [0.7, 0.3]

    return CompositeTrigger(
        triggers=triggers,
        mode='weighted',
        weights=weights,
        threshold=0.3,
        name='ids_composite'
    )
