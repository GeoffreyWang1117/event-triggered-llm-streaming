"""
Bayesian trigger using Beta-Binomial conjugate prior.

Maintains a posterior P(anomaly | observations) and triggers
when the posterior probability exceeds a credible threshold.
"""
import numpy as np
from typing import Dict, Any, Tuple, List
from collections import deque

from .base import EventTrigger, TriggerResult, TriggerReason
from ..models.base import ModelOutput
from ..data.base import StreamSample


class BayesianTrigger(EventTrigger):
    """
    Bayesian trigger using Beta-Binomial conjugate prior.

    Maintains posterior P(anomaly_rate | data) ~ Beta(alpha, beta).
    Triggers when the posterior probability that the anomaly rate
    exceeds a critical threshold is above a confidence level.

    The Beta-Binomial model provides:
    - Natural uncertainty quantification via credible intervals
    - Sequential Bayesian updating (O(1) per step)
    - Calibrated probability estimates
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        anomaly_threshold: float = 0.5,
        credible_level: float = 0.95,
        trigger_prob: float = 0.7,
        window_size: int = 100,
        decay_factor: float = 0.995,
        cooldown_steps: int = 10,
        min_evidence_window: int = 5,
    ):
        """
        Initialize Bayesian trigger.

        Args:
            prior_alpha: Beta prior alpha (pseudo-count for anomalies)
            prior_beta: Beta prior beta (pseudo-count for normal)
            anomaly_threshold: Threshold on anomaly score to consider as anomaly
            credible_level: Credible interval level (e.g., 0.95 for 95%)
            trigger_prob: Posterior probability threshold for triggering
            window_size: Window for ECE computation
            decay_factor: Exponential decay for posterior (handles non-stationarity)
            cooldown_steps: Minimum steps between triggers
            min_evidence_window: Minimum evidence window
        """
        super().__init__(cooldown_steps, min_evidence_window)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.anomaly_threshold = anomaly_threshold
        self.credible_level = credible_level
        self.trigger_prob = trigger_prob
        self.window_size = window_size
        self.decay_factor = decay_factor

        # Posterior parameters
        self.alpha = prior_alpha
        self.beta_param = prior_beta  # avoid shadowing beta

        # Tracking
        self._step_count = 0
        self._predicted_probs = deque(maxlen=window_size)
        self._observed_labels = deque(maxlen=window_size)
        self._score_history = deque(maxlen=window_size)

    def evaluate(
        self,
        sample: StreamSample,
        model_output: ModelOutput,
    ) -> TriggerResult:
        """Evaluate Bayesian trigger condition."""
        self._step_count += 1
        score = model_output.anomaly_score
        self._score_history.append(score)

        # Classify current observation as anomaly or normal
        is_anomaly = score > self.anomaly_threshold

        # Apply decay to handle non-stationarity
        self.alpha = self.decay_factor * self.alpha + (1 - self.decay_factor) * self.prior_alpha
        self.beta_param = self.decay_factor * self.beta_param + (1 - self.decay_factor) * self.prior_beta

        # Bayesian update: Beta(alpha, beta) posterior
        if is_anomaly:
            self.alpha += 1.0
        else:
            self.beta_param += 1.0

        # Compute posterior statistics
        posterior_mean, posterior_var = self.get_posterior()
        lower, upper = self.get_credible_interval(1 - self.credible_level)

        # Track for calibration
        self._predicted_probs.append(posterior_mean)
        if sample.label is not None:
            # Use label as ground truth if available
            true_anomaly = float(sample.label < 30) if isinstance(sample.label, (int, float)) else 0.0
            self._observed_labels.append(true_anomaly)
        else:
            self._observed_labels.append(float(is_anomaly))

        # Trigger decision: fire if posterior probability of high anomaly rate
        # exceeds trigger_prob (using the lower credible bound for conservatism)
        should_trigger = lower > self.trigger_prob or posterior_mean > self.trigger_prob * 1.5

        if should_trigger:
            reason = TriggerReason.ANOMALY_SCORE
            confidence = float(np.clip(posterior_mean, 0, 1))
        else:
            reason = TriggerReason.NONE
            confidence = float(np.clip(posterior_mean / self.trigger_prob, 0, 1))

        return TriggerResult(
            should_trigger=should_trigger,
            reason=reason,
            confidence=confidence,
            statistics={
                'posterior_mean': float(posterior_mean),
                'posterior_var': float(posterior_var),
                'credible_lower': float(lower),
                'credible_upper': float(upper),
                'alpha': float(self.alpha),
                'beta': float(self.beta_param),
                'anomaly_score': float(score),
                'is_anomaly': bool(is_anomaly),
                'step_count': self._step_count,
            },
        )

    def get_posterior(self) -> Tuple[float, float]:
        """
        Get posterior mean and variance.

        Returns:
            (mean, variance) of Beta(alpha, beta) posterior
        """
        total = self.alpha + self.beta_param
        mean = self.alpha / total
        var = (self.alpha * self.beta_param) / (total ** 2 * (total + 1))
        return float(mean), float(var)

    def get_credible_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Get credible interval for the anomaly rate.

        Args:
            alpha: Significance level (0.05 for 95% CI)

        Returns:
            (lower, upper) bounds of the credible interval
        """
        from scipy import stats
        dist = stats.beta(self.alpha, self.beta_param)
        lower = dist.ppf(alpha / 2)
        upper = dist.ppf(1 - alpha / 2)
        return float(lower), float(upper)

    def calibration_error(self, true_labels: List[float] = None) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            true_labels: Optional external ground truth. If None, uses tracked labels.

        Returns:
            ECE value (lower is better)
        """
        if true_labels is not None:
            probs = list(self._predicted_probs)[:len(true_labels)]
            labels = true_labels[:len(probs)]
        else:
            probs = list(self._predicted_probs)
            labels = list(self._observed_labels)

        if len(probs) < 10:
            return 1.0

        probs = np.array(probs)
        labels = np.array(labels)

        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            bin_confidence = probs[mask].mean()
            bin_accuracy = labels[mask].mean()
            ece += mask.sum() / len(probs) * abs(bin_accuracy - bin_confidence)

        return float(ece)

    def reset(self) -> None:
        """Reset trigger state."""
        self._steps_since_trigger = self.cooldown_steps
        self._evidence_buffer = []
        self.alpha = self.prior_alpha
        self.beta_param = self.prior_beta
        self._step_count = 0
        self._predicted_probs.clear()
        self._observed_labels.clear()
        self._score_history.clear()

    def get_state(self) -> Dict[str, Any]:
        """Get trigger state."""
        mean, var = self.get_posterior()
        return {
            'alpha': self.alpha,
            'beta': self.beta_param,
            'posterior_mean': mean,
            'posterior_var': var,
            'step_count': self._step_count,
            'anomaly_threshold': self.anomaly_threshold,
            'trigger_prob': self.trigger_prob,
        }
