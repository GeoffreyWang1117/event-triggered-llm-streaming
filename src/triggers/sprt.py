"""
Sequential Probability Ratio Test (SPRT) based trigger.
Optimal sequential hypothesis testing.
"""
import numpy as np
from typing import Dict, Any, Optional

from .base import EventTrigger, TriggerResult, TriggerReason
from ..models.base import ModelOutput
from ..data.base import StreamSample


class SPRTTrigger(EventTrigger):
    """
    SPRT-based trigger for sequential hypothesis testing.

    Tests:
    - H0: Fast model is sufficient (normal operation)
    - H1: LLM diagnosis is needed (anomaly/change detected)

    Uses the log-likelihood ratio:
    - L_t = sum_{i=1}^{t} log(p(x_i | H1) / p(x_i | H0))

    Decision:
    - Reject H0 (trigger LLM) if L_t >= log((1-beta)/alpha)
    - Accept H0 if L_t <= log(beta/(1-alpha))
    - Continue otherwise
    """

    def __init__(
        self,
        alpha: float = 0.05,  # Type I error (false alarm)
        beta: float = 0.10,   # Type II error (missed detection)
        h0_mean: Optional[float] = None,  # Mean under H0
        h1_mean: Optional[float] = None,  # Mean under H1
        std: float = 1.0,
        warmup_samples: int = 50,
        cooldown_steps: int = 20,
        min_evidence_window: int = 10,
    ):
        """
        Initialize SPRT trigger.

        Args:
            alpha: Type I error probability
            beta: Type II error probability
            h0_mean: Expected value under H0 (normal)
            h1_mean: Expected value under H1 (anomaly)
            std: Standard deviation (assumed equal under both hypotheses)
            warmup_samples: Samples for parameter estimation
            cooldown_steps: Minimum steps between triggers
            min_evidence_window: Minimum evidence window
        """
        super().__init__(cooldown_steps, min_evidence_window)
        self.alpha = alpha
        self.beta = beta
        self.std = std
        self.warmup_samples = warmup_samples

        # SPRT thresholds
        self.upper_threshold = np.log((1 - beta) / alpha)
        self.lower_threshold = np.log(beta / (1 - alpha))

        # Mean parameters
        self._h0_mean = h0_mean
        self._h1_mean = h1_mean

        # Log-likelihood ratio
        self._llr = 0.0

        # Warmup
        self._warmup_data = []

    def _log_likelihood_ratio(self, x: float) -> float:
        """
        Compute log-likelihood ratio for Gaussian distributions.

        LLR = log(p(x|H1)) - log(p(x|H0))
            = ((x - μ0)² - (x - μ1)²) / (2σ²)
            = (μ1 - μ0)(x - (μ0 + μ1)/2) / σ²
        """
        if self._h0_mean is None or self._h1_mean is None:
            return 0.0

        mu_diff = self._h1_mean - self._h0_mean
        return mu_diff * (x - (self._h0_mean + self._h1_mean) / 2) / (self.std ** 2)

    def evaluate(
        self,
        sample: StreamSample,
        model_output: ModelOutput,
    ) -> TriggerResult:
        """Evaluate SPRT hypothesis test."""
        x = model_output.anomaly_score

        # Warmup phase
        if self._h0_mean is None:
            self._warmup_data.append(x)
            if len(self._warmup_data) >= self.warmup_samples:
                # Estimate H0 parameters from "normal" data
                self._h0_mean = np.mean(self._warmup_data)
                self.std = np.std(self._warmup_data) + 1e-6
                # H1 mean: assume 2 sigma shift
                self._h1_mean = self._h0_mean + 2 * self.std
                self._warmup_data = []

            return TriggerResult(
                should_trigger=False,
                reason=TriggerReason.NONE,
                confidence=0.0,
                statistics={'phase': 'warmup', 'samples': len(self._warmup_data)},
            )

        # Update log-likelihood ratio
        llr_increment = self._log_likelihood_ratio(x)
        self._llr += llr_increment

        # Decision
        if self._llr >= self.upper_threshold:
            # Reject H0: trigger LLM
            should_trigger = True
            reason = TriggerReason.SPRT_REJECT
            confidence = min(1.0, self._llr / self.upper_threshold)
            self._llr = 0.0  # Reset after decision
        elif self._llr <= self.lower_threshold:
            # Accept H0: continue without LLM
            should_trigger = False
            reason = TriggerReason.NONE
            confidence = 0.0
            self._llr = 0.0  # Reset after decision
        else:
            # Continue sampling
            should_trigger = False
            reason = TriggerReason.NONE
            # Confidence based on proximity to thresholds
            range_size = self.upper_threshold - self.lower_threshold
            confidence = (self._llr - self.lower_threshold) / range_size

        return TriggerResult(
            should_trigger=should_trigger,
            reason=reason,
            confidence=confidence,
            statistics={
                'llr': self._llr,
                'upper_threshold': self.upper_threshold,
                'lower_threshold': self.lower_threshold,
                'h0_mean': self._h0_mean,
                'h1_mean': self._h1_mean,
                'std': self.std,
                'llr_increment': llr_increment,
            },
        )

    def reset(self) -> None:
        """Reset trigger state."""
        super().reset()
        self._llr = 0.0
        if self._h0_mean is None:  # Only reset if not manually set
            self._warmup_data = []

    def get_state(self) -> Dict[str, Any]:
        """Get trigger state."""
        return {
            'llr': self._llr,
            'h0_mean': self._h0_mean,
            'h1_mean': self._h1_mean,
            'std': self.std,
            'alpha': self.alpha,
            'beta': self.beta,
            'upper_threshold': self.upper_threshold,
            'lower_threshold': self.lower_threshold,
        }

    def set_hypotheses(self, h0_mean: float, h1_mean: float, std: float):
        """Manually set hypothesis parameters."""
        self._h0_mean = h0_mean
        self._h1_mean = h1_mean
        self.std = std
        self._warmup_data = []
