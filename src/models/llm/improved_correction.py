"""
Improved LLM Correction Mechanism.

Fixes:
1. Better prompts with domain-specific knowledge
2. Output correction factors instead of absolute values
3. Confidence-based correction gating
4. Learning from correction history
"""
import os
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()


@dataclass
class CorrectionResult:
    """Result of LLM correction."""
    should_correct: bool
    correction_factor: float  # Multiply prediction by this
    correction_direction: str  # 'increase', 'decrease', 'none'
    confidence: float
    reasoning: str
    evidence_used: List[str]


@dataclass
class CorrectionHistory:
    """Track correction performance for learning."""
    original_prediction: float
    corrected_prediction: float
    true_value: Optional[float] = None
    correction_improved: Optional[bool] = None


class DomainKnowledge:
    """Domain-specific knowledge for grounding LLM corrections."""

    CMAPSS_KNOWLEDGE = """
## Turbofan Engine RUL Prediction Domain Knowledge

### Sensor Interpretation:
- T2 (Total temperature at fan inlet): Normal range 520-650R. High values indicate thermal stress.
- T24 (Total temperature at LPC outlet): Normal 620-650R. Deviation indicates compressor issues.
- T30 (Total temperature at HPC outlet): Normal 1500-1600R. Critical for engine health.
- T50 (Total temperature at LPT outlet): High values indicate turbine degradation.
- P2 (Pressure at fan inlet): Should be stable. Fluctuations indicate inlet disturbances.
- P15 (Total pressure in bypass duct): Relates to bypass ratio efficiency.
- P30 (Total pressure at HPC outlet): Critical for compression efficiency.
- Nf (Physical fan speed): Decreasing trend indicates degradation.
- Nc (Physical core speed): Should correlate with Nf. Divergence is concerning.
- epr (Engine pressure ratio): Key performance indicator. Decreasing = degradation.
- Ps30 (Static pressure at HPC outlet): Relates to compressor efficiency.
- phi (Ratio of fuel flow to Ps30): Efficiency indicator.
- NRf (Corrected fan speed): Normalized for comparison.
- NRc (Corrected core speed): Normalized for comparison.
- BPR (Bypass ratio): Should be stable. Changes indicate flow path issues.

### Degradation Patterns:
- Early degradation: Slight efficiency drops, increased temperatures
- Mid-life: Noticeable performance degradation, increased fuel consumption
- Near failure: Rapid changes in multiple sensors, unstable readings

### RUL Estimation Guidelines:
- If sensors show stable patterns: Model prediction likely accurate
- If temperature sensors trending up: Reduce RUL estimate by 10-20%
- If pressure ratios declining: Reduce RUL estimate by 15-25%
- If multiple anomalies: Consider significant reduction (20-40%)
- If readings are stable near historical norms: Trust model prediction
"""

    CICIDS_KNOWLEDGE = """
## Network Intrusion Detection Domain Knowledge

### Traffic Feature Interpretation:
- Flow Duration: Very short (<1s) or very long (>3600s) flows are suspicious
- Total Packets: Imbalanced forward/backward ratio may indicate scanning
- Packet Length: Unusually small or large packets can indicate attacks
- Flow Bytes/s: Extremely high rates indicate potential DDoS
- Flag Counts: High SYN without ACK indicates SYN flood
- Init_Win_bytes: Abnormal window sizes can indicate crafted packets

### Attack Patterns:
- Port Scan: Many short connections, incremental port numbers
- DDoS: High volume, similar packet sizes, single target
- Brute Force: Many failed authentications, same destination
- Web Attack: SQL injection patterns, unusual URL lengths
- Botnet: Periodic communication patterns, C2 signatures

### Classification Guidelines:
- High-volume short flows to many ports: Likely scan (80% confidence)
- Sustained high-bandwidth single target: Likely DDoS (85% confidence)
- Normal flow patterns with slight anomalies: Likely benign (70% confidence)
- Mixed patterns: Requires careful analysis, consider as suspicious
"""

    @classmethod
    def get_knowledge(cls, domain: str) -> str:
        if domain.lower() == 'cmapss':
            return cls.CMAPSS_KNOWLEDGE
        elif domain.lower() in ['cicids', 'ids', 'intrusion']:
            return cls.CICIDS_KNOWLEDGE
        else:
            return ""


class ImprovedLLMCorrector:
    """
    Improved LLM-based prediction corrector.

    Key improvements:
    1. Uses domain knowledge for grounded corrections
    2. Outputs correction factors (not absolute values)
    3. Learns from correction history
    4. Confidence-based gating
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        correction_confidence_threshold: float = 0.7,
        max_correction_factor: float = 0.3,  # Max 30% correction
        domain: str = "cmapss"
    ):
        self.model = model
        self.temperature = temperature
        self.correction_confidence_threshold = correction_confidence_threshold
        self.max_correction_factor = max_correction_factor
        self.domain = domain

        self.client = openai.OpenAI()
        self.domain_knowledge = DomainKnowledge.get_knowledge(domain)

        # Learning from history
        self._correction_history: deque = deque(maxlen=100)
        self._correction_success_rate = 0.5  # Prior

        # Statistics
        self._total_calls = 0
        self._total_corrections = 0
        self._successful_corrections = 0

    def correct_prediction(
        self,
        prediction: float,
        uncertainty: float,
        features: np.ndarray,
        context: Dict[str, Any] = None,
    ) -> Tuple[float, CorrectionResult]:
        """
        Correct a prediction using LLM.

        Args:
            prediction: Original model prediction
            uncertainty: Model uncertainty estimate
            features: Current feature values
            context: Additional context (e.g., trigger reason)

        Returns:
            corrected_prediction, CorrectionResult
        """
        self._total_calls += 1

        # Build prompt
        prompt = self._build_correction_prompt(prediction, uncertainty, features, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=500,
            )

            result = self._parse_response(response.choices[0].message.content, prediction)

            # Apply correction with gating
            corrected_prediction = self._apply_correction(prediction, result)

            return corrected_prediction, result

        except Exception as e:
            print(f"LLM correction error: {e}")
            return prediction, CorrectionResult(
                should_correct=False,
                correction_factor=1.0,
                correction_direction='none',
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                evidence_used=[]
            )

    def _get_system_prompt(self) -> str:
        return f"""You are an expert system for correcting machine learning predictions in {self.domain.upper()} domain.

Your task is to analyze whether a model's prediction should be adjusted based on the sensor data and context.

IMPORTANT GUIDELINES:
1. You must output a correction FACTOR (0.7-1.3), NOT an absolute value
2. Factor < 1.0 means decrease the prediction
3. Factor > 1.0 means increase the prediction
4. Factor = 1.0 means no correction needed
5. Only suggest corrections when you have HIGH CONFIDENCE
6. Cite specific sensor values as evidence

{self.domain_knowledge}

Respond in JSON format:
{{
    "should_correct": true/false,
    "correction_factor": 0.8-1.2,
    "correction_direction": "increase"/"decrease"/"none",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "evidence": ["sensor1: value indicates X", "sensor2: trend shows Y"]
}}
"""

    def _build_correction_prompt(
        self,
        prediction: float,
        uncertainty: float,
        features: np.ndarray,
        context: Dict[str, Any] = None,
    ) -> str:
        context = context or {}

        if self.domain == 'cmapss':
            sensor_names = [
                'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
                'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
                'PCNfR_dmd', 'W31', 'W32'
            ]
            feature_str = self._format_features(features, sensor_names)

            return f"""
## Current Prediction Analysis

**Model Prediction:** RUL = {prediction:.1f} cycles
**Model Uncertainty:** {uncertainty:.2f}
**Trigger Reason:** {context.get('trigger_reason', 'uncertainty_exceeded')}

**Current Sensor Readings:**
{feature_str}

**Historical Correction Success Rate:** {self._correction_success_rate:.1%}

Based on the sensor readings and domain knowledge, should this RUL prediction be adjusted?
If the sensors indicate more/less degradation than the model predicts, suggest a correction factor.
"""
        else:  # CICIDS
            return f"""
## Current Prediction Analysis

**Model Prediction:** {'Attack' if prediction == 1 else 'Normal'} (class={prediction})
**Model Confidence:** {1 - uncertainty:.2%}
**Trigger Reason:** {context.get('trigger_reason', 'low_confidence')}

**Current Network Features:**
{self._format_features(features, None)}

Based on the network traffic features, should this classification be changed?
"""

    def _format_features(self, features: np.ndarray, names: List[str] = None) -> str:
        if features.ndim > 1:
            features = features[-1] if features.ndim == 2 else features.flatten()[-len(names) if names else -20:]

        lines = []
        n_features = min(len(features), len(names) if names else 20)

        for i in range(n_features):
            name = names[i] if names and i < len(names) else f"feature_{i}"
            value = features[i]
            lines.append(f"  {name}: {value:.4f}")

        return "\n".join(lines)

    def _parse_response(self, response_text: str, original_prediction: float) -> CorrectionResult:
        """Parse LLM response into CorrectionResult."""
        import re

        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                correction_factor = float(data.get('correction_factor', 1.0))
                # Clamp correction factor
                correction_factor = max(
                    1 - self.max_correction_factor,
                    min(1 + self.max_correction_factor, correction_factor)
                )

                return CorrectionResult(
                    should_correct=data.get('should_correct', False),
                    correction_factor=correction_factor,
                    correction_direction=data.get('correction_direction', 'none'),
                    confidence=float(data.get('confidence', 0.5)),
                    reasoning=data.get('reasoning', ''),
                    evidence_used=data.get('evidence', [])
                )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse LLM response: {e}")

        return CorrectionResult(
            should_correct=False,
            correction_factor=1.0,
            correction_direction='none',
            confidence=0.0,
            reasoning="Failed to parse response",
            evidence_used=[]
        )

    def _apply_correction(
        self,
        prediction: float,
        result: CorrectionResult
    ) -> float:
        """Apply correction with confidence gating."""
        if not result.should_correct:
            return prediction

        if result.confidence < self.correction_confidence_threshold:
            # Low confidence: reduce correction magnitude
            reduced_factor = 1.0 + (result.correction_factor - 1.0) * result.confidence
            self._total_corrections += 1
            return prediction * reduced_factor

        self._total_corrections += 1
        return prediction * result.correction_factor

    def record_outcome(
        self,
        original_prediction: float,
        corrected_prediction: float,
        true_value: float
    ):
        """Record correction outcome for learning."""
        original_error = abs(original_prediction - true_value)
        corrected_error = abs(corrected_prediction - true_value)

        improved = corrected_error < original_error

        history = CorrectionHistory(
            original_prediction=original_prediction,
            corrected_prediction=corrected_prediction,
            true_value=true_value,
            correction_improved=improved
        )
        self._correction_history.append(history)

        if improved:
            self._successful_corrections += 1

        # Update success rate
        if len(self._correction_history) > 10:
            recent_success = sum(
                1 for h in list(self._correction_history)[-20:]
                if h.correction_improved
            )
            self._correction_success_rate = recent_success / min(20, len(self._correction_history))

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_calls': self._total_calls,
            'total_corrections': self._total_corrections,
            'successful_corrections': self._successful_corrections,
            'correction_success_rate': self._correction_success_rate,
            'correction_confidence_threshold': self.correction_confidence_threshold,
        }


class CorrectionEnsemble:
    """
    Ensemble of correction strategies for robust prediction adjustment.
    """

    def __init__(self, domain: str = "cmapss"):
        self.domain = domain

        # Multiple correction strategies
        self.llm_corrector = ImprovedLLMCorrector(
            domain=domain,
            correction_confidence_threshold=0.75
        )

        # Rule-based corrections as fallback
        self._rule_corrections = self._initialize_rules(domain)

    def _initialize_rules(self, domain: str) -> Dict[str, callable]:
        """Initialize rule-based correction functions."""
        if domain == 'cmapss':
            return {
                'high_temperature': lambda pred, feat: pred * 0.9 if feat[2] > 1550 else pred,
                'low_efficiency': lambda pred, feat: pred * 0.85 if feat[9] < 1.0 else pred,
                'stable_sensors': lambda pred, feat: pred  # Trust model
            }
        else:
            return {}

    def correct(
        self,
        prediction: float,
        uncertainty: float,
        features: np.ndarray,
        context: Dict[str, Any] = None,
        use_llm: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apply ensemble correction.

        Uses LLM when confidence is needed, rules as fallback.
        """
        corrections = []
        metadata = {'strategies_used': []}

        # Rule-based corrections (fast, always available)
        rule_pred = prediction
        for rule_name, rule_fn in self._rule_corrections.items():
            try:
                new_pred = rule_fn(rule_pred, features.flatten())
                if new_pred != rule_pred:
                    corrections.append(('rule', rule_name, new_pred))
                    rule_pred = new_pred
                    metadata['strategies_used'].append(f'rule:{rule_name}')
            except Exception:
                pass

        # LLM correction (when enabled and uncertainty is high)
        if use_llm and uncertainty > 0.3:
            llm_pred, llm_result = self.llm_corrector.correct_prediction(
                prediction, uncertainty, features, context
            )
            if llm_result.should_correct and llm_result.confidence > 0.6:
                corrections.append(('llm', llm_result.reasoning, llm_pred))
                metadata['strategies_used'].append('llm')
                metadata['llm_result'] = {
                    'factor': llm_result.correction_factor,
                    'confidence': llm_result.confidence,
                    'reasoning': llm_result.reasoning,
                }

        # Combine corrections
        if not corrections:
            return prediction, metadata

        # Use weighted average based on confidence
        if len(corrections) == 1:
            final_prediction = corrections[0][2]
        else:
            # Simple average for now
            final_prediction = np.mean([c[2] for c in corrections])

        metadata['original_prediction'] = prediction
        metadata['final_prediction'] = final_prediction
        metadata['correction_applied'] = True

        return final_prediction, metadata
