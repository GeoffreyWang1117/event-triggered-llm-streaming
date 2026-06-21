"""
Hybrid LLM Correction: Semantic Diagnosis + Learned Adjustment.

Key insight: LLMs excel at semantic understanding but struggle with numeric adjustments.
Solution: LLM provides diagnosis, learned model does correction.

Components:
1. LLM Diagnostician - Provides semantic analysis and correction direction
2. Learned Corrector - Trained to map features + diagnosis to correction factor
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
from dotenv import load_dotenv
import openai

load_dotenv()


@dataclass
class DiagnosisResult:
    """Result of LLM diagnosis (not correction)."""
    degradation_level: str  # 'low', 'moderate', 'high', 'critical'
    confidence: float
    key_indicators: List[str]
    reasoning: str
    suggested_direction: str  # 'trust_model', 'reduce_rul', 'increase_rul'


class LLMDiagnostician:
    """
    LLM provides semantic diagnosis, NOT numeric corrections.

    This is what LLMs are actually good at - understanding patterns
    and providing qualitative assessments.
    """

    DOMAIN_KNOWLEDGE = """
## Turbofan Engine Health Assessment

### Sensor Interpretation (Critical for RUL):
- T30 (HPC outlet temp): >1580R indicates significant wear
- T50 (LPT outlet temp): Rising trend = turbine degradation
- Nf/Nc divergence: Core-fan speed mismatch = mechanical issues
- epr decline: Engine pressure ratio drop = efficiency loss
- BPR change: Bypass ratio shift = flow path degradation

### Degradation Stages:
1. HEALTHY: All sensors within ±5% of nominal, stable trends
2. EARLY WEAR: Slight efficiency drops, T30 trending up 1-2%
3. MODERATE: Multiple sensors deviating, fuel consumption up
4. SEVERE: Rapid multi-sensor degradation, unstable readings
5. CRITICAL: Near-failure indicators, immediate attention needed

### Assessment Guidelines:
- If most sensors stable near nominal → Trust model prediction
- If temperatures elevated but stable → Model likely accurate
- If multiple sensors show acceleration → Model may be optimistic
- If readings are erratic → High uncertainty, reduce confidence
"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI()

    def diagnose(
        self,
        prediction: float,
        uncertainty: float,
        features: np.ndarray,
        feature_names: List[str] = None,
    ) -> DiagnosisResult:
        """Get semantic diagnosis from LLM."""

        prompt = self._build_prompt(prediction, uncertainty, features, feature_names)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=300,
            )

            return self._parse_response(response.choices[0].message.content)

        except Exception as e:
            print(f"LLM diagnosis error: {e}")
            return DiagnosisResult(
                degradation_level='unknown',
                confidence=0.0,
                key_indicators=[],
                reasoning=f"Error: {str(e)}",
                suggested_direction='trust_model'
            )

    def _get_system_prompt(self) -> str:
        return f"""You are an expert turbofan engine health analyst.

Your task: Assess engine degradation level based on sensor readings.
You are NOT asked to predict RUL - just assess current health status.

{self.DOMAIN_KNOWLEDGE}

Respond in JSON:
{{
    "degradation_level": "low/moderate/high/critical",
    "confidence": 0.0-1.0,
    "key_indicators": ["indicator1", "indicator2"],
    "reasoning": "Brief explanation",
    "suggested_direction": "trust_model/reduce_rul/increase_rul"
}}

IMPORTANT: Be conservative. Only suggest adjustments when you have HIGH confidence.
"""

    def _build_prompt(
        self,
        prediction: float,
        uncertainty: float,
        features: np.ndarray,
        feature_names: List[str] = None,
    ) -> str:
        if feature_names is None:
            feature_names = [
                'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
                'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
                'PCNfR_dmd', 'W31', 'W32'
            ]

        if features.ndim > 1:
            features = features[-1]

        sensor_lines = []
        for i, name in enumerate(feature_names[:len(features)]):
            sensor_lines.append(f"  {name}: {features[i]:.4f}")

        return f"""## Engine Health Assessment Request

Model's RUL Prediction: {prediction:.1f} cycles
Model Uncertainty: {uncertainty:.3f}

Current Sensor Readings:
{chr(10).join(sensor_lines)}

Based on the sensor readings, what is your assessment of the engine's degradation level?
Should we trust the model's prediction, or does the sensor data suggest adjustment?
"""

    def _parse_response(self, response_text: str) -> DiagnosisResult:
        import re

        try:
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return DiagnosisResult(
                    degradation_level=data.get('degradation_level', 'unknown'),
                    confidence=float(data.get('confidence', 0.5)),
                    key_indicators=data.get('key_indicators', []),
                    reasoning=data.get('reasoning', ''),
                    suggested_direction=data.get('suggested_direction', 'trust_model')
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse diagnosis: {e}")

        return DiagnosisResult(
            degradation_level='unknown',
            confidence=0.0,
            key_indicators=[],
            reasoning='Parse failed',
            suggested_direction='trust_model'
        )


class LearnedCorrector(nn.Module):
    """
    Learned correction model that maps:
    (features, prediction, diagnosis_encoding) -> correction_factor

    Trained on historical data where we know the true RUL.
    """

    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()

        # Diagnosis encoding: 4 levels + 3 directions = 7 dims
        diagnosis_dim = 7

        # Input: features + prediction + uncertainty + diagnosis
        input_dim = n_features + 2 + diagnosis_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()  # Output in [-1, 1], will be scaled to correction factor
        )

        self.max_correction = 0.3  # Max 30% correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output correction factor in [0.7, 1.3]."""
        raw = self.network(x)
        return 1.0 + raw * self.max_correction

    @staticmethod
    def encode_diagnosis(diagnosis: DiagnosisResult) -> np.ndarray:
        """Encode diagnosis result as feature vector."""
        # Degradation level one-hot
        level_encoding = np.zeros(4)
        level_map = {'low': 0, 'moderate': 1, 'high': 2, 'critical': 3}
        if diagnosis.degradation_level in level_map:
            level_encoding[level_map[diagnosis.degradation_level]] = 1

        # Direction one-hot
        direction_encoding = np.zeros(3)
        direction_map = {'trust_model': 0, 'reduce_rul': 1, 'increase_rul': 2}
        if diagnosis.suggested_direction in direction_map:
            direction_encoding[direction_map[diagnosis.suggested_direction]] = diagnosis.confidence

        return np.concatenate([level_encoding, direction_encoding])


class HybridCorrector:
    """
    Main hybrid correction system.

    1. LLM provides semantic diagnosis
    2. Learned model does numeric correction
    3. Falls back to simple rules if LLM unavailable
    """

    def __init__(
        self,
        n_features: int = 21,
        use_llm: bool = True,
        device: str = 'auto'
    ):
        self.n_features = n_features
        self.use_llm = use_llm

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Components
        if use_llm:
            self.diagnostician = LLMDiagnostician()
        else:
            self.diagnostician = None

        self.corrector = LearnedCorrector(n_features).to(self.device)

        # Training state
        self._is_trained = False
        self._training_history = []

        # Statistics
        self._total_calls = 0
        self._llm_calls = 0
        self._corrections_applied = 0

    def fit(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        true_values: np.ndarray,
        uncertainties: np.ndarray = None,
        epochs: int = 100,
        lr: float = 0.001,
    ):
        """
        Train the correction model on historical data.

        Args:
            features: (n_samples, n_features)
            predictions: Model predictions
            true_values: Ground truth RUL
            uncertainties: Model uncertainties (optional)
        """
        print("Training hybrid corrector...")

        n_samples = len(predictions)

        if uncertainties is None:
            uncertainties = np.ones(n_samples) * 0.5

        # Calculate target correction factors
        # correction_factor = true_value / prediction (clamped)
        with np.errstate(divide='ignore', invalid='ignore'):
            target_factors = true_values / (predictions + 1e-6)
            target_factors = np.clip(target_factors, 0.5, 2.0)
            # Normalize to [-1, 1] range for tanh output
            target_normalized = (target_factors - 1.0) / self.corrector.max_correction
            target_normalized = np.clip(target_normalized, -1, 1)

        # Create simple diagnosis encodings (without LLM for training)
        diagnosis_encodings = self._create_rule_based_diagnoses(
            features, predictions, true_values
        )

        # Prepare training data
        X = np.concatenate([
            features,
            predictions.reshape(-1, 1),
            uncertainties.reshape(-1, 1),
            diagnosis_encodings
        ], axis=1)

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(target_normalized).to(self.device)

        # Train
        optimizer = torch.optim.Adam(self.corrector.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.corrector.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass through just the network (before tanh scaling)
            raw_output = self.corrector.network(X_tensor).squeeze()
            loss = criterion(raw_output, y_tensor)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        self._is_trained = True

        # Evaluate training fit
        self.corrector.eval()
        with torch.no_grad():
            pred_factors = self.corrector(X_tensor).cpu().numpy().flatten()
            corrected = predictions * pred_factors

            orig_mae = np.mean(np.abs(predictions - true_values))
            corr_mae = np.mean(np.abs(corrected - true_values))

            print(f"Training fit - Original MAE: {orig_mae:.2f}, Corrected MAE: {corr_mae:.2f}")
            print(f"Improvement: {(orig_mae - corr_mae) / orig_mae * 100:.1f}%")

    def _create_rule_based_diagnoses(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        true_values: np.ndarray,
    ) -> np.ndarray:
        """Create diagnosis encodings using rules (for training without LLM)."""
        n_samples = len(predictions)
        encodings = np.zeros((n_samples, 7))

        for i in range(n_samples):
            feat = features[i] if features.ndim > 1 else features
            pred = predictions[i]
            true_val = true_values[i]

            # Determine degradation level based on prediction
            if pred > 100:
                level = 0  # low
            elif pred > 50:
                level = 1  # moderate
            elif pred > 20:
                level = 2  # high
            else:
                level = 3  # critical

            encodings[i, level] = 1

            # Determine suggested direction based on error
            error = pred - true_val
            if abs(error) < pred * 0.1:  # Within 10%
                direction = 0  # trust_model
            elif error > 0:  # Over-predicting
                direction = 1  # reduce_rul
            else:
                direction = 2  # increase_rul

            encodings[i, 4 + direction] = 0.8  # Confidence

        return encodings

    def correct(
        self,
        prediction: float,
        uncertainty: float,
        features: np.ndarray,
        use_llm: bool = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Correct a prediction using hybrid approach.

        Returns:
            corrected_prediction, metadata
        """
        self._total_calls += 1
        use_llm = use_llm if use_llm is not None else self.use_llm

        metadata = {
            'original_prediction': prediction,
            'used_llm': False,
            'diagnosis': None,
            'correction_factor': 1.0,
        }

        # Get diagnosis
        if use_llm and self.diagnostician is not None:
            try:
                diagnosis = self.diagnostician.diagnose(
                    prediction, uncertainty, features
                )
                self._llm_calls += 1
                metadata['used_llm'] = True
                metadata['diagnosis'] = {
                    'level': diagnosis.degradation_level,
                    'confidence': diagnosis.confidence,
                    'direction': diagnosis.suggested_direction,
                    'reasoning': diagnosis.reasoning,
                }

                # If LLM says trust model with high confidence, skip correction
                if (diagnosis.suggested_direction == 'trust_model' and
                    diagnosis.confidence > 0.7):
                    metadata['skipped_reason'] = 'llm_trusts_model'
                    return prediction, metadata

            except Exception as e:
                print(f"LLM diagnosis failed: {e}")
                diagnosis = self._rule_based_diagnosis(prediction, features)
        else:
            diagnosis = self._rule_based_diagnosis(prediction, features)

        # Apply learned correction
        if self._is_trained:
            diagnosis_encoding = LearnedCorrector.encode_diagnosis(diagnosis)

            if features.ndim > 1:
                features = features[-1]

            X = np.concatenate([
                features.flatten(),
                [prediction, uncertainty],
                diagnosis_encoding
            ])

            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)

            self.corrector.eval()
            with torch.no_grad():
                correction_factor = self.corrector(X_tensor).item()

            corrected = prediction * correction_factor

            metadata['correction_factor'] = correction_factor
            metadata['corrected_prediction'] = corrected

            self._corrections_applied += 1

            return corrected, metadata
        else:
            # Fallback to simple rules if not trained
            return self._apply_simple_rules(prediction, diagnosis), metadata

    def _rule_based_diagnosis(
        self,
        prediction: float,
        features: np.ndarray
    ) -> DiagnosisResult:
        """Simple rule-based diagnosis when LLM not available."""
        if prediction > 100:
            level = 'low'
        elif prediction > 50:
            level = 'moderate'
        elif prediction > 20:
            level = 'high'
        else:
            level = 'critical'

        return DiagnosisResult(
            degradation_level=level,
            confidence=0.5,
            key_indicators=[],
            reasoning='Rule-based diagnosis',
            suggested_direction='trust_model'
        )

    def _apply_simple_rules(
        self,
        prediction: float,
        diagnosis: DiagnosisResult
    ) -> float:
        """Simple rule-based correction fallback."""
        if diagnosis.suggested_direction == 'reduce_rul':
            return prediction * 0.9
        elif diagnosis.suggested_direction == 'increase_rul':
            return prediction * 1.1
        return prediction

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_calls': self._total_calls,
            'llm_calls': self._llm_calls,
            'corrections_applied': self._corrections_applied,
            'is_trained': self._is_trained,
        }
