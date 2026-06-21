"""
Evidence Grounding for LLM Diagnosis.
Ensures LLM outputs are grounded in provided evidence.
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class GroundingScore:
    """Grounding quality assessment."""
    overall_score: float  # [0, 1]
    evidence_coverage: float  # How much evidence was used
    claim_support: float  # How well claims are supported
    hallucination_risk: float  # Risk of hallucination
    details: Dict[str, Any] = None


class EvidenceGrounder:
    """
    Evidence grounding module for LLM diagnosis.

    Ensures that:
    1. Diagnosis is based on provided evidence
    2. Claims are traceable to evidence
    3. Hallucinations are minimized
    """

    def __init__(
        self,
        min_evidence_support: float = 0.5,
        max_claim_novelty: float = 0.7,
    ):
        """
        Initialize evidence grounder.

        Args:
            min_evidence_support: Minimum evidence support required
            max_claim_novelty: Maximum novelty (too novel = potential hallucination)
        """
        self.min_evidence_support = min_evidence_support
        self.max_claim_novelty = max_claim_novelty

    def ground_diagnosis(
        self,
        diagnosis: str,
        evidence_window: np.ndarray,
        event_packet: Dict[str, Any],
    ) -> Tuple[str, GroundingScore]:
        """
        Ground diagnosis in evidence.

        Args:
            diagnosis: Raw diagnosis from LLM
            evidence_window: Evidence observations
            event_packet: Full event packet

        Returns:
            (grounded_diagnosis, grounding_score)
        """
        # Analyze evidence
        evidence_stats = self._analyze_evidence(evidence_window)

        # Check diagnosis grounding
        grounding_score = self._compute_grounding_score(
            diagnosis, evidence_stats, event_packet
        )

        # Add grounding annotations
        grounded_diagnosis = self._annotate_diagnosis(
            diagnosis, evidence_stats, grounding_score
        )

        return grounded_diagnosis, grounding_score

    def _analyze_evidence(self, evidence: np.ndarray) -> Dict[str, Any]:
        """Analyze evidence window for grounding."""
        if evidence is None or len(evidence) == 0:
            return {'empty': True}

        return {
            'empty': False,
            'n_samples': len(evidence),
            'mean': np.mean(evidence, axis=0).tolist(),
            'std': np.std(evidence, axis=0).tolist(),
            'min': np.min(evidence, axis=0).tolist(),
            'max': np.max(evidence, axis=0).tolist(),
            'trend': self._compute_trend(evidence),
            'anomaly_indicators': self._detect_anomaly_indicators(evidence),
        }

    def _compute_trend(self, evidence: np.ndarray) -> Dict[str, Any]:
        """Compute trend in evidence."""
        if len(evidence) < 3:
            return {'direction': 'unknown', 'magnitude': 0}

        # Simple linear trend per feature
        x = np.arange(len(evidence))
        trends = []
        for i in range(evidence.shape[1]):
            slope = np.polyfit(x, evidence[:, i], 1)[0]
            trends.append(slope)

        avg_trend = np.mean(trends)

        if avg_trend > 0.1:
            direction = 'increasing'
        elif avg_trend < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'

        return {
            'direction': direction,
            'magnitude': float(np.abs(avg_trend)),
            'feature_trends': trends,
        }

    def _detect_anomaly_indicators(self, evidence: np.ndarray) -> List[str]:
        """Detect potential anomaly indicators in evidence."""
        indicators = []

        if len(evidence) < 2:
            return indicators

        # Check for sudden changes
        diff = np.diff(evidence, axis=0)
        max_diff = np.max(np.abs(diff))
        if max_diff > 2 * np.std(evidence):
            indicators.append('sudden_change')

        # Check for increasing variance
        first_half = evidence[:len(evidence)//2]
        second_half = evidence[len(evidence)//2:]
        if np.std(second_half) > 1.5 * np.std(first_half):
            indicators.append('increasing_variance')

        # Check for drift
        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)
        if abs(mean_second - mean_first) > np.std(evidence):
            indicators.append('drift')

        return indicators

    def _compute_grounding_score(
        self,
        diagnosis: str,
        evidence_stats: Dict[str, Any],
        event_packet: Dict[str, Any],
    ) -> GroundingScore:
        """Compute grounding quality score."""
        if evidence_stats.get('empty', True):
            return GroundingScore(
                overall_score=0.0,
                evidence_coverage=0.0,
                claim_support=0.0,
                hallucination_risk=1.0,
            )

        # Evidence coverage: does diagnosis reference evidence?
        evidence_keywords = ['anomaly', 'score', 'trend', 'increase', 'decrease',
                           'deviation', 'change', 'observation', 'sensor']
        coverage = sum(1 for kw in evidence_keywords if kw.lower() in diagnosis.lower())
        evidence_coverage = min(1.0, coverage / 5)

        # Claim support: are specific claims supported?
        # Check if mentioned trends match evidence
        claim_support = 0.5  # Default moderate support

        trend = evidence_stats.get('trend', {})
        trend_dir = trend.get('direction', 'unknown')

        if trend_dir == 'increasing' and 'increas' in diagnosis.lower():
            claim_support += 0.25
        elif trend_dir == 'decreasing' and 'decreas' in diagnosis.lower():
            claim_support += 0.25

        if evidence_stats.get('anomaly_indicators'):
            indicators = evidence_stats['anomaly_indicators']
            for ind in indicators:
                if ind.replace('_', ' ') in diagnosis.lower():
                    claim_support += 0.1

        claim_support = min(1.0, claim_support)

        # Hallucination risk: specific numbers or facts not in evidence
        hallucination_risk = 0.2  # Base risk

        # Check for specific numbers that don't match evidence
        import re
        numbers = re.findall(r'\d+\.?\d*', diagnosis)
        anomaly_score = event_packet.get('anomaly_score', 0)
        for num in numbers:
            try:
                n = float(num)
                # Allow some tolerance
                if abs(n - anomaly_score) > 0.5 and n > 1:
                    hallucination_risk += 0.1
            except ValueError:
                continue

        hallucination_risk = min(1.0, hallucination_risk)

        # Overall score
        overall_score = (
            0.3 * evidence_coverage +
            0.4 * claim_support +
            0.3 * (1 - hallucination_risk)
        )

        return GroundingScore(
            overall_score=overall_score,
            evidence_coverage=evidence_coverage,
            claim_support=claim_support,
            hallucination_risk=hallucination_risk,
            details={
                'evidence_stats': evidence_stats,
                'matched_keywords': coverage,
            }
        )

    def _annotate_diagnosis(
        self,
        diagnosis: str,
        evidence_stats: Dict[str, Any],
        grounding_score: GroundingScore,
    ) -> str:
        """Add grounding annotations to diagnosis."""
        annotations = []

        # Add evidence summary
        if not evidence_stats.get('empty', True):
            trend = evidence_stats.get('trend', {})
            annotations.append(
                f"[Evidence: {evidence_stats['n_samples']} samples, "
                f"trend={trend.get('direction', 'unknown')}]"
            )

        # Add confidence note based on grounding
        if grounding_score.overall_score < 0.5:
            annotations.append("[Note: Limited evidence support]")
        elif grounding_score.hallucination_risk > 0.5:
            annotations.append("[Note: Verify specific claims]")

        if annotations:
            return diagnosis + "\n\n" + "\n".join(annotations)
        return diagnosis


class HistoricalRetriever:
    """
    Retrieves relevant historical events for context.
    Helps LLM make better diagnoses by providing similar past cases.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize retriever.

        Args:
            max_history: Maximum events to store
        """
        self.max_history = max_history
        self._history: List[Dict[str, Any]] = []
        self._embeddings: List[np.ndarray] = []

    def add_event(
        self,
        event_packet: Dict[str, Any],
        diagnosis_result: Dict[str, Any],
    ):
        """Add event to history."""
        self._history.append({
            'event': event_packet,
            'diagnosis': diagnosis_result,
            'timestamp': event_packet.get('trigger_timestamp', 0),
        })

        # Compute simple embedding (could use sentence transformer)
        embedding = self._compute_embedding(event_packet)
        self._embeddings.append(embedding)

        # Trim history
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
            self._embeddings = self._embeddings[-self.max_history:]

    def retrieve_similar(
        self,
        event_packet: Dict[str, Any],
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar historical events.

        Args:
            event_packet: Current event
            k: Number of events to retrieve

        Returns:
            List of similar historical events with diagnoses
        """
        if len(self._history) == 0:
            return []

        # Compute embedding for current event
        current_emb = self._compute_embedding(event_packet)

        # Compute similarities
        similarities = []
        for i, hist_emb in enumerate(self._embeddings):
            sim = self._cosine_similarity(current_emb, hist_emb)
            similarities.append((i, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        for i, sim in similarities[:k]:
            result = self._history[i].copy()
            result['similarity'] = sim
            results.append(result)

        return results

    def _compute_embedding(self, event_packet: Dict[str, Any]) -> np.ndarray:
        """Compute simple embedding for event."""
        # Simple feature-based embedding
        features = []

        # Anomaly score
        features.append(event_packet.get('anomaly_score', 0))

        # Uncertainty
        features.append(event_packet.get('uncertainty_estimate', 0))

        # Trigger reason encoding
        trigger_map = {
            'anomaly_score_exceeded': 1,
            'uncertainty_exceeded': 2,
            'cusum_change_detected': 3,
            'sprt_h0_rejected': 4,
            'information_gain_high': 5,
            'optimal_stopping_criterion': 6,
        }
        trigger_reason = event_packet.get('trigger_reason', 'unknown')
        features.append(trigger_map.get(trigger_reason, 0))

        # Evidence statistics (if available)
        evidence = event_packet.get('evidence_window', [])
        if isinstance(evidence, (list, np.ndarray)) and len(evidence) > 0:
            evidence = np.array(evidence)
            features.extend([np.mean(evidence), np.std(evidence)])
        else:
            features.extend([0, 0])

        return np.array(features)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
