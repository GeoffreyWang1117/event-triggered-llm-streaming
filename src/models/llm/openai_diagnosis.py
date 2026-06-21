"""
Real LLM Diagnosis Engine using OpenAI API.
Provides grounded semantic diagnosis for streaming robot systems.
"""
import os
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(v) for v in obj]
    return obj

# Load environment variables
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Run: pip install openai")


@dataclass
class DiagnosisResult:
    """Result of LLM diagnosis."""
    diagnosis: str
    confidence: float
    severity: str  # low, medium, high, critical
    recommended_actions: List[str]
    evidence_summary: str
    grounding_score: float  # 0-1, how well grounded in evidence
    latency_ms: float
    tokens_used: int
    raw_response: str


@dataclass
class EventPacket:
    """Event packet sent to LLM for diagnosis."""
    timestamp: int
    evidence_window: List[Dict[str, Any]]  # Recent observations
    anomaly_scores: List[float]
    uncertainty_scores: List[float]
    fast_model_prediction: Any
    trigger_reason: str
    context: Dict[str, Any]  # Domain-specific context


class OpenAIDiagnosisEngine:
    """
    Real LLM diagnosis engine using OpenAI API.
    Provides grounded semantic diagnosis with evidence verification.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",  # Cost-effective for streaming
        temperature: float = 0.3,  # Lower for more consistent diagnosis
        max_tokens: int = 500,
        timeout: float = 30.0,
    ):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI package not available")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Statistics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_latency = 0.0

    def diagnose(
        self,
        event_packet: EventPacket,
        domain: str = "general"
    ) -> DiagnosisResult:
        """
        Perform semantic diagnosis on event packet.

        Args:
            event_packet: Evidence and context for diagnosis
            domain: Domain type (cmapss, geolife, cicids, general)

        Returns:
            DiagnosisResult with grounded diagnosis
        """
        # Build grounded prompt
        system_prompt = self._build_system_prompt(domain)
        user_prompt = self._build_user_prompt(event_packet, domain)

        # Call OpenAI API
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            raw_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            # Update statistics
            self.total_calls += 1
            self.total_tokens += tokens_used
            self.total_latency += latency_ms

            # Parse JSON response
            result = self._parse_response(raw_response, event_packet, latency_ms, tokens_used)
            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return DiagnosisResult(
                diagnosis=f"Error: {str(e)}",
                confidence=0.0,
                severity="unknown",
                recommended_actions=["Check system connectivity"],
                evidence_summary="Diagnosis failed",
                grounding_score=0.0,
                latency_ms=latency_ms,
                tokens_used=0,
                raw_response=str(e)
            )

    def _build_system_prompt(self, domain: str) -> str:
        """Build system prompt for specific domain."""
        base_prompt = """You are an expert diagnostic system for streaming robot/IoT systems.
Your task is to analyze sensor data and provide grounded, evidence-based diagnosis.

CRITICAL RULES:
1. Your diagnosis MUST be grounded in the provided evidence
2. Do NOT speculate beyond what the data supports
3. Reference specific sensor values or patterns in your diagnosis
4. Provide actionable recommendations

You must respond in JSON format with these fields:
{
    "diagnosis": "Clear description of the detected issue",
    "confidence": 0.0-1.0,
    "severity": "low|medium|high|critical",
    "recommended_actions": ["action1", "action2"],
    "evidence_summary": "Summary of evidence supporting the diagnosis",
    "key_indicators": ["indicator1", "indicator2"]
}"""

        domain_prompts = {
            "cmapss": """
DOMAIN: Turbofan Engine Degradation (NASA CMAPSS)
- You are monitoring turbofan engine health
- Key sensors: Temperature, Pressure, Fan Speed, Fuel Flow
- Goal: Detect degradation patterns and predict remaining useful life
- Critical: Late detection of failure is dangerous""",

            "geolife": """
DOMAIN: GPS Trajectory Anomaly Detection (GeoLife)
- You are monitoring GPS trajectories for anomalies
- Key features: Position, Speed, Acceleration, Heading Change
- Goal: Detect unusual movement patterns
- Consider: Traffic, stops, route deviations""",

            "cicids": """
DOMAIN: Network Intrusion Detection (CIC-IDS)
- You are monitoring network traffic for attacks
- Key features: Flow duration, packet sizes, protocol flags
- Attack types: DDoS, DoS, Brute Force, Port Scan
- Critical: Fast detection prevents damage""",

            "general": """
DOMAIN: General Streaming System Monitoring
- Monitor sensor readings for anomalies
- Detect unusual patterns or degradation
- Provide actionable diagnostic insights"""
        }

        return base_prompt + domain_prompts.get(domain, domain_prompts["general"])

    def _build_user_prompt(self, event_packet: EventPacket, domain: str) -> str:
        """Build user prompt with evidence."""
        # Format evidence window
        evidence_str = self._format_evidence(event_packet.evidence_window, domain)

        # Format anomaly scores
        recent_anomalies = event_packet.anomaly_scores[-10:] if event_packet.anomaly_scores else []
        anomaly_str = ", ".join([f"{s:.3f}" for s in recent_anomalies])

        # Format uncertainty
        recent_uncertainty = event_packet.uncertainty_scores[-5:] if event_packet.uncertainty_scores else []
        uncertainty_str = ", ".join([f"{u:.3f}" for u in recent_uncertainty])

        # Calculate mean anomaly score safely
        mean_anomaly = f"{sum(recent_anomalies)/len(recent_anomalies):.3f}" if recent_anomalies else "N/A"

        prompt = f"""TRIGGER ALERT at timestamp {event_packet.timestamp}

TRIGGER REASON: {event_packet.trigger_reason}

EVIDENCE WINDOW (recent observations):
{evidence_str}

ANOMALY SCORES (recent): [{anomaly_str}]
Mean: {mean_anomaly}

UNCERTAINTY SCORES: [{uncertainty_str}]

FAST MODEL PREDICTION: {event_packet.fast_model_prediction}

CONTEXT: {json.dumps(convert_to_native(event_packet.context), indent=2)}

Based on this evidence, provide your diagnosis in JSON format."""

        return prompt

    def _format_evidence(self, evidence_window: List[Dict], domain: str) -> str:
        """Format evidence window for prompt."""
        if not evidence_window:
            return "No evidence available"

        # Take last 5 observations for brevity
        recent = evidence_window[-5:]

        lines = []
        for i, obs in enumerate(recent):
            if domain == "cmapss":
                lines.append(f"  t-{len(recent)-1-i}: sensors={obs.get('features', obs)[:5]}...")
            elif domain == "geolife":
                lines.append(f"  t-{len(recent)-1-i}: pos=({obs.get('lat', 'N/A')}, {obs.get('lon', 'N/A')}), speed={obs.get('speed', 'N/A')}")
            elif domain == "cicids":
                lines.append(f"  t-{len(recent)-1-i}: flow_duration={obs.get('flow_duration', 'N/A')}, packets={obs.get('packets', 'N/A')}")
            else:
                lines.append(f"  t-{len(recent)-1-i}: {str(obs)[:100]}...")

        return "\n".join(lines)

    def _parse_response(
        self,
        raw_response: str,
        event_packet: EventPacket,
        latency_ms: float,
        tokens_used: int
    ) -> DiagnosisResult:
        """Parse LLM response and compute grounding score."""
        try:
            data = json.loads(raw_response)

            # Compute grounding score
            grounding_score = self._compute_grounding_score(
                data.get("evidence_summary", ""),
                data.get("key_indicators", []),
                event_packet
            )

            return DiagnosisResult(
                diagnosis=data.get("diagnosis", "No diagnosis provided"),
                confidence=float(data.get("confidence", 0.5)),
                severity=data.get("severity", "medium"),
                recommended_actions=data.get("recommended_actions", []),
                evidence_summary=data.get("evidence_summary", ""),
                grounding_score=grounding_score,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                raw_response=raw_response
            )
        except json.JSONDecodeError:
            return DiagnosisResult(
                diagnosis=raw_response[:500],
                confidence=0.3,
                severity="unknown",
                recommended_actions=[],
                evidence_summary="Failed to parse structured response",
                grounding_score=0.0,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                raw_response=raw_response
            )

    def _compute_grounding_score(
        self,
        evidence_summary: str,
        key_indicators: List[str],
        event_packet: EventPacket
    ) -> float:
        """
        Compute how well the diagnosis is grounded in evidence.

        Checks:
        1. Does diagnosis reference specific values?
        2. Does diagnosis mention anomaly scores?
        3. Does diagnosis reference the trigger reason?
        """
        score = 0.0
        checks = 0

        # Check if evidence summary references numbers
        import re
        numbers = re.findall(r'\d+\.?\d*', evidence_summary)
        if numbers:
            score += 0.3
        checks += 1

        # Check if key indicators are provided
        if key_indicators and len(key_indicators) > 0:
            score += 0.3
        checks += 1

        # Check if trigger reason is acknowledged
        if event_packet.trigger_reason.lower() in evidence_summary.lower():
            score += 0.2
        checks += 1

        # Check for specific domain terms
        domain_terms = ["sensor", "anomaly", "pattern", "degradation", "attack", "trajectory"]
        if any(term in evidence_summary.lower() for term in domain_terms):
            score += 0.2
        checks += 1

        return min(1.0, score)

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency,
            "avg_latency_ms": self.total_latency / max(1, self.total_calls),
            "avg_tokens_per_call": self.total_tokens / max(1, self.total_calls),
        }


class MockLLMDiagnosisEngine:
    """Mock LLM engine for testing without API calls."""

    def __init__(self):
        self.total_calls = 0

    def diagnose(self, event_packet: EventPacket, domain: str = "general") -> DiagnosisResult:
        """Return mock diagnosis."""
        self.total_calls += 1
        time.sleep(0.05)  # Simulate latency

        return DiagnosisResult(
            diagnosis=f"Mock diagnosis for {domain} at t={event_packet.timestamp}",
            confidence=0.8,
            severity="medium",
            recommended_actions=["Monitor closely", "Check thresholds"],
            evidence_summary="Based on anomaly score pattern",
            grounding_score=0.7,
            latency_ms=50.0,
            tokens_used=100,
            raw_response="{}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        return {"total_calls": self.total_calls}
