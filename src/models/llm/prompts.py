"""
Prompt templates for LLM diagnosis.
"""
from typing import Dict, Any, List, Optional
from string import Template


class PromptBuilder:
    """
    Builder for structured diagnosis prompts.
    Supports different domains and task types.
    """

    SYSTEM_PROMPT = """You are an expert diagnostic system for robotic and industrial systems.
Your role is to analyze sensor data, anomalies, and system events to provide:
1. Accurate diagnosis of issues
2. Severity assessment
3. Actionable recommendations

CRITICAL RULES:
- Base ALL conclusions ONLY on provided evidence
- If evidence is insufficient, say so explicitly
- Never fabricate sensor readings or statistics
- Quantify uncertainty in your assessments
"""

    CMAPSS_DOMAIN_CONTEXT = """
## Domain: Turbofan Engine Health Monitoring (CMAPSS)

You are analyzing sensor data from aircraft turbofan engines. The system monitors:
- Operational settings (altitude, Mach number, throttle position)
- Engine sensors (temperatures, pressures, speeds, flows)

Key degradation indicators:
- Increasing temperature trends at critical points
- Decreasing efficiency ratios
- Abnormal vibration patterns
- Pressure ratio changes

RUL (Remaining Useful Life) is the primary concern - predicting cycles until failure.
"""

    CICIDS_DOMAIN_CONTEXT = """
## Domain: Network Intrusion Detection (CIC-IDS)

You are analyzing network traffic for potential security threats. Monitored features include:
- Flow statistics (duration, packet counts, byte counts)
- Protocol information
- Flag patterns
- Inter-arrival times

Key threat indicators:
- Unusual traffic patterns
- Protocol anomalies
- Suspicious connection behaviors
- Potential attack signatures
"""

    GEOLIFE_DOMAIN_CONTEXT = """
## Domain: GPS Trajectory Analysis (GeoLife)

You are analyzing GPS trajectory data for anomaly detection. Features include:
- Position (latitude, longitude, altitude)
- Time intervals
- Speed and direction
- Movement patterns

Key anomaly indicators:
- Unusual movement patterns
- Sudden location jumps
- Abnormal speeds
- Route deviations
"""

    def __init__(self, domain: str = "general"):
        """
        Initialize prompt builder.

        Args:
            domain: Domain context ('cmapss', 'cicids', 'geolife', 'general')
        """
        self.domain = domain
        self._domain_contexts = {
            'cmapss': self.CMAPSS_DOMAIN_CONTEXT,
            'cicids': self.CICIDS_DOMAIN_CONTEXT,
            'geolife': self.GEOLIFE_DOMAIN_CONTEXT,
            'general': "",
        }

    def build_diagnosis_prompt(
        self,
        event_packet: Dict[str, Any],
        historical_context: Optional[List[Dict]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """
        Build full diagnosis prompt.

        Args:
            event_packet: Event packet from edge
            historical_context: Similar historical events
            feature_names: Names of features in evidence

        Returns:
            Complete prompt string
        """
        parts = [
            self.SYSTEM_PROMPT,
            self._domain_contexts.get(self.domain, ""),
            self._format_event(event_packet, feature_names),
        ]

        if historical_context:
            parts.append(self._format_history(historical_context))

        parts.append(self._format_task())

        return "\n\n".join(parts)

    def _format_event(
        self,
        event_packet: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """Format event information."""
        trigger_reason = event_packet.get('trigger_reason', 'unknown')
        anomaly_score = event_packet.get('anomaly_score', 0)
        uncertainty = event_packet.get('uncertainty_estimate', 0)
        prediction = event_packet.get('fast_model_prediction', None)
        context = event_packet.get('system_context', {})

        event_text = f"""## Current Event

**Trigger Reason**: {trigger_reason}
**Anomaly Score**: {anomaly_score:.4f}
**Model Uncertainty**: {uncertainty:.4f}
**Fast Model Prediction**: {prediction}

### Trigger Context
"""
        for key, value in context.items():
            event_text += f"- {key}: {value}\n"

        # Format evidence
        evidence = event_packet.get('evidence_window', [])
        if evidence is not None and len(evidence) > 0:
            event_text += "\n### Evidence Window (Recent Observations)\n"
            event_text += "```\n"

            # Header
            if feature_names:
                header = "t   " + "  ".join(f"{n[:8]:>8}" for n in feature_names[:5])
                if len(feature_names) > 5:
                    header += "  ..."
                event_text += header + "\n"

            # Data
            for i, obs in enumerate(evidence[-10:]):  # Last 10 observations
                t_label = f"t-{len(evidence)-1-i}"
                if hasattr(obs, '__len__'):
                    values = "  ".join(f"{v:8.3f}" for v in obs[:5])
                    if len(obs) > 5:
                        values += "  ..."
                else:
                    values = f"{obs:8.3f}"
                event_text += f"{t_label:4} {values}\n"

            event_text += "```\n"

        return event_text

    def _format_history(self, historical: List[Dict]) -> str:
        """Format historical context."""
        if not historical:
            return ""

        hist_text = "## Similar Historical Events\n\n"

        for i, event in enumerate(historical[:3]):
            similarity = event.get('similarity', 0)
            diagnosis = event.get('diagnosis', {})

            hist_text += f"### Past Event {i+1} (Similarity: {similarity:.2f})\n"
            hist_text += f"- Diagnosis: {diagnosis.get('diagnosis', 'N/A')}\n"
            hist_text += f"- Severity: {diagnosis.get('severity', 'N/A')}\n"
            hist_text += f"- Outcome: {diagnosis.get('outcome', 'N/A')}\n\n"

        return hist_text

    def _format_task(self) -> str:
        """Format task instructions."""
        return """## Your Task

Analyze the current event and provide a structured diagnosis.

**Required Output Format**:
```
DIAGNOSIS: [Your diagnosis of what is happening and why]

SEVERITY: [low/medium/high/critical]

CONFIDENCE: [0.0-1.0]

REASONING: [Step-by-step reasoning based on evidence]

ACTIONS:
- [Recommended action 1]
- [Recommended action 2]
- [Recommended action 3]

EVIDENCE_SUMMARY: [Summary of key evidence points used]
```

Remember:
- Only reference data you can see in the evidence
- Be specific about which features/values support your conclusions
- If uncertain, clearly state the uncertainty
"""

    def build_followup_prompt(
        self,
        original_diagnosis: str,
        feedback: str,
    ) -> str:
        """
        Build follow-up prompt for clarification.

        Args:
            original_diagnosis: Previous diagnosis
            feedback: User feedback or additional question

        Returns:
            Follow-up prompt
        """
        return f"""## Follow-up Analysis

Your previous diagnosis:
{original_diagnosis}

User feedback/question:
{feedback}

Please provide clarification or updated analysis based on this feedback.
Maintain the same structured output format.
"""
