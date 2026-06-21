#!/usr/bin/env python3
"""
Ollama LLM Client: Real LLM integration using local Ollama API.

Ollama API Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import time
import httpx
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"  # Default model, can be changed
    timeout: float = 60.0
    temperature: float = 0.3
    max_tokens: int = 500


class OllamaClient:
    """
    Ollama API Client for LLM-based robot diagnosis.

    Usage:
        client = OllamaClient(base_url="http://localhost:11434", model="llama3.2")
        result = client.diagnose(state_vector, trigger_reason, domain_knowledge)
    """

    def __init__(self, config: Optional[OllamaConfig] = None, **kwargs):
        """
        Initialize Ollama client.

        Args:
            config: OllamaConfig object
            **kwargs: Override config values (base_url, model, timeout, etc.)
        """
        if config is None:
            config = OllamaConfig()

        # Allow kwargs to override config
        self.base_url = kwargs.get('base_url', config.base_url)
        self.model = kwargs.get('model', config.model)
        self.timeout = kwargs.get('timeout', config.timeout)
        self.temperature = kwargs.get('temperature', config.temperature)
        self.max_tokens = kwargs.get('max_tokens', config.max_tokens)

        # Statistics
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0

        logger.info(f"Ollama client initialized: {self.base_url}, model={self.model}")

    def check_health(self) -> bool:
        """Check if Ollama server is available."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []

    def _build_diagnosis_prompt(self, state_vector: np.ndarray,
                                 trigger_reason: str,
                                 domain_knowledge: str = "") -> str:
        """Build diagnosis prompt for LLM."""
        # Format state vector
        if len(state_vector) > 0:
            state_str = ", ".join([f"{v:.4f}" for v in state_vector[:10]])
            if len(state_vector) > 10:
                state_str += f", ... ({len(state_vector)} total)"
        else:
            state_str = "No state data"

        prompt = f"""You are a robot diagnostic system. Analyze the following robot state and provide diagnosis.

## Robot State
State Vector: [{state_str}]
- Position (x, y): ({state_vector[0]:.3f}, {state_vector[1]:.3f}) if available
- Velocity (linear, angular): ({state_vector[2]:.3f}, {state_vector[3]:.3f}) if available
- Min obstacle distance: {state_vector[4]:.3f}m if available

## Trigger Information
Trigger Reason: {trigger_reason}

## Domain Knowledge
{domain_knowledge if domain_knowledge else "Standard robot navigation monitoring."}

## Task
Analyze the robot state and provide diagnosis. Respond with ONLY a JSON object in this exact format:

```json
{{
    "diagnosis": "Brief description of the detected issue or status",
    "severity": "LOW|MEDIUM|HIGH|CRITICAL",
    "recommended_action": "CONTINUE|SLOW_DOWN|STOP|REPLAN|RETURN_HOME",
    "confidence": 0.85,
    "explanation": "Detailed explanation of the analysis and reasoning"
}}
```

Important:
- severity must be one of: LOW, MEDIUM, HIGH, CRITICAL
- recommended_action must be one of: CONTINUE, SLOW_DOWN, STOP, REPLAN, RETURN_HOME
- confidence must be a float between 0.0 and 1.0
- Respond with ONLY the JSON object, no additional text"""

        return prompt

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON diagnosis."""
        # Try to extract JSON from response
        text = response_text.strip()

        # Remove markdown code blocks if present
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        # Try to parse JSON
        try:
            result = json.loads(text)

            # Validate required fields
            required_fields = ['diagnosis', 'severity', 'recommended_action', 'confidence']
            for field in required_fields:
                if field not in result:
                    result[field] = self._get_default_value(field)

            # Validate severity
            valid_severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            if result['severity'] not in valid_severities:
                result['severity'] = 'MEDIUM'

            # Validate action
            valid_actions = ['CONTINUE', 'SLOW_DOWN', 'STOP', 'REPLAN', 'RETURN_HOME']
            if result['recommended_action'] not in valid_actions:
                result['recommended_action'] = 'CONTINUE'

            # Validate confidence
            try:
                result['confidence'] = float(result['confidence'])
                result['confidence'] = max(0.0, min(1.0, result['confidence']))
            except (TypeError, ValueError):
                result['confidence'] = 0.5

            return result

        except json.JSONDecodeError:
            # If JSON parsing fails, create a default response
            logger.warning(f"Failed to parse JSON from response: {text[:100]}...")
            return {
                'diagnosis': f"LLM response parsing failed. Raw: {text[:200]}",
                'severity': 'MEDIUM',
                'recommended_action': 'CONTINUE',
                'confidence': 0.3,
                'explanation': 'Could not parse LLM response as JSON'
            }

    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields."""
        defaults = {
            'diagnosis': 'Unable to generate diagnosis',
            'severity': 'MEDIUM',
            'recommended_action': 'CONTINUE',
            'confidence': 0.5,
            'explanation': 'No explanation provided'
        }
        return defaults.get(field, None)

    def diagnose(self, state_vector: np.ndarray, trigger_reason: str,
                 domain_knowledge: str = "") -> Dict[str, Any]:
        """
        Get diagnosis from Ollama LLM.

        Args:
            state_vector: Robot state as numpy array
            trigger_reason: Reason for triggering diagnosis
            domain_knowledge: Optional domain-specific context

        Returns:
            Dictionary with diagnosis, severity, recommended_action, confidence, etc.
        """
        start_time = time.time()

        # Build prompt
        prompt = self._build_diagnosis_prompt(state_vector, trigger_reason, domain_knowledge)

        try:
            # Call Ollama API
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens
                        }
                    }
                )
                response.raise_for_status()

                data = response.json()
                response_text = data.get('response', '')

            # Parse response
            result = self._parse_response(response_text)

            # Add metadata
            latency_ms = (time.time() - start_time) * 1000
            result['latency_ms'] = latency_ms
            result['is_mock'] = False
            result['model'] = self.model
            result['provider'] = 'ollama'

            # Update statistics
            self.request_count += 1
            self.total_latency += latency_ms

            logger.info(f"Diagnosis complete: {result['severity']} - {result['recommended_action']} "
                       f"({latency_ms:.1f}ms)")

            return result

        except httpx.TimeoutException:
            self.error_count += 1
            latency_ms = (time.time() - start_time) * 1000
            return {
                'diagnosis': 'LLM request timed out',
                'severity': 'MEDIUM',
                'recommended_action': 'CONTINUE',
                'confidence': 0.0,
                'explanation': f'Request timed out after {self.timeout}s',
                'latency_ms': latency_ms,
                'is_mock': False,
                'error': 'timeout'
            }

        except Exception as e:
            self.error_count += 1
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Ollama diagnosis failed: {e}")
            return {
                'diagnosis': f'LLM error: {str(e)}',
                'severity': 'MEDIUM',
                'recommended_action': 'CONTINUE',
                'confidence': 0.0,
                'explanation': str(e),
                'latency_ms': latency_ms,
                'is_mock': False,
                'error': str(e)
            }

    def diagnose_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple diagnosis requests.

        Args:
            requests: List of dicts with 'state_vector', 'trigger_reason', 'domain_knowledge'

        Returns:
            List of diagnosis results
        """
        results = []
        for req in requests:
            result = self.diagnose(
                np.array(req.get('state_vector', [])),
                req.get('trigger_reason', ''),
                req.get('domain_knowledge', '')
            )
            results.append(result)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'total_latency_ms': self.total_latency,
            'avg_latency_ms': self.total_latency / max(1, self.request_count),
            'error_rate': self.error_count / max(1, self.request_count),
            'model': self.model,
            'base_url': self.base_url
        }


# Domain knowledge templates
ROBOT_DOMAIN_KNOWLEDGE = """
## Robot Navigation Domain Knowledge

### State Vector Interpretation
- Position (x, y): Robot's location in meters
- Velocity (linear): Safe range 0.0-0.5 m/s, warning >0.3 m/s near obstacles
- Velocity (angular): Normal range -1.0 to 1.0 rad/s
- Min obstacle distance: Critical if <0.3m, warning if <0.5m

### Trigger Conditions
- ET (Event-Triggered): Fires when state change exceeds threshold
- OS (Optimal Stopping): Fires when expected miss cost > invoke cost
- SPRT: Statistical test detects anomaly with controlled error rates

### Critical Situations
- Obstacle distance <0.3m: CRITICAL, recommend STOP or SLOW_DOWN
- Robot stationary >10s while navigating: Likely STUCK, recommend REPLAN
- Sudden position jump: Localization error, recommend SLOW_DOWN
- High sensor noise: Degraded sensing, recommend SLOW_DOWN

### Actions
- CONTINUE: No immediate intervention needed
- SLOW_DOWN: Reduce velocity by 50-70%
- STOP: Emergency stop
- REPLAN: Request new navigation path
- RETURN_HOME: Abort mission and return to start
"""


def test_ollama_client():
    """Test Ollama client with sample data."""
    print("Testing Ollama Client...")
    print("=" * 60)

    client = OllamaClient()

    # Check health
    print("\n1. Health Check...")
    if not client.check_health():
        print("   ERROR: Ollama server not available!")
        print("   Please start Ollama with: ollama serve")
        print("   And ensure you have a model: ollama pull llama3.2")
        return False
    print("   OK: Ollama server is running")

    # List models
    print("\n2. Available Models...")
    models = client.list_models()
    if models:
        print(f"   Found {len(models)} models: {models[:5]}")
    else:
        print("   WARNING: No models found. Run: ollama pull llama3.2")

    # Test diagnosis
    print("\n3. Test Diagnosis...")
    state_vector = np.array([1.5, 2.0, 0.3, 0.1, 0.45, 1.2, 0.0, 0.0, 0.0, 0.0])
    trigger_reason = "ET: state change detected, prediction change=0.15"

    result = client.diagnose(state_vector, trigger_reason, ROBOT_DOMAIN_KNOWLEDGE)

    print(f"   Diagnosis: {result['diagnosis']}")
    print(f"   Severity: {result['severity']}")
    print(f"   Action: {result['recommended_action']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Latency: {result['latency_ms']:.1f}ms")

    # Statistics
    print("\n4. Statistics...")
    stats = client.get_stats()
    print(f"   Requests: {stats['request_count']}")
    print(f"   Errors: {stats['error_count']}")
    print(f"   Avg Latency: {stats['avg_latency_ms']:.1f}ms")

    print("\n" + "=" * 60)
    print("Test Complete!")
    return True


if __name__ == '__main__':
    test_ollama_client()
