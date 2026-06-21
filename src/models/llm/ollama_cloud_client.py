#!/usr/bin/env python3
"""
Ollama Cloud LLM Client: Uses Ollama Cloud API for real LLM experiments.

Supports multiple large-scale models for robot diagnosis experiments.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

try:
    from ollama import Client
except ImportError:
    raise ImportError("Please install ollama: pip install ollama")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Available cloud models for experiments
AVAILABLE_MODELS = {
    # Ultra-large models (>400B)
    'kimi-k2:1t': {'size': '1T', 'type': 'ultra-large', 'description': 'Kimi K2 1T parameters'},
    'kimi-k2-thinking': {'size': '1T', 'type': 'ultra-large', 'description': 'Kimi K2 with reasoning'},
    'deepseek-v3.2': {'size': '671B', 'type': 'ultra-large', 'description': 'DeepSeek V3.2'},
    'cogito-2.1:671b': {'size': '671B', 'type': 'ultra-large', 'description': 'Cogito 2.1'},
    'mistral-large-3:675b': {'size': '675B', 'type': 'ultra-large', 'description': 'Mistral Large 3'},
    'qwen3-coder:480b': {'size': '480B', 'type': 'ultra-large', 'description': 'Qwen3 Coder'},

    # Large models (100-400B)
    'minimax-m2.1': {'size': '230B', 'type': 'large', 'description': 'MiniMax M2.1'},
    'devstral-2:123b': {'size': '123B', 'type': 'large', 'description': 'Devstral 2'},
    'gpt-oss:120b': {'size': '120B', 'type': 'large', 'description': 'GPT-OSS 120B'},
    'qwen3-next:80b': {'size': '80B', 'type': 'large', 'description': 'Qwen3 Next'},

    # Medium models (20-80B)
    'devstral-small-2:24b': {'size': '24B', 'type': 'medium', 'description': 'Devstral Small 2'},
    'nemotron-3-nano:30b': {'size': '30B', 'type': 'medium', 'description': 'Nemotron 3 Nano'},
    'gpt-oss:20b': {'size': '20B', 'type': 'medium', 'description': 'GPT-OSS 20B'},
    'gemma3:27b': {'size': '27B', 'type': 'medium', 'description': 'Gemma 3 27B'},

    # Small models (<20B)
    'ministral-3:14b': {'size': '14B', 'type': 'small', 'description': 'Ministral 3 14B'},
    'ministral-3:8b': {'size': '8B', 'type': 'small', 'description': 'Ministral 3 8B'},
    'gemma3:12b': {'size': '12B', 'type': 'small', 'description': 'Gemma 3 12B'},
    'rnj-1:8b': {'size': '8B', 'type': 'small', 'description': 'RNJ-1 8B'},
}

# Recommended models for different experiment types
EXPERIMENT_MODELS = {
    # High-quality ablation study - use best model
    'ablation_primary': 'deepseek-v3.2',

    # Main experiments - balance quality and speed
    'main_experiments': 'gpt-oss:120b',

    # Latency comparison - fast model
    'latency_test': 'nemotron-3-nano:30b',

    # Model scaling study
    'scaling_study': ['gpt-oss:20b', 'gpt-oss:120b', 'deepseek-v3.2'],
}


@dataclass
class OllamaCloudConfig:
    """Configuration for Ollama Cloud client."""
    host: str = "https://ollama.com"
    api_key: str = ""
    model: str = "gpt-oss:120b"
    timeout: float = 120.0
    temperature: float = 0.3
    max_tokens: int = 800


class OllamaCloudClient:
    """
    Ollama Cloud API Client for LLM-based robot diagnosis.

    Usage:
        client = OllamaCloudClient(api_key="your-key", model="deepseek-v3.2")
        result = client.diagnose(state_vector, trigger_reason)
    """

    def __init__(self, config: Optional[OllamaCloudConfig] = None, **kwargs):
        """
        Initialize Ollama Cloud client.

        Args:
            config: OllamaCloudConfig object
            **kwargs: Override config values (api_key, model, etc.)
        """
        if config is None:
            config = OllamaCloudConfig()

        self.host = kwargs.get('host', config.host)
        self.api_key = kwargs.get('api_key', config.api_key) or os.environ.get('OLLAMA_API_KEY', '')
        self.model = kwargs.get('model', config.model)
        self.timeout = kwargs.get('timeout', config.timeout)
        self.temperature = kwargs.get('temperature', config.temperature)
        self.max_tokens = kwargs.get('max_tokens', config.max_tokens)

        if not self.api_key:
            raise ValueError("OLLAMA_API_KEY is required. Set via environment or api_key parameter.")

        # Initialize client
        self.client = Client(
            host=self.host,
            headers={'Authorization': f'Bearer {self.api_key}'}
        )

        # Statistics
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        self.model_stats: Dict[str, Dict] = {}

        logger.info(f"Ollama Cloud client initialized: model={self.model}")

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            result = self.client.list()
            models = []
            for m in result.models:
                models.append(m.model)
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def _build_diagnosis_prompt(self, state_vector: np.ndarray,
                                 trigger_reason: str,
                                 domain_knowledge: str = "") -> List[Dict[str, str]]:
        """Build chat messages for diagnosis."""
        # Format state vector
        if len(state_vector) > 0:
            state_parts = []
            labels = ['pos_x', 'pos_y', 'vel_lin', 'vel_ang', 'min_dist',
                      'mean_dist', 'dist_std', 'front', 'right', 'back',
                      'left', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            for i, v in enumerate(state_vector[:len(labels)]):
                state_parts.append(f"{labels[i]}={v:.4f}")
            state_str = ", ".join(state_parts)
        else:
            state_str = "No state data available"

        system_prompt = """You are an expert robot diagnostic system. Your task is to analyze robot sensor data and provide actionable diagnosis.

You must respond with ONLY a valid JSON object in this exact format:
{
    "diagnosis": "Brief description of the issue",
    "severity": "LOW|MEDIUM|HIGH|CRITICAL",
    "recommended_action": "CONTINUE|SLOW_DOWN|STOP|REPLAN|RETURN_HOME",
    "confidence": 0.85,
    "explanation": "Detailed reasoning"
}

Severity guidelines:
- LOW: Minor deviations, normal operation
- MEDIUM: Noticeable issues, monitor closely
- HIGH: Significant problems, intervention needed
- CRITICAL: Immediate danger, emergency response

Action guidelines:
- CONTINUE: No intervention needed
- SLOW_DOWN: Reduce velocity by 50%
- STOP: Emergency stop
- REPLAN: Request new navigation path
- RETURN_HOME: Abort mission"""

        user_prompt = f"""Analyze this robot state and provide diagnosis.

## Sensor Data
{state_str}

## Trigger Event
{trigger_reason}

## Domain Context
{domain_knowledge if domain_knowledge else "Standard mobile robot navigation."}

Respond with ONLY the JSON object, no other text."""

        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract diagnosis."""
        text = response_text.strip()

        # Remove markdown code blocks
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

        # Find JSON object
        if '{' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            if end > start:
                text = text[start:end]

        try:
            result = json.loads(text)

            # Validate and fix fields
            valid_severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            if result.get('severity') not in valid_severities:
                result['severity'] = 'MEDIUM'

            valid_actions = ['CONTINUE', 'SLOW_DOWN', 'STOP', 'REPLAN', 'RETURN_HOME']
            if result.get('recommended_action') not in valid_actions:
                result['recommended_action'] = 'CONTINUE'

            try:
                result['confidence'] = float(result.get('confidence', 0.5))
                result['confidence'] = max(0.0, min(1.0, result['confidence']))
            except:
                result['confidence'] = 0.5

            if 'diagnosis' not in result:
                result['diagnosis'] = 'Diagnosis generated'
            if 'explanation' not in result:
                result['explanation'] = ''

            return result

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {text[:200]}...")
            return {
                'diagnosis': 'Parse error',
                'severity': 'MEDIUM',
                'recommended_action': 'CONTINUE',
                'confidence': 0.3,
                'explanation': f'Could not parse: {text[:100]}'
            }

    def diagnose(self, state_vector: np.ndarray, trigger_reason: str,
                 domain_knowledge: str = "", model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get diagnosis from Ollama Cloud LLM.

        Args:
            state_vector: Robot state as numpy array
            trigger_reason: Reason for triggering diagnosis
            domain_knowledge: Optional domain context
            model: Override default model for this request

        Returns:
            Dictionary with diagnosis results
        """
        use_model = model or self.model
        start_time = time.time()

        messages = self._build_diagnosis_prompt(state_vector, trigger_reason, domain_knowledge)

        try:
            # Call Ollama Cloud API
            response = self.client.chat(
                model=use_model,
                messages=messages,
                stream=False,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )

            response_text = response['message']['content']
            result = self._parse_response(response_text)

            # Add metadata
            latency_ms = (time.time() - start_time) * 1000
            result['latency_ms'] = latency_ms
            result['is_mock'] = False
            result['model'] = use_model
            result['provider'] = 'ollama_cloud'

            # Update statistics
            self.request_count += 1
            self.total_latency += latency_ms

            if use_model not in self.model_stats:
                self.model_stats[use_model] = {'count': 0, 'latency': 0.0, 'errors': 0}
            self.model_stats[use_model]['count'] += 1
            self.model_stats[use_model]['latency'] += latency_ms

            logger.info(f"[{use_model}] {result['severity']} - {result['recommended_action']} "
                       f"(conf={result['confidence']:.2f}, {latency_ms:.0f}ms)")

            return result

        except Exception as e:
            self.error_count += 1
            if use_model in self.model_stats:
                self.model_stats[use_model]['errors'] += 1
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Diagnosis failed: {e}")
            return {
                'diagnosis': f'Error: {str(e)}',
                'severity': 'MEDIUM',
                'recommended_action': 'CONTINUE',
                'confidence': 0.0,
                'explanation': str(e),
                'latency_ms': latency_ms,
                'is_mock': False,
                'model': use_model,
                'error': str(e)
            }

    def diagnose_streaming(self, state_vector: np.ndarray, trigger_reason: str,
                           domain_knowledge: str = "", model: Optional[str] = None):
        """
        Get diagnosis with streaming response.

        Yields partial responses as they arrive.
        """
        use_model = model or self.model
        start_time = time.time()

        messages = self._build_diagnosis_prompt(state_vector, trigger_reason, domain_knowledge)

        try:
            full_response = ""
            for part in self.client.chat(use_model, messages=messages, stream=True):
                content = part['message']['content']
                full_response += content
                yield {'partial': content, 'done': False}

            # Parse final response
            result = self._parse_response(full_response)
            latency_ms = (time.time() - start_time) * 1000
            result['latency_ms'] = latency_ms
            result['is_mock'] = False
            result['model'] = use_model

            self.request_count += 1
            self.total_latency += latency_ms

            yield {'result': result, 'done': True}

        except Exception as e:
            self.error_count += 1
            yield {'error': str(e), 'done': True}

    def compare_models(self, state_vector: np.ndarray, trigger_reason: str,
                       models: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare diagnosis results from multiple models.

        Args:
            state_vector: Robot state
            trigger_reason: Trigger reason
            models: List of model names to compare

        Returns:
            Dictionary mapping model names to their results
        """
        results = {}
        for model_name in models:
            if model_name in AVAILABLE_MODELS:
                logger.info(f"Running diagnosis with {model_name}...")
                result = self.diagnose(state_vector, trigger_reason, model=model_name)
                results[model_name] = result
            else:
                logger.warning(f"Model {model_name} not in available models list")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'total_latency_ms': self.total_latency,
            'avg_latency_ms': self.total_latency / max(1, self.request_count),
            'error_rate': self.error_count / max(1, self.request_count),
            'current_model': self.model,
            'per_model_stats': {}
        }

        for model, data in self.model_stats.items():
            stats['per_model_stats'][model] = {
                'count': data['count'],
                'avg_latency_ms': data['latency'] / max(1, data['count']),
                'errors': data['errors']
            }

        return stats


# Domain knowledge for robot experiments
ROBOT_DOMAIN_KNOWLEDGE = """
## Mobile Robot Navigation Domain

### Sensor Interpretation
- Position (pos_x, pos_y): Robot location in meters, origin at start
- Velocity (vel_lin): Linear velocity, safe range 0.0-0.5 m/s
- Velocity (vel_ang): Angular velocity, normal range -1.0 to 1.0 rad/s
- Obstacle distance (min_dist): Closest obstacle, CRITICAL if <0.3m

### Trigger Mechanisms
- ET (Event-Triggered): State/prediction change exceeds threshold
- OS (Optimal Stopping): Expected miss cost exceeds invoke cost
- SPRT: Sequential test detects statistical anomaly

### Critical Conditions
- min_dist < 0.3m: Collision risk, STOP or SLOW_DOWN
- vel_lin > 0.5 near obstacles: Dangerous, SLOW_DOWN
- Stationary > 10s while navigating: STUCK, need REPLAN
- Sudden position jump: Localization error, SLOW_DOWN
"""


def test_ollama_cloud():
    """Test Ollama Cloud client."""
    print("=" * 60)
    print("Testing Ollama Cloud Client")
    print("=" * 60)

    # Get API key
    api_key = os.environ.get('OLLAMA_API_KEY', '')

    client = OllamaCloudClient(api_key=api_key, model='gpt-oss:120b')

    # List models
    print("\n1. Available Models:")
    models = client.list_models()
    for m in models[:10]:
        info = AVAILABLE_MODELS.get(m, {})
        print(f"   - {m} ({info.get('size', '?')}) - {info.get('description', '')}")

    # Test diagnosis
    print("\n2. Testing Diagnosis...")
    state_vector = np.array([1.5, 2.0, 0.35, 0.1, 0.28, 1.2, 0.15,
                             0.5, 0.8, 1.5, 0.6, 0.1, -0.05, 9.8, 0.0, 0.0, 0.02])
    trigger_reason = "SPRT: Anomaly detected, LLR=2.5 > threshold=2.0"

    result = client.diagnose(state_vector, trigger_reason, ROBOT_DOMAIN_KNOWLEDGE)

    print(f"\n   Model: {result.get('model')}")
    print(f"   Diagnosis: {result['diagnosis']}")
    print(f"   Severity: {result['severity']}")
    print(f"   Action: {result['recommended_action']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Latency: {result['latency_ms']:.0f}ms")
    print(f"   Explanation: {result.get('explanation', '')[:100]}...")

    # Statistics
    print("\n3. Statistics:")
    stats = client.get_stats()
    print(f"   Requests: {stats['request_count']}")
    print(f"   Avg Latency: {stats['avg_latency_ms']:.0f}ms")

    print("\n" + "=" * 60)
    return client


if __name__ == '__main__':
    test_ollama_cloud()
