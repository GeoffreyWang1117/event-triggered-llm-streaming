"""
Semantic Diagnosis Engine using Local LLM.
Runs on host machine (RTX 3090) for grounded diagnosis.
"""
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class DiagnosisResult:
    """Result from LLM semantic diagnosis."""
    diagnosis: str  # Main diagnosis text
    confidence: float  # Confidence in diagnosis [0, 1]
    severity: str  # 'low', 'medium', 'high', 'critical'
    actionable_items: List[str]  # Recommended actions
    evidence_summary: str  # Summary of evidence used
    reasoning: str  # Chain-of-thought reasoning

    # Metadata
    latency_ms: float = 0.0
    tokens_used: int = 0
    model_name: str = ""

    # Grounding information
    is_grounded: bool = True
    grounding_score: float = 1.0
    cited_evidence: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'diagnosis': self.diagnosis,
            'confidence': self.confidence,
            'severity': self.severity,
            'actionable_items': self.actionable_items,
            'evidence_summary': self.evidence_summary,
            'reasoning': self.reasoning,
            'latency_ms': self.latency_ms,
            'tokens_used': self.tokens_used,
            'model_name': self.model_name,
            'is_grounded': self.is_grounded,
            'grounding_score': self.grounding_score,
        }


class SemanticDiagnosisEngine:
    """
    LLM-based semantic diagnosis engine.

    Key features:
    - Grounded diagnosis based on evidence packet
    - Structured output for actionable recommendations
    - Confidence estimation
    - Hallucination mitigation
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        use_vllm: bool = True,
    ):
        """
        Initialize diagnosis engine.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run on
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            use_vllm: Whether to use vLLM for inference
        """
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_vllm = use_vllm

        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        # Statistics
        self._total_diagnoses = 0
        self._avg_latency_ms = 0.0

    def load_model(self):
        """Load the LLM model."""
        if self._is_loaded:
            return

        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_transformers()

        self._is_loaded = True

    def _load_vllm(self):
        """Load model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
            self._model = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                dtype="half",
            )
            self._sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except ImportError:
            print("vLLM not available, falling back to transformers")
            self._load_transformers()

    def _load_transformers(self):
        """Load model using transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.use_vllm = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self._model = None

    def diagnose(self, event_packet: Dict[str, Any]) -> DiagnosisResult:
        """
        Perform semantic diagnosis on event packet.

        Args:
            event_packet: Event packet from edge device

        Returns:
            DiagnosisResult with diagnosis and recommendations
        """
        start_time = time.perf_counter()

        # Build prompt
        prompt = self._build_prompt(event_packet)

        # Generate response
        if self._model is None:
            # Mock response for testing
            response = self._mock_diagnosis(event_packet)
        elif self.use_vllm:
            response = self._generate_vllm(prompt)
        else:
            response = self._generate_transformers(prompt)

        # Parse response
        result = self._parse_response(response, event_packet)

        # Update metrics
        latency_ms = (time.perf_counter() - start_time) * 1000
        result.latency_ms = latency_ms
        result.model_name = self.model_name

        self._total_diagnoses += 1
        self._avg_latency_ms = (
            (self._avg_latency_ms * (self._total_diagnoses - 1) + latency_ms)
            / self._total_diagnoses
        )

        return result

    def _build_prompt(self, event_packet: Dict[str, Any]) -> str:
        """Build diagnosis prompt from event packet."""
        trigger_reason = event_packet.get('trigger_reason', 'unknown')
        anomaly_score = event_packet.get('anomaly_score', 0)
        uncertainty = event_packet.get('uncertainty_estimate', 0)
        context = event_packet.get('system_context', {})
        prediction = event_packet.get('fast_model_prediction', None)

        # Format evidence window
        evidence = event_packet.get('evidence_window', [])
        if isinstance(evidence, list) and len(evidence) > 0:
            evidence_str = f"Recent observations ({len(evidence)} timesteps):\n"
            for i, obs in enumerate(evidence[-5:]):  # Last 5 observations
                evidence_str += f"  t-{len(evidence)-1-i}: {obs}\n"
        else:
            evidence_str = "Evidence: Not available"

        prompt = f"""You are an expert diagnostic system for robotic/industrial systems.
Analyze the following event and provide a diagnosis.

## Event Information
- Trigger Reason: {trigger_reason}
- Anomaly Score: {anomaly_score:.4f}
- Prediction Uncertainty: {uncertainty:.4f}
- Fast Model Prediction: {prediction}

## Context
{context}

## Evidence
{evidence_str}

## Task
Based on the above information, provide:
1. A diagnosis explaining the likely cause of this event
2. Severity assessment (low/medium/high/critical)
3. Confidence level (0-1)
4. Recommended actions

IMPORTANT: Base your diagnosis ONLY on the provided evidence. Do not make assumptions about data you haven't seen.

## Response Format
DIAGNOSIS: [Your diagnosis]
SEVERITY: [low/medium/high/critical]
CONFIDENCE: [0.0-1.0]
REASONING: [Your reasoning based on evidence]
ACTIONS:
- [Action 1]
- [Action 2]
EVIDENCE_SUMMARY: [Summary of evidence used]
"""
        return prompt

    def _generate_vllm(self, prompt: str) -> str:
        """Generate using vLLM."""
        outputs = self._model.generate([prompt], self._sampling_params)
        return outputs[0].outputs[0].text

    def _generate_transformers(self, prompt: str) -> str:
        """Generate using transformers."""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response

    def _mock_diagnosis(self, event_packet: Dict[str, Any]) -> str:
        """Generate mock diagnosis for testing."""
        anomaly_score = event_packet.get('anomaly_score', 0)
        trigger_reason = event_packet.get('trigger_reason', 'unknown')

        if anomaly_score > 2.0:
            severity = "high"
            diagnosis = "Significant anomaly detected indicating potential system degradation"
        elif anomaly_score > 1.0:
            severity = "medium"
            diagnosis = "Moderate deviation from normal operation detected"
        else:
            severity = "low"
            diagnosis = "Minor fluctuation in system behavior"

        return f"""DIAGNOSIS: {diagnosis}. Trigger reason: {trigger_reason}.
SEVERITY: {severity}
CONFIDENCE: 0.75
REASONING: Based on the anomaly score of {anomaly_score:.2f} and trigger reason '{trigger_reason}', the system shows signs of deviation from normal behavior.
ACTIONS:
- Monitor system closely for the next observation window
- Check sensor readings for any physical issues
- Review recent operational changes
EVIDENCE_SUMMARY: Analyzed recent observations showing anomaly score of {anomaly_score:.2f} with trigger type {trigger_reason}.
"""

    def _parse_response(
        self,
        response: str,
        event_packet: Dict[str, Any]
    ) -> DiagnosisResult:
        """Parse LLM response into structured result."""
        lines = response.strip().split('\n')

        diagnosis = ""
        severity = "low"
        confidence = 0.5
        reasoning = ""
        actions = []
        evidence_summary = ""

        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('DIAGNOSIS:'):
                diagnosis = line[10:].strip()
                current_section = 'diagnosis'
            elif line.startswith('SEVERITY:'):
                severity = line[9:].strip().lower()
                current_section = 'severity'
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line[11:].strip())
                except ValueError:
                    confidence = 0.5
                current_section = 'confidence'
            elif line.startswith('REASONING:'):
                reasoning = line[10:].strip()
                current_section = 'reasoning'
            elif line.startswith('ACTIONS:'):
                current_section = 'actions'
            elif line.startswith('EVIDENCE_SUMMARY:'):
                evidence_summary = line[17:].strip()
                current_section = 'evidence'
            elif line.startswith('- ') and current_section == 'actions':
                actions.append(line[2:].strip())
            elif current_section == 'diagnosis' and line:
                diagnosis += ' ' + line
            elif current_section == 'reasoning' and line:
                reasoning += ' ' + line

        return DiagnosisResult(
            diagnosis=diagnosis,
            confidence=min(1.0, max(0.0, confidence)),
            severity=severity if severity in ['low', 'medium', 'high', 'critical'] else 'low',
            actionable_items=actions,
            evidence_summary=evidence_summary,
            reasoning=reasoning,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'total_diagnoses': self._total_diagnoses,
            'avg_latency_ms': self._avg_latency_ms,
            'model_name': self.model_name,
            'is_loaded': self._is_loaded,
        }
