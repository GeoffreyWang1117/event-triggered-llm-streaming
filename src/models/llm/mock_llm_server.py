#!/usr/bin/env python3
"""
Mock LLM Server: Provides LLM-like responses for testing without API calls.

Runs as a FastAPI server that mimics LLM diagnosis behavior.
"""

import json
import time
import random
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiagnosisRequest(BaseModel):
    """Request format for diagnosis."""
    request_id: str
    state_vector: list
    trigger_reason: str
    domain_knowledge: Optional[str] = ""


class DiagnosisResponse(BaseModel):
    """Response format for diagnosis."""
    request_id: str
    diagnosis: str
    severity: str
    recommended_action: str
    confidence: float
    explanation: str
    latency_ms: float
    is_mock: bool = True


# Diagnosis templates based on severity
DIAGNOSIS_TEMPLATES = {
    'CRITICAL': [
        "Critical system anomaly detected. Immediate intervention required.",
        "Severe degradation in sensor readings. Robot safety may be compromised.",
        "Critical threshold exceeded. Emergency stop recommended.",
    ],
    'HIGH': [
        "Significant deviation from normal operation detected.",
        "High-priority anomaly identified. Close monitoring required.",
        "Substantial sensor drift observed. Course correction needed.",
    ],
    'MEDIUM': [
        "Moderate deviation in system parameters.",
        "Noticeable change in operational characteristics.",
        "Sensor readings show concerning trends.",
    ],
    'LOW': [
        "Minor fluctuation detected. Within acceptable limits.",
        "Slight deviation observed. Continued monitoring advised.",
        "Small perturbation in sensor data. No immediate action needed.",
    ]
}

ACTIONS = {
    'CRITICAL': ['STOP', 'RETURN_HOME'],
    'HIGH': ['SLOW_DOWN', 'REPLAN'],
    'MEDIUM': ['SLOW_DOWN', 'CONTINUE'],
    'LOW': ['CONTINUE']
}


app = FastAPI(title="Mock LLM Server", version="1.0.0")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "mode": "mock"}


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    """Generate mock diagnosis."""
    start_time = time.time()

    # Simulate LLM processing time (100-500ms)
    latency = random.uniform(0.1, 0.5)
    time.sleep(latency)

    # Analyze trigger reason to determine severity
    trigger_lower = request.trigger_reason.lower()

    if 'critical' in trigger_lower or 'sprt' in trigger_lower:
        severity = 'CRITICAL'
        confidence = random.uniform(0.85, 0.98)
    elif 'anomaly' in trigger_lower or 'high' in trigger_lower:
        severity = 'HIGH'
        confidence = random.uniform(0.7, 0.9)
    elif 'change' in trigger_lower or 'drift' in trigger_lower:
        severity = 'MEDIUM'
        confidence = random.uniform(0.5, 0.75)
    else:
        severity = 'LOW'
        confidence = random.uniform(0.3, 0.6)

    # Analyze state vector if available
    if request.state_vector:
        state = np.array(request.state_vector)
        state_magnitude = np.linalg.norm(state)

        # Adjust severity based on state magnitude
        if state_magnitude > 10:
            if severity == 'LOW':
                severity = 'MEDIUM'
            elif severity == 'MEDIUM':
                severity = 'HIGH'

    # Select diagnosis and action
    diagnosis = random.choice(DIAGNOSIS_TEMPLATES[severity])
    action = random.choice(ACTIONS[severity])

    # Generate explanation
    explanation = f"Analysis of {len(request.state_vector)} state variables. "
    if request.domain_knowledge:
        explanation += "Applied domain knowledge constraints. "
    explanation += f"Trigger pattern: {request.trigger_reason[:50]}..."

    latency_ms = (time.time() - start_time) * 1000

    logger.info(f"Request {request.request_id}: {severity} - {action} ({latency_ms:.1f}ms)")

    return DiagnosisResponse(
        request_id=request.request_id,
        diagnosis=diagnosis,
        severity=severity,
        recommended_action=action,
        confidence=confidence,
        explanation=explanation,
        latency_ms=latency_ms,
        is_mock=True
    )


@app.post("/batch_diagnose")
async def batch_diagnose(requests: list):
    """Batch diagnosis for multiple requests."""
    responses = []
    for req in requests:
        request = DiagnosisRequest(**req)
        response = await diagnose(request)
        responses.append(response.dict())
    return {"responses": responses}


@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    return {
        "mode": "mock",
        "avg_latency_ms": 250,
        "requests_served": 0,  # Would track in production
        "uptime_seconds": time.time()
    }


def main():
    """Run the mock LLM server."""
    import os

    host = os.environ.get('LLM_HOST', '0.0.0.0')
    port = int(os.environ.get('LLM_PORT', 8000))

    logger.info(f"Starting Mock LLM Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == '__main__':
    main()
