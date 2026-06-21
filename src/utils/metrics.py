"""
Metrics for evaluation.
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_rul_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    max_rul: int = 125,
) -> Dict[str, float]:
    """
    Compute RUL prediction metrics.

    Args:
        predictions: Predicted RUL values
        targets: True RUL values
        max_rul: Maximum RUL value

    Returns:
        Dictionary of metrics
    """
    predictions = np.clip(predictions, 0, max_rul)
    targets = np.clip(targets, 0, max_rul)

    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))

    # Scoring function (asymmetric - penalizes late predictions more)
    errors = predictions - targets
    score = np.sum(np.where(
        errors < 0,
        np.exp(-errors / 13) - 1,  # Early prediction
        np.exp(errors / 10) - 1     # Late prediction
    ))

    # Percentage within X cycles
    within_5 = np.mean(np.abs(errors) <= 5) * 100
    within_10 = np.mean(np.abs(errors) <= 10) * 100
    within_20 = np.mean(np.abs(errors) <= 20) * 100

    # Timeliness metrics
    early_pct = np.mean(errors < 0) * 100
    late_pct = np.mean(errors > 0) * 100

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'score': float(score / len(predictions)),
        'within_5_cycles': float(within_5),
        'within_10_cycles': float(within_10),
        'within_20_cycles': float(within_20),
        'early_prediction_pct': float(early_pct),
        'late_prediction_pct': float(late_pct),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
    }


def compute_diagnosis_metrics(
    diagnoses: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute diagnosis quality metrics.

    Args:
        diagnoses: List of diagnosis results
        ground_truth: Optional ground truth labels

    Returns:
        Dictionary of metrics
    """
    if not diagnoses:
        return {
            'total_diagnoses': 0,
            'avg_confidence': 0.0,
            'avg_latency_ms': 0.0,
        }

    confidences = [d.get('confidence', 0) for d in diagnoses]
    latencies = [d.get('latency_ms', 0) for d in diagnoses]

    # Severity distribution
    severities = [d.get('severity', 'low') for d in diagnoses]
    severity_counts = {
        'low': sum(1 for s in severities if s == 'low'),
        'medium': sum(1 for s in severities if s == 'medium'),
        'high': sum(1 for s in severities if s == 'high'),
        'critical': sum(1 for s in severities if s == 'critical'),
    }

    # Grounding metrics (if available)
    grounding_scores = [
        d.get('grounding_score', 1.0) for d in diagnoses
    ]

    metrics = {
        'total_diagnoses': len(diagnoses),
        'avg_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences)),
        'avg_latency_ms': float(np.mean(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'severity_distribution': severity_counts,
        'avg_grounding_score': float(np.mean(grounding_scores)),
    }

    # Actionability: percentage with actionable items
    has_actions = [
        len(d.get('actionable_items', [])) > 0 for d in diagnoses
    ]
    metrics['actionability_rate'] = float(np.mean(has_actions))

    return metrics


def compute_trigger_metrics(
    trigger_events: List[Dict[str, Any]],
    true_anomalies: List[int] = None,
) -> Dict[str, float]:
    """
    Compute trigger mechanism metrics.

    Args:
        trigger_events: List of trigger events
        true_anomalies: Optional ground truth anomaly timestamps

    Returns:
        Dictionary of metrics
    """
    if not trigger_events:
        return {
            'total_triggers': 0,
            'trigger_rate': 0.0,
        }

    # Trigger timing statistics
    timestamps = [e.get('timestamp', 0) for e in trigger_events]
    if len(timestamps) > 1:
        intervals = np.diff(timestamps)
        avg_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))
    else:
        avg_interval = 0.0
        std_interval = 0.0

    # Trigger reason distribution
    reasons = [e.get('trigger_reason', 'unknown') for e in trigger_events]
    reason_counts = {}
    for r in reasons:
        reason_counts[r] = reason_counts.get(r, 0) + 1

    # Confidence statistics
    confidences = [e.get('trigger_confidence', 0) for e in trigger_events]

    metrics = {
        'total_triggers': len(trigger_events),
        'avg_trigger_interval': avg_interval,
        'std_trigger_interval': std_interval,
        'reason_distribution': reason_counts,
        'avg_trigger_confidence': float(np.mean(confidences)),
    }

    # If ground truth available, compute precision/recall
    if true_anomalies is not None and len(true_anomalies) > 0:
        # Simple window-based matching
        window = 5
        detected = 0
        for anomaly_t in true_anomalies:
            for trigger_t in timestamps:
                if abs(trigger_t - anomaly_t) <= window:
                    detected += 1
                    break

        recall = detected / len(true_anomalies)

        # False positives: triggers not near any anomaly
        false_positives = 0
        for trigger_t in timestamps:
            near_anomaly = any(
                abs(trigger_t - a) <= window for a in true_anomalies
            )
            if not near_anomaly:
                false_positives += 1

        precision = (len(timestamps) - false_positives) / max(1, len(timestamps))

        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(2 * precision * recall / max(0.001, precision + recall))

    return metrics


def compute_system_metrics(
    edge_metrics: Dict[str, Any],
    host_metrics: Dict[str, Any] = None,
) -> Dict[str, float]:
    """
    Compute system-level metrics.

    Args:
        edge_metrics: Metrics from edge device
        host_metrics: Metrics from host (LLM server)

    Returns:
        Dictionary of system metrics
    """
    metrics = {
        'edge_total_samples': edge_metrics.get('total_samples', 0),
        'edge_total_triggers': edge_metrics.get('total_triggers', 0),
        'edge_trigger_rate': edge_metrics.get('avg_trigger_rate', 0),
        'edge_avg_latency_ms': edge_metrics.get('avg_inference_time_ms', 0),
        'edge_max_latency_ms': edge_metrics.get('max_inference_time_ms', 0),
    }

    if host_metrics:
        metrics.update({
            'host_total_diagnoses': host_metrics.get('total_diagnoses', 0),
            'host_avg_latency_ms': host_metrics.get('avg_latency_ms', 0),
        })

        # End-to-end metrics
        if metrics['edge_total_triggers'] > 0:
            metrics['e2e_avg_latency_ms'] = (
                metrics['edge_avg_latency_ms'] +
                metrics.get('host_avg_latency_ms', 0)
            )

    # Efficiency: samples per trigger
    if metrics['edge_total_triggers'] > 0:
        metrics['samples_per_trigger'] = (
            metrics['edge_total_samples'] / metrics['edge_total_triggers']
        )

    return metrics
