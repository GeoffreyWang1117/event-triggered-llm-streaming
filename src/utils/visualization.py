"""
Visualization utilities for experiments.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path


def plot_rul_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    trigger_times: Optional[List[int]] = None,
    title: str = "RUL Predictions",
    save_path: Optional[str] = None,
):
    """
    Plot RUL predictions vs ground truth.

    Args:
        predictions: Predicted RUL values
        targets: True RUL values
        trigger_times: Optional list of LLM trigger timestamps
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: Predictions vs Truth
    ax1 = axes[0]
    ax1.plot(targets, 'b-', alpha=0.7, label='True RUL', linewidth=1)
    ax1.plot(predictions, 'r-', alpha=0.7, label='Predicted RUL', linewidth=1)

    if trigger_times:
        for t in trigger_times:
            ax1.axvline(x=t, color='g', alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.axvline(x=-1, color='g', alpha=0.5, linestyle='--',
                    linewidth=1, label='LLM Trigger')

    ax1.set_ylabel('RUL (cycles)')
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom: Prediction Error
    ax2 = axes[1]
    errors = predictions - targets
    ax2.plot(errors, 'k-', alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    ax2.fill_between(range(len(errors)), errors, 0, alpha=0.3,
                     color=np.where(errors >= 0, 'orange', 'blue'))

    if trigger_times:
        for t in trigger_times:
            ax2.axvline(x=t, color='g', alpha=0.3, linestyle='--', linewidth=0.5)

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Prediction Error')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_trigger_events(
    anomaly_scores: List[float],
    trigger_events: List[Dict[str, Any]],
    threshold: float = 2.0,
    title: str = "Event-Triggered LLM Invocations",
    save_path: Optional[str] = None,
):
    """
    Plot anomaly scores with trigger events.

    Args:
        anomaly_scores: List of anomaly scores
        trigger_events: List of trigger event dictionaries
        threshold: Anomaly threshold line
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: Anomaly scores
    ax1 = axes[0]
    ax1.plot(anomaly_scores, 'b-', alpha=0.7, linewidth=1, label='Anomaly Score')
    ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.7,
                label=f'Threshold ({threshold})')

    # Mark trigger points
    trigger_times = [e.get('timestamp', 0) for e in trigger_events]
    trigger_scores = []
    for t in trigger_times:
        if t < len(anomaly_scores):
            trigger_scores.append(anomaly_scores[t])
        else:
            trigger_scores.append(threshold)

    ax1.scatter(trigger_times, trigger_scores, c='red', s=50, zorder=5,
                label='LLM Trigger', marker='v')

    ax1.set_ylabel('Anomaly Score')
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom: Trigger reason distribution over time
    ax2 = axes[1]

    # Create trigger reason timeline
    reasons = [e.get('trigger_reason', 'unknown') for e in trigger_events]
    unique_reasons = list(set(reasons))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_reasons)))
    reason_colors = {r: c for r, c in zip(unique_reasons, colors)}

    for t, reason in zip(trigger_times, reasons):
        ax2.axvline(x=t, color=reason_colors[reason], alpha=0.7, linewidth=2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=reason_colors[r], linewidth=2, label=r)
        for r in unique_reasons
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Trigger Events')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison(
    results: List[Dict[str, Any]],
    metric: str = 'mae',
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
):
    """
    Plot comparison of different methods.

    Args:
        results: List of result dictionaries
        metric: Metric to compare
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = []
    values = []
    colors = []

    for r in results:
        method = r.get('method', 'unknown')
        model = r.get('model', 'N/A')

        if method == 'baseline':
            label = f"{model} (baseline)"
            color = 'blue'
        elif method == 'periodic_llm':
            period = r.get('period', 0)
            label = f"{model}+periodic_{period}"
            color = 'orange'
        else:
            trigger = r.get('trigger', 'unknown')
            label = f"{model}+{trigger}"
            color = 'green'

        methods.append(label)
        values.append(r.get(metric, 0))
        colors.append(color)

    bars = ax.bar(range(len(methods)), values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_llm_efficiency(
    results: List[Dict[str, Any]],
    title: str = "LLM Invocation Efficiency",
    save_path: Optional[str] = None,
):
    """
    Plot LLM invocation efficiency vs accuracy tradeoff.

    Args:
        results: List of result dictionaries
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for r in results:
        method = r.get('method', 'unknown')
        mae = r.get('mae', 0)
        llm_rate = r.get('llm_rate', 0) * 100

        if method == 'baseline':
            marker = 's'
            color = 'blue'
            label = f"{r.get('model')} baseline"
        elif method == 'periodic_llm':
            marker = '^'
            color = 'orange'
            label = f"periodic_{r.get('period')}"
        else:
            marker = 'o'
            color = 'green'
            label = f"{r.get('trigger')}"

        ax.scatter(llm_rate, mae, marker=marker, c=color, s=100, label=label, alpha=0.7)

    ax.set_xlabel('LLM Invocation Rate (%)')
    ax.set_ylabel('MAE (cycles)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add Pareto frontier line
    # (Simplified: connect points in order of LLM rate)
    sorted_results = sorted(results, key=lambda x: x.get('llm_rate', 0))
    llm_rates = [r.get('llm_rate', 0) * 100 for r in sorted_results]
    maes = [r.get('mae', 0) for r in sorted_results]

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
