"""
Experiment 7: Stronger Baselines Comparison
Implements RouteLLM-style, cost-aware bandit, and other baselines.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from collections import deque
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.uai_experiments.config import UAIExperimentConfig, set_seed, save_results, get_data_dir
from src.data.cmapss import CMAPSSDataset, create_sequences
from src.models.fast.gru import GRUModel
from src.data.base import StreamSample

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'ecml_pkdd'

C_LLM = 1.0
C_MISS = 10.0


def collect_stream(config: UAIExperimentConfig, seed: int) -> List[Dict]:
    """Collect model outputs as a stream."""
    set_seed(seed)
    data_dir = get_data_dir()

    dataset = CMAPSSDataset(data_dir, subset='FD001', split='train')
    X, y, _ = create_sequences(dataset, seq_length=config.seq_length)
    if len(X) == 0:
        return []

    model = GRUModel(n_features=X.shape[2], hidden_size=config.hidden_size, device=config.device)
    model.fit(X, y, epochs=config.epochs, batch_size=config.batch_size, learning_rate=config.lr)

    test_dataset = CMAPSSDataset(data_dir, subset='FD001', split='test')
    model.reset_state()

    stream = []
    for sample in test_dataset:
        if sample.metadata['cycle'] == 1:
            model.reset_state()
        output = model.step(sample.features)
        rul = sample.label if sample.label is not None else 100.0
        stream.append({
            'sample': sample,
            'output': output,
            'anomaly_score': output.anomaly_score,
            'uncertainty': output.uncertainty,
            'prediction': float(output.prediction.flatten()[0]) if hasattr(output.prediction, 'flatten') else float(output.prediction),
            'rul': rul,
            'is_critical': rul < 30,
        })

    return stream


def evaluate_policy(stream: List[Dict], decisions: List[bool]) -> Dict:
    """Evaluate a trigger policy given binary decisions."""
    n_total = len(stream)
    n_invocations = sum(decisions)
    n_critical = sum(1 for s in stream if s['is_critical'])
    n_misses = sum(1 for s, d in zip(stream, decisions) if s['is_critical'] and not d)

    total_cost = C_LLM * n_invocations + C_MISS * n_misses
    inv_rate = n_invocations / n_total if n_total > 0 else 0
    miss_rate = n_misses / n_critical if n_critical > 0 else 0

    # F1
    tp = sum(1 for s, d in zip(stream, decisions) if s['is_critical'] and d)
    fp = sum(1 for s, d in zip(stream, decisions) if not s['is_critical'] and d)
    fn = n_misses
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'n_invocations': n_invocations,
        'n_misses': n_misses,
        'n_critical': n_critical,
        'total_cost': total_cost,
        'invocation_rate': inv_rate,
        'miss_rate': miss_rate,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


# ==================== Baselines ====================

def periodic_baseline(stream: List[Dict], target_inv_rate: float = 0.05) -> List[bool]:
    """Invoke every K steps to match target invocation rate."""
    K = max(1, int(1.0 / target_inv_rate))
    return [i % K == 0 for i in range(len(stream))]


def random_baseline(stream: List[Dict], target_inv_rate: float = 0.05) -> List[bool]:
    """Random invocation with probability p."""
    return [np.random.random() < target_inv_rate for _ in stream]


def routellm_baseline(stream: List[Dict], train_fraction: float = 0.3) -> List[bool]:
    """RouteLLM-style: train a router on features to predict critical events."""
    train_size = int(train_fraction * len(stream))
    train_data = stream[:train_size]
    test_data = stream[train_size:]

    # Build features
    def build_features(data: List[Dict], window: int = 10) -> np.ndarray:
        features = []
        anomaly_buf = deque(maxlen=window)
        prev_pred = 0
        for item in data:
            anomaly_buf.append(item['anomaly_score'])
            d_pred = item['prediction'] - prev_pred
            prev_pred = item['prediction']
            features.append([
                item['anomaly_score'],
                item['uncertainty'],
                item['prediction'],
                d_pred,
                np.mean(list(anomaly_buf)),
                np.std(list(anomaly_buf)) if len(anomaly_buf) > 1 else 0,
            ])
        return np.array(features)

    # Labels: 1 if within 5 steps of critical transition
    def build_labels(data: List[Dict]) -> np.ndarray:
        labels = np.zeros(len(data))
        for i in range(len(data)):
            # Check if any of next 5 steps transitions to critical
            for j in range(max(0, i - 5), min(len(data), i + 6)):
                if data[j]['is_critical']:
                    labels[i] = 1
                    break
        return labels

    X_train = build_features(train_data)
    y_train = build_labels(train_data)

    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        clf.fit(X_train, y_train)

        X_test = build_features(test_data)
        predictions = clf.predict(X_test)

        # Full decisions: False for train period, predictions for test
        decisions = [False] * train_size + list(predictions.astype(bool))
    except ImportError:
        # Fallback: simple threshold
        decisions = [item['anomaly_score'] + item['uncertainty'] > 1.0 for item in stream]

    return decisions


def linucb_baseline(stream: List[Dict], alpha: float = 1.0) -> List[bool]:
    """Cost-aware contextual bandit with LinUCB."""
    d = 3  # context dimension: [anomaly, uncertainty, 1/steps_since_invoke]
    # Two arms: 0=don't invoke, 1=invoke
    A = [np.eye(d) for _ in range(2)]
    b = [np.zeros(d) for _ in range(2)]

    decisions = []
    steps_since_invoke = 100

    for item in stream:
        # Context
        ctx = np.array([item['anomaly_score'], item['uncertainty'],
                        1.0 / (steps_since_invoke + 1)])

        # UCB for each arm
        ucbs = []
        for arm in range(2):
            A_inv = np.linalg.inv(A[arm])
            theta = A_inv @ b[arm]
            ucb = ctx @ theta + alpha * np.sqrt(ctx @ A_inv @ ctx)
            ucbs.append(ucb)

        # Choose arm with higher UCB
        chosen = int(ucbs[1] > ucbs[0])
        decisions.append(bool(chosen))

        # Compute reward
        is_critical = item['is_critical']
        if chosen == 1:  # Invoke
            reward = -C_LLM + (C_MISS if is_critical else 0)  # Save miss cost if critical
            steps_since_invoke = 0
        else:  # Don't invoke
            reward = -C_MISS if is_critical else 0
            steps_since_invoke += 1

        # Update chosen arm
        A[chosen] += np.outer(ctx, ctx)
        b[chosen] += reward * ctx

    return decisions


def fixed_percentile_baseline(stream: List[Dict], percentile: float = 95) -> List[bool]:
    """Invoke when anomaly+uncertainty exceeds running percentile."""
    window = deque(maxlen=200)
    decisions = []

    for item in stream:
        score = item['anomaly_score'] + item['uncertainty']
        window.append(score)

        if len(window) < 20:
            decisions.append(False)
        else:
            threshold = np.percentile(list(window), percentile)
            decisions.append(score > threshold)

    return decisions


def speculative_baseline(stream: List[Dict],
                         low_thresh: float = 0.3, high_thresh: float = 1.0) -> List[bool]:
    """Two-threshold speculative execution."""
    decisions = []
    for item in stream:
        score = item['anomaly_score'] + item['uncertainty']
        # Only invoke at high threshold
        decisions.append(score > high_thresh)

    return decisions


# ==================== Paper's methods ====================

def threshold_trigger(stream: List[Dict], theta: float = 1.0) -> List[bool]:
    """Paper's threshold trigger."""
    from src.triggers.threshold import ThresholdTrigger
    trigger = ThresholdTrigger(
        anomaly_threshold=theta, uncertainty_threshold=theta * 0.5,
        cooldown_steps=5, min_evidence_window=3)

    decisions = []
    for item in stream:
        result = trigger.step(item['sample'], item['output'])
        decisions.append(result.should_trigger)
    return decisions


def cusum_trigger(stream: List[Dict], h: float = 5.0) -> List[bool]:
    """Paper's CUSUM trigger."""
    from src.triggers.cusum import CUSUMTrigger
    trigger = CUSUMTrigger(
        slack=0.5, threshold=h,
        warmup_samples=30, cooldown_steps=5, min_evidence_window=3)

    decisions = []
    for item in stream:
        result = trigger.step(item['sample'], item['output'])
        decisions.append(result.should_trigger)
    return decisions


def optimal_stopping_trigger(stream: List[Dict], c: float = 1.0) -> List[bool]:
    """Paper's optimal stopping trigger."""
    from src.triggers.optimal_stopping import OptimalStoppingTrigger
    trigger = OptimalStoppingTrigger(
        llm_cost=c, risk_weight=0.1,
        cooldown_steps=5, min_evidence_window=3)

    decisions = []
    for item in stream:
        result = trigger.step(item['sample'], item['output'])
        decisions.append(result.should_trigger)
    return decisions


def generate_latex_table(results: Dict) -> str:
    """Generate comparison table."""
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Comparison with stronger baselines on CMAPSS FD001 (5 seeds). '
        r'$c_{\text{LLM}}=1$, $c_{\text{miss}}=10$. Best result per metric in \textbf{bold}.}',
        r'\label{tab:stronger_baselines}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{llccccc}',
        r'\toprule',
        r'Category & Method & Inv.\ Rate$\downarrow$ & Miss Rate$\downarrow$ & F1$\uparrow$ & Total Cost$\downarrow$ & Type \\',
        r'\midrule',
    ]

    agg = results.get('aggregated', {})

    # Find best values
    all_costs = [d.get('total_cost_mean', float('inf')) for d in agg.values()]
    all_f1s = [d.get('f1_mean', 0) for d in agg.values()]
    all_miss = [d.get('miss_rate_mean', 1) for d in agg.values()]
    best_cost = min(all_costs) if all_costs else 0
    best_f1 = max(all_f1s) if all_f1s else 0
    best_miss = min(all_miss) if all_miss else 0

    categories = {
        'Ours': ['threshold', 'cusum', 'optimal_stopping'],
        'Non-adaptive': ['periodic', 'random', 'fixed_percentile_95', 'fixed_percentile_99'],
        'Learning': ['routellm', 'linucb'],
        'Hybrid': ['speculative'],
    }

    for cat, methods in categories.items():
        for method in methods:
            if method not in agg:
                continue
            d = agg[method]

            cost_str = f"{d['total_cost_mean']:.1f}$\\pm${d['total_cost_std']:.1f}"
            if abs(d['total_cost_mean'] - best_cost) < 0.1:
                cost_str = r'\textbf{' + cost_str + '}'

            f1_str = f"{d['f1_mean']:.3f}"
            if abs(d['f1_mean'] - best_f1) < 0.001:
                f1_str = r'\textbf{' + f1_str + '}'

            miss_str = f"{d['miss_rate_mean']:.3f}"
            if abs(d['miss_rate_mean'] - best_miss) < 0.001:
                miss_str = r'\textbf{' + miss_str + '}'

            lines.append(
                f'  {cat} & {method} '
                f'& {d["invocation_rate_mean"]:.3f} '
                f'& {miss_str} '
                f'& {f1_str} '
                f'& {cost_str} '
                f'& {cat} \\\\'
            )
            cat = ''  # Only show category once

        lines.append(r'  \midrule')

    lines[-1] = r'  \bottomrule'
    lines.extend([
        r'\end{tabular}%',
        r'}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def run_experiment():
    """Run all baseline comparisons."""
    print("=" * 60)
    print("ECML-PKDD Experiment 7: Stronger Baselines")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = UAIExperimentConfig(quick=False)
    config.n_seeds = 5
    config.epochs = 50

    methods = {
        # Paper's methods
        'threshold': lambda s: threshold_trigger(s, theta=1.0),
        'cusum': lambda s: cusum_trigger(s, h=5.0),
        'optimal_stopping': lambda s: optimal_stopping_trigger(s, c=1.0),
        # Baselines
        'periodic': lambda s: periodic_baseline(s, target_inv_rate=0.05),
        'random': lambda s: random_baseline(s, target_inv_rate=0.05),
        'routellm': lambda s: routellm_baseline(s, train_fraction=0.3),
        'linucb': lambda s: linucb_baseline(s, alpha=1.0),
        'fixed_percentile_95': lambda s: fixed_percentile_baseline(s, percentile=95),
        'fixed_percentile_99': lambda s: fixed_percentile_baseline(s, percentile=99),
        'speculative': lambda s: speculative_baseline(s, low_thresh=0.3, high_thresh=1.0),
    }

    all_results = {'per_seed': {}, 'aggregated': {}}

    for seed_idx in range(config.n_seeds):
        seed = config.seed + seed_idx
        print(f"\n  Seed {seed} ({seed_idx + 1}/{config.n_seeds})...")

        stream = collect_stream(config, seed)
        if not stream:
            print(f"    No data for seed {seed}")
            continue

        seed_results = {}
        for method_name, method_fn in methods.items():
            print(f"    Running {method_name}...")
            set_seed(seed)  # Reset seed for stochastic methods
            decisions = method_fn(stream)
            result = evaluate_policy(stream, decisions)
            seed_results[method_name] = result

        all_results['per_seed'][str(seed)] = seed_results

    # Aggregate
    for method_name in methods:
        values = {
            'total_cost': [], 'invocation_rate': [], 'miss_rate': [], 'f1': [],
            'precision': [], 'recall': [], 'n_invocations': [], 'n_misses': [],
        }
        for seed_data in all_results['per_seed'].values():
            if method_name in seed_data:
                for k in values:
                    values[k].append(seed_data[method_name].get(k, 0))

        if values['total_cost']:
            all_results['aggregated'][method_name] = {
                f'{k}_mean': float(np.mean(v)) for k, v in values.items()
            }
            all_results['aggregated'][method_name].update({
                f'{k}_std': float(np.std(v)) for k, v in values.items()
            })
            all_results['aggregated'][method_name]['n_seeds'] = len(values['total_cost'])

    # Save
    save_results(all_results, str(RESULTS_DIR / 'stronger_baselines.json'))

    # Generate LaTeX
    latex = generate_latex_table(all_results)
    with open(RESULTS_DIR / 'stronger_baselines_table.tex', 'w') as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {RESULTS_DIR / 'stronger_baselines_table.tex'}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by total cost)")
    print("=" * 60)
    sorted_methods = sorted(all_results['aggregated'].items(),
                            key=lambda x: x[1].get('total_cost_mean', float('inf')))
    for method, data in sorted_methods:
        print(f"  {method:25s}: cost={data['total_cost_mean']:7.1f}+/-{data['total_cost_std']:5.1f}, "
              f"inv={data['invocation_rate_mean']:.3f}, miss={data['miss_rate_mean']:.3f}, "
              f"f1={data['f1_mean']:.3f}")

    return all_results


if __name__ == '__main__':
    run_experiment()
