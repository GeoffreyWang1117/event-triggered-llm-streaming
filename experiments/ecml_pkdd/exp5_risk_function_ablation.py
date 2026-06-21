"""
Experiment 5: Risk Function R(H_t) Ablation Study
Compares different constructions of the risk function and their Pareto frontiers.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Callable
from collections import deque
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.uai_experiments.config import UAIExperimentConfig, set_seed, save_results, get_data_dir
from src.data.cmapss import CMAPSSDataset, create_sequences
from src.models.fast.gru import GRUModel
from src.data.base import StreamSample


RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'ecml_pkdd'


class RiskFunction:
    """Base class for risk function variants."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def reset(self):
        pass

    def compute(self, anomaly_score: float, uncertainty: float,
                t: int, t_since_trigger: int) -> float:
        raise NotImplementedError


class R_LinearCombo(RiskFunction):
    """R1: alpha * anomaly + beta * uncertainty"""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__('R1_linear', f'Linear: {alpha}*a + {beta}*u')
        self.alpha = alpha
        self.beta = beta

    def compute(self, anomaly_score, uncertainty, t, t_since_trigger):
        return self.alpha * anomaly_score + self.beta * uncertainty


class R_AnomalyOnly(RiskFunction):
    """R2: anomaly_score only"""

    def __init__(self):
        super().__init__('R2_anomaly_only', 'Anomaly score only')

    def compute(self, anomaly_score, uncertainty, t, t_since_trigger):
        return anomaly_score


class R_UncertaintyOnly(RiskFunction):
    """R3: uncertainty only"""

    def __init__(self):
        super().__init__('R3_uncertainty_only', 'Uncertainty only')

    def compute(self, anomaly_score, uncertainty, t, t_since_trigger):
        return uncertainty


class R_Multiplicative(RiskFunction):
    """R4: anomaly * uncertainty"""

    def __init__(self):
        super().__init__('R4_multiplicative', 'Anomaly * Uncertainty')

    def compute(self, anomaly_score, uncertainty, t, t_since_trigger):
        return anomaly_score * uncertainty


class R_Max(RiskFunction):
    """R5: max(anomaly, uncertainty)"""

    def __init__(self):
        super().__init__('R5_max', 'max(anomaly, uncertainty)')

    def compute(self, anomaly_score, uncertainty, t, t_since_trigger):
        return max(anomaly_score, uncertainty)


class R_EWMA(RiskFunction):
    """R6: Exponential weighted moving average"""

    def __init__(self, gamma: float = 0.9):
        super().__init__('R6_ewma', f'EWMA (gamma={gamma})')
        self.gamma = gamma
        self._prev = 0.0

    def reset(self):
        self._prev = 0.0

    def compute(self, anomaly_score, uncertainty, t, t_since_trigger):
        instant = anomaly_score + uncertainty
        self._prev = self.gamma * self._prev + (1 - self.gamma) * instant
        return self._prev


class R_Learned(RiskFunction):
    """R7: Learned logistic regression"""

    def __init__(self):
        super().__init__('R7_learned', 'Learned logistic regressor')
        self.weights = None
        self.bias = 0.0
        self._prev_anomaly = 0.0
        self._prev_uncertainty = 0.0

    def reset(self):
        self._prev_anomaly = 0.0
        self._prev_uncertainty = 0.0

    def train(self, features: np.ndarray, labels: np.ndarray):
        """Train logistic regression on features -> critical label."""
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(features, labels)
            self.weights = clf.coef_[0]
            self.bias = clf.intercept_[0]
        except Exception:
            self.weights = np.ones(features.shape[1]) * 0.5
            self.bias = 0.0

    def compute(self, anomaly_score, uncertainty, t, t_since_trigger):
        if self.weights is None:
            return anomaly_score + uncertainty

        d_anomaly = anomaly_score - self._prev_anomaly
        d_uncertainty = uncertainty - self._prev_uncertainty
        self._prev_anomaly = anomaly_score
        self._prev_uncertainty = uncertainty

        features = np.array([anomaly_score, uncertainty, d_anomaly, d_uncertainty, t_since_trigger])
        logit = np.dot(self.weights, features) + self.bias
        return 1.0 / (1.0 + np.exp(-logit))


class R_Quantile(RiskFunction):
    """R8: Running percentile rank"""

    def __init__(self, window: int = 200):
        super().__init__('R8_quantile', 'Quantile-based (running)')
        self.window = window
        self._anomaly_history = deque(maxlen=window)
        self._uncertainty_history = deque(maxlen=window)

    def reset(self):
        self._anomaly_history.clear()
        self._uncertainty_history.clear()

    def compute(self, anomaly_score, uncertainty, t, t_since_trigger):
        self._anomaly_history.append(anomaly_score)
        self._uncertainty_history.append(uncertainty)

        if len(self._anomaly_history) < 10:
            return (anomaly_score + uncertainty) / 2

        a_rank = np.mean(np.array(list(self._anomaly_history)) <= anomaly_score)
        u_rank = np.mean(np.array(list(self._uncertainty_history)) <= uncertainty)
        return 0.5 * a_rank + 0.5 * u_rank


def collect_outputs_with_labels(config: UAIExperimentConfig, seed: int) -> List[Dict]:
    """Collect model outputs with critical event labels."""
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

    outputs = []
    for sample in test_dataset:
        if sample.metadata['cycle'] == 1:
            model.reset_state()
        output = model.step(sample.features)
        rul = sample.label if sample.label is not None else 100.0
        outputs.append({
            'anomaly_score': output.anomaly_score,
            'uncertainty': output.uncertainty,
            'prediction': float(output.prediction.flatten()[0]) if hasattr(output.prediction, 'flatten') else float(output.prediction),
            'rul': rul,
            'is_critical': rul < 30,
            'unit_id': sample.metadata.get('unit_id', 0),
        })

    return outputs


def evaluate_risk_function(risk_fn: RiskFunction, outputs: List[Dict],
                           theta_grid: np.ndarray) -> Dict:
    """Evaluate a risk function across a grid of thresholds."""
    # Compute risk scores
    risk_fn.reset()
    risk_scores = []
    t_since_trigger = 100  # Start high

    for t, item in enumerate(outputs):
        r = risk_fn.compute(item['anomaly_score'], item['uncertainty'], t, t_since_trigger)
        risk_scores.append(r)
        t_since_trigger += 1

    risk_scores = np.array(risk_scores)
    is_critical = np.array([item['is_critical'] for item in outputs])

    # Sweep thresholds
    pareto_points = []
    for theta in theta_grid:
        triggered = risk_scores >= theta
        n_invocations = int(np.sum(triggered))
        n_total = len(outputs)
        n_critical = int(np.sum(is_critical))

        # Miss = critical and not triggered (with cooldown simulation)
        n_misses = 0
        cooldown = 0
        for i in range(n_total):
            if cooldown > 0:
                cooldown -= 1
                continue
            if is_critical[i] and not triggered[i]:
                n_misses += 1
            if triggered[i]:
                cooldown = 5

        inv_rate = n_invocations / n_total if n_total > 0 else 0
        miss_rate = n_misses / n_critical if n_critical > 0 else 0

        # F1 of trigger (treat critical as positive)
        tp = int(np.sum(triggered & is_critical))
        fp = int(np.sum(triggered & ~is_critical))
        fn = n_misses
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        pareto_points.append({
            'theta': float(theta),
            'invocation_rate': inv_rate,
            'miss_rate': miss_rate,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        })

    # Compute Pareto frontier (invocation_rate vs miss_rate)
    frontier = []
    for p in pareto_points:
        dominated = False
        for q in pareto_points:
            if q['invocation_rate'] <= p['invocation_rate'] and q['miss_rate'] <= p['miss_rate'] and \
               (q['invocation_rate'] < p['invocation_rate'] or q['miss_rate'] < p['miss_rate']):
                dominated = True
                break
        if not dominated:
            frontier.append(p)

    frontier_sorted = sorted(frontier, key=lambda x: x['invocation_rate'])

    # AUC of Pareto curve (lower = better)
    if len(frontier_sorted) > 1:
        x = [p['invocation_rate'] for p in frontier_sorted]
        y = [p['miss_rate'] for p in frontier_sorted]
        auc = float(np.trapz(y, x))
    else:
        auc = 1.0

    # Best F1
    best_f1 = max(p['f1'] for p in pareto_points)

    # Invocation rate at miss <= 5%
    inv_at_5pct = min(
        (p['invocation_rate'] for p in pareto_points if p['miss_rate'] <= 0.05),
        default=1.0
    )

    return {
        'pareto_points': pareto_points,
        'pareto_frontier': frontier_sorted,
        'auc': auc,
        'best_f1': best_f1,
        'inv_rate_at_miss_5pct': inv_at_5pct,
        'risk_score_stats': {
            'mean': float(np.mean(risk_scores)),
            'std': float(np.std(risk_scores)),
            'min': float(np.min(risk_scores)),
            'max': float(np.max(risk_scores)),
        },
    }


def generate_latex_table(results: Dict) -> str:
    """Generate comparison table."""
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Risk function $R(\mathcal{H}_t)$ ablation on CMAPSS FD001 (5 seeds). '
        r'AUC: area under Pareto curve (lower=better). Inv.\ Rate at Miss$\leq$5\%: '
        r'minimum invocation rate achieving $\leq$5\% miss rate.}',
        r'\label{tab:risk_ablation}',
        r'\begin{tabular}{llcccc}',
        r'\toprule',
        r'Variant & Description & AUC$\downarrow$ & Best F1$\uparrow$ & Inv.@Miss$\leq$5\% & Rank \\',
        r'\midrule',
    ]

    agg = results.get('aggregated', {})
    # Sort by AUC
    sorted_variants = sorted(agg.items(), key=lambda x: x[1].get('auc_mean', 1.0))

    for rank, (name, data) in enumerate(sorted_variants, 1):
        desc = data.get('description', '')[:25]
        auc = data.get('auc_mean', 0)
        auc_std = data.get('auc_std', 0)
        f1 = data.get('best_f1_mean', 0)
        inv = data.get('inv_at_5pct_mean', 1)

        bold = r'\textbf{' if rank == 1 else ''
        end_bold = '}' if rank == 1 else ''

        lines.append(
            f'  {bold}{name}{end_bold} & {desc} '
            f'& {bold}{auc:.4f}$\\pm${auc_std:.4f}{end_bold} '
            f'& {f1:.3f} & {inv:.3f} & {rank} \\\\'
        )

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def run_experiment():
    """Run risk function ablation study."""
    print("=" * 60)
    print("ECML-PKDD Experiment 5: Risk Function Ablation")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = UAIExperimentConfig(quick=False)
    config.n_seeds = 5
    config.epochs = 50

    risk_functions = [
        R_LinearCombo(1.0, 1.0),
        R_AnomalyOnly(),
        R_UncertaintyOnly(),
        R_Multiplicative(),
        R_Max(),
        R_EWMA(0.9),
        R_Learned(),
        R_Quantile(200),
    ]

    all_results = {'per_seed': {}, 'aggregated': {}}

    for seed_idx in range(config.n_seeds):
        seed = config.seed + seed_idx
        print(f"\n  Seed {seed} ({seed_idx + 1}/{config.n_seeds})...")

        outputs = collect_outputs_with_labels(config, seed)
        if not outputs:
            print(f"    No data for seed {seed}")
            continue

        # Train learned risk function on first 30%
        train_size = int(0.3 * len(outputs))
        train_outputs = outputs[:train_size]
        eval_outputs = outputs[train_size:]

        # Prepare features for learned model
        features = []
        prev_a, prev_u = 0, 0
        for i, item in enumerate(train_outputs):
            features.append([
                item['anomaly_score'], item['uncertainty'],
                item['anomaly_score'] - prev_a, item['uncertainty'] - prev_u,
                min(i, 100),  # t_since_trigger proxy
            ])
            prev_a, prev_u = item['anomaly_score'], item['uncertainty']
        features = np.array(features)
        labels = np.array([item['is_critical'] for item in train_outputs]).astype(int)

        # Train R7
        for rf in risk_functions:
            if isinstance(rf, R_Learned):
                rf.train(features, labels)

        # Compute theta grid based on risk score range
        # First pass to get range
        temp_rf = R_LinearCombo()
        temp_scores = [temp_rf.compute(o['anomaly_score'], o['uncertainty'], 0, 0) for o in eval_outputs]
        theta_min = max(0, np.percentile(temp_scores, 50))
        theta_max = np.percentile(temp_scores, 99.9)

        seed_results = {}
        for rf in risk_functions:
            print(f"    Evaluating {rf.name}...")
            rf.reset()

            # Compute range for this specific R
            rf_scores = []
            for t, item in enumerate(eval_outputs):
                rf_scores.append(rf.compute(item['anomaly_score'], item['uncertainty'], t, 100))
            rf.reset()

            rf_scores = np.array(rf_scores)
            t_min = np.percentile(rf_scores, 30)
            t_max = np.percentile(rf_scores, 99.5)
            theta_grid = np.linspace(t_min, t_max, 20)

            result = evaluate_risk_function(rf, eval_outputs, theta_grid)
            result['description'] = rf.description
            seed_results[rf.name] = result

        all_results['per_seed'][str(seed)] = seed_results

    # Aggregate across seeds
    rf_names = [rf.name for rf in risk_functions]
    for rf_name in rf_names:
        aucs, f1s, invs = [], [], []
        desc = ''
        for seed_data in all_results['per_seed'].values():
            if rf_name in seed_data:
                aucs.append(seed_data[rf_name]['auc'])
                f1s.append(seed_data[rf_name]['best_f1'])
                invs.append(seed_data[rf_name]['inv_rate_at_miss_5pct'])
                desc = seed_data[rf_name].get('description', '')

        if aucs:
            all_results['aggregated'][rf_name] = {
                'description': desc,
                'auc_mean': float(np.mean(aucs)),
                'auc_std': float(np.std(aucs)),
                'best_f1_mean': float(np.mean(f1s)),
                'best_f1_std': float(np.std(f1s)),
                'inv_at_5pct_mean': float(np.mean(invs)),
                'inv_at_5pct_std': float(np.std(invs)),
                'n_seeds': len(aucs),
            }

    # Save
    save_results(all_results, str(RESULTS_DIR / 'risk_function_ablation.json'))

    # Generate LaTeX
    latex = generate_latex_table(all_results)
    with open(RESULTS_DIR / 'risk_function_ablation_table.tex', 'w') as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {RESULTS_DIR / 'risk_function_ablation_table.tex'}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by AUC)")
    print("=" * 60)
    sorted_rf = sorted(all_results['aggregated'].items(), key=lambda x: x[1]['auc_mean'])
    for rank, (name, data) in enumerate(sorted_rf, 1):
        print(f"  #{rank} {name}: AUC={data['auc_mean']:.4f}+/-{data['auc_std']:.4f}, "
              f"F1={data['best_f1_mean']:.3f}, Inv@Miss5%={data['inv_at_5pct_mean']:.3f}")

    return all_results


if __name__ == '__main__':
    run_experiment()
