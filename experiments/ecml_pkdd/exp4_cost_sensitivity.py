"""
Experiment 4: Cost and Latency Sensitivity Analysis
Sweeps c_LLM and latency to show how optimal threshold and trigger behavior change.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.uai_experiments.config import UAIExperimentConfig, set_seed, save_results, get_data_dir
from src.data.cmapss import CMAPSSDataset, create_sequences
from src.models.fast.gru import GRUModel
from src.triggers.threshold import ThresholdTrigger
from src.triggers.cusum import CUSUMTrigger
from src.triggers.sprt import SPRTTrigger
from src.triggers.optimal_stopping import OptimalStoppingTrigger
from src.data.base import StreamSample

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'ecml_pkdd'

C_MISS = 100.0  # Cost of missing a critical event


def collect_model_outputs(config: UAIExperimentConfig, seed: int) -> List[Dict]:
    """Train model and collect all outputs for downstream trigger evaluation."""
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
        outputs.append({
            'sample': sample,
            'output': output,
            'rul': sample.label if sample.label is not None else 100.0,
        })

    return outputs


def evaluate_trigger_at_cost(outputs: List[Dict], trigger_type: str,
                             c_llm: float, threshold_value: float) -> Dict:
    """Evaluate a trigger at given cost and threshold."""
    if trigger_type == 'threshold':
        trigger = ThresholdTrigger(
            anomaly_threshold=threshold_value,
            uncertainty_threshold=threshold_value * 0.5,
            cooldown_steps=5, min_evidence_window=3)
    elif trigger_type == 'cusum':
        trigger = CUSUMTrigger(
            slack=0.5, threshold=threshold_value,
            warmup_samples=30, cooldown_steps=5, min_evidence_window=3)
    elif trigger_type == 'sprt':
        trigger = SPRTTrigger(
            alpha=0.05, beta=0.10,
            warmup_samples=30, cooldown_steps=5, min_evidence_window=3)
    elif trigger_type == 'optimal_stopping':
        trigger = OptimalStoppingTrigger(
            llm_cost=c_llm, risk_weight=0.1,
            risk_threshold=threshold_value,
            cooldown_steps=5, min_evidence_window=3)
    else:
        return {}

    n_invocations = 0
    n_misses = 0
    n_critical = 0

    for item in outputs:
        result = trigger.step(item['sample'], item['output'])
        is_critical = item['rul'] < 30

        if is_critical:
            n_critical += 1
            if not result.should_trigger:
                n_misses += 1

        if result.should_trigger:
            n_invocations += 1

    total_cost = c_llm * n_invocations + C_MISS * n_misses
    n_total = len(outputs)

    return {
        'invocation_rate': n_invocations / n_total if n_total > 0 else 0,
        'miss_rate': n_misses / n_critical if n_critical > 0 else 0,
        'n_invocations': n_invocations,
        'n_misses': n_misses,
        'n_critical': n_critical,
        'total_cost': total_cost,
        'invocation_cost': c_llm * n_invocations,
        'miss_cost': C_MISS * n_misses,
    }


def cost_sensitivity_sweep(config: UAIExperimentConfig) -> Dict:
    """Sweep c_LLM to find optimal threshold at each cost level."""
    print("Running cost sensitivity sweep...")
    c_llm_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    trigger_types = ['threshold', 'cusum', 'optimal_stopping']
    theta_grid = np.linspace(0.1, 10.0, 20)

    results = {'c_llm_values': c_llm_values, 'triggers': {}}

    for trigger_type in trigger_types:
        print(f"  Trigger: {trigger_type}")
        trigger_results = {'per_cost': {}}

        for seed_idx in range(config.n_seeds):
            seed = config.seed + seed_idx
            outputs = collect_model_outputs(config, seed)
            if not outputs:
                continue

            for c_llm in c_llm_values:
                key = f"c{c_llm}"
                if key not in trigger_results['per_cost']:
                    trigger_results['per_cost'][key] = {'seeds': {}}

                # Sweep threshold to find optimal
                best_cost = float('inf')
                best_theta = theta_grid[0]
                best_result = {}
                all_theta_results = []

                for theta in theta_grid:
                    r = evaluate_trigger_at_cost(outputs, trigger_type, c_llm, theta)
                    all_theta_results.append({
                        'theta': float(theta),
                        'total_cost': r['total_cost'],
                        'invocation_rate': r['invocation_rate'],
                        'miss_rate': r['miss_rate'],
                    })
                    if r['total_cost'] < best_cost:
                        best_cost = r['total_cost']
                        best_theta = theta
                        best_result = r

                trigger_results['per_cost'][key]['seeds'][str(seed)] = {
                    'optimal_theta': float(best_theta),
                    'optimal_cost': best_cost,
                    **best_result,
                    'theta_sweep': all_theta_results,
                }

        # Aggregate
        for key in trigger_results['per_cost']:
            seeds = trigger_results['per_cost'][key]['seeds']
            if seeds:
                thetas = [s['optimal_theta'] for s in seeds.values()]
                costs = [s['optimal_cost'] for s in seeds.values()]
                inv_rates = [s['invocation_rate'] for s in seeds.values()]
                miss_rates = [s['miss_rate'] for s in seeds.values()]
                trigger_results['per_cost'][key]['summary'] = {
                    'optimal_theta_mean': float(np.mean(thetas)),
                    'optimal_theta_std': float(np.std(thetas)),
                    'optimal_cost_mean': float(np.mean(costs)),
                    'invocation_rate_mean': float(np.mean(inv_rates)),
                    'miss_rate_mean': float(np.mean(miss_rates)),
                }

        results['triggers'][trigger_type] = trigger_results

    return results


def latency_sensitivity(config: UAIExperimentConfig) -> Dict:
    """Analyze throughput degradation at different LLM latencies."""
    print("Running latency sensitivity analysis...")
    latencies_ms = [10, 50, 100, 500, 1000, 5000, 10000]
    fast_inference_ms = 5.0  # GRU inference time

    # Get typical invocation rates from a single seed
    seed = config.seed
    outputs = collect_model_outputs(config, seed)
    if not outputs:
        return {}

    results = {'latencies_ms': latencies_ms, 'analyses': []}

    trigger = ThresholdTrigger(
        anomaly_threshold=1.0, uncertainty_threshold=0.5,
        cooldown_steps=5, min_evidence_window=3)

    n_triggers = 0
    for item in outputs:
        result = trigger.step(item['sample'], item['output'])
        if result.should_trigger:
            n_triggers += 1

    inv_rate = n_triggers / len(outputs)

    for lat in latencies_ms:
        # Average time per sample = fast_time + inv_rate * llm_time
        avg_time_ms = fast_inference_ms + inv_rate * lat
        throughput = 1000.0 / avg_time_ms  # samples/sec
        baseline_throughput = 1000.0 / fast_inference_ms
        degradation = throughput / baseline_throughput

        results['analyses'].append({
            'llm_latency_ms': lat,
            'avg_time_per_sample_ms': avg_time_ms,
            'throughput_samples_per_sec': throughput,
            'baseline_throughput': baseline_throughput,
            'throughput_ratio': degradation,
            'invocation_rate': inv_rate,
        })

    return results


def cost_robustness(config: UAIExperimentConfig) -> Dict:
    """Test robustness to cost uncertainty."""
    print("Running cost robustness analysis...")
    nominal_costs = [0.1, 1.0, 10.0]
    noise_levels = [0.1, 0.3, 0.5, 1.0]

    seed = config.seed
    outputs = collect_model_outputs(config, seed)
    if not outputs:
        return {}

    results = {'analyses': []}

    for c_nominal in nominal_costs:
        # First get optimal theta at known cost
        theta_grid = np.linspace(0.1, 10.0, 20)
        best_theta = 1.0
        best_cost = float('inf')
        for theta in theta_grid:
            r = evaluate_trigger_at_cost(outputs, 'threshold', c_nominal, theta)
            if r['total_cost'] < best_cost:
                best_cost = r['total_cost']
                best_theta = theta

        known_cost_result = evaluate_trigger_at_cost(outputs, 'threshold', c_nominal, best_theta)

        for sigma in noise_levels:
            # Simulate cost uncertainty
            np.random.seed(42)
            noisy_costs = []
            for _ in range(100):
                c_actual = c_nominal * np.exp(np.random.randn() * sigma)
                r = evaluate_trigger_at_cost(outputs, 'threshold', c_actual, best_theta)
                noisy_costs.append(r['total_cost'])

            results['analyses'].append({
                'c_nominal': c_nominal,
                'sigma': sigma,
                'known_cost': known_cost_result['total_cost'],
                'noisy_cost_mean': float(np.mean(noisy_costs)),
                'noisy_cost_std': float(np.std(noisy_costs)),
                'cost_degradation_ratio': float(np.mean(noisy_costs) / max(known_cost_result['total_cost'], 1e-6)),
                'optimal_theta': float(best_theta),
            })

    return results


def pareto_frontier(config: UAIExperimentConfig) -> Dict:
    """Compute joint cost-quality Pareto frontier."""
    print("Computing Pareto frontier...")
    c_llm_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    theta_grid = np.linspace(0.1, 10.0, 30)

    seed = config.seed
    outputs = collect_model_outputs(config, seed)
    if not outputs:
        return {}

    all_points = []
    for c_llm in c_llm_values:
        for theta in theta_grid:
            r = evaluate_trigger_at_cost(outputs, 'threshold', c_llm, theta)
            all_points.append({
                'c_llm': c_llm,
                'theta': float(theta),
                'total_cost': r['total_cost'],
                'miss_rate': r['miss_rate'],
                'invocation_rate': r['invocation_rate'],
            })

    # Find Pareto frontier (minimize both total_cost and miss_rate)
    pareto = []
    for p in all_points:
        dominated = False
        for q in all_points:
            if q['total_cost'] <= p['total_cost'] and q['miss_rate'] <= p['miss_rate'] and \
               (q['total_cost'] < p['total_cost'] or q['miss_rate'] < p['miss_rate']):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    # Find knee point (largest curvature change)
    pareto_sorted = sorted(pareto, key=lambda x: x['total_cost'])
    knee_idx = 0
    if len(pareto_sorted) > 2:
        costs = [p['total_cost'] for p in pareto_sorted]
        misses = [p['miss_rate'] for p in pareto_sorted]
        # Normalized curvature
        max_curvature = 0
        for i in range(1, len(pareto_sorted) - 1):
            dc1 = costs[i] - costs[i - 1]
            dc2 = costs[i + 1] - costs[i]
            dm1 = misses[i] - misses[i - 1]
            dm2 = misses[i + 1] - misses[i]
            curvature = abs(dc2 * dm1 - dc1 * dm2)
            if curvature > max_curvature:
                max_curvature = curvature
                knee_idx = i

    return {
        'all_points': all_points[:500],  # Limit size
        'pareto_frontier': pareto_sorted,
        'knee_point': pareto_sorted[knee_idx] if pareto_sorted else {},
        'n_pareto_points': len(pareto_sorted),
    }


def generate_latex_table(results: Dict) -> str:
    """Generate cost sensitivity LaTeX table."""
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Optimal threshold $\theta^*$ and performance at different LLM costs $c_{\text{LLM}}$. '
        r'$c_{\text{miss}}=100$ fixed. Results averaged over 5 seeds on CMAPSS FD001.}',
        r'\label{tab:cost_sensitivity}',
        r'\begin{tabular}{rcccccc}',
        r'\toprule',
        r'$c_{\text{LLM}}$ & \multicolumn{2}{c}{Threshold} & \multicolumn{2}{c}{CUSUM} & \multicolumn{2}{c}{Opt. Stopping} \\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}',
        r'& $\theta^*$ & Miss\% & $\theta^*$ & Miss\% & $\theta^*$ & Miss\% \\',
        r'\midrule',
    ]

    cost_sweep = results.get('cost_sweep', {})
    c_values = cost_sweep.get('c_llm_values', [])

    for c_llm in c_values:
        key = f"c{c_llm}"
        row = f'  {c_llm}'

        for trig in ['threshold', 'cusum', 'optimal_stopping']:
            tdata = cost_sweep.get('triggers', {}).get(trig, {}).get('per_cost', {}).get(key, {})
            summary = tdata.get('summary', {})
            theta = summary.get('optimal_theta_mean', 0)
            miss = summary.get('miss_rate_mean', 0)
            row += f' & {theta:.2f} & {miss:.1%}'

        row += r' \\'
        lines.append(row)

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    # Latency table
    lines.extend([
        '', '',
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Throughput degradation at different LLM latencies (invocation rate from threshold trigger).}',
        r'\label{tab:latency_sensitivity}',
        r'\begin{tabular}{rccc}',
        r'\toprule',
        r'LLM Latency (ms) & Avg. Time/Sample (ms) & Throughput (samples/s) & Ratio \\',
        r'\midrule',
    ])

    for a in results.get('latency', {}).get('analyses', []):
        lines.append(
            f'  {a["llm_latency_ms"]} & {a["avg_time_per_sample_ms"]:.1f} '
            f'& {a["throughput_samples_per_sec"]:.1f} & {a["throughput_ratio"]:.3f} \\\\'
        )

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def run_experiment():
    """Run all cost/latency sensitivity experiments."""
    print("=" * 60)
    print("ECML-PKDD Experiment 4: Cost & Latency Sensitivity Analysis")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = UAIExperimentConfig(quick=False)
    config.n_seeds = 5
    config.epochs = 50

    all_results = {}

    # 1. Cost sensitivity sweep
    all_results['cost_sweep'] = cost_sensitivity_sweep(config)

    # 2. Latency sensitivity
    all_results['latency'] = latency_sensitivity(config)

    # 3. Pareto frontier
    all_results['pareto'] = pareto_frontier(config)

    # 4. Cost robustness
    all_results['robustness'] = cost_robustness(config)

    # Save
    save_results(all_results, str(RESULTS_DIR / 'cost_sensitivity.json'))

    # Generate LaTeX
    latex = generate_latex_table(all_results)
    with open(RESULTS_DIR / 'cost_sensitivity_table.tex', 'w') as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {RESULTS_DIR / 'cost_sensitivity_table.tex'}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Cost sweep summary
    cost_sweep = all_results.get('cost_sweep', {})
    for trig in ['threshold', 'cusum', 'optimal_stopping']:
        tdata = cost_sweep.get('triggers', {}).get(trig, {}).get('per_cost', {})
        print(f"\n  {trig}:")
        for key in sorted(tdata.keys()):
            s = tdata[key].get('summary', {})
            print(f"    {key}: theta*={s.get('optimal_theta_mean', 0):.2f}, "
                  f"inv_rate={s.get('invocation_rate_mean', 0):.3f}, "
                  f"miss_rate={s.get('miss_rate_mean', 0):.3f}")

    # Pareto knee
    knee = all_results.get('pareto', {}).get('knee_point', {})
    if knee:
        print(f"\n  Pareto knee: c_LLM={knee.get('c_llm', 0)}, theta={knee.get('theta', 0):.2f}, "
              f"cost={knee.get('total_cost', 0):.1f}, miss_rate={knee.get('miss_rate', 0):.3f}")

    return all_results


if __name__ == '__main__':
    run_experiment()
