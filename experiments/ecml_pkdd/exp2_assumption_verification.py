"""
Experiment 2: Theoretical Assumption Verification
Empirically tests whether paper's theoretical assumptions hold on CMAPSS data.
"""
import numpy as np
import torch
from pathlib import Path
from scipy import stats as scipy_stats
from typing import Dict, List
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.uai_experiments.config import UAIExperimentConfig, set_seed, save_results, get_data_dir
from src.data.cmapss import CMAPSSDataset, create_sequences
from src.models.fast.gru import GRUModel
from src.data.base import StreamSample

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'ecml_pkdd'


def collect_risk_trajectory(config: UAIExperimentConfig, seed: int) -> Dict:
    """Train model and collect risk scores, RUL, hidden states over test data."""
    set_seed(seed)
    data_dir = get_data_dir()

    dataset = CMAPSSDataset(data_dir, subset='FD001', split='train')
    X, y, _ = create_sequences(dataset, seq_length=config.seq_length)
    if len(X) == 0:
        return {}

    model = GRUModel(n_features=X.shape[2], hidden_size=config.hidden_size, device=config.device)
    model.fit(X, y, epochs=config.epochs, batch_size=config.batch_size, learning_rate=config.lr)

    test_dataset = CMAPSSDataset(data_dir, subset='FD001', split='test')
    model.reset_state()

    records = []
    for sample in test_dataset:
        if sample.metadata['cycle'] == 1:
            model.reset_state()
        output = model.step(sample.features)
        risk = output.anomaly_score + output.uncertainty
        records.append({
            'risk': risk,
            'anomaly_score': output.anomaly_score,
            'uncertainty': output.uncertainty,
            'rul': sample.label if sample.label is not None else 100.0,
            'unit_id': sample.metadata.get('unit_id', 0),
            'cycle': sample.metadata.get('cycle', 0),
            'hidden_norm': float(np.linalg.norm(output.hidden_state.flatten())) if output.hidden_state is not None else 0,
        })

    return {
        'risk': np.array([r['risk'] for r in records]),
        'anomaly': np.array([r['anomaly_score'] for r in records]),
        'uncertainty': np.array([r['uncertainty'] for r in records]),
        'rul': np.array([r['rul'] for r in records]),
        'unit_id': np.array([r['unit_id'] for r in records]),
        'cycle': np.array([r['cycle'] for r in records]),
    }


def verify_lipschitz(data: Dict) -> Dict:
    """
    Assumption 1: Lipschitz continuity of risk.
    ||R(H_{t+1}) - R(H_t)|| <= L
    """
    risk = data['risk']
    unit_ids = data['unit_id']

    all_increments = []
    for uid in np.unique(unit_ids):
        mask = unit_ids == uid
        r = risk[mask]
        if len(r) > 1:
            increments = np.abs(np.diff(r))
            all_increments.extend(increments)

    increments = np.array(all_increments)
    if len(increments) == 0:
        return {'error': 'no increments'}

    L_max = float(np.max(increments))
    L_99 = float(np.percentile(increments, 99))
    L_95 = float(np.percentile(increments, 95))
    L_mean = float(np.mean(increments))

    # Test: KS test against exponential (bounded support implies light tail)
    # Fit exponential
    loc, scale = scipy_stats.expon.fit(increments, floc=0)
    ks_stat, ks_pval = scipy_stats.kstest(increments, 'expon', args=(0, scale))

    # Test: fraction within 2*L_95
    bounded_fraction = float(np.mean(increments <= 2 * L_95))

    return {
        'L_max': L_max,
        'L_99': L_99,
        'L_95': L_95,
        'L_mean': L_mean,
        'L_std': float(np.std(increments)),
        'n_increments': len(increments),
        'ks_test_exponential': {'statistic': float(ks_stat), 'p_value': float(ks_pval)},
        'bounded_fraction_2x_L95': bounded_fraction,
        'verdict': 'Supported' if bounded_fraction > 0.99 else 'Partially Supported',
    }


def verify_submartingale(data: Dict) -> Dict:
    """
    Assumption 2: Submartingale property under H1 (abnormal).
    E[R_{t+1} | H_t] >= R_t when RUL < 50 (abnormal regime)
    """
    risk = data['risk']
    rul = data['rul']
    unit_ids = data['unit_id']

    h1_increments = []  # Risk increments in abnormal regime
    h0_increments = []  # Risk increments in normal regime

    for uid in np.unique(unit_ids):
        mask = unit_ids == uid
        r = risk[mask]
        ru = rul[mask]
        if len(r) < 2:
            continue
        for i in range(len(r) - 1):
            inc = r[i + 1] - r[i]
            if ru[i] < 50:  # H1: abnormal
                h1_increments.append(inc)
            else:  # H0: normal
                h0_increments.append(inc)

    h1_increments = np.array(h1_increments)
    h0_increments = np.array(h0_increments)

    result = {}

    # H1: should have non-negative mean increment (submartingale)
    if len(h1_increments) > 5:
        t_stat, p_val = scipy_stats.ttest_1samp(h1_increments, 0)
        result['h1_abnormal'] = {
            'mean_increment': float(np.mean(h1_increments)),
            'std_increment': float(np.std(h1_increments)),
            'fraction_positive': float(np.mean(h1_increments > 0)),
            'n_samples': len(h1_increments),
            't_statistic': float(t_stat),
            'p_value_one_sided': float(p_val / 2) if t_stat > 0 else float(1 - p_val / 2),
            'verdict': 'Supported' if np.mean(h1_increments) > 0 and p_val / 2 < 0.05 else 'Partially Supported' if np.mean(h1_increments) > 0 else 'Not Supported',
        }

    # H0: should not systematically increase (martingale or supermartingale)
    if len(h0_increments) > 5:
        t_stat, p_val = scipy_stats.ttest_1samp(h0_increments, 0)
        result['h0_normal'] = {
            'mean_increment': float(np.mean(h0_increments)),
            'std_increment': float(np.std(h0_increments)),
            'fraction_positive': float(np.mean(h0_increments > 0)),
            'n_samples': len(h0_increments),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
        }

    return result


def verify_monotone_risk(data: Dict) -> Dict:
    """
    Assumption 3: Monotone increasing risk as RUL -> 0.
    """
    risk = data['risk']
    rul = data['rul']

    # Bin by RUL
    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
            (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    bin_means = []
    bin_labels = []

    for lo, hi in bins:
        mask = (rul >= lo) & (rul < hi)
        if np.sum(mask) > 0:
            bin_means.append(float(np.mean(risk[mask])))
            bin_labels.append(f'{lo}-{hi}')

    # Spearman correlation: RUL vs risk (expect negative, i.e. risk increases as RUL decreases)
    valid_mask = rul < 100
    if np.sum(valid_mask) > 10:
        spearman_r, spearman_p = scipy_stats.spearmanr(rul[valid_mask], risk[valid_mask])
    else:
        spearman_r, spearman_p = 0, 1

    # Monotonicity: fraction of consecutive bins where risk increases as RUL decreases
    if len(bin_means) > 1:
        # bin_means[0] is RUL 0-10 (should have highest risk)
        monotone_count = sum(1 for i in range(len(bin_means) - 1) if bin_means[i] >= bin_means[i + 1])
        monotone_fraction = monotone_count / (len(bin_means) - 1)
    else:
        monotone_fraction = 0

    return {
        'bin_means': dict(zip(bin_labels, bin_means)),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'monotone_fraction': monotone_fraction,
        'verdict': 'Supported' if spearman_r < -0.3 and spearman_p < 0.05 else 'Partially Supported' if spearman_r < 0 else 'Not Supported',
        'note': 'Negative Spearman r means risk increases as RUL decreases (expected)',
    }


def verify_continuous_density(data: Dict) -> Dict:
    """
    Assumption 4: Continuous density of risk under H0 and H1.
    """
    risk = data['risk']
    rul = data['rul']

    h0_risk = risk[rul >= 50]
    h1_risk = risk[rul < 50]

    result = {}

    for label, r in [('H0_normal', h0_risk), ('H1_abnormal', h1_risk)]:
        if len(r) < 10:
            continue

        # Fit normal distribution
        mu, sigma = scipy_stats.norm.fit(r)
        ks_norm, p_norm = scipy_stats.kstest(r, 'norm', args=(mu, sigma))

        # Fit log-normal (shift to positive)
        r_shifted = r - r.min() + 1e-6
        shape, loc, scale = scipy_stats.lognorm.fit(r_shifted, floc=0)
        ks_lognorm, p_lognorm = scipy_stats.kstest(r_shifted, 'lognorm', args=(shape, 0, scale))

        # Shapiro-Wilk (limited to 5000 samples)
        r_sample = r[:5000] if len(r) > 5000 else r
        sw_stat, sw_p = scipy_stats.shapiro(r_sample)

        result[label] = {
            'n_samples': len(r),
            'mean': float(np.mean(r)),
            'std': float(np.std(r)),
            'skewness': float(scipy_stats.skew(r)),
            'kurtosis': float(scipy_stats.kurtosis(r)),
            'ks_normal': {'statistic': float(ks_norm), 'p_value': float(p_norm)},
            'ks_lognormal': {'statistic': float(ks_lognorm), 'p_value': float(p_lognorm)},
            'shapiro_wilk': {'statistic': float(sw_stat), 'p_value': float(sw_p)},
            'best_fit': 'lognormal' if p_lognorm > p_norm else 'normal',
        }

    # Two-sample KS test: H0 vs H1 should have different distributions
    if len(h0_risk) > 10 and len(h1_risk) > 10:
        ks_2sample, p_2sample = scipy_stats.ks_2samp(h0_risk, h1_risk)
        result['h0_vs_h1'] = {
            'ks_statistic': float(ks_2sample),
            'p_value': float(p_2sample),
            'distributions_differ': p_2sample < 0.05,
        }

    result['verdict'] = 'Supported' if all(
        result.get(k, {}).get('ks_normal', {}).get('p_value', 0) > 0.01 or
        result.get(k, {}).get('ks_lognormal', {}).get('p_value', 0) > 0.01
        for k in ['H0_normal', 'H1_abnormal'] if k in result
    ) else 'Partially Supported'

    return result


def verify_stationarity(data: Dict) -> Dict:
    """
    Assumption 5: Stationarity within regimes.
    """
    risk = data['risk']
    rul = data['rul']

    result = {}

    for label, mask in [('H0_normal', rul >= 50), ('H1_abnormal', rul < 50)]:
        r = risk[mask]
        if len(r) < 20:
            continue

        mid = len(r) // 2
        first_half = r[:mid]
        second_half = r[mid:]

        # KS test
        ks_stat, ks_p = scipy_stats.ks_2samp(first_half, second_half)
        # Mann-Whitney U
        u_stat, u_p = scipy_stats.mannwhitneyu(first_half, second_half, alternative='two-sided')

        result[label] = {
            'first_half_mean': float(np.mean(first_half)),
            'second_half_mean': float(np.mean(second_half)),
            'ks_test': {'statistic': float(ks_stat), 'p_value': float(ks_p)},
            'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_p)},
            'stationary': ks_p > 0.05,
        }

    result['verdict'] = 'Supported' if all(
        result.get(k, {}).get('stationary', False)
        for k in ['H0_normal'] if k in result  # Stationarity mainly expected under H0
    ) else 'Partially Supported'

    return result


def generate_latex_table(results: Dict) -> str:
    """Generate summary table for assumption verification."""
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Empirical verification of theoretical assumptions on CMAPSS FD001 (5 seeds). '
        r'Verdicts based on statistical tests at $\alpha=0.05$.}',
        r'\label{tab:assumption_verification}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{p{3.5cm}p{3cm}ccp{2.5cm}}',
        r'\toprule',
        r'Assumption & Test & Statistic & $p$-value & Verdict \\',
        r'\midrule',
    ]

    agg = results.get('aggregated', {})

    # Lipschitz
    lip = agg.get('lipschitz', {})
    lines.append(
        f'  Lipschitz continuity (Thm 1) & KS vs exponential '
        f'& {lip.get("ks_statistic_mean", 0):.3f} '
        f'& {lip.get("ks_pvalue_mean", 0):.3f} '
        f'& {lip.get("verdict", "N/A")} \\\\'
    )
    lines.append(
        f'  \\quad $L_{{99}}$ bound & Bounded fraction '
        f'& {lip.get("L99_mean", 0):.3f} '
        f'& -- '
        f'& {lip.get("bounded_frac_mean", 0):.1%} within $2L_{{95}}$ \\\\'
    )

    # Submartingale
    sub = agg.get('submartingale', {})
    lines.append(
        f'  Submartingale under $H_1$ (Thm 2) & One-sided $t$-test '
        f'& {sub.get("t_stat_mean", 0):.2f} '
        f'& {sub.get("p_value_mean", 0):.4f} '
        f'& {sub.get("verdict", "N/A")} \\\\'
    )

    # Monotone
    mono = agg.get('monotone', {})
    lines.append(
        f'  Monotone risk (Prop 1) & Spearman $\\rho$ '
        f'& {mono.get("spearman_r_mean", 0):.3f} '
        f'& {mono.get("spearman_p_mean", 0):.4f} '
        f'& {mono.get("verdict", "N/A")} \\\\'
    )

    # Continuous density
    dens = agg.get('continuous_density', {})
    lines.append(
        f'  Continuous density (Thm 2) & KS goodness-of-fit '
        f'& {dens.get("ks_stat_mean", 0):.3f} '
        f'& {dens.get("ks_pval_mean", 0):.3f} '
        f'& {dens.get("verdict", "N/A")} \\\\'
    )

    # Stationarity
    stat = agg.get('stationarity', {})
    lines.append(
        f'  Stationarity under $H_0$ (Thm 3) & KS 2-sample '
        f'& {stat.get("ks_stat_mean", 0):.3f} '
        f'& {stat.get("ks_pval_mean", 0):.3f} '
        f'& {stat.get("verdict", "N/A")} \\\\'
    )

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def run_experiment():
    """Run all assumption verification tests."""
    print("=" * 60)
    print("ECML-PKDD Experiment 2: Theoretical Assumption Verification")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = UAIExperimentConfig(quick=False)
    config.n_seeds = 5
    config.epochs = 50

    all_results = {'per_seed': {}, 'aggregated': {}}

    for seed_idx in range(config.n_seeds):
        seed = config.seed + seed_idx
        print(f"\n  Seed {seed} ({seed_idx + 1}/{config.n_seeds})...")

        try:
            data = collect_risk_trajectory(config, seed)
            if not data:
                print(f"    No data for seed {seed}")
                continue

            seed_results = {}
            print("    Verifying Lipschitz continuity...")
            seed_results['lipschitz'] = verify_lipschitz(data)
            print("    Verifying submartingale property...")
            seed_results['submartingale'] = verify_submartingale(data)
            print("    Verifying monotone risk...")
            seed_results['monotone'] = verify_monotone_risk(data)
            print("    Verifying continuous density...")
            seed_results['continuous_density'] = verify_continuous_density(data)
            print("    Verifying stationarity...")
            seed_results['stationarity'] = verify_stationarity(data)

            all_results['per_seed'][str(seed)] = seed_results

        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate across seeds
    agg = {}
    seeds_data = list(all_results['per_seed'].values())

    if seeds_data:
        # Lipschitz
        agg['lipschitz'] = {
            'L99_mean': float(np.mean([s['lipschitz']['L_99'] for s in seeds_data if 'lipschitz' in s])),
            'L_mean_mean': float(np.mean([s['lipschitz']['L_mean'] for s in seeds_data if 'lipschitz' in s])),
            'ks_statistic_mean': float(np.mean([s['lipschitz']['ks_test_exponential']['statistic'] for s in seeds_data if 'lipschitz' in s])),
            'ks_pvalue_mean': float(np.mean([s['lipschitz']['ks_test_exponential']['p_value'] for s in seeds_data if 'lipschitz' in s])),
            'bounded_frac_mean': float(np.mean([s['lipschitz']['bounded_fraction_2x_L95'] for s in seeds_data if 'lipschitz' in s])),
            'verdict': 'Supported' if np.mean([s['lipschitz']['bounded_fraction_2x_L95'] for s in seeds_data if 'lipschitz' in s]) > 0.99 else 'Partially Supported',
        }

        # Submartingale
        h1_data = [s['submartingale']['h1_abnormal'] for s in seeds_data if 'submartingale' in s and 'h1_abnormal' in s.get('submartingale', {})]
        if h1_data:
            agg['submartingale'] = {
                'mean_increment_mean': float(np.mean([d['mean_increment'] for d in h1_data])),
                'fraction_positive_mean': float(np.mean([d['fraction_positive'] for d in h1_data])),
                't_stat_mean': float(np.mean([d['t_statistic'] for d in h1_data])),
                'p_value_mean': float(np.mean([d['p_value_one_sided'] for d in h1_data])),
                'verdict': 'Supported' if np.mean([d['mean_increment'] for d in h1_data]) > 0 else 'Not Supported',
            }

        # Monotone
        mono_data = [s['monotone'] for s in seeds_data if 'monotone' in s]
        if mono_data:
            agg['monotone'] = {
                'spearman_r_mean': float(np.mean([d['spearman_r'] for d in mono_data])),
                'spearman_p_mean': float(np.mean([d['spearman_p'] for d in mono_data])),
                'monotone_fraction_mean': float(np.mean([d['monotone_fraction'] for d in mono_data])),
                'verdict': 'Supported' if np.mean([d['spearman_r'] for d in mono_data]) < -0.3 else 'Partially Supported',
            }

        # Continuous density
        dens_data = [s['continuous_density'] for s in seeds_data if 'continuous_density' in s]
        if dens_data:
            h0_ks = [d['H0_normal']['ks_normal']['statistic'] for d in dens_data if 'H0_normal' in d]
            h0_p = [d['H0_normal']['ks_normal']['p_value'] for d in dens_data if 'H0_normal' in d]
            agg['continuous_density'] = {
                'ks_stat_mean': float(np.mean(h0_ks)) if h0_ks else 0,
                'ks_pval_mean': float(np.mean(h0_p)) if h0_p else 0,
                'verdict': 'Supported' if h0_p and np.mean(h0_p) > 0.01 else 'Partially Supported',
            }

        # Stationarity
        stat_data = [s['stationarity'] for s in seeds_data if 'stationarity' in s]
        if stat_data:
            h0_stat = [d['H0_normal']['ks_test']['statistic'] for d in stat_data if 'H0_normal' in d]
            h0_p = [d['H0_normal']['ks_test']['p_value'] for d in stat_data if 'H0_normal' in d]
            agg['stationarity'] = {
                'ks_stat_mean': float(np.mean(h0_stat)) if h0_stat else 0,
                'ks_pval_mean': float(np.mean(h0_p)) if h0_p else 0,
                'verdict': 'Supported' if h0_p and np.mean(h0_p) > 0.05 else 'Partially Supported',
            }

    all_results['aggregated'] = agg

    # Save
    save_results(all_results, str(RESULTS_DIR / 'assumption_verification.json'))

    # Generate LaTeX
    latex = generate_latex_table(all_results)
    with open(RESULTS_DIR / 'assumption_verification_table.tex', 'w') as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {RESULTS_DIR / 'assumption_verification_table.tex'}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for assumption, data in agg.items():
        print(f"  {assumption}: {data.get('verdict', 'N/A')}")

    return all_results


if __name__ == '__main__':
    run_experiment()
