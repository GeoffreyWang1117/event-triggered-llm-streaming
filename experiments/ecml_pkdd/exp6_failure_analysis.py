"""
Experiment 6: LLM Diagnosis Failure Analysis
Analyzes failure cases from real LLM experiments to identify failure modes,
safety implications, and suggest fallback strategies.
"""
import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'ecml_pkdd'
REAL_LLM_DIR = Path(__file__).resolve().parents[2] / 'results' / 'real_llm'


def load_all_diagnoses() -> List[Dict]:
    """Load all diagnosis files from real LLM results."""
    all_diags = []
    if not REAL_LLM_DIR.exists():
        print(f"  Warning: {REAL_LLM_DIR} not found")
        return all_diags

    for fpath in sorted(REAL_LLM_DIR.glob('diag_*.json')):
        try:
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, list):
                # Extract metadata from filename
                # Pattern: diag_cmapss_FD001_threshold_ollama_llama3.1_8b.json
                fname = fpath.stem.replace('diag_', '')
                parts = fname.split('_')
                dataset = '_'.join(parts[:2])  # cmapss_FD001
                # Find trigger and backend
                trigger = parts[2] if len(parts) > 2 else 'unknown'
                backend = '_'.join(parts[3:]) if len(parts) > 3 else 'unknown'

                for d in data:
                    d['_source_file'] = fpath.name
                    d['_dataset'] = dataset
                    d['_trigger_type'] = trigger
                    d['_backend'] = backend
                    all_diags.append(d)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: Could not load {fpath.name}: {e}")

    return all_diags


def categorize_grounding(score: float) -> str:
    """Categorize grounding score."""
    if score >= 1.0:
        return 'perfect'
    elif score >= 0.75:
        return 'good'
    elif score >= 0.5:
        return 'partial'
    elif score >= 0.25:
        return 'poor'
    else:
        return 'failure'


def analyze_grounding_component_failures(diag: Dict) -> List[str]:
    """Determine which grounding components failed."""
    failures = []
    response = diag.get('response', {})
    anomaly = diag.get('anomaly', 0)

    # Component 1: Numeric evidence
    txt = response.get('evidence_summary', '') + ' ' + response.get('diagnosis', '')
    if len(re.findall(r'\d+\.?\d*', txt)) < 2:
        failures.append('no_numeric_evidence')

    # Component 2: Key indicators
    if not response.get('key_indicators'):
        failures.append('no_key_indicators')

    # Component 3: Severity match
    sev = response.get('severity', '').lower()
    severity_match = False
    if anomaly > 2.0 and sev in ('high', 'critical'):
        severity_match = True
    elif anomaly < 1.0 and sev in ('low', 'medium'):
        severity_match = True
    elif 1.0 <= anomaly <= 2.0 and sev == 'medium':
        severity_match = True
    if not severity_match:
        failures.append(f'severity_mismatch(anomaly={anomaly:.2f},sev={sev})')

    # Component 4: Confidence
    c = response.get('confidence', 0.5)
    if not (0.1 < c < 0.99):
        failures.append(f'unreasonable_confidence({c:.2f})')

    return failures


def run_analysis() -> Dict:
    """Run complete failure analysis."""
    print("Loading diagnoses...")
    all_diags = load_all_diagnoses()
    print(f"  Loaded {len(all_diags)} diagnoses from {len(set(d['_source_file'] for d in all_diags))} files")

    if not all_diags:
        print("  No diagnoses found. Generating synthetic analysis for demonstration.")
        return _synthetic_analysis()

    results = {
        'total_diagnoses': len(all_diags),
        'distribution': {},
        'by_trigger': {},
        'by_dataset': {},
        'by_backend': {},
        'failure_modes': {},
        'safety_critical_cases': [],
        'representative_failures': [],
        'fallback_recommendations': [],
    }

    # 1. Overall grounding distribution
    groundings = [d.get('grounding', 0) for d in all_diags]
    categories = [categorize_grounding(g) for g in groundings]
    for cat in ['perfect', 'good', 'partial', 'poor', 'failure']:
        count = categories.count(cat)
        results['distribution'][cat] = {
            'count': count,
            'fraction': count / len(all_diags),
        }

    results['overall_stats'] = {
        'mean_grounding': float(np.mean(groundings)),
        'std_grounding': float(np.std(groundings)),
        'median_grounding': float(np.median(groundings)),
        'min_grounding': float(np.min(groundings)),
    }

    # 2. By trigger type
    trigger_groups = defaultdict(list)
    for d in all_diags:
        trigger_groups[d['_trigger_type']].append(d)

    for trigger, diags in trigger_groups.items():
        gs = [d.get('grounding', 0) for d in diags]
        results['by_trigger'][trigger] = {
            'count': len(diags),
            'mean_grounding': float(np.mean(gs)),
            'std_grounding': float(np.std(gs)),
            'low_grounding_fraction': float(np.mean([g < 0.75 for g in gs])),
        }

    # 3. By dataset
    dataset_groups = defaultdict(list)
    for d in all_diags:
        dataset_groups[d['_dataset']].append(d)

    for dataset, diags in dataset_groups.items():
        gs = [d.get('grounding', 0) for d in diags]
        results['by_dataset'][dataset] = {
            'count': len(diags),
            'mean_grounding': float(np.mean(gs)),
            'std_grounding': float(np.std(gs)),
        }

    # 4. By backend
    backend_groups = defaultdict(list)
    for d in all_diags:
        backend_groups[d['_backend']].append(d)

    for backend, diags in backend_groups.items():
        gs = [d.get('grounding', 0) for d in diags]
        lats = [d.get('latency_ms', 0) for d in diags]
        results['by_backend'][backend] = {
            'count': len(diags),
            'mean_grounding': float(np.mean(gs)),
            'std_grounding': float(np.std(gs)),
            'mean_latency_ms': float(np.mean(lats)),
        }

    # 5. Failure mode analysis (grounding < 0.75)
    low_grounding = [d for d in all_diags if d.get('grounding', 0) < 0.75]
    failure_mode_counts = defaultdict(int)
    failure_mode_examples = defaultdict(list)

    for d in low_grounding:
        failures = analyze_grounding_component_failures(d)
        for f in failures:
            # Extract category (before parenthesis)
            cat = f.split('(')[0]
            failure_mode_counts[cat] += 1
            if len(failure_mode_examples[cat]) < 3:
                failure_mode_examples[cat].append({
                    'file': d['_source_file'],
                    'timestamp': d.get('timestamp', 'N/A'),
                    'anomaly': d.get('anomaly', 0),
                    'grounding': d.get('grounding', 0),
                    'detail': f,
                })

    results['failure_modes'] = {
        'total_low_grounding': len(low_grounding),
        'fraction_low_grounding': len(low_grounding) / len(all_diags) if all_diags else 0,
        'mode_counts': dict(failure_mode_counts),
        'mode_examples': {k: v for k, v in failure_mode_examples.items()},
    }

    # 6. Safety-critical cases: high anomaly + wrong severity
    for d in all_diags:
        anomaly = d.get('anomaly', 0)
        response = d.get('response', {})
        sev = response.get('severity', '').lower()

        # Dangerous: high anomaly but LLM says low severity
        if anomaly > 2.0 and sev in ('low', 'medium', 'normal', ''):
            results['safety_critical_cases'].append({
                'file': d['_source_file'],
                'dataset': d['_dataset'],
                'timestamp': d.get('timestamp', 'N/A'),
                'anomaly': anomaly,
                'severity_reported': sev,
                'grounding': d.get('grounding', 0),
                'diagnosis_snippet': response.get('diagnosis', '')[:200],
                'risk': 'HIGH - LLM underestimates severity',
            })

    results['safety_critical_cases'] = results['safety_critical_cases'][:20]  # Limit

    # 7. Representative failure cases for paper table
    # Select diverse failures across modes
    seen_modes = set()
    for d in sorted(low_grounding, key=lambda x: x.get('grounding', 0)):
        failures = analyze_grounding_component_failures(d)
        mode_key = tuple(sorted(f.split('(')[0] for f in failures))
        if mode_key not in seen_modes and len(results['representative_failures']) < 8:
            seen_modes.add(mode_key)
            response = d.get('response', {})
            results['representative_failures'].append({
                'dataset': d['_dataset'],
                'trigger': d['_trigger_type'],
                'backend': d['_backend'],
                'timestamp': d.get('timestamp', 'N/A'),
                'anomaly': round(d.get('anomaly', 0), 3),
                'grounding': d.get('grounding', 0),
                'severity': response.get('severity', 'N/A'),
                'confidence': response.get('confidence', 'N/A'),
                'failure_modes': failures,
                'diagnosis_snippet': response.get('diagnosis', '')[:150],
            })

    # 8. Latency vs grounding correlation
    lats = np.array([d.get('latency_ms', 0) for d in all_diags])
    grnds = np.array([d.get('grounding', 0) for d in all_diags])
    if len(lats) > 5:
        from scipy import stats as scipy_stats
        corr, p_val = scipy_stats.spearmanr(lats, grnds)
        results['latency_grounding_correlation'] = {
            'spearman_r': float(corr),
            'p_value': float(p_val),
        }

    # 9. Fallback recommendations
    results['fallback_recommendations'] = [
        {
            'strategy': 'Severity Cross-Check',
            'description': 'When LLM reports low severity but anomaly_score > 2.0, '
                          'automatically escalate to high severity or re-query LLM.',
            'addresses': 'severity_mismatch failures',
            'estimated_coverage': f'{failure_mode_counts.get("severity_mismatch", 0)} cases',
        },
        {
            'strategy': 'Confidence Thresholding',
            'description': 'Reject LLM diagnoses with confidence < 0.1 or > 0.99; '
                          'fall back to fast model prediction.',
            'addresses': 'unreasonable_confidence failures',
            'estimated_coverage': f'{failure_mode_counts.get("unreasonable_confidence", 0)} cases',
        },
        {
            'strategy': 'Evidence Requirement',
            'description': 'Require LLM to cite at least 2 numeric values from sensor data; '
                          'if not, request re-generation with explicit prompt.',
            'addresses': 'no_numeric_evidence failures',
            'estimated_coverage': f'{failure_mode_counts.get("no_numeric_evidence", 0)} cases',
        },
        {
            'strategy': 'Human-in-Loop Escalation',
            'description': 'When grounding < 0.5, flag for human review before acting on diagnosis.',
            'addresses': 'All severe failures',
            'estimated_coverage': f'{results["distribution"].get("poor", {}).get("count", 0) + results["distribution"].get("failure", {}).get("count", 0)} cases',
        },
    ]

    return results


def _synthetic_analysis() -> Dict:
    """Generate synthetic failure analysis for testing."""
    np.random.seed(42)
    return {
        'total_diagnoses': 0,
        'note': 'No real LLM diagnosis files found. Run real_llm_experiment.py first.',
    }


def generate_latex_table(results: Dict) -> str:
    """Generate LaTeX failure case table."""
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Representative LLM diagnosis failure cases. Anomaly: fast model anomaly score; '
        r'Grounding: diagnosis quality (0--1); Failure modes identify which grounding components failed.}',
        r'\label{tab:failure_cases}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{llcccll}',
        r'\toprule',
        r'Dataset & Backend & Anomaly & Grounding & Severity & Failure Mode & Snippet \\',
        r'\midrule',
    ]

    for case in results.get('representative_failures', [])[:8]:
        modes = ', '.join(m.split('(')[0] for m in case.get('failure_modes', []))
        snippet = case.get('diagnosis_snippet', '')[:60].replace('&', r'\&').replace('%', r'\%')
        lines.append(
            f'  {case.get("dataset", "")} & {case.get("backend", "")[:15]} '
            f'& {case.get("anomaly", 0):.2f} & {case.get("grounding", 0):.2f} '
            f'& {case.get("severity", "")} & {modes[:30]} & {snippet}... \\\\'
        )

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        r'\end{table}',
    ])

    # Safety summary table
    n_safety = len(results.get('safety_critical_cases', []))
    lines.extend([
        '', '',
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Failure mode distribution and proposed fallback strategies.}',
        r'\label{tab:fallback_strategies}',
        r'\begin{tabular}{lcp{6cm}}',
        r'\toprule',
        r'Failure Mode & Count & Fallback Strategy \\',
        r'\midrule',
    ])

    mode_counts = results.get('failure_modes', {}).get('mode_counts', {})
    fallbacks = results.get('fallback_recommendations', [])
    for fb in fallbacks:
        mode_key = fb['addresses'].replace(' failures', '')
        count = mode_counts.get(mode_key, 'N/A')
        lines.append(f'  {mode_key} & {count} & {fb["description"][:80]}... \\\\')

    lines.extend([
        f'  \\midrule',
        f'  Safety-critical (severity underestimate) & {n_safety} & Auto-escalate when anomaly $> 2.0$ \\\\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def run_experiment():
    """Run failure analysis experiment."""
    print("=" * 60)
    print("ECML-PKDD Experiment 6: LLM Diagnosis Failure Analysis")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_analysis()

    # Save JSON
    from experiments.uai_experiments.config import save_results
    save_results(results, str(RESULTS_DIR / 'failure_analysis.json'))

    # Generate LaTeX
    latex = generate_latex_table(results)
    with open(RESULTS_DIR / 'failure_cases_table.tex', 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved to {RESULTS_DIR / 'failure_cases_table.tex'}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    dist = results.get('distribution', {})
    for cat in ['perfect', 'good', 'partial', 'poor', 'failure']:
        d = dist.get(cat, {})
        print(f"  {cat:10s}: {d.get('count', 0):4d} ({d.get('fraction', 0):.1%})")

    print(f"\n  Safety-critical cases: {len(results.get('safety_critical_cases', []))}")
    print(f"  Failure modes:")
    for mode, count in results.get('failure_modes', {}).get('mode_counts', {}).items():
        print(f"    {mode}: {count}")

    return results


if __name__ == '__main__':
    run_experiment()
