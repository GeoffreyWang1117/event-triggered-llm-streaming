"""
Run all ECML-PKDD 2026 supplementary experiments.
Usage: python experiments/ecml_pkdd/run_all_ecml.py [--quick] [--exp N]
"""
import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def run_exp(name, module_path):
    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"{'#' * 70}\n")
    start = time.time()
    try:
        mod = __import__(module_path, fromlist=['run_experiment'])
        mod.run_experiment()
        elapsed = time.time() - start
        print(f"\n  Completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, help='Run only experiment N (1,2,4,5,6,7)')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer seeds)')
    args = parser.parse_args()

    experiments = [
        (6, 'Exp 6: Failure Analysis (post-processing)', 'experiments.ecml_pkdd.exp6_failure_analysis'),
        (1, 'Exp 1: Statistical Significance Tests', 'experiments.ecml_pkdd.exp1_statistical_tests'),
        (2, 'Exp 2: Assumption Verification', 'experiments.ecml_pkdd.exp2_assumption_verification'),
        (4, 'Exp 4: Cost Sensitivity Analysis', 'experiments.ecml_pkdd.exp4_cost_sensitivity'),
        (5, 'Exp 5: Risk Function Ablation', 'experiments.ecml_pkdd.exp5_risk_function_ablation'),
        (7, 'Exp 7: Stronger Baselines', 'experiments.ecml_pkdd.exp7_stronger_baselines'),
    ]

    if args.exp:
        experiments = [(n, name, mod) for n, name, mod in experiments if n == args.exp]
        if not experiments:
            print(f"Unknown experiment: {args.exp}")
            sys.exit(1)

    results = {}
    total_start = time.time()

    for num, name, module in experiments:
        success = run_exp(name, module)
        results[num] = success

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE ({total_elapsed:.0f}s)")
    print(f"{'=' * 70}")
    for num, success in results.items():
        status = 'OK' if success else 'FAILED'
        print(f"  Exp {num}: {status}")


if __name__ == '__main__':
    main()
