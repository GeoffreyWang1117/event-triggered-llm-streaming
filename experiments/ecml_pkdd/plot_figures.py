"""
Generate publication-quality figures for ECML-PKDD paper.
Figure 2: (a) Pareto curves by method, (b) Cost sensitivity.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'ecml_pkdd'
FIGURES_DIR = Path(__file__).resolve().parents[2] / 'submissions' / 'ECML_PKDD_2026' / 'paper' / 'figures'

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.3,
})


def load_json(name):
    with open(RESULTS_DIR / f'{name}.json') as f:
        return json.load(f)


def plot_pareto_panel(ax):
    """Panel (a): Pareto curves from risk function ablation + baselines."""

    # --- Risk function ablation data: extract Pareto points per R variant ---
    ablation = load_json('risk_function_ablation')

    # Pick representative seed for curves (use first available)
    first_seed = list(ablation['per_seed'].keys())[0]
    seed_data = ablation['per_seed'][first_seed]

    # Plot top 4 R variants as curves
    r_configs = [
        ('R2_anomaly_only', 'Anomaly score ($R_2$)', '#2166ac', '-', 'o'),
        ('R1_linear', 'Linear combo ($R_1$)', '#b2182b', '-', 's'),
        ('R4_multiplicative', 'Multiplicative ($R_4$)', '#762a83', '--', 'D'),
        ('R6_ewma', 'EWMA ($R_6$)', '#1b7837', '--', '^'),
    ]

    for r_name, label, color, ls, marker in r_configs:
        if r_name not in seed_data:
            continue
        points = seed_data[r_name].get('pareto_points', [])
        if not points:
            continue
        # Sort by invocation rate
        points = sorted(points, key=lambda p: p['invocation_rate'])
        inv_rates = [p['invocation_rate'] * 100 for p in points]
        miss_rates = [p['miss_rate'] * 100 for p in points]
        ax.plot(inv_rates, miss_rates, color=color, linestyle=ls, marker=marker,
                markevery=3, label=label, zorder=3)

    # --- Baselines as individual points ---
    baselines = load_json('stronger_baselines')
    agg = baselines.get('aggregated', {})

    baseline_configs = [
        ('linucb', 'LinUCB', '#e66101', '*', 9),
        ('routellm', 'RouteLLM-style', '#5e3c99', 'P', 7),
        ('periodic', 'Periodic', '#969696', 'X', 7),
        ('random', 'Random', '#bdbdbd', 'v', 6),
    ]

    for method, label, color, marker, ms in baseline_configs:
        if method not in agg:
            continue
        d = agg[method]
        inv = d['invocation_rate_mean'] * 100
        miss = d['miss_rate_mean'] * 100
        ax.scatter(inv, miss, color=color, marker=marker, s=ms**2,
                   label=label, zorder=5, edgecolors='k', linewidths=0.3)

    # Ideal corner annotation
    ax.annotate('ideal', xy=(0, 0), xytext=(3, 5),
                fontsize=7, fontstyle='italic', color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    ax.set_xlabel('Invocation Rate (%)')
    ax.set_ylabel('Miss Rate (%)')
    ax.set_xlim(-0.5, 32)
    ax.set_ylim(-2, 52)
    # Add a break indicator for truncated axis
    ax.annotate('', xy=(30, 50), xytext=(32, 52), fontsize=6, color='gray')
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='0.7',
              ncol=1, columnspacing=0.8, handletextpad=0.4, borderpad=0.4)
    ax.grid(True, alpha=0.25)
    ax.set_title('(a) Pareto curves by method', fontsize=10, pad=6)


def plot_cost_panel(ax):
    """Panel (b): Cost sensitivity — optimal theta vs c_LLM."""
    cost_data = load_json('cost_sensitivity')
    cost_sweep = cost_data.get('cost_sweep', {})
    c_values = sorted(cost_sweep.get('c_llm_values', []))

    trigger_configs = [
        ('threshold', 'Threshold', '#2166ac', '-', 'o'),
        ('cusum', 'CUSUM', '#b2182b', '--', 's'),
        ('optimal_stopping', 'Opt.\\ Stopping', '#1b7837', '-.', 'D'),
    ]

    for trig_name, label, color, ls, marker in trigger_configs:
        tdata = cost_sweep.get('triggers', {}).get(trig_name, {}).get('per_cost', {})
        if not tdata:
            continue

        cs, thetas, miss_rates = [], [], []
        for c in c_values:
            key = f"c{c}"
            s = tdata.get(key, {}).get('summary', {})
            if s:
                cs.append(c)
                thetas.append(s.get('optimal_theta_mean', 0))
                miss_rates.append(s.get('miss_rate_mean', 0) * 100)

        if cs:
            ax.plot(cs, thetas, color=color, linestyle=ls, marker=marker,
                    markersize=4, label=label)

    ax.set_xscale('log')

    # Stability region shading (after setting scale and limits)
    ax.axvspan(0.008, 5, alpha=0.07, color='#4daf4a', zorder=0)
    ax.text(0.15, 9.3, 'stable region', fontsize=6.5,
            color='#4daf4a', alpha=0.8, ha='center', va='top', fontstyle='italic')
    ax.set_xlabel('$c_{\\mathrm{LLM}}$')
    ax.set_ylabel('Optimal $\\theta^*$')
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='0.8',
              handletextpad=0.4)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('(b) Cost sensitivity', fontsize=10, pad=6)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))
    fig.subplots_adjust(wspace=0.38)

    plot_pareto_panel(ax1)
    plot_cost_panel(ax2)

    out_pdf = FIGURES_DIR / 'pareto_frontier.pdf'
    out_png = FIGURES_DIR / 'pareto_frontier.png'
    fig.savefig(out_pdf, format='pdf')
    fig.savefig(out_png, format='png')
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == '__main__':
    main()
