# Uncertainty-Aware Sequential Decision Rules for Event-Triggered LLM Invocation

This repository contains code sufficient to reproduce all experimental results reported in the paper.
All hyperparameters correspond exactly to those listed in Appendix B of the submission.

## Setup

```bash
pip install -r requirements.txt
```

For local LLM experiments, install [Ollama](https://ollama.ai) and pull the model:
```bash
ollama pull llama3.1:8b
```

## Reproducing Results

All experiment scripts are in `experiments/ecml_pkdd/`. Run all experiments:

```bash
python experiments/ecml_pkdd/run_all_ecml.py
```

Or reproduce individual results:

| Paper Reference | Script | Output |
|----------------|--------|--------|
| Table 1 (assumptions + inter-event) | `exp2_assumption_verification.py` | `results/ecml_pkdd/assumption_verification.json` |
| Table 2 (calibration + adaptive) | `exp1_statistical_tests.py` | `results/ecml_pkdd/statistical_tests.json` |
| Table 3 (regret analysis) | `exp1_statistical_tests.py` | `results/ecml_pkdd/statistical_tests.json` |
| Fig. 2 (risk ablation) | `exp5_risk_function_ablation.py` | `results/ecml_pkdd/risk_function_ablation.json` |
| Fig. 3 (baselines) | `exp7_stronger_baselines.py` | `results/ecml_pkdd/stronger_baselines.json` |
| Fig. 4 (real LLM grounding) | `exp6_failure_analysis.py` | `results/ecml_pkdd/failure_analysis.json` |
| Table 4 (LLM comparison + CIC-IDS) | `exp6_failure_analysis.py` | `results/ecml_pkdd/failure_analysis.json` |
| Cost sensitivity analysis | `exp4_cost_sensitivity.py` | `results/ecml_pkdd/cost_sensitivity.json` |

To regenerate all figures:
```bash
python experiments/ecml_pkdd/plot_figures.py
```

## Code Structure

```
src/
├── models/
│   ├── fast/          # GRU, ensemble, MC Dropout implementations
│   └── llm/           # LLM client, grounding evaluation, prompts
├── triggers/          # All trigger implementations
│   ├── threshold.py   # Basic threshold trigger
│   ├── cusum.py       # CUSUM trigger
│   ├── sprt.py        # Sequential probability ratio test
│   ├── optimal_stopping.py
│   ├── bayesian_trigger.py
│   ├── adaptive_threshold.py  # OGD and LinUCB
│   └── composite.py   # Composite trigger combining multiple signals
└── utils/             # Metrics, visualization utilities
```

## Data

- **C-MAPSS**: Download from [NASA Prognostics Repository](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq). Place in `data/cmapss/`.
- **CIC-IDS2017**: Download from [UNB](https://www.unb.ca/cic/datasets/ids-2017.html). Place in `data/cicids2017/`.

## Hardware

Experiments were run on NVIDIA RTX 4090 (24 GB), AMD Ryzen 9 7950X, 64 GB RAM.
Total runtime: ~12 hours including real LLM API calls.
