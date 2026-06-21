# Uncertainty-Aware Sequential Decision Rules for Event-Triggered LLM Invocation in Streaming Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20783298.svg)](https://doi.org/10.5281/zenodo.20783298)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Conference](https://img.shields.io/badge/ECML%20PKDD-2026-blue.svg)](https://ecmlpkdd.org/2026/)

Code and reproduction artifacts for the ECML PKDD 2026 (Research Track) paper.
Archived release: **[10.5281/zenodo.20783298](https://doi.org/10.5281/zenodo.20783298)**.

Streaming inference pipelines increasingly pair lightweight **fast models** with
**Large Language Models (LLMs)** that provide rich semantic understanding at
substantial cost. The central question of *when* to invoke the LLM has received
limited formal treatment. We cast it as a **risk-based sequential stopping
problem**: a trigger policy π fires when a risk functional R(H_t) exceeds a
threshold θ.

Within this framework we prove six results:

1. a minimum inter-event time bound excluding trigger **chattering**;
2. **optimality of threshold policies** via smooth pasting;
3. approximate **SPRT** guarantees under estimated parameters;
4. **O(√(T log T)) regret** for stationary streams (→ O(√((C_T+1)·T log T)) under C_T changepoints);
5. **O(1/√T) convergence** of online gradient descent for adaptive thresholds;
6. a **calibration-to-miss-rate transfer** inequality.

Classical trigger families — event-triggered, optimal stopping, SPRT, CUSUM, and
Bayesian triggers — arise as special cases. On turbofan degradation data (CMAPSS)
with real LLM calls, we verify the assumptions, ablate the risk-function design,
compare against six baselines (incl. a RouteLLM-style router and contextual
bandits), and analyze cost sensitivity and LLM failure modes.

**Keywords:** event-triggered systems · LLM invocation · sequential
decision-making · uncertainty calibration · streaming inference.

## Repository layout

```
src/triggers/      trigger families (threshold, CUSUM, SPRT, Bayesian, adaptive, composite)
src/models/fast/   fast edge models (GRU/TCN/Transformer, contrastive IDS)
src/models/llm/    LLM backends + grounding/diagnosis (keys via env vars only)
experiments/ecml_pkdd/   experiment drivers (see REPRODUCE.md for the table→script map)
results/ecml_pkdd/       cached result JSON/TeX referenced by the paper
configs/                 dataset + experiment configs
```

See [`REPRODUCE.md`](REPRODUCE.md) for the exact script→table/figure mapping.

## Quick start

```bash
pip install -r requirements.txt
# LLM backends read credentials from the environment — never hard-code them:
export OLLAMA_API_KEY=...        # or OPENAI_API_KEY for the cloud backend
python experiments/ecml_pkdd/run_all_ecml.py
```

## Data

Datasets are **not** bundled (license/size). Download separately:

- **CMAPSS** (turbofan, primary domain): NASA Prognostics Data Repository → `data/cmapss/`
- **CIC-IDS2017** (cross-domain sanity check; ~2.8M flows, we sample a stratified 252K subset): UNB CIC → `data/cicids2017/`

## Citation

```bibtex
@inproceedings{wang2026uncertainty,
  title     = {Uncertainty-Aware Sequential Decision Rules for Event-Triggered
               LLM Invocation in Streaming Systems},
  author    = {Wang, Zhaohui},
  booktitle = {Machine Learning and Knowledge Discovery in Databases (ECML PKDD)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {Research Track}
}
```

Or cite the archived code release directly:

```bibtex
@software{wang2026etllm_code,
  author    = {Wang, Zhaohui},
  title     = {Event-Triggered LLM Invocation in Streaming Systems (code release)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20783298},
  url       = {https://doi.org/10.5281/zenodo.20783298}
}
```

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)). The paper text and
figures are © the author / Springer Nature under the proceedings Licence to Publish.
