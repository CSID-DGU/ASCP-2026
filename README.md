# ASCP-2026

Reinforcement-learning and integer-programming code for airline crew pairing research.

## Repository layout

- `RL/`: environments, constraints, loaders, rollout, and shared utilities
- `model/`: encoder, decoder, and FiLM modules
- `experiments/`: training and sweep entry points
- `evaluation/`: checkpoint evaluation and set-partitioning/IP routines
- `baselines/`: baseline evaluation entry points
- `analysis/`: dataset and experiment analysis
- `diagnose/`: model and constraint diagnostics

See [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) for the working conventions.

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data and outputs

Datasets, checkpoints, experiment outputs, generated figures, papers, and working notes are
kept out of Git. Put local datasets under `RL/data/`; scripts should write run artifacts to a
`results/`, `runs/`, `logs/`, `checkpoints/`, or `paper_runs/` subdirectory as appropriate.

The repository tracks code and reproducibility documentation only. External datasets and
baseline repositories must be obtained separately according to their original licenses.
