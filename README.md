# Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models

Reference implementation for **Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models**

This repo contains the minimal code to reproduce the two main experiments:

- **Sudoku infill** — train on a generated Sudoku dataset, then evaluate exact-solve
  accuracy on a held-out validation set.
- **N-Queens infill** — train on a generated N-Queens dataset, then evaluate solution
  accuracy and coverage (distinct valid solutions found), binned by the number of
  solutions per puzzle.

Both datasets are generated deterministically in-memory from a seed — there is no
dataset to download. (Sudoku additionally caches its generated shards under `data/`
to skip regeneration on reruns; that cache is git-ignored and rebuilt automatically.)

## Setup

```bash
pip install -r requirements.txt
# flash-attn must be installed separately (build flags):
pip install flash-attn==2.8.3 --no-build-isolation
```

A CUDA GPU is required. Weights & Biases logging is optional — run with
`WANDB_MODE=offline` (used below) and no account or API key is needed.


## Reproducing the main results

Everything runs through `python main.py` (Hydra). The two sweep scripts in
`reproduce/` drive training and per-checkpoint evaluation; every hyperparameter is
overridable via environment variables (see the header of each script).

### Sudoku

```bash
export WANDB_MODE=offline
# Trains FLM (infill) over training-set sizes x conditioning_prob_clean, then
# runs sudoku_eval on every checkpoint. Default: difficulty=hard, N in {10000,1000}.
./reproduce/sweep_infill_sudoku_gen.sh

# Examples:
DIFFICULTY=easy SUBSET_SIZES="1000 10000" ./reproduce/sweep_infill_sudoku_gen.sh
EVAL_ONLY=1 ./reproduce/sweep_infill_sudoku_gen.sh   # re-eval existing checkpoints only
```

### N-Queens

```bash
export WANDB_MODE=offline
# Trains FLM (infill) over conditioning_prob_clean, runs nqueens_eval on every
# checkpoint, and renders accuracy/coverage-vs-#solutions plots. Default: 10x10.
./reproduce/sweep_infill_nqueens.sh

# Example:
CONDITIONING_PROBS_CLEAN="0.5 1.0" ./reproduce/sweep_infill_nqueens.sh
```

### Where results land

Checkpoints and evaluation outputs are written under each run directory:

```
outputs/<sweep>/<run_name>/checkpoints/*.ckpt
outputs/<sweep>/<run_name>/sudoku_eval/<ckpt>/results.json     # Sudoku
outputs/<sweep>/<run_name>/nqueens_eval/<ckpt>/results.json    # N-Queens (+ plots)
```

Each `results.json` reports the accuracy (and, for N-Queens, coverage) for that
checkpoint.

## Repository layout

```
main.py            Hydra entry point: train / sudoku_eval / nqueens_eval
algo.py            FLM algorithm (flow objective, conditional infill sampling)
trainer_base.py    PyTorch-Lightning training base
dataloader.py      In-memory Sudoku / N-Queens dataset generation + batching
dataset_code/      Deterministic puzzle generators
models/            DiT backbone (dit.py) + EMA
configs/           Hydra configs (algo / model / data / trainer ...)
reproduce/         Sweep scripts + smoke_test.sh
plot_nqueens.py    Renders the N-Queens accuracy/coverage figures
```
