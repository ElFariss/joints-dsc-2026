# Reproducibility

## What this repository contains

The notebook is an **inference-only technical report**. It loads saved model state and creates new predictions; it does not retrain the models when run from top to bottom. Historical training, validation, and ablation results are recorded in the notebook and identified there as previously executed results.

This repository intentionally contains only the notebook, achievement and pipeline images, and project documentation. It does **not** contain the competition ZIP, the model-state package, or the auxiliary input CSV files. A fresh clone therefore cannot run the inference cells until those inputs are obtained separately.

## Environment

The notebook specifies Python 3.12 and pins the packages in its first setup cell. Install the same pinned stack with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The PyTorch CPU wheel index is included in `requirements.txt`. The notebook can use CUDA only when the environment has a compatible CUDA-enabled PyTorch installation; its default device is CPU.

## Required input files

The inference workflow expects these files, which are not tracked here:

| File | Purpose |
| --- | --- |
| `datajointsxinspire.zip` | Official competition archive containing `train.csv`, `test_history.csv`, `test.csv`, `movies.csv`, `holidays.csv`, `ticket_prices.csv`, and `sample_submission.csv`. |
| `Tembok Ratapan Solo.npz` | Packaged, hash-checked inference weights and runtime code for the three approaches. |
| `cinepoint_pre_cutoff.csv` | Historical external context snapshot checked against the September 30, 2025 cutoff. |
| `kalender_libur_pre_cutoff.csv` | Calendar input matched against the calendar embedded in the model package. |

Place the model package and two CSV files in `datasets/` or the repository root. The notebook resolves those three files from `datasets/` first, then the repository root. Place the official ZIP in either location, or set `JOINTS_OFFICIAL_ZIP` to its path. Do not commit competition data, model weights, or external input files unless their redistribution is explicitly permitted.

## Run

Start the notebook server from the repository root, open `notebook/Tembok Ratapan Solo.ipynb`, and run the cells in order. The notebook detects the repository root so its relative paths remain stable when opened from the `notebook/` directory. By default it uses four CPU threads; set `JOINTS_THREADS` to change this. `JOINTS_DEVICE=cpu` is the default, and `JOINTS_DEVICE=cuda` requests a CUDA device.

The notebook writes generated predictions and verification receipts to `joints_submission_output/`. It checks the output schema, finite non-negative predictions, unique IDs, and sample-submission row order before writing the receipt.

## Validation boundary

The recorded pooled development MASE values are 0.438788 for actuarial LightGBM, 0.484760 for the Negative Binomial plus neural approach, and 0.434681 for the learned credibility mixture. These are chronological development results documented by the notebook. They are not hidden-test or leaderboard scores. The separately described compact router was selected adaptively and its configuration-agreement checks are not an independent accuracy estimate.

For an auditable rerun, retain the exact hashes of the official ZIP, model package, and both CSV files along with the generated `inference_verification.json` receipt. The notebook records the model and input hashes in that receipt.
