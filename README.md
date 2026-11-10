# Tembok Ratapan Solo — JOINTS 2026 Movie-Ticket Forecasting

## Achievement

<p align="center">
  <img src="assets/certificate_aditya_akbar.png" alt="JOINTS 2026 finalist certificate for Aditya Akbar" width="32%">
  <img src="assets/certificate_muhammad_faris_el_hakim.png" alt="JOINTS 2026 finalist certificate for Muhammad Faris El Hakim" width="32%">
  <img src="assets/certificate_taqiyudin_miftah_adn.png" alt="JOINTS 2026 finalist certificate for Taqiyudin Miftah Adn" width="32%">
</p>

**Tembok Ratapan Solo was a finalist in the student category of the JOINTS 2026 Data Science Competition.** The team is Aditya Akbar, Muhammad Faris El Hakim, and Taqiyudin Miftah Adn of Universitas Brawijaya.

## Abstract

The task is to forecast movie-ticket demand for days 4–10 (D4–D10) using information available from the first three days (D1–D3). This requires estimating a changing demand curve across films and cinemas while accounting for opening scale, staged cinema openings, zeros, calendar effects, and limited history.

The notebook compares three actuarial machine-learning approaches: actuarial features with LightGBM, a Negative Binomial prior with a neural correction, and a learned credibility mixture. The mixture combines five complementary experts—direct LightGBM, hurdle LightGBM, a Negative Binomial plus neural expert, individual projection, and collective projection. A softmax gate assigns non-negative weights that sum to one using information about the available history and expert disagreement.

## Problem Setting

Each prediction is a `total_ticket` value for a movie–cinema pair and a forecast day from D4 through D10. The notebook normalizes absolute errors by the mean ticket count over D1–D3, with a minimum scale of one, and reports the competition's MASE metric. Its inference path checks the official sample-submission IDs and produces 72,611 predictions in the original ID order.

Chronological development validation uses five monthly folds from May through September 2025, with film-level separation. Model and feature choices are evaluated on earlier periods before later periods are scored. The notebook also records per-fold comparisons, ablations, calibration, and interpretability diagnostics.

## Modeling Pipeline

![Five-expert learned credibility mixture pipeline](assets/modeling_pipeline.png)

The figure summarizes the learned mixture. Each expert produces a forecast; a learned credibility gate combines them, followed by horizon-specific calibration. The displayed MASE comparison is a **development result**: 0.438788 for calibrated LightGBM and 0.434681 for the learned mixture.

## Development Results

| Approach | Pooled development MASE |
| --- | ---: |
| Actuarial features + LightGBM | 0.438788 |
| Negative Binomial prior + neural correction | 0.484760 |
| Learned credibility mixture | **0.434681** |

The learned mixture's score is about **0.94% lower** than the calibrated LightGBM result on this development comparison. These numbers are from the notebook's recorded chronological development runs; they are not hidden-test scores or a claim about final leaderboard performance. The notebook also marks its final compact router as an adaptively selected configuration, so its agreement diagnostics should not be read as an independent generalization estimate.

## Repository Contents

| Path | Description |
| --- | --- |
| [notebook/Tembok Ratapan Solo.ipynb](notebook/Tembok%20Ratapan%20Solo.ipynb) | Technical report, saved development results, and inference-only workflow. |
| [assets/](assets/) | Three finalist certificates and the five-expert modeling-pipeline figure. |
| [datasets/](datasets/) | Local-only location for competition and auxiliary input files; data are not versioned. |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Environment, input contract, and run instructions. |
| [requirements.txt](requirements.txt) | Pinned Python dependencies matching the notebook's setup cell. |

## Scope and Limitations

- The repository does not include competition data, the packaged model weights, or the auxiliary CSV inputs required by the inference cells.
- The notebook runs inference from saved state; running it does not retrain the models or repeat the original hyperparameter search.
- Development MASE is not hidden-test performance. Prior exploration and adaptive router selection may contribute selection bias.
- The external historical context has limited coverage and does not directly match test titles; the notebook limits and audits its contribution.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact files needed to run the notebook.
