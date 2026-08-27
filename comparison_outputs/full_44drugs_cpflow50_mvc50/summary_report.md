# CPFLOW combined vs original MultiVCDiff

Evaluation environment: `/disk1/nicole/miniconda3/envs/MultiVCDiff/bin/python`

Dataset: `/data1/dataset/stem_cell/CPgenes/DiT_input_512_train_full_local.h5ad`

Models:

| Model | Checkpoint | Sampling | Drugs | Samples per drug |
|---|---|---:|---:|---:|
| CPFLOW combined | `CPFLOW/results_combined/000-CPFlow-B-2-noise/checkpoints/0008000.pt` | 50-step Heun ODE | 44 | 16 |
| MultiVCDiff original | `/data1/dataset/stem_cell/CPgenes/0100000.pt` | 50-step respaced diffusion | 44 | 16 |

## Overall RNA-generation metrics

| Model | Mean Pearson | Median Pearson | Mean MSE | Median MSE | Mean MAE | Median MAE |
|---|---:|---:|---:|---:|---:|---:|
| CPFLOW combined | 0.981894 | 0.982517 | 0.161875 | 0.143540 | 0.306280 | 0.294396 |
| MultiVCDiff original | 0.463640 | 0.450557 | 2340.681100 | 1925.997314 | 36.059734 | 33.944551 |

## Selected showcase drugs

The showcase drugs were selected by largest per-drug Pearson gain of CPFLOW over MultiVCDiff.

| Drug | CPFLOW Pearson | MultiVCDiff Pearson | Delta Pearson | CPFLOW MSE | MultiVCDiff MSE |
|---|---:|---:|---:|---:|---:|
| mg-132 | 0.982135 | 0.268262 | 0.713873 | 0.128899 | 2588.561523 |
| doxorubicin | 0.977119 | 0.264263 | 0.712857 | 0.166223 | 2486.645020 |
| forskolin | 0.984131 | 0.307393 | 0.676738 | 0.117413 | 848.387390 |
| colchicine | 0.948713 | 0.273262 | 0.675451 | 0.635397 | 739.387085 |

## Generated files

| File | Content |
|---|---|
| `model_summary.csv` | Overall model-level metrics |
| `per_drug_metrics.csv` | Per-drug metrics for each model |
| `per_drug_deltas.csv` | Paired CPFLOW-minus-MultiVCDiff deltas |
| `showcase_selected_drugs.csv` | Selected showcase drug table |
| `pred_means_cpflow_combined.csv` | CPFLOW predicted mean RNA profiles |
| `pred_means_multivcdiff_original.csv` | MultiVCDiff predicted mean RNA profiles |
| `figure_model_comparison.svg/pdf/png` | Overall comparison figure |
| `figure_drug_showcase.svg/pdf/png` | Showcase drug figure |

Note: MultiVCDiff was evaluated with 50-step respaced diffusion for a tractable all-drug comparison. Its original sampling script defaults to 1000 steps, which would require substantially longer GPU time.
