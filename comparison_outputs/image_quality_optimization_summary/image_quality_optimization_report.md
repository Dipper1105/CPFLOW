# CPFLOW image-quality optimization summary

This summary compares morphology-aware samples from the original combined CPFLOW checkpoint
against samples from the image-focused fine-tuned checkpoint.

Original candidate directory: `CPFLOW/comparison_outputs/morphology_aware_cpflow`
Optimized candidate directory: `CPFLOW/comparison_outputs/morphology_aware_cpflow_imageft`
Image-finetuned checkpoint: `CPFLOW/results_image_finetune/000-CPFlow-B-2-imageft/checkpoints/last.pt`

Mean best-candidate score gain across drugs: `0.299`.

| Drug | Original score | Optimized score | Delta score | Foreground error gain | Blockiness error gain |
|---|---:|---:|---:|---:|---:|
| anisomycin | -0.700 | -0.229 | 0.471 | 0.078 | 0.018 |
| monastrol | -0.410 | -0.244 | 0.166 | -0.022 | 0.030 |
| puromycin | -0.439 | -0.179 | 0.260 | -0.024 | 0.021 |

Generated files:

- `figure_cpflow_image_quality_before_after.png/svg/pdf/tiff`
- `figure_cpflow_image_quality_metrics.png/svg/pdf/tiff`
- `image_quality_optimization_metrics.csv`

Interpretation:

The image-focused checkpoint produces visibly more cell-like samples than the original checkpoint,
with clearer nuclei/cytoplasm separation and candidate scores closer to real-image statistics.
Residual artifacts remain, especially local block texture and density mismatch, so the result should be
reported as improved but not yet article-grade image synthesis.
