# Best 6 generated perturbation images across 44 drugs

The optimized image-finetuned CPFLOW checkpoint was sampled across all drugs, with an additional
high-candidate search for the strongest initial image-quality hits when multiple score files are merged.
Candidates were ranked against real Cell Painting image statistics. The 6 drugs below are the
globally best best-candidate examples among all generated perturbation images.

Input run directory: `CPFLOW/comparison_outputs/full44_imageft2_morphology_candidates`
Total drugs evaluated: `44`
Total candidates scored: `1408`
Candidates per drug detected: `32` median, `32` max

| Rank | Drug | Candidate | Score | Foreground | Real foreground | Sharpness | Real sharpness | Blockiness | Real blockiness |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | nystatin | 30 | -0.096 | 0.245 | 0.248 | 0.048 | 0.052 | 0.482 | 0.498 |
| 2 | puromycin | 7 | -0.098 | 0.689 | 0.700 | 0.067 | 0.066 | 0.492 | 0.498 |
| 3 | pd-98059 | 24 | -0.103 | 0.799 | 0.800 | 0.062 | 0.060 | 0.501 | 0.500 |
| 4 | hydroxyurea | 12 | -0.104 | 0.581 | 0.586 | 0.059 | 0.060 | 0.503 | 0.498 |
| 5 | quercetin | 14 | -0.105 | 0.340 | 0.330 | 0.056 | 0.054 | 0.492 | 0.502 |
| 6 | cyclophosphamide | 9 | -0.112 | 0.414 | 0.406 | 0.058 | 0.061 | 0.497 | 0.501 |

Score sources:

- `CPFLOW/comparison_outputs/full44_imageft2_morphology_candidates/candidate_quality_scores.csv`

Generated files:

- `figure_best6_generated_perturbation_images.png/svg/pdf/tiff`
- `figure_all44_generated_image_quality_ranking.png/svg/pdf/tiff`
- `best6_generated_perturbation_images.csv`
- `all44_best_generated_image_quality.csv`

Selection note:

This ranking is based on image morphology/display statistics, not RNA Pearson. It is intended to
select the clearest and most normal-looking generated perturbation images for visual showcase.
