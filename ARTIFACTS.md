# CPFLOW research artifacts

This private repository keeps the complete implementation and the result set
needed to inspect the main CPFLOW findings without turning Git history into a
model-checkpoint archive.

## Versioned in the repository

- Model, flow, encoder, loss, dataset, training, sampling, and evaluation code
- Default configuration, runnable shell scripts, and Python dependencies
- Training and evaluation logs
- The 44-drug RNA comparison reports, tables, and figures
- Image-quality optimization evidence
- Final ImageFT2 best-image rankings and full-44-drug score summary
- Image-derived phenotype tables and figures
- Publication-ready PNG, PDF, and SVG exports

## Model weights

Model weights are intentionally not included in the current GitHub snapshot.
The final local weights remain available at:

```text
results_combined/000-CPFlow-B-2-noise/checkpoints/0008000.pt
results_image_finetune_round2/000-CPFlow-B-2-imageft2/checkpoints/0001000.pt
```

A representative final weight can be published separately through a GitHub
Release, Hugging Face, or Zenodo if direct inference or archival distribution
becomes part of the release goal.

## Deliberately excluded

All checkpoints, optimizer-heavy `last.pt` snapshots, copied `start_from_*.pt`
states, raw candidate-image sweeps, duplicate TIFF exports, IDE metadata,
Python caches, and transient distributed-training state are not part of the
GitHub copy. The retained metrics, reports, and figures preserve the result
evidence while keeping the repository focused and lightweight.
