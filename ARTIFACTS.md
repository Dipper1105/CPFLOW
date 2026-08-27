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

## Stored in the private GitHub Release

| Asset | Purpose |
|---|---|
| `cpflow-combined-0008000.pt` | Final combined model used for the main RNA comparison |
| `cpflow-imageft2-0001000.pt` | Final image-finetuned model used for the strongest image results |
| `checkpoints-manifest.sha256` | SHA-256 integrity checks for both model assets |

The Release assets use descriptive names; their original project paths are:

```text
results_combined/000-CPFlow-B-2-noise/checkpoints/0008000.pt
results_image_finetune_round2/000-CPFlow-B-2-imageft2/checkpoints/0001000.pt
```

## Deliberately excluded

Intermediate checkpoints, optimizer-heavy `last.pt` snapshots, copied
`start_from_*.pt` states, raw candidate-image sweeps, duplicate TIFF exports,
IDE metadata, Python caches, and transient distributed-training state are not
part of the GitHub copy. They do not add evidence beyond the retained final
models, metrics, reports, and figures.

