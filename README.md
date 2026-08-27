# CPFLOW: Contrastive Rectified-Flow for Multimodal Virtual Cells

CPFLOW re-architects [MultiVCDiff](../MultiVCDiff) around three ideas drawn from
the CellFlux paper series (v1, v2, and the RL post-training paper), targeting
**cross-modal representation quality** rather than only image fidelity.

It jointly generates **cell-morphology images** and **L1000 transcriptomes**
under a drug perturbation, sharing a single transformer and a single flow time
`t` so the two modalities stay coupled.

## What changed vs MultiVCDiff

| Axis | MultiVCDiff | CPFLOW |
|---|---|---|
| Generative process | 1000-step Gaussian DDPM (`diffusion/`) | **Rectified flow / stochastic interpolants** (`cpflow/flow.py`) |
| Source distribution | Gaussian noise only | **noise → treated** (stage 1) then **control → treated** (stage 2) |
| Cross-modal fusion | `torch.cat` + shared self-attn only | shared self-attn **+ bidirectional cross-attention** (`CrossAttnFusion`) |
| RNA representation | 1 token | **K latent tokens** (`RNATokenizer`, default 8) |
| Drug conditioning | `Linear(1024, D)` | `DrugEncoder` MLP, optional aux (ChemBERTa/Gene2Vec) slot |
| Representation learning | none | **InfoNCE** image↔RNA alignment (`ProjectionHead` + `losses.info_nce_loss`) |
| Biological regularisation | none | **gene-covariance loss** (`losses.rna_correlation_loss`) |
| Loss weighting | parsed but **ignored** | applied; optional uncertainty weighting |
| Guidance | none | **classifier-free guidance** (learned null-drug embedding) |

Data conventions (RNA min-max to `[-1,1]`, Morgan FCFP4 fingerprints, SD-VAE
latent `*0.18215`, image normalisation) are kept **identical** to MultiVCDiff, so
the same `.h5ad` / VAE work unchanged.

## Layout

```
CPFLOW/
├── cpflow/
│   ├── flow.py                # FlowTransport: interpolants + ODE sampler
│   ├── models.py              # CPFlowModel + CPFlow_models registry
│   ├── encoders.py            # DrugEncoder, RNATokenizer/Detokenizer, ProjectionHead
│   ├── losses.py              # flow matching + InfoNCE + covariance reg
│   ├── multimodal_dataset.py  # PairedCellDataset (same-batch ctrl/treated pairing)
│   └── pos_embed.py           # 2-D sin-cos position embeddings
├── train.py                   # DDP two-stage curriculum trainer
├── sample.py                  # ctrl→treated (or noise→treated) ODE inference
├── scripts/                   # runnable stage1 / stage2 / sample shells
└── configs/default.yaml       # documented defaults
```

## Method

**Flow objective.** For each modality we linearly interpolate source `x0` and
target `x1`, optionally injecting a vanishing-at-endpoints noise term:

```
x_t = (1-t) x0 + t x1 + sin²(πt)·σ·ε
v_t = (x1 - x0) + π·sin(2πt)·σ·ε      # regression target
L_flow = ‖ v_θ(x_t, t, c) − v_t ‖²
```

Image latent and RNA share `t` and the drug condition `c = t_emb + drug_emb`.

**Two-stage curriculum.**
1. `--stage noise`: `x0` = Gaussian noise. Warms up the generator on all treated
   samples.
2. `--stage paired`: `x0` = a **same-batch control cell** (image + RNA). The model
   now learns the perturbation displacement and implicitly corrects batch effects.
   Resume the stage-1 checkpoint.

**Contrastive alignment.** Pooled image and RNA tokens are projected to unit-norm
128-d vectors; a symmetric InfoNCE loss pulls the matched image/RNA of a cell
together and pushes mismatched pairs apart, sharpening cross-modal representation.

**Covariance regulariser.** The predicted target RNA (recovered as
`x_t + (1-t)·v`) is constrained to match the real target's gene-gene covariance,
discouraging biologically implausible independent-gene samples.

## Usage

```bash
pip install -r requirements.txt

# Stage 1 — noise -> treated
bash scripts/train_stage1_noise.sh

# Stage 2 — control -> treated (pass the stage-1 checkpoint)
bash scripts/train_stage2_paired.sh results_cpflow/000-CPFlow-XL-2-noise/checkpoints/0050000.pt

# Sample
bash scripts/sample.sh results_cpflow/001-CPFlow-XL-2-paired/checkpoints/0050000.pt
```

Edit the `--h5ad-path`, `--image-dir`, and `--batch-column` flags in the scripts
to match your dataset. Set `--fusion-every 0` and `--rna-tokens 1` to ablate back
toward the MultiVCDiff architecture.

## Not yet included

The DiffusionNFT reward post-training (CellFluxRL) and pretrained scGPT/MolFormer
encoders are described in the improvement plan but left as follow-ups; the
`DrugEncoder` aux slot and `ProjectionHead` are the hooks for them.
