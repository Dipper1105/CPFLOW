"""
Paired control/treated dataset for CPFLOW.

Extends MultiVCDiff's H5ADDataset (multimodal_dataset.py). Key addition: for the
control->treated flow (stage 2 of the CellFlux-v2 curriculum) each treated sample
is paired with a *control* (e.g. DMSO / vehicle) sample drawn from the same batch
(plate). This is what lets the model learn the perturbation *displacement* rather
than generating from scratch, and it decouples true drug effects from plate-level
batch effects.

The dataset works in two modes:
  * mode="noise"  : stage-1 warmup. __getitem__ returns treated image+RNA only;
                    the source is Gaussian noise (sampled in the training loop).
  * mode="paired" : stage-2. __getitem__ additionally returns a control image+RNA
                    from the same batch, used as the flow source.

RNA normalisation and Morgan-fingerprint drug encoding are kept identical to
MultiVCDiff so checkpoints/data are interchangeable.
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import scanpy as sc
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
rdBase.DisableLog("rdApp.warning")


def drug_encoder(smiles_list, num_bits=1024, comb_num=1):
    """Morgan (FCFP4) fingerprint encoder — identical to MultiVCDiff."""
    out = np.zeros((len(smiles_list), num_bits), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        parts = smiles.split("+") if comb_num != 1 else [smiles]
        for smi in parts:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            bits = AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, useFeatures=True, nBits=num_bits
            ).ToBitString()
            vec = np.array(list(bits), dtype=np.float32)
            if comb_num == 1:
                out[i] = vec
            else:
                out[i] += vec
    return out


class PairedCellDataset(Dataset):
    def __init__(
        self,
        h5ad_path,
        image_dir=None,
        image_size=512,
        image_col="merged_image",
        drug_col="compound",
        smiles_col="smiles",
        batch_col=None,          # plate/well-plate column for same-batch pairing
        control_value="DMSO",    # value of drug_col marking control samples
        mode="paired",           # "paired" or "noise"
        transform=None,
        normalize_rna=True,
        fp_size=1024,
        comb_num=1,
        seed=0,
        ctrl_rna_h5ad=None,      # optional external control-RNA source (experiment C)
    ):
        self.adata = sc.read_h5ad(h5ad_path)
        self.image_dir = image_dir
        self.image_col = image_col
        self.drug_col = drug_col
        self.smiles_col = smiles_col
        self.batch_col = batch_col
        self.control_value = control_value
        self.mode = mode
        self.fp_size = fp_size
        self.comb_num = comb_num
        self._rng = np.random.default_rng(seed)

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.normalize_rna = normalize_rna
        if normalize_rna:
            X = self._dense(self.adata.X)
            self.rna_min = X.min(0)
            self.rna_max = X.max(0)
            self.rna_range = self.rna_max - self.rna_min
            self.rna_range[self.rna_range == 0] = 1.0

        # precompute drug fingerprints (matches MultiVCDiff)
        smiles = self.adata.obs[self.smiles_col].astype(str).tolist()
        self.drug_embeddings = torch.from_numpy(
            drug_encoder(smiles, num_bits=fp_size, comb_num=comb_num)
        ).float()

        # optional external control-RNA source (experiment C: real DMSO RNA as the
        # RNA flow source while images stay noise-sourced, since this corpus has no
        # control images). Normalised with the *treated* stats so both live on the
        # same [-1,1] scale and the same 977-gene order.
        self.ctrl_rna = None
        if ctrl_rna_h5ad is not None:
            cad = sc.read_h5ad(ctrl_rna_h5ad)
            assert cad.n_vars == self.adata.n_vars, (
                f"control RNA gene count {cad.n_vars} != treated {self.adata.n_vars}"
            )
            Xc = self._dense(cad.X).astype(np.float32)
            if self.normalize_rna:
                Xc = 2 * ((Xc - self.rna_min) / self.rna_range) - 1
            self.ctrl_rna = torch.from_numpy(Xc).float()

        self._build_index()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _dense(X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    def _build_index(self):
        obs = self.adata.obs
        is_ctrl = (obs[self.drug_col].astype(str) == self.control_value).values

        if self.mode == "paired":
            # treated samples are the ones we iterate over
            self.sample_idx = np.where(~is_ctrl)[0]
            # map batch -> list of control-row indices, for same-plate pairing
            self.ctrl_by_batch = {}
            ctrl_idx = np.where(is_ctrl)[0]
            if self.batch_col is not None and self.batch_col in obs.columns:
                batches = obs[self.batch_col].astype(str).values
                for ci in ctrl_idx:
                    self.ctrl_by_batch.setdefault(batches[ci], []).append(ci)
                self._batches = batches
            else:
                self.ctrl_by_batch = None
            self._all_ctrl = ctrl_idx
            if len(ctrl_idx) == 0:
                raise ValueError(
                    f"mode='paired' but no control rows found "
                    f"({self.drug_col}=={self.control_value!r})."
                )
        else:  # noise mode uses every row as a treated target
            self.sample_idx = np.arange(self.adata.n_obs)

    def _load_row(self, idx):
        X = self._dense(self.adata.X[idx]).flatten()
        if self.normalize_rna:
            X = 2 * ((X - self.rna_min) / self.rna_range) - 1
        rna = torch.from_numpy(X).float()

        img_path = self.adata.obs[self.image_col].iloc[idx]
        if self.image_dir:
            img_path = os.path.join(self.image_dir, img_path)
        img = self.transform(Image.open(img_path).convert("RGB"))
        return img, rna

    def _pick_control(self, treated_idx):
        if self.ctrl_by_batch is not None:
            batch = self._batches[treated_idx]
            pool = self.ctrl_by_batch.get(batch)
            if not pool:                      # fall back to any control
                pool = self._all_ctrl
        else:
            pool = self._all_ctrl
        return int(pool[self._rng.integers(len(pool))])

    # ------------------------------------------------------------------ dataset
    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, i):
        idx = int(self.sample_idx[i])
        img1, rna1 = self._load_row(idx)             # treated (target)
        drug_fp = self.drug_embeddings[idx]
        drug_name = self.adata.obs[self.drug_col].iloc[idx]

        item = {
            "image": img1,
            "rna": rna1,
            "drug_embedding": drug_fp,
            "drugname": drug_name,
        }

        if self.mode == "paired":
            cidx = self._pick_control(idx)
            img0, rna0 = self._load_row(cidx)        # control (source)
            item["image_ctrl"] = img0
            item["rna_ctrl"] = rna0

        # experiment C: attach a real control-RNA vector as the RNA flow source
        if self.ctrl_rna is not None:
            j = int(self._rng.integers(self.ctrl_rna.shape[0]))
            item["rna_ctrl"] = self.ctrl_rna[j]

        return item
