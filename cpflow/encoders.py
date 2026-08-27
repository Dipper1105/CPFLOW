"""
Pluggable conditioning / representation encoders for CPFLOW.

These replace MultiVCDiff's minimal `nn.Linear(1024, D)` drug embedder and the
2-layer `RNAEmbedder`. Everything degrades gracefully: with default flags the
behaviour matches MultiVCDiff (Morgan fingerprint -> linear, RNA -> MLP), but
richer options can be switched on without touching the rest of the model.

Design goals (improvement B in the plan):
  * Drug side  : Morgan fingerprint (+ optional Gene2Vec-style aux vector) fused
                 through an MLP; optional extra precomputed molecule embedding
                 (ChemBERTa / MolFormer) concatenated in.
  * RNA side   : encode the 977-d landmark vector into K latent tokens (Perceiver
                 style) instead of a single token, fixing the 1-vs-T asymmetry
                 with the image patch tokens.

None of these require external weights to run; precomputed embeddings (if used)
are passed in as plain tensors from the dataset.
"""
import torch
import torch.nn as nn


def _mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class DrugEncoder(nn.Module):
    """
    Maps a drug fingerprint (+ optional auxiliary molecule embedding) to a single
    conditioning vector of size `hidden_size`.

    fp_size:       Morgan fingerprint width (default 1024, matches MultiVCDiff)
    aux_size:      width of an optional precomputed embedding (e.g. ChemBERTa 768).
                   0 disables it.
    """
    def __init__(self, hidden_size, fp_size=1024, aux_size=0, depth=2):
        super().__init__()
        self.aux_size = aux_size
        in_size = fp_size + aux_size
        sizes = [in_size] + [hidden_size] * depth
        self.net = _mlp(sizes)

    def forward(self, drug_fp, drug_aux=None):
        if self.aux_size > 0 and drug_aux is not None:
            x = torch.cat([drug_fp, drug_aux], dim=-1)
        else:
            x = drug_fp
        return self.net(x)


class RNATokenizer(nn.Module):
    """
    Encode the RNA expression vector into `num_tokens` latent tokens of width
    `hidden_size`, so RNA participates in cross-modal attention on equal footing
    with the image patch tokens.

    Set num_tokens=1 to recover MultiVCDiff's single-token behaviour.
    """
    def __init__(self, num_rna_features, hidden_size, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.encoder = _mlp([num_rna_features, hidden_size, hidden_size * num_tokens])
        self.token_pos = nn.Parameter(torch.zeros(1, num_tokens, hidden_size))
        nn.init.normal_(self.token_pos, std=0.02)

    def forward(self, rna):
        h = self.encoder(rna)                                  # (N, T*D)
        h = h.view(-1, self.num_tokens, self.hidden_size)      # (N, T, D)
        return h + self.token_pos


class RNADetokenizer(nn.Module):
    """
    Collapse the RNA output tokens back to a velocity/prediction over the
    `num_rna_features` genes. adaLN-conditioned, mirroring MultiVCDiff's
    RNAFinalLayer but pooling over multiple tokens.
    """
    def __init__(self, hidden_size, num_rna_features, num_tokens=8):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.proj = nn.Linear(hidden_size * num_tokens, num_rna_features)
        self.num_tokens = num_tokens

    def forward(self, tokens, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        x = self.norm(tokens)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x.flatten(1)                                       # (N, T*D)
        return self.proj(x)                                    # (N, num_rna_features)


class ProjectionHead(nn.Module):
    """
    Non-linear projection head for contrastive alignment (improvement A1).
    Maps a pooled modality representation to a unit-norm embedding.
    """
    def __init__(self, in_size, hidden_size, out_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)
