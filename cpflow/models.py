"""
CPFLOW model: a rectified-flow multimodal transformer that jointly denoises
(transports) a cell-morphology image latent and an L1000 RNA vector, conditioned
on a drug.

Relationship to MultiVCDiff/models.py (DiTMultimodal):
  * Same DiT backbone: patch embed, adaLN-Zero DiT blocks, sin-cos pos embed.
  * The network now predicts a *velocity field* (flow matching) instead of the
    DDPM epsilon; `learn_sigma` is dropped (flow has no variance head), so the
    image head outputs exactly `in_channels` and the RNA head `num_rna_features`.

Three architectural improvements over DiTMultimodal:
  A2. Cross-attention fusion: every `fusion_every` blocks, image tokens attend to
      RNA tokens and vice-versa, giving explicit fine-grained cross-modal
      interaction instead of relying only on shared self-attention.
  B.  RNA is tokenised into K tokens (RNATokenizer) rather than 1, fixing the
      1-vs-T asymmetry; drug conditioning goes through a richer DrugEncoder.
  A1. Two projection heads (image, RNA) expose pooled unit-norm embeddings for a
      contrastive alignment loss computed in train.py.

Classifier-free guidance: the drug condition is randomly dropped to a learned
null embedding during training (prob `drug_dropout_prob`); `velocity()` exposes
`force_uncond` for guided sampling.
"""
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from .pos_embed import get_2d_sincos_pos_embed
from .encoders import DrugEncoder, RNATokenizer, RNADetokenizer, ProjectionHead


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds a continuous flow time t in [0,1] into a vector (sinusoidal + MLP).

    Note vs MultiVCDiff: t is now a float in [0,1] rather than an int in
    [0,1000); we scale it up before the sinusoidal features so the frequency
    range is well used.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, time_scale=1000.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.time_scale = time_scale

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        t_freq = self.timestep_embedding(t * self.time_scale, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    """adaLN-Zero self-attention block (verbatim behaviour from MultiVCDiff)."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kw):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **kw)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden,
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CrossAttnFusion(nn.Module):
    """
    Bidirectional cross-attention between image and RNA tokens (improvement A2).

    img' = img + gate * MHA(q=img, kv=rna);  rna' = rna + gate * MHA(q=rna, kv=img)

    Gates are adaLN-Zero initialised so the module starts as identity and the
    network learns to introduce cross-modal coupling gradually (same trick as
    DiT's zero-init, keeps early training stable).
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_rna = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attends_rna = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.rna_attends_img = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        nn.init.constant_(self.gate[-1].weight, 0)
        nn.init.constant_(self.gate[-1].bias, 0)

    def forward(self, img_tokens, rna_tokens, c):
        gate_img, gate_rna = self.gate(c).chunk(2, dim=1)
        qi, ki = self.norm_img(img_tokens), self.norm_rna(rna_tokens)
        img_upd, _ = self.img_attends_rna(qi, ki, ki, need_weights=False)
        rna_upd, _ = self.rna_attends_img(ki, qi, qi, need_weights=False)
        img_tokens = img_tokens + gate_img.unsqueeze(1) * img_upd
        rna_tokens = rna_tokens + gate_rna.unsqueeze(1) * rna_upd
        return img_tokens, rna_tokens


class FinalLayer(nn.Module):
    """adaLN final layer for the image branch (outputs patchified velocity)."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class CPFlowModel(nn.Module):
    def __init__(
        self,
        input_size=64,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_rna_features=977,
        drug_fp_size=1024,
        drug_aux_size=0,
        rna_tokens=8,
        fusion_every=4,
        drug_dropout_prob=0.1,
        proj_dim=128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels          # velocity, no learn_sigma
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_rna_features = num_rna_features
        self.rna_tokens = rna_tokens
        self.drug_dropout_prob = drug_dropout_prob

        # --- embedders -------------------------------------------------------
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.drug_embedder = DrugEncoder(hidden_size, fp_size=drug_fp_size, aux_size=drug_aux_size)
        self.rna_tokenizer = RNATokenizer(num_rna_features, hidden_size, num_tokens=rna_tokens)

        # learned null-drug embedding for classifier-free guidance
        self.null_drug = nn.Parameter(torch.zeros(1, hidden_size))

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # --- backbone --------------------------------------------------------
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.fusion_layers = nn.ModuleList()
        self.fusion_at = set()
        if fusion_every and fusion_every > 0:
            for i in range(depth):
                if (i + 1) % fusion_every == 0:
                    self.fusion_at.add(i)
            self.fusion_layers = nn.ModuleList(
                [CrossAttnFusion(hidden_size, num_heads) for _ in range(len(self.fusion_at))]
            )
        self._fusion_index = {blk: k for k, blk in enumerate(sorted(self.fusion_at))}

        # --- heads -----------------------------------------------------------
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.rna_detokenizer = RNADetokenizer(hidden_size, num_rna_features, num_tokens=rna_tokens)

        # projection heads for contrastive alignment
        self.img_proj = ProjectionHead(hidden_size, hidden_size, proj_dim)
        self.rna_proj = ProjectionHead(hidden_size, hidden_size, proj_dim)

        self.initialize_weights()

    # ------------------------------------------------------------------ init
    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.null_drug, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        # zero-init RNA head so it starts as identity velocity too
        nn.init.constant_(self.rna_detokenizer.adaLN[-1].weight, 0)
        nn.init.constant_(self.rna_detokenizer.adaLN[-1].bias, 0)
        nn.init.constant_(self.rna_detokenizer.proj.weight, 0)
        nn.init.constant_(self.rna_detokenizer.proj.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    # ------------------------------------------------------------------ drug cond
    def _drug_cond(self, drug_fp, drug_aux, train, force_uncond=False):
        emb = self.drug_embedder(drug_fp, drug_aux)
        if force_uncond:
            return self.null_drug.expand(emb.shape[0], -1)
        if train and self.drug_dropout_prob > 0:
            drop = torch.rand(emb.shape[0], device=emb.device) < self.drug_dropout_prob
            emb = torch.where(drop.unsqueeze(1), self.null_drug.expand_as(emb), emb)
        return emb

    # ------------------------------------------------------------------ forward
    def forward(self, img_t, rna_t, t, drug_fp, drug_aux=None, force_uncond=False,
                return_features=False):
        """
        Predict the joint velocity field.

        Returns (img_v, rna_v) and, if return_features, also a dict with pooled
        unit-norm projections {'z_img', 'z_rna'} for the contrastive loss.
        """
        img_tokens = self.x_embedder(img_t) + self.pos_embed          # (N, T, D)
        T = img_tokens.shape[1]
        rna_tokens = self.rna_tokenizer(rna_t)                        # (N, K, D)

        t_emb = self.t_embedder(t)                                    # (N, D)
        drug_emb = self._drug_cond(drug_fp, drug_aux, self.training, force_uncond)
        c = t_emb + drug_emb                                          # (N, D)

        for i, block in enumerate(self.blocks):
            tokens = torch.cat([img_tokens, rna_tokens], dim=1)       # (N, T+K, D)
            tokens = block(tokens, c)
            img_tokens, rna_tokens = tokens[:, :T], tokens[:, T:]
            if i in self.fusion_at:
                fusion = self.fusion_layers[self._fusion_index[i]]
                img_tokens, rna_tokens = fusion(img_tokens, rna_tokens, c)

        img_out = self.final_layer(img_tokens, c)                     # (N, T, p*p*C)
        img_v = self.unpatchify(img_out)                              # (N, C, H, W)
        rna_v = self.rna_detokenizer(rna_tokens, c)                   # (N, num_rna_features)

        if return_features:
            feats = {
                "z_img": self.img_proj(img_tokens.mean(dim=1)),
                "z_rna": self.rna_proj(rna_tokens.mean(dim=1)),
            }
            return img_v, rna_v, feats
        return img_v, rna_v

    def velocity(self, img_z, rna_z, t, drug_fp, drug_aux=None, force_uncond=False):
        """Inference-time convenience wrapper (no feature return)."""
        return self.forward(img_z, rna_z, t, drug_fp, drug_aux, force_uncond=force_uncond)


# --------------------------------------------------------------------- configs
def _cfg(depth, hidden, patch, heads):
    def build(**kw):
        return CPFlowModel(depth=depth, hidden_size=hidden, patch_size=patch, num_heads=heads, **kw)
    return build


CPFlow_models = {
    "CPFlow-XL/2": _cfg(28, 1152, 2, 16), "CPFlow-XL/4": _cfg(28, 1152, 4, 16), "CPFlow-XL/8": _cfg(28, 1152, 8, 16),
    "CPFlow-L/2":  _cfg(24, 1024, 2, 16), "CPFlow-L/4":  _cfg(24, 1024, 4, 16), "CPFlow-L/8":  _cfg(24, 1024, 8, 16),
    "CPFlow-B/2":  _cfg(12, 768, 2, 12),  "CPFlow-B/4":  _cfg(12, 768, 4, 12),  "CPFlow-B/8":  _cfg(12, 768, 8, 12),
    "CPFlow-S/2":  _cfg(12, 384, 2, 6),   "CPFlow-S/4":  _cfg(12, 384, 4, 6),   "CPFlow-S/8":  _cfg(12, 384, 8, 6),
}
