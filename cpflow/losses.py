"""
Loss terms for CPFLOW.

  flow_matching_loss : per-modality velocity regression (replaces MultiVCDiff's
                       mse_img/vb_img/mse_rna/vb_rna DDPM losses).
  info_nce_loss      : symmetric InfoNCE aligning the per-sample image and RNA
                       projections (improvement A1). Positives are the matched
                       image/RNA of the same cell; negatives are other cells in
                       the batch.
  rna_correlation_loss : keeps the gene-gene covariance of the *predicted target*
                       RNA close to that of the real target RNA (improvement D1),
                       discouraging biologically implausible independent-gene
                       samples.

train.py combines these with configurable weights (and fixes MultiVCDiff's bug
where --img/--rna-loss-weight were parsed but never applied).
"""
import torch
import torch.nn.functional as F


def flow_matching_loss(pred_v, target_v):
    """Mean-squared velocity error, averaged over all non-batch dims."""
    diff = (pred_v - target_v) ** 2
    return diff.flatten(1).mean(dim=1)          # (N,)


def info_nce_loss(z_img, z_rna, temperature=0.1):
    """
    Symmetric InfoNCE / CLIP loss between L2-normalised projections.
    z_img, z_rna: (N, d) unit-norm. Returns a scalar.
    """
    logits = z_img @ z_rna.t() / temperature     # (N, N)
    labels = torch.arange(z_img.shape[0], device=z_img.device)
    loss_i2r = F.cross_entropy(logits, labels)
    loss_r2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2r + loss_r2i)


def _predicted_target(x_t, v_pred, t):
    """
    From the flow relation x_t = (1-t) x0 + t x1 and v = x1 - x0, the endpoint is
        x1 = x_t + (1 - t) v.
    Used to recover the predicted target RNA for the correlation regulariser.
    """
    view = (-1,) + (1,) * (x_t.dim() - 1)
    tb = t.view(view)
    return x_t + (1.0 - tb) * v_pred


def rna_correlation_loss(rna_t, rna_v_pred, rna_v_target, t):
    """
    Frobenius distance between the gene covariance of predicted vs real target
    RNA within the batch. Both are recovered as endpoints from the flow relation
    so the loss is well defined at every t.
    """
    x1_pred = _predicted_target(rna_t, rna_v_pred, t)
    x1_real = _predicted_target(rna_t, rna_v_target, t)

    def cov(x):
        x = x - x.mean(dim=0, keepdim=True)
        return (x.t() @ x) / max(x.shape[0] - 1, 1)

    c_pred, c_real = cov(x1_pred), cov(x1_real)
    return ((c_pred - c_real) ** 2).mean()
