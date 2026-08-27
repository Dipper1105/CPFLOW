"""
Rectified-flow / stochastic-interpolant transport for CPFLOW.

This module replaces MultiVCDiff's 1000-step Gaussian DDPM
(diffusion/gaussian_diffusion.py) with a continuous-time flow that transports a
*source* distribution (DMSO control cells) to a *target* distribution (drug
perturbed cells), jointly over the image latent and the RNA vector.

Core ideas ported from CellFlux v2:
  * distribution-to-distribution linear interpolation  x_t = (1-t) x0 + t x1
  * target velocity                                     v   = x1 - x0
  * "noisy interpolants" (stochastic interpolants) to regularise the path:
        x_t = (1-t) x0 + t x1 + sin^2(pi t) * sigma * eps
        v   = (x1 - x0) + pi * sin(2 pi t) * sigma * eps
    which vanishes at t=0 and t=1 (endpoints stay clean) but injects noise in
    the middle of the trajectory, empirically improving sample quality.

Both modalities (image latent, RNA) share the *same* scalar t per sample so the
two generative trajectories stay coupled -- this is what makes the joint model
learn cross-modal coherence rather than two independent generators.
"""
import math
import torch


class FlowTransport:
    def __init__(self, sigma: float = 0.0, t_eps: float = 1e-4):
        """
        Args:
            sigma:  std of the noisy-interpolant term. 0.0 recovers plain
                    rectified flow. CellFlux v2 uses a small positive value.
            t_eps:  clamp t away from the exact endpoints for numerical safety.
        """
        self.sigma = sigma
        self.t_eps = t_eps

    def sample_t(self, batch_size, device):
        t = torch.rand(batch_size, device=device)
        return t.clamp(self.t_eps, 1.0 - self.t_eps)

    def interpolate(self, x0, x1, t, noise=None):
        """
        Build the interpolated point x_t and its target velocity v_t for one
        modality. `t` is a 1-D tensor (N,); x0/x1 are (N, ...) with arbitrary
        trailing dims. Returns (x_t, v_t).
        """
        # broadcast t over trailing dims
        view = (-1,) + (1,) * (x0.dim() - 1)
        tb = t.view(view)

        x_t = (1.0 - tb) * x0 + tb * x1
        v_t = x1 - x0

        if self.sigma > 0.0:
            if noise is None:
                noise = torch.randn_like(x0)
            # sin^2(pi t) envelope -> 0 at both endpoints
            env = torch.sin(math.pi * tb) ** 2
            denv = math.pi * torch.sin(2.0 * math.pi * tb)  # d/dt sin^2(pi t)
            x_t = x_t + env * self.sigma * noise
            v_t = v_t + denv * self.sigma * noise
        return x_t, v_t

    def build_batch(self, img0, img1, rna0, rna1, device):
        """
        Draw a shared t and produce interpolants + targets for both modalities.

        Returns dict with img_t, rna_t (model inputs), img_v, rna_v (regression
        targets) and t.
        """
        n = img1.shape[0]
        t = self.sample_t(n, device)
        img_t, img_v = self.interpolate(img0, img1, t)
        rna_t, rna_v = self.interpolate(rna0, rna1, t)
        return {"t": t, "img_t": img_t, "rna_t": rna_t, "img_v": img_v, "rna_v": rna_v}

    @torch.no_grad()
    def sample_ode(
        self,
        model,
        img0,
        rna0,
        drug_fp,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        method: str = "heun",
    ):
        """
        Integrate dz/dt = v_theta(z_t, t, c) from t=0 (source) to t=1 (target).

        img0/rna0 are the *source* samples (control-cell latent + control RNA,
        or Gaussian noise when running the noise->target stage-1 model).

        Supports classifier-free guidance: cfg_scale > 1 extrapolates between the
        conditional and unconditional (null-drug) velocity. `method` is 'euler'
        or 'heun' (2nd-order, matches CellFluxRL's sampler).
        """
        device = img0.device
        n = img0.shape[0]
        img = img0.clone()
        rna = rna0.clone()
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        def velocity(img_z, rna_z, t_scalar):
            t_vec = torch.full((n,), float(t_scalar), device=device)
            img_v, rna_v = model.velocity(img_z, rna_z, t_vec, drug_fp)
            if cfg_scale != 1.0:
                img_vu, rna_vu = model.velocity(img_z, rna_z, t_vec, drug_fp, force_uncond=True)
                img_v = img_vu + cfg_scale * (img_v - img_vu)
                rna_v = rna_vu + cfg_scale * (rna_v - rna_vu)
            return img_v, rna_v

        for i in range(num_steps):
            t0, t1 = ts[i], ts[i + 1]
            dt = (t1 - t0)
            img_k1, rna_k1 = velocity(img, rna, t0)
            if method == "euler":
                img = img + dt * img_k1
                rna = rna + dt * rna_k1
            elif method == "heun":
                img_e = img + dt * img_k1
                rna_e = rna + dt * rna_k1
                img_k2, rna_k2 = velocity(img_e, rna_e, t1)
                img = img + 0.5 * dt * (img_k1 + img_k2)
                rna = rna + 0.5 * dt * (rna_k1 + rna_k2)
            else:
                raise ValueError(f"unknown ODE method: {method}")
        return img, rna
