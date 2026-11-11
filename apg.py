"""Adaptive Projected Guidance (APG) for Stable Audio Open, in place of the CFG step of cfg.py.

Sadat et al., "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models", ICLR 2025
(arXiv 2410.02416), Algorithm 1. The update D_cond - D_uncond is
  1. averaged with reverse momentum:   avg <- diff + momentum * avg,  diff <- avg     (momentum < 0)
  2. capped in norm:                   diff <- diff * min(1, norm_threshold / ||diff||)
  3. split along D_cond:               diff <- diff_orthogonal + eta * diff_parallel
and the guided prediction is D_cond + (guidance_scale - 1) * diff.

The paper applies this to the denoised prediction D, not to the raw network output. Stable Audio Open's scheduler
(CosineDPMSolverMultistepScheduler, v-prediction, EDM preconditioning) gets D = c_skip * z + c_out * F from the
network output F, so the two outputs are converted to D, guided, and the result is converted back to F for
scheduler.step. With eta = 1, norm_threshold = 0 and momentum = 0 this is the same as cfg.py.

Norms and inner products are taken per clip over all non-batch dimensions, as in the paper's code.
"""

import torch


class APG:
    def __init__(self, eta=0.0, norm_threshold=0.0, momentum=-0.75):
        self.eta = eta
        self.norm_threshold = norm_threshold  # 0 = no cap
        self.momentum = momentum  # 0 = no momentum
        self.reset()

    def reset(self):
        """Call before each clip: clears the momentum average and the per-step norm log."""
        self.running_average = 0
        self.norms = []  # per step: ||D_cond - D_uncond|| and the norm that the cap compares with r

    def __call__(self, pred_uncond, pred_cond, guidance_scale, latents, scheduler):
        """pred_uncond, pred_cond: network outputs (1, C, L); latents: the current sample z (1, C, L).
        Returns the guided network output for scheduler.step."""
        dtype = pred_cond.dtype
        sigma = scheduler.sigmas[scheduler.step_index].double()
        sigma_data = scheduler.config.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5  # v-prediction
        z = latents.double()
        d_uncond = c_skip * z + c_out * pred_uncond.double()
        d_cond = c_skip * z + c_out * pred_cond.double()
        dims = list(range(1, d_cond.ndim))

        diff = d_cond - d_uncond
        raw_norm = diff.norm(dim=dims).item()
        if self.momentum != 0:
            self.running_average = diff + self.momentum * self.running_average
            diff = self.running_average
        capped_norm = diff.norm(dim=dims, keepdim=True)
        self.norms.append({"raw": raw_norm, "before_cap": capped_norm.item()})
        if self.norm_threshold > 0:
            diff = diff * torch.clamp(self.norm_threshold / capped_norm, max=1.0)
        unit = torch.nn.functional.normalize(d_cond.flatten(1), dim=1).view_as(d_cond)
        parallel = (diff * unit).sum(dim=dims, keepdim=True) * unit
        diff = (diff - parallel) + self.eta * parallel

        d_guided = d_cond + (guidance_scale - 1) * diff
        return ((d_guided - c_skip * z) / c_out).to(dtype)
