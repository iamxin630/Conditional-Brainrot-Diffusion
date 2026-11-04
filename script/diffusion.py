"""
Gaussian Diffusion — noise schedule, forward process, training loss, DDIM sampling.

Implements:
  - Cosine beta schedule (Improved DDPM, Nichol & Dhariwal 2021)
  - Forward process  q(x_t | x_0)
  - Training loss  (epsilon prediction, MSE)
  - DDIM deterministic sampling with Classifier-Free Guidance
"""

import math

import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from "Improved Denoising Diffusion Probabilistic Models".

    Returns betas of shape [timesteps].
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


class GaussianDiffusion:
    """
    Handles the diffusion forward process, loss computation, and DDIM sampling.

    Parameters
    ----------
    timesteps : int
        Total number of diffusion timesteps T.
    noise_schedule : str
        Type of noise schedule ('cosine').
    device : torch.device
        Device to place precomputed tensors on.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        noise_schedule: str = "cosine",
        device: torch.device = torch.device("cpu"),
    ):
        self.timesteps = timesteps
        self.device = device

        # Compute betas and derived quantities
        if noise_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store as buffers (not parameters — we never optimize these)
        self.betas = betas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    def to(self, device: torch.device):
        """Move all precomputed tensors to a new device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    # ----- Forward process -----

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1-ᾱ_t) I).

        Parameters
        ----------
        x_0 : [B, C, H, W]  clean image in [-1, 1]
        t   : [B]  timestep indices
        noise : optional pre-sampled noise [B, C, H, W]

        Returns
        -------
        x_t : [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    # ----- Training loss -----

    def training_loss(
        self, model, x_0: torch.Tensor,
        animal_ids: torch.Tensor, object_ids: torch.Tensor, pair_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute epsilon-prediction MSE loss.

        Randomly samples timestep and noise, then computes:
            loss = MSE(model(x_t, t, cond), noise)
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        pred_noise = model(x_t, t, animal_ids, object_ids, pair_ids)
        loss = F.mse_loss(pred_noise, noise)
        return loss

    # ----- DDIM Sampling -----

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape,
        animal_ids: torch.Tensor,
        object_ids: torch.Tensor,
        pair_ids: torch.Tensor,
        null_animal_ids: torch.Tensor,
        null_object_ids: torch.Tensor,
        null_pair_ids: torch.Tensor,
        ddim_steps: int = 50,
        cfg_scale: float = 1.5,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling with Classifier-Free Guidance.

        Parameters
        ----------
        model : nn.Module
            The noise prediction model.
        shape : tuple
            Shape of the output (B, C, H, W).
        animal_ids, object_ids, pair_ids : [B]
            Condition labels.
        null_animal_ids, null_object_ids, null_pair_ids : [B]
            Null (unconditional) labels for CFG.
        ddim_steps : int
            Number of DDIM sampling steps.
        cfg_scale : float
            Classifier-free guidance scale. 1.0 = no guidance.
        eta : float
            DDIM stochasticity (0 = deterministic).

        Returns
        -------
        x_0 : [B, C, H, W] in [-1, 1]
        """
        device = self.device
        B = shape[0]

        # Build sub-sequence of timesteps: evenly spaced from T-1 down to 0
        step_size = self.timesteps // ddim_steps
        timestep_seq = list(range(0, self.timesteps, step_size))  # ascending
        timestep_seq = list(reversed(timestep_seq))  # descending: T-1, ..., 0

        # Start from pure noise
        x = torch.randn(shape, device=device)

        for i, t_cur in enumerate(timestep_seq):
            t_tensor = torch.full((B,), t_cur, device=device, dtype=torch.long)

            # Conditional prediction
            noise_cond = model(x, t_tensor, animal_ids, object_ids, pair_ids)

            # CFG
            if cfg_scale != 1.0:
                noise_uncond = model(x, t_tensor, null_animal_ids, null_object_ids, null_pair_ids)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = noise_cond

            # Current and previous alpha_bar
            alpha_bar_t = self.alphas_cumprod[t_cur]

            if i < len(timestep_seq) - 1:
                t_prev = timestep_seq[i + 1]
                alpha_bar_t_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_bar_t_prev = torch.tensor(1.0, device=device)

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-1, 1)

            # DDIM update
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma ** 2) * noise_pred
            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt

            if sigma > 0 and i < len(timestep_seq) - 1:
                x = x + sigma * torch.randn_like(x)

        return x


def build_diffusion_from_config(config: dict) -> GaussianDiffusion:
    """Construct GaussianDiffusion from config dict."""
    diff_cfg = config["diffusion"]
    return GaussianDiffusion(
        timesteps=diff_cfg["timesteps"],
        noise_schedule=diff_cfg["noise_schedule_type"],
        device=torch.device(diff_cfg["device"]),
    )
