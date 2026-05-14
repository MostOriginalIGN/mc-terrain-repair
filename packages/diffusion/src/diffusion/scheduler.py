"""Simple Gaussian diffusion scheduler utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SchedulerConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class GaussianDiffusionScheduler:
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.config = SchedulerConfig(num_train_timesteps, beta_start, beta_end)
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.posterior_variance = self.betas * (1.0 - prev) / (1.0 - self.alphas_cumprod)

    def to(self, device: torch.device | str) -> 'GaussianDiffusionScheduler':
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(0, self.config.num_train_timesteps, (batch_size,), device=device)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def predict_x0_from_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
        return (x_t - sqrt_one_minus * noise) / sqrt_alpha.clamp(min=1e-6)

    def step(self, noise_pred: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        betas_t = self._extract(self.betas, timesteps, sample.shape)
        alphas_t = self._extract(self.alphas, timesteps, sample.shape)
        alpha_prod_t = self._extract(self.alphas_cumprod, timesteps, sample.shape)
        prev_t = torch.clamp(timesteps - 1, min=0)
        alpha_prod_prev = self._extract(self.alphas_cumprod, prev_t, sample.shape)

        pred_x0 = self.predict_x0_from_noise(sample, timesteps, noise_pred).clamp(0.0, 1.0)
        coeff_x0 = (torch.sqrt(alpha_prod_prev) * betas_t) / (1.0 - alpha_prod_t)
        coeff_xt = (torch.sqrt(alphas_t) * (1.0 - alpha_prod_prev)) / (1.0 - alpha_prod_t)
        mean = coeff_x0 * pred_x0 + coeff_xt * sample

        variance = self._extract(self.posterior_variance, timesteps, sample.shape)
        noise = torch.randn_like(sample)
        nonzero_mask = (timesteps > 0).float().view(-1, *([1] * (sample.ndim - 1)))
        return mean + nonzero_mask * torch.sqrt(variance.clamp(min=1e-20)) * noise

    def inference_timesteps(self, device: torch.device | str) -> torch.Tensor:
        return torch.arange(self.config.num_train_timesteps - 1, -1, -1, device=device, dtype=torch.long)

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.to(timesteps.device)[timesteps]
        return out.view(-1, *([1] * (len(shape) - 1)))
