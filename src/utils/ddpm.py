from typing import Optional

import torch


def gather(v: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    v = torch.gather(input=v, dim=-1, index=index)
    return v.reshape(-1, 1, 1, 1)


class Diffusion:
    def __init__(self, beta_range: list[float, float], T: int) -> None:
        beta_1, beta_T = beta_range
        self.beta = torch.linspace(beta_1, beta_T, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.T = T
        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> "Diffusion":
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.device = device

        return self

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gathered = gather(v=self.alpha_bar, index=t)
        mean = gathered**0.5 * x0
        var = 1 - gathered

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0, device=self.device)

        mean, var = self.q_xt_x0(x0, t)

        return mean + (var**0.5) * eps

    def sample_t(self, size: tuple[int]) -> torch.Tensor:
        return torch.randint(low=0, high=self.T, size=size, device=self.device)

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, eps_theta: torch.Tensor) -> torch.Tensor:
        alpha = gather(v=self.alpha, index=t)
        alpha_bar = gather(v=self.alpha_bar, index=t)
        eps_coeff = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha**0.5) * (xt - eps_coeff * eps_theta)
        var = gather(v=self.beta, index=t)
        eps = torch.randn(xt.shape, device=self.device)

        return mean + (var**0.5) * eps
