from typing import Dict, Tuple
import torch
import torch.nn as nn
import math

class DDPM(nn.Module):
    def __init__(
        self,
        n_T: int,
        schedule_type: str = "cosine",   # "cosine" or "linear"
        betas: Tuple[float, float] = (1e-4, 0.02),  # used only for linear schedule
        s: float = 0.008,                # small constant for cosine schedule
    ) -> None:
        super(DDPM, self).__init__()
        if schedule_type == "cosine":
            schedule = cosine_ddpm_schedules(n_T, s)
        elif schedule_type == "linear":
            schedule = ddpm_schedules(betas[0], betas[1], n_T)
        else:
            raise ValueError("schedule_type must be either 'cosine' or 'linear'")
        for k, v in schedule.items():
            self.register_buffer(k, v)
        self.n_T = n_T

    def get_noisy_images_and_noise(self, x: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given the original images x (x₀) and a tensor of timesteps (each in [1, T]),
        compute:
          - xₜ: the noisy image at time t,
          - xₜ₋₁: the noisy image at time t-1 (with t=1 returning x₀),
          - eps: the added Gaussian noise.
        """
        device = x.device
        # Move schedule buffers to the same device as x
        self.sqrtab = self.sqrtab.to(device)
        self.sqrtmab = self.sqrtmab.to(device)

        if (timesteps == 0).any():
            raise ValueError("Timesteps should be in the range [1, T]. Received a timestep of 0.")

        eps = torch.randn_like(x, device=device)

        # Compute x_t using precomputed sqrt(ab) and sqrt(mab)
        sqrtab_t = self.sqrtab[timesteps].view(-1, 1, 1)
        sqrtmab_t = self.sqrtmab[timesteps].view(-1, 1, 1)
        x_t = sqrtab_t * x + sqrtmab_t * eps

        # Compute x_{t-1}
        timesteps_prev = timesteps - 1  # For t=1, this returns index 0 (i.e. x₀).
        sqrtab_t_prev = self.sqrtab[timesteps_prev].view(-1, 1, 1)
        sqrtmab_t_prev = self.sqrtmab[timesteps_prev].view(-1, 1, 1)
        x_t_minus_1 = sqrtab_t_prev * x + sqrtmab_t_prev * eps

        return x_t, x_t_minus_1, eps

def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Compute the DDPM schedules using a linear beta schedule.

    Args:
        beta1 (float): Starting value of beta.
        beta2 (float): Final value of beta.
        T (int): Total number of timesteps.

    Returns:
        Dict[str, torch.Tensor]: Dictionary with schedule tensors.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    # Linear schedule for beta_t over [0, T]
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,               
        "oneover_sqrta": oneover_sqrta,     
        "sqrt_beta_t": torch.sqrt(beta_t),  
        "alphabar_t": alphabar_t,           
        "sqrtab": sqrtab,                 
        "sqrtmab": sqrtmab,               
        "mab_over_sqrtmab": mab_over_sqrtmab,
    }
def cosine_ddpm_schedules(T: int, s: float = 0.008) -> Dict[str, torch.Tensor]:
    """
    Compute the DDPM schedules using a cosine noise schedule.

    Args:
        T (int): Total number of timesteps.
        s (float): A small offset (typically 0.008) to avoid singularities.

    Returns:
        Dict[str, torch.Tensor]: Dictionary with schedule tensors.
    """
    steps = torch.arange(0, T + 1, dtype=torch.float32)
    # Define the cosine function f(t)
    f = lambda t: torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    f_steps = f(steps)  # Compute f for all steps as a tensor
    alphabar_t = f_steps / f_steps[0]  # Normalize so that alphabar_0 = 1

    beta_t = torch.empty(T + 1, dtype=torch.float32)
    beta_t[0] = 0.0
    # Compute beta_t from the cosine schedule
    for t in range(1, T + 1):
        beta_t[t] = min(1 - alphabar_t[t] / alphabar_t[t - 1], 0.999)
    
    alpha_t = 1 - beta_t
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": torch.sqrt(beta_t),
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab,
    }
