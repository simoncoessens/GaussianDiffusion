import torch

def denormalize_parameters(W_normalized, param_ranges):
    """Denormalizes parameters from [-1, 1] back to the original range."""
    W_denormalized = torch.zeros_like(W_normalized)
    for i in range(W_normalized.shape[-1]):
        min_val, max_val = param_ranges[i]
        if min_val == max_val:
            W_denormalized[..., i] = min_val
        else:
            # Update only the i-th feature slice
            W_denormalized[..., i] = ((W_normalized[..., i] + 1) / 2) * (max_val - min_val) + min_val
    return W_denormalized
