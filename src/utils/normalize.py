import torch

def normalize_parameters(W, param_ranges):
    """Normalizes parameters to the range [-1, 1]."""
    W_normalized = torch.zeros_like(W)
    for i in range(W.shape[-1]):
        min_val, max_val = param_ranges[i]
        if min_val == max_val:
            W_normalized[..., i] = 0.0  # Using ... allows for more general shapes.
        else:
            W_normalized[..., i] = 2 * (W[..., i] - min_val) / (max_val - min_val) - 1
    return W_normalized