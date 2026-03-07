"""
DiT-style Gaussian Transformer for diffusion in 2D Gaussian splatting latent space.

Architecture follows DiT (Peebles & Xie, ICCV 2023):
  - AdaLN-Zero conditioning: timestep modulates each block via shift/scale/gate
  - Pre-norm residual blocks with gated outputs
  - No positional encodings on set elements (permutation equivariant)
  - No BatchNorm (only LayerNorm)
"""

import math

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Sinusoidal positional encoding + 2-layer MLP for timestep conditioning."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def sinusoidal_embedding(t, dim):
        """Create sinusoidal positional embeddings from scalar timesteps.

        Args:
            t: (B,) tensor of timestep values
            dim: embedding dimension (must be even)

        Returns:
            (B, dim) sinusoidal embeddings
        """
        half_dim = dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)  # (B, half_dim)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)

    def forward(self, t):
        """
        Args:
            t: (B,) tensor of timestep indices

        Returns:
            (B, hidden_size) conditioning vector
        """
        t_freq = self.sinusoidal_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning (DiT pattern).

    Each block applies:
        x = x + gate_attn * attn(modulate(norm1(x), shift_attn, scale_attn))
        x = x + gate_mlp  * mlp(modulate(norm2(x), shift_mlp, scale_mlp))

    The modulation projection is zero-initialized so each block starts as identity.
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        # AdaLN modulation: projects conditioning to 6 vectors
        # (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Zero-initialize so block starts as identity
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        """
        Args:
            x: (B, K, hidden_size) set of token embeddings
            c: (B, hidden_size) conditioning vector (from timestep embedder)

        Returns:
            (B, K, hidden_size)
        """
        # Compute 6 modulation parameters from conditioning
        mod = self.adaLN_modulation(c).unsqueeze(1)  # (B, 1, 6*H)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Self-attention with AdaLN
        h = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_attn * attn_out

        # FFN with AdaLN
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(h)

        return x


class LabelEmbedder(nn.Module):
    """Embeds class labels into conditioning vectors (for CFG)."""

    def __init__(self, num_classes, hidden_size, dropout_prob=0.0):
        super().__init__()
        use_cfg = dropout_prob > 0
        # +1 for the "null" / unconditional class token
        self.embedding_table = nn.Embedding(num_classes + use_cfg, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """During training, randomly replace labels with null token."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train=False, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class GaussianTransformer(nn.Module):
    """DiT-style transformer for diffusion on sets of 2D Gaussians.

    Permutation equivariant: no positional encodings on set elements.
    Timestep conditioning via AdaLN-Zero in every block.
    Optional class conditioning for Classifier-Free Guidance (CFG).

    Constructor interface kept compatible with the original model.
    """

    def __init__(self, input_dim, time_emb_dim, feature_dim, num_timestamps,
                 num_transformer_blocks=6, num_heads=8,
                 num_classes=0, class_dropout_prob=0.0):
        """
        Args:
            input_dim: sequence length K (kept for API compatibility, unused internally)
            time_emb_dim: hidden size for the transformer and timestep MLP
            feature_dim: per-Gaussian feature dimension (e.g. 6)
            num_timestamps: max T for the diffusion schedule
            num_transformer_blocks: number of DiT blocks
            num_heads: attention heads per block
            num_classes: number of classes (0 = unconditional, no label embedder)
            class_dropout_prob: probability of dropping class label (for CFG training)
        """
        super().__init__()
        hidden_size = time_emb_dim  # use time_emb_dim as the hidden size
        self.num_timesteps = num_timestamps
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Timestep conditioning
        self.time_embed = TimestepEmbedder(hidden_size)

        # Class conditioning (optional, for CFG)
        if num_classes > 0:
            self.label_embed = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Simple input/output projections
        self.input_proj = nn.Linear(feature_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, feature_dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads)
            for _ in range(num_transformer_blocks)
        ])

        # Final AdaLN before output projection
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        # Zero-init final modulation
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        # Zero-init output projection so initial predictions are zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following DiT conventions."""
        # Initialize all linear layers and LayerNorms
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # Skip layers we already zero-initialized
                if module.weight.abs().sum() == 0:
                    return
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def forward(self, gaussians, t, y=None):
        """
        Args:
            gaussians: (B, K, feature_dim) noisy Gaussian parameters
            t: (B,) timestep indices
            y: (B,) class labels (optional, for CFG)

        Returns:
            (B, K, feature_dim) predicted noise
        """
        # Timestep conditioning
        t_norm = t.float() / self.num_timesteps
        c = self.time_embed(t_norm)  # (B, hidden_size)

        # Add class conditioning if available
        if self.num_classes > 0 and y is not None:
            c = c + self.label_embed(y, train=self.training)

        # Project input features to hidden dimension
        x = self.input_proj(gaussians)  # (B, K, hidden_size)

        # DiT blocks
        for block in self.blocks:
            x = block(x, c)

        # Final AdaLN + output projection
        mod = self.final_adaLN(c).unsqueeze(1)  # (B, 1, 2*H)
        shift, scale = mod.chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        output = self.output_proj(x)  # (B, K, feature_dim)

        return output


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    batch_size = 32
    num_gaussians = 70
    feature_dim = 6
    time_emb_dim = 512
    num_blocks = 32
    num_heads = 64
    num_timestamps = 200

    model = GaussianTransformer(
        input_dim=num_gaussians,
        time_emb_dim=time_emb_dim,
        feature_dim=feature_dim,
        num_timestamps=num_timestamps,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
    )

    gaussian_inputs = torch.randn(batch_size, num_gaussians, feature_dim)
    t = torch.randint(low=1, high=num_timestamps + 1, size=(batch_size,), dtype=torch.long)

    predicted_noise = model(gaussian_inputs, t.float())
    print("Predicted noise shape:", predicted_noise.shape)
    print("Total trainable parameters:", f"{count_parameters(model):,}")

    # Verify permutation equivariance
    perm = torch.randperm(num_gaussians)
    out_original = model(gaussian_inputs, t.float())
    out_permuted = model(gaussian_inputs[:, perm], t.float())
    equivariance_error = (out_original[:, perm] - out_permuted).abs().max().item()
    print(f"Permutation equivariance error: {equivariance_error:.2e}")
