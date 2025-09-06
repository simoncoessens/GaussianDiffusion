import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------
# Simpler Time Embedding Network (TimeSiren)
# ---------------------
class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int, target_dim: int) -> None:
        super(TimeSiren, self).__init__()
        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)
        self.lin3 = nn.Linear(emb_dim, emb_dim)
        self.lin4 = nn.Linear(emb_dim, emb_dim)
        self.lin_out = nn.Linear(emb_dim, target_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed to be in the range [0, 1] after normalization.
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.activation(self.lin2(x))
        x = self.activation(self.lin3(x))
        x = self.lin4(x)
        x = self.lin_out(x)
        return x

# ---------------------
# Multihead Attention Module
# ---------------------
class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(MultiheadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        return self.norm(x + attn_output)  # Residual connection

# ---------------------
# Add & Norm Helper Module
# ---------------------
class AddNorm(nn.Module):
    def __init__(self, dim):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, sublayer):
        return self.norm(x + sublayer(x))

# ---------------------
# ConvProjectionBlock (PointNet-style using Conv1d)
# ---------------------
class ConvProjectionBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvProjectionBlock, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # x shape: (B, L, C)
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)  # (B, L, out_features)
        return x

# ---------------------
# Initial Projection using ConvProjectionBlock
# ---------------------
class InitialProjection(nn.Module):
    def __init__(self, in_dim, final_dim=512):
        super(InitialProjection, self).__init__()
        self.layer1 = ConvProjectionBlock(in_dim, 64)
        self.layer2 = ConvProjectionBlock(64, 128)
        self.layer3 = ConvProjectionBlock(128, 256)
        self.layer4 = ConvProjectionBlock(256, final_dim)
        self.shortcut = nn.Conv1d(in_dim, final_dim, kernel_size=1)

    def forward(self, x):
        # x shape: (B, L, in_dim)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        sc = x.transpose(1, 2)  # (B, in_dim, L)
        sc = self.shortcut(sc)
        sc = sc.transpose(1, 2)  # (B, L, final_dim)
        return out + sc

# ---------------------
# Updated Output Projection using ConvProjectionBlock (Deep Projection)
# ---------------------
class OutputProjection(nn.Module):
    def __init__(self, in_dim=512, out_dim=7):
        super(OutputProjection, self).__init__()
        self.layer1 = ConvProjectionBlock(in_dim, 256)
        self.layer2 = ConvProjectionBlock(256, 128)
        self.layer3 = ConvProjectionBlock(128, 64)
        self.layer4 = ConvProjectionBlock(64, out_dim)
        self.shortcut = nn.Conv1d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        # x shape: (B, L, in_dim)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        sc = x.transpose(1, 2)
        sc = self.shortcut(sc)
        sc = sc.transpose(1, 2)
        return out + sc

# ---------------------
# Transformer Block with Multihead Attention and Expanded MLP
# ---------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.add_norm1 = AddNorm(dim)
        self.multihead_attention = MultiheadAttention(dim, num_heads)
        self.add_norm2 = AddNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        x = self.add_norm1(x, lambda inp: self.multihead_attention(inp))
        x = self.add_norm2(x, lambda inp: self.mlp(inp))
        return x

# ---------------------
# Gaussian Transformer with Multihead Attention and Time Normalization
# ---------------------
class GaussianTransformer(nn.Module):
    def __init__(self, input_dim, time_emb_dim, feature_dim, num_transformer_blocks=6, num_heads=8, num_timesteps=1000):
        """
        Args:
            input_dim: Not used directly in this design (the model operates on sets of Gaussians).
            time_emb_dim: Hidden dimension for the time embedding.
            feature_dim: Dimension of each Gaussian (should be 7).
            num_transformer_blocks: Number of Transformer blocks.
            num_heads: Number of attention heads.
            num_timesteps: Total number of timesteps in the diffusion process (T).
        """
        super(GaussianTransformer, self).__init__()
        self.num_timesteps = num_timesteps  # For time normalization
        self.time_embed = TimeSiren(time_emb_dim, feature_dim)
        self.initial_projection = InitialProjection(in_dim=feature_dim, final_dim=512)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(dim=512, num_heads=num_heads) for _ in range(num_transformer_blocks)]
        )
        self.output_projection = OutputProjection(in_dim=512, out_dim=feature_dim)

    def forward(self, gaussians, t):
        """
        Args:
            gaussians: Tensor of shape (B, L, feature_dim)
            t: Tensor of shape (B,) (raw time values in [0, num_timesteps])
        Returns:
            Predicted noise tensor of shape (B, L, feature_dim)
        """
        # Normalize t to [0, 1]
        t_norm = t.float() / self.num_timesteps  
        # Expand time embedding over the set
        t_emb = self.time_embed(t_norm).unsqueeze(1).expand(-1, gaussians.size(1), -1)
        gaussians = gaussians + t_emb
        x = self.initial_projection(gaussians)
        for block in self.transformer_blocks:
            x = block(x)
        output = self.output_projection(x)
        return output

# ---------------------
# Helper: Count Model Parameters
# ---------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------
# Example Usage
# ---------------------
if __name__ == "__main__":
    batch_size = 32
    num_gaussians = 70
    feature_dim = 7   # Each Gaussian is represented by 7 features.
    time_emb_dim = 32
    num_blocks = 16
    num_heads = 64
    num_timesteps = 1000  # Total timesteps

    model = GaussianTransformer(
        input_dim=num_gaussians,  # not used directly here
        time_emb_dim=time_emb_dim,
        feature_dim=feature_dim,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
        num_timesteps=num_timesteps
    )
    
    gaussian_inputs = torch.randn(batch_size, num_gaussians, feature_dim)
    t = torch.randint(low=1, high=num_timesteps + 1, size=(batch_size,), dtype=torch.long)

    predicted_noise = model(gaussian_inputs, t)
    print("Predicted noise shape:", predicted_noise.shape)
    print("\nTotal number of trainable parameters:", count_parameters(model))