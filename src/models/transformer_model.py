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
        return self.norm(x + attn_output)  # Add residual connection

# ---------------------
# Add & Norm helper module
# ---------------------
class AddNorm(nn.Module):
    def __init__(self, dim):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, sublayer):
        return self.norm(x + sublayer(x))

# ---------------------
# Projection Block
# ---------------------
class ProjectionBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectionBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        x = F.gelu(x)
        return x
    
# ---------------------
# Initial Projection using ConvProjectionBlock
# ---------------------
class InitialProjection(nn.Module):
    def __init__(self, in_dim, final_dim=512):
        super(InitialProjection, self).__init__()
        self.layer1 = ProjectionBlock(in_dim, final_dim)
        self.layer2 = ProjectionBlock(final_dim, final_dim)
        self.layer3 = ProjectionBlock(final_dim, final_dim)
        self.layer4 = ProjectionBlock(final_dim, final_dim)
        self.layer5 = ProjectionBlock(final_dim, final_dim)
        self.shortcut = nn.Conv1d(in_dim, final_dim, kernel_size=1)

    def forward(self, x):
        # x shape: (B, L, in_dim)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
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
        self.layer1 = ProjectionBlock(in_dim, out_dim)
        self.layer2 = ProjectionBlock(out_dim, out_dim)
        self.layer3 = ProjectionBlock(out_dim, out_dim)
        self.layer4 = ProjectionBlock(out_dim, out_dim)
        self.layer5 = ProjectionBlock(out_dim, out_dim)
        self.shortcut = nn.Conv1d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        # x shape: (B, L, in_dim)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        sc = x.transpose(1, 2)
        sc = self.shortcut(sc)
        sc = sc.transpose(1, 2)
        return out + sc

# # ---------------------
# # Initial Projection
# # ---------------------
# class InitialProjection(nn.Module):
#     def __init__(self, in_dim, final_dim=512):
#         super(InitialProjection, self).__init__()
#         self.layer1 = ProjectionBlock(in_dim, 256)
#         self.layer2 = ProjectionBlock(256, final_dim)
#         self.shortcut = nn.Linear(in_dim, final_dim)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         sc = self.shortcut(x)
#         return out + sc

# # ---------------------
# # Output Projection
# # ---------------------
# class OutputProjection(nn.Module):
#     def __init__(self, in_dim=512, out_dim=7):
#         super(OutputProjection, self).__init__()
#         self.layer1 = ProjectionBlock(in_dim, 256)
#         self.layer2 = ProjectionBlock(256, out_dim)
#         self.shortcut = nn.Linear(in_dim, out_dim)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         sc = self.shortcut(x)
#         return out + sc
    
# # ---------------------
# # Self-Attention Module (single-head)
# # ---------------------
# class SelfAttention(nn.Module):
#     def __init__(self, dim):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(dim, dim)
#         self.key = nn.Linear(dim, dim)
#         self.value = nn.Linear(dim, dim)
#         self.scale = dim ** -0.5

#     def forward(self, x):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#         attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
#         out = torch.bmm(attn_weights, V)
#         return out

# # ---------------------
# # Transformer Block with enhanced MLP capacity 
# # ---------------------    
# class TransformerBlock(nn.Module):
#     def __init__(self, dim):
#         super(TransformerBlock, self).__init__()
#         self.add_norm1 = AddNorm(dim)
#         self.attention = SelfAttention(dim)
#         self.add_norm2 = AddNorm(dim)
#         # Expanded MLP: increasing intermediate dimensions for better capacity.
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.GELU(),
#             nn.Linear(dim * 2, dim * 4),
#             nn.GELU(),
#             nn.Linear(dim * 4, dim * 2),
#             nn.GELU(),
#             nn.Linear(dim * 2, dim)
#         )

#     def forward(self, x):
#         x = self.add_norm1(x, lambda inp: self.attention(inp))
#         x = self.add_norm2(x, lambda inp: self.mlp(inp))
#         return x

# ---------------------
# Transformer Block with Multihead Attention
# ---------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.add_norm1 = AddNorm(dim)
        self.multihead_attention = MultiheadAttention(dim, num_heads)
        self.add_norm2 = AddNorm(dim)
        
        # Expanded MLP
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
# Gaussian Transformer with Multihead Attention
# ---------------------
class GaussianTransformer(nn.Module):
    def __init__(self, input_dim, time_emb_dim, feature_dim, num_timestamps, num_transformer_blocks=6, num_heads=8):
        super(GaussianTransformer, self).__init__()
        self.time_embed = TimeSiren(time_emb_dim, feature_dim)
        self.num_timesteps = num_timestamps
        self.input_dim = input_dim
        self.initial_projection = InitialProjection(in_dim=feature_dim, final_dim=512)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(dim=512, num_heads=num_heads) for _ in range(num_transformer_blocks)]
        )
        self.output_projection = OutputProjection(in_dim=512, out_dim=feature_dim)

    def forward(self, gaussians, t):
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
# Helper: Count model parameters
# ---------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------
# Example Usage
# ---------------------
if __name__ == "__main__":
    batch_size = 32
    num_gaussians = 70
    feature_dim = 7
    time_emb_dim = 512
    num_blocks = 16 
    num_heads = 64
    num_timestamps = 1000 

    model = GaussianTransformer(
        input_dim=num_gaussians,
        time_emb_dim=time_emb_dim,
        feature_dim=feature_dim,
        num_timestamps=num_timestamps,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
    )
    
    gaussian_inputs = torch.randn(batch_size, num_gaussians, feature_dim)
    t = torch.randint(low=1, high=1001, size=(batch_size,), dtype=torch.long)

    predicted_noise = model(gaussian_inputs, t.float())
    print("Predicted noise shape:", predicted_noise.shape)
    print("\nTotal number of trainable parameters:", count_parameters(model))