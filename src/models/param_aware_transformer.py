# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ---------------------
# # Simpler Time Embedding Network (TimeSiren)
# # ---------------------
# class TimeSiren(nn.Module):
#     def __init__(self, emb_dim: int, target_dim: int) -> None:
#         super(TimeSiren, self).__init__()
#         self.lin1 = nn.Linear(1, emb_dim, bias=False)
#         self.lin2 = nn.Linear(emb_dim, emb_dim)
#         self.lin3 = nn.Linear(emb_dim, emb_dim)
#         self.lin4 = nn.Linear(emb_dim, emb_dim)
#         self.lin_out = nn.Linear(emb_dim, target_dim)
#         self.activation = nn.GELU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.view(-1, 1)
#         x = torch.sin(self.lin1(x))
#         x = self.activation(self.lin2(x))
#         x = self.activation(self.lin3(x))
#         x = self.lin4(x)
#         x = self.lin_out(x)
#         return x

# # ---------------------
# # Parameter Type Embedding
# # ---------------------
# class ParameterTypeEmbedding(nn.Module):
#     def __init__(self, feature_dim, embedding_dim):
#         super(ParameterTypeEmbedding, self).__init__()
#         # Create learnable embeddings for each parameter type
#         self.param_embeddings = nn.Parameter(torch.zeros(feature_dim, embedding_dim))
#         self.param_norms = nn.ModuleList([nn.LayerNorm(1) for _ in range(feature_dim)])
        
#         # Parameter-specific MLP projections
#         self.param_projectors = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(1, embedding_dim // 2),
#                 nn.GELU(),
#                 nn.Linear(embedding_dim // 2, embedding_dim)
#             ) for _ in range(feature_dim)
#         ])
        
#         # Initialize the parameter embeddings
#         nn.init.normal_(self.param_embeddings, mean=0.0, std=0.02)

#     def forward(self, x):
#         # x shape: [batch_size, num_gaussians, feature_dim]
#         batch_size, num_gaussians, feature_dim = x.shape
        
#         # Process each parameter type separately
#         processed_features = []
        
#         for i in range(feature_dim):
#             # Extract parameter i
#             param_i = x[:, :, i:i+1]  # Shape: [batch_size, num_gaussians, 1]
            
#             # Normalize this parameter specifically
#             param_i = self.param_norms[i](param_i)
            
#             # Project using parameter-specific network
#             param_projected = self.param_projectors[i](param_i)
            
#             # Add parameter type embedding
#             param_with_type = param_projected + self.param_embeddings[i].view(1, 1, -1)
            
#             processed_features.append(param_with_type)
        
#         # Concatenate all processed parameters
#         return torch.cat(processed_features, dim=-1)

# # ---------------------
# # Multihead Attention Module
# # ---------------------
# class MultiheadAttention(nn.Module):
#     def __init__(self, dim, num_heads=8):
#         super(MultiheadAttention, self).__init__()
#         self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         attn_output, _ = self.mha(x, x, x)
#         return self.norm(x + attn_output)  # Add residual connection

# # ---------------------
# # Add & Norm helper module
# # ---------------------
# class AddNorm(nn.Module):
#     def __init__(self, dim):
#         super(AddNorm, self).__init__()
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x, sublayer):
#         return self.norm(x + sublayer(x))

# # ---------------------
# # Projection Block
# # ---------------------
# class ProjectionBlock(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(ProjectionBlock, self).__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         self.bn = nn.BatchNorm1d(out_features)

#     def forward(self, x):
#         x = self.linear(x)
#         x = x.transpose(1, 2)
#         x = self.bn(x)
#         x = x.transpose(1, 2)
#         x = F.gelu(x)
#         return x
    
# # ---------------------
# # Initial Projection using ConvProjectionBlock
# # ---------------------
# class InitialProjection(nn.Module):
#     def __init__(self, in_dim, final_dim=1024):  # Changed to 1024
#         super(InitialProjection, self).__init__()
#         self.layer1 = ProjectionBlock(in_dim, final_dim)
#         self.layer2 = ProjectionBlock(final_dim, final_dim)
#         self.layer3 = ProjectionBlock(final_dim, final_dim)
#         self.layer4 = ProjectionBlock(final_dim, final_dim)
#         self.layer5 = ProjectionBlock(final_dim, final_dim)
#         self.shortcut = nn.Conv1d(in_dim, final_dim, kernel_size=1)

#     def forward(self, x):
#         # x shape: (B, L, in_dim)
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         sc = x.transpose(1, 2)  # (B, in_dim, L)
#         sc = self.shortcut(sc)
#         sc = sc.transpose(1, 2)  # (B, L, final_dim)
#         return out + sc

# # ---------------------
# # Updated Output Projection using ConvProjectionBlock - UPDATED FOR 1024 INPUT
# # ---------------------
# class OutputProjection(nn.Module):
#     def __init__(self, in_dim=1024, out_dim=7):  # Changed to 1024
#         super(OutputProjection, self).__init__()
#         self.layer1 = ProjectionBlock(in_dim, out_dim)
#         self.layer2 = ProjectionBlock(out_dim, out_dim)
#         self.layer3 = ProjectionBlock(out_dim, out_dim)
#         self.layer4 = ProjectionBlock(out_dim, out_dim)
#         self.layer5 = ProjectionBlock(out_dim, out_dim)
#         self.shortcut = nn.Conv1d(in_dim, out_dim, kernel_size=1)

#     def forward(self, x):
#         # x shape: (B, L, in_dim)
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         sc = x.transpose(1, 2)
#         sc = self.shortcut(sc)
#         sc = sc.transpose(1, 2)
#         return out + sc

# # ---------------------
# # Transformer Block with Multihead Attention
# # ---------------------
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads=8):
#         super(TransformerBlock, self).__init__()
#         self.add_norm1 = AddNorm(dim)
#         self.multihead_attention = MultiheadAttention(dim, num_heads)
#         self.add_norm2 = AddNorm(dim)
        
#         # Expanded MLP
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
#         x = self.add_norm1(x, lambda inp: self.multihead_attention(inp))
#         x = self.add_norm2(x, lambda inp: self.mlp(inp))
#         return x

# # ---------------------
# # Parameter-Aware Gaussian Transformer with Multihead Attention - USING OUTPUT PROJECTION
# # ---------------------
# class ParameterAwareGaussianTransformer(nn.Module):
#     def __init__(self, input_dim, time_emb_dim, feature_dim, num_timestamps, num_transformer_blocks=6, num_heads=8):
#         super(ParameterAwareGaussianTransformer, self).__init__()
#         self.time_embed = TimeSiren(time_emb_dim, feature_dim)
#         self.num_timesteps = num_timestamps
#         self.input_dim = input_dim
#         self.feature_dim = feature_dim
        
#         # Parameter-specific embedding
#         embedding_dim = 64  # Size for each parameter's embedding
#         self.param_embedding = ParameterTypeEmbedding(feature_dim, embedding_dim)
        
#         # Adjust initial projection to account for expanded feature dimension
#         self.initial_projection = InitialProjection(in_dim=embedding_dim * feature_dim, final_dim=1024)

#         self.transformer_blocks = nn.ModuleList(
#             [TransformerBlock(dim=1024, num_heads=num_heads) for _ in range(num_transformer_blocks)]
#         )
        
#         # Use OutputProjection for final layer
#         self.output_projection = OutputProjection(in_dim=1024, out_dim=feature_dim)

#     def forward(self, gaussians, t):
#         # Normalize t to [0, 1]
#         t_norm = t.float() / self.num_timesteps
        
#         # Expand time embedding over the set
#         t_emb = self.time_embed(t_norm).unsqueeze(1).expand(-1, gaussians.size(1), -1)
        
#         # Add time embedding to input
#         gaussians_with_time = gaussians + t_emb
        
#         # Process each parameter type separately
#         parameter_aware_features = self.param_embedding(gaussians_with_time)
        
#         # Project to transformer dimension
#         x = self.initial_projection(parameter_aware_features)
        
#         # Process through transformer blocks
#         for block in self.transformer_blocks:
#             x = block(x)
        
#         # Use unified output projection instead of parameter-specific projections
#         output = self.output_projection(x)
        
#         return output

# # ---------------------
# # Helper: Count model parameters
# # ---------------------
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # ---------------------
# # Example Usage
# # ---------------------
# if __name__ == "__main__":
#     batch_size = 64
#     num_gaussians = 200
#     feature_dim = 7  # 7 parameters per Gaussian
#     time_emb_dim = 256
#     num_blocks = 24
#     num_heads = 16
#     num_timestamps = 500

#     # Use the new parameter-aware transformer model
#     model = ParameterAwareGaussianTransformer(
#         input_dim=num_gaussians,
#         time_emb_dim=time_emb_dim,
#         feature_dim=feature_dim,
#         num_timestamps=num_timestamps,
#         num_transformer_blocks=num_blocks,
#         num_heads=num_heads,
#     )
    
#     gaussian_inputs = torch.randn(batch_size, num_gaussians, feature_dim)
#     t = torch.randint(low=1, high=501, size=(batch_size,), dtype=torch.long)

#     predicted_noise = model(gaussian_inputs, t.float())
#     print("Predicted noise shape:", predicted_noise.shape)
#     print("\nTotal number of trainable parameters:", count_parameters(model))
    
#     # Keep the original GaussianTransformer for comparison
#     original_model = GaussianTransformer(
#         input_dim=num_gaussians,
#         time_emb_dim=time_emb_dim,
#         feature_dim=feature_dim,
#         num_timestamps=num_timestamps,
#         num_transformer_blocks=num_blocks,
#         num_heads=num_heads,
#     )
    
#     print("\nOriginal model parameters:", count_parameters(original_model))
#     print("Parameter-aware model parameters:", count_parameters(model))

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
        self.lin_out = nn.Linear(emb_dim, target_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.activation(self.lin2(x))
        x = self.lin_out(x)
        return x

# ---------------------
# Parameter Type Embedding
# ---------------------
class ParameterTypeEmbedding(nn.Module):
    def __init__(self, feature_dim, embedding_dim):
        super(ParameterTypeEmbedding, self).__init__()
        # Create learnable embeddings for each parameter type
        self.param_embeddings = nn.Parameter(torch.zeros(feature_dim, embedding_dim))
        
        # Simpler normalization - using LayerNorm for the whole feature dimension
        self.norm = nn.LayerNorm(feature_dim)
        
        # Simplified parameter projection - just one linear layer
        self.param_projectors = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(feature_dim)
        ])
        
        # Initialize the parameter embeddings
        nn.init.normal_(self.param_embeddings, mean=0.0, std=0.02)

    def forward(self, x):
        # x shape: [batch_size, num_gaussians, feature_dim]
        batch_size, num_gaussians, feature_dim = x.shape
        
        # Normalize all features together
        x = self.norm(x)
        
        # Process each parameter type separately
        processed_features = []
        
        for i in range(feature_dim):
            # Extract parameter i
            param_i = x[:, :, i:i+1]  # Shape: [batch_size, num_gaussians, 1]
            
            # Project using parameter-specific network (simpler)
            param_projected = self.param_projectors[i](param_i)
            
            # Add parameter type embedding
            param_with_type = param_projected + self.param_embeddings[i].view(1, 1, -1)
            
            processed_features.append(param_with_type)
        
        # Concatenate all processed parameters
        return torch.cat(processed_features, dim=-1)

# ---------------------
# Multihead Attention Module - Simplified
# ---------------------
class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(MultiheadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        return attn_output  # No residual - handled in TransformerBlock

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
# Simplified Projection Block
# ---------------------
class ProjectionBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectionBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)  # LayerNorm instead of BatchNorm
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
    
# ---------------------
# Simplified Initial Projection - fewer layers
# ---------------------
class InitialProjection(nn.Module):
    def __init__(self, in_dim, final_dim=512):  # Reduced to 512
        super(InitialProjection, self).__init__()
        self.layer1 = ProjectionBlock(in_dim, final_dim // 2)
        self.layer2 = ProjectionBlock(final_dim // 2, final_dim)
        self.shortcut = nn.Linear(in_dim, final_dim)  # Direct linear projection instead of Conv1d

    def forward(self, x):
        # x shape: (B, L, in_dim)
        out = self.layer1(x)
        out = self.layer2(out)
        sc = self.shortcut(x)  # Simple linear shortcut
        return out + sc

# ---------------------
# Simplified Output Projection - fewer layers
# ---------------------
class OutputProjection(nn.Module):
    def __init__(self, in_dim=512, out_dim=7):
        super(OutputProjection, self).__init__()
        self.layer1 = ProjectionBlock(in_dim, in_dim // 2)
        self.layer2 = ProjectionBlock(in_dim // 2, out_dim)
        self.shortcut = nn.Linear(in_dim, out_dim)  # Direct linear projection

    def forward(self, x):
        # x shape: (B, L, in_dim)
        out = self.layer1(x)
        out = self.layer2(out)
        sc = self.shortcut(x)  # Simple linear shortcut
        return out + sc

# ---------------------
# Simplified Transformer Block
# ---------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        # Simplified MLP - shallower
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        # Pre-norm architecture with explicit residuals
        attn_out = self.attn(self.norm1(x))
        x = x + attn_out
        
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x

# ---------------------
# Parameter-Aware Gaussian Transformer - Simplified
# ---------------------
class ParameterAwareGaussianTransformer(nn.Module):
    def __init__(self, input_dim, time_emb_dim, feature_dim, num_timestamps, num_transformer_blocks=6, num_heads=8):
        super(ParameterAwareGaussianTransformer, self).__init__()
        self.time_embed = TimeSiren(time_emb_dim, feature_dim)
        self.num_timesteps = num_timestamps
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        # Parameter-specific embedding
        embedding_dim = 32  # Reduced from 64 to 32 for simplicity
        self.param_embedding = ParameterTypeEmbedding(feature_dim, embedding_dim)
        
        # Reduced transformer dimension to 512 from 1024
        transformer_dim = 512
        
        # Adjust initial projection - simplified
        self.initial_projection = InitialProjection(in_dim=embedding_dim * feature_dim, final_dim=transformer_dim)

        # Reduce number of transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(dim=transformer_dim, num_heads=num_heads) for _ in range(num_transformer_blocks)]
        )
        
        # Simplified output projection
        self.output_projection = OutputProjection(in_dim=transformer_dim, out_dim=feature_dim)

    def forward(self, gaussians, t):
        # Normalize t to [0, 1]
        t_norm = t.float() / self.num_timesteps
        
        # Expand time embedding over the set
        t_emb = self.time_embed(t_norm).unsqueeze(1).expand(-1, gaussians.size(1), -1)
        
        # Add time embedding to input
        gaussians_with_time = gaussians + t_emb
        
        # Process each parameter type separately
        parameter_aware_features = self.param_embedding(gaussians_with_time)
        
        # Project to transformer dimension
        x = self.initial_projection(parameter_aware_features)
        
        # Process through transformer blocks 
        for block in self.transformer_blocks:
            x = block(x)
        
        # Use unified output projection
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
    batch_size = 64
    num_gaussians = 200
    feature_dim = 7  # 7 parameters per Gaussian
    time_emb_dim = 128  # Reduced from 256
    num_blocks = 12    # Reduced from 24
    num_heads = 8      # Reduced from 16
    num_timestamps = 500

    # Use the simplified parameter-aware transformer model
    model = ParameterAwareGaussianTransformer(
        input_dim=num_gaussians,
        time_emb_dim=time_emb_dim,
        feature_dim=feature_dim,
        num_timestamps=num_timestamps,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
    )
    
    gaussian_inputs = torch.randn(batch_size, num_gaussians, feature_dim)
    t = torch.randint(low=1, high=501, size=(batch_size,), dtype=torch.long)

    predicted_noise = model(gaussian_inputs, t.float())
    print("Predicted noise shape:", predicted_noise.shape)
    print("\nTotal number of trainable parameters:", count_parameters(model))