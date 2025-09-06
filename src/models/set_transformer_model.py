import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------
# Time Embedding (TimeSiren)
# ---------------------
class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int, target_dim: int):
        """
        Embeds a scalar time input into a vector of dimension target_dim.
        """
        super(TimeSiren, self).__init__()
        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)
        self.lin3 = nn.Linear(emb_dim, emb_dim)
        self.lin4 = nn.Linear(emb_dim, emb_dim)
        self.lin_out = nn.Linear(emb_dim, target_dim)
        self.activation = nn.GELU()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1) -> reshape to (B, 1)
        t = t.view(-1, 1)
        x = torch.sin(self.lin1(t))
        x = self.activation(self.lin2(x))
        x = self.activation(self.lin3(x))
        x = self.lin4(x)
        x = self.lin_out(x)
        return x

# ---------------------
# MAB Block (Multihead Attention Block) with Dropout and Residual Connections
# ---------------------
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True, dropout=0.1):
        """
        Implements the Multihead Attention Block (MAB) with dropout.
        Projects Q, K, V into a latent space, splits them into multiple heads,
        computes scaled dot-product attention, and then merges the heads.
        Optionally applies layer normalization.
        """
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.dropout = nn.Dropout(dropout)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

    def forward(self, Q, K):
        # Q: (B, L_q, dim_Q), K: (B, L_k, dim_K)
        Q = self.fc_q(Q)  # (B, L_q, dim_V)
        K = self.fc_k(K)  # (B, L_k, dim_V)
        V = self.fc_v(K)  # (B, L_k, dim_V)

        dim_split = self.dim_V // self.num_heads
        # Split into heads and merge batch dimension
        Q_ = torch.cat(Q.split(dim_split, dim=-1), dim=0)  # (B*num_heads, L_q, dim_split)
        K_ = torch.cat(K.split(dim_split, dim=-1), dim=0)  # (B*num_heads, L_k, dim_split)
        V_ = torch.cat(V.split(dim_split, dim=-1), dim=0)  # (B*num_heads, L_k, dim_split)

        attn = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), dim=-1)
        attn = self.dropout(attn)
        O_ = Q_ + attn.bmm(V_)  # Residual connection inside attention
        # Merge heads back
        O = torch.cat(O_.split(Q.size(0), dim=0), dim=-1)
        if hasattr(self, 'ln0'):
            O = self.ln0(O)
        O = O + F.gelu(self.fc_o(O))  # Post-attention MLP with residual
        O = self.dropout(O)
        if hasattr(self, 'ln1'):
            O = self.ln1(O)
        return O

# ---------------------
# SAB Block (Self-Attention Block) using MAB
# ---------------------
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=True, dropout=0.1):
        """
        SAB uses MAB with Q = K = input to model full self-attention.
        """
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X):
        return self.mab(X, X)

# ---------------------
# ISAB Block (Induced Set Attention Block) using MAB
# ---------------------
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inducing, ln=True, dropout=0.1):
        """
        ISAB introduces a fixed set of inducing points to reduce the cost of attention.
        First, the inducing points attend to the input, then the input attends to the inducing representation.
        """
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inducing, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X):
        batch_size = X.size(0)
        I_expanded = self.I.expand(batch_size, -1, -1)  # (B, num_inducing, dim_out)
        H = self.mab0(I_expanded, X)
        return self.mab1(X, H)

# ---------------------
# Updated Initial Projection (Deeper and Wider)
# ---------------------
class InitialProjection(nn.Module):
    def __init__(self, in_dim, final_dim):
        """
        Projects the input features (dimension in_dim) to a higher dimension (final_dim)
        using a deeper transformation with four linear layers and a residual shortcut.
        """
        super(InitialProjection, self).__init__()
        self.layer1 = nn.Linear(in_dim, final_dim)
        self.layer2 = nn.Linear(final_dim, final_dim)
        self.layer3 = nn.Linear(final_dim, final_dim)
        self.layer4 = nn.Linear(final_dim, final_dim)
        self.shortcut = nn.Linear(in_dim, final_dim)

    def forward(self, x):
        out = F.gelu(self.layer1(x))
        out = F.gelu(self.layer2(out))
        out = F.gelu(self.layer3(out))
        out = F.gelu(self.layer4(out))
        return out + self.shortcut(x)

# ---------------------
# Updated Output Projection (Deeper, matching the InitialProjection scheme)
# ---------------------
class OutputProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        Projects from the transformer dimension (in_dim) back to the original feature dimension (out_dim)
        using a deeper transformation with multiple linear layers and a residual shortcut.
        """
        super(OutputProjection, self).__init__()
        self.layer1 = nn.Linear(in_dim, in_dim)
        self.layer2 = nn.Linear(in_dim, in_dim)
        self.layer3 = nn.Linear(in_dim, in_dim)
        self.final_proj = nn.Linear(in_dim, out_dim)
        self.shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = F.gelu(self.layer1(x))
        out = F.gelu(self.layer2(out))
        out = F.gelu(self.layer3(out))
        out = F.gelu(self.final_proj(out))
        return out + self.shortcut(x)

# ---------------------
# Gaussian Transformer Model using Repeated Set Transformer Blocks
# ---------------------
class GaussianTransformer(nn.Module):
    def __init__(self, feature_dim, time_emb_dim, num_heads=32,
                 num_mab_layers=4, num_isab_layers=2, num_sab_layers=4,
                 model_dim=1024, T=1000, dropout=0.1, num_inducing=32):
        """
        Args:
            feature_dim: Dimension of each input Gaussian (should be 7).
            time_emb_dim: Hidden dimension for the time embedding.
            num_heads: Number of attention heads.
            num_mab_layers: Number of MAB blocks.
            num_isab_layers: Number of ISAB blocks.
            num_sab_layers: Number of SAB blocks.
            model_dim: Internal model (embedding) dimension (e.g. 1024).
            T: Total number of timesteps (used for normalizing time).
            dropout: Dropout probability for attention blocks.
            num_inducing: Number of inducing points in ISAB blocks.
        """
        super(GaussianTransformer, self).__init__()
        self.T = T  # Total timesteps
        # Time embedding: maps normalized time to a vector of size feature_dim.
        self.time_embed = TimeSiren(time_emb_dim, feature_dim)
        # Project input (of size feature_dim) to model dimension.
        self.initial_projection = InitialProjection(in_dim=feature_dim, final_dim=model_dim)
        # Repeated MAB blocks.
        self.mab_blocks = nn.ModuleList(
            [MAB(model_dim, model_dim, model_dim, num_heads, ln=True, dropout=dropout) for _ in range(num_mab_layers)]
        )
        # Repeated ISAB blocks.
        self.isab_blocks = nn.ModuleList(
            [ISAB(model_dim, model_dim, num_heads, num_inducing=num_inducing, ln=True, dropout=dropout) for _ in range(num_isab_layers)]
        )
        # Repeated SAB blocks.
        self.sab_blocks = nn.ModuleList(
            [SAB(model_dim, model_dim, num_heads, ln=True, dropout=dropout) for _ in range(num_sab_layers)]
        )
        # Project back to the original feature dimension.
        self.output_projection = OutputProjection(in_dim=model_dim, out_dim=feature_dim)

    def forward(self, gaussians: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gaussians: Tensor of shape (B, 70, feature_dim) where feature_dim is typically 7.
            t: Tensor of shape (B,) or (B,1) representing the timestep.
        Returns:
            Predicted noise tensor of shape (B, 70, feature_dim)
        """
        # Normalize t to [0, 1] using T.
        t_norm = t.float() / self.T  
        t_emb = self.time_embed(t_norm)           # (B, feature_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, gaussians.size(1), -1)  # (B, 70, feature_dim)
        # Condition the input with the time embedding.
        x = gaussians + t_emb
        # Project to the model dimension.
        x = self.initial_projection(x)  # (B, 70, model_dim)
        # Apply repeated MAB blocks.
        for mab in self.mab_blocks:
            x = mab(x, x)
        # Apply repeated ISAB blocks.
        for isab in self.isab_blocks:
            x = isab(x)
        # Apply repeated SAB blocks.
        for sab in self.sab_blocks:
            x = sab(x)
        # Project back to the original feature dimension.
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
    feature_dim = 7    # Each Gaussian is represented by 7 features.
    time_emb_dim = 256  # Increased time embedding dimension for expressiveness.
    num_heads = 64
    num_mab_layers = 6
    num_isab_layers = 4
    num_sab_layers = 6
    model_dim = 1024   # Increased model (embedding) dimension.
    T = 1000          # Total timesteps.
    dropout = 0.1
    num_inducing = 32  # More inducing points in ISAB.

    model = GaussianTransformer(feature_dim=feature_dim, time_emb_dim=time_emb_dim, num_heads=num_heads,
                                num_mab_layers=num_mab_layers, num_isab_layers=num_isab_layers,
                                num_sab_layers=num_sab_layers, model_dim=model_dim, T=T, dropout=dropout,
                                num_inducing=num_inducing)
    gaussian_inputs = torch.randn(batch_size, num_gaussians, feature_dim)
    t = torch.randint(low=1, high=T+1, size=(batch_size,), dtype=torch.float)
    
    predicted_noise = model(gaussian_inputs, t)
    print("Predicted noise shape:", predicted_noise.shape)
    print("\nTotal number of trainable parameters:", count_parameters(model))