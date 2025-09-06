import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int, target_dim: int) -> None:
        super(TimeSiren, self).__init__()
        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)
        self.lin3 = nn.Linear(emb_dim, emb_dim)
        self.lin4 = nn.Linear(emb_dim, emb_dim)
        self.lin_out = nn.Linear(emb_dim, target_dim)  # Linear layer to match feature dim
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.activation(self.lin2(x))
        x = self.activation(self.lin3(x))
        x = self.lin4(x)
        x = self.lin_out(x)  # Convert to required feature dimension
        return x

class GaussianSplatProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, time_emb_dim, feature_dim, pooling='max'):
        super(GaussianSplatProcessor, self).__init__()

        self.time_emb_dim = time_emb_dim
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.time_embed = TimeSiren(self.time_emb_dim, self.feature_dim)

        self.local_mlp = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(16, self.hidden_dim, kernel_size=1)
        )

        if pooling == 'mean':
            self.pooling = lambda x: x.mean(dim=2)
        elif pooling == 'max':
            self.pooling = lambda x: x.max(dim=2)[0]
        else:
            raise ValueError("Unsupported pooling method. Use 'mean' or 'max'.")

        self.output_mlp = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, output_dim, kernel_size=1)
        )

    def forward(self, gaussians, t):
        t = t.float()
        # print(t.shape)
        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, gaussians.size(1), -1)
        # print(t_emb.shape)
        gaussians = gaussians + t_emb
        # print(gaussians.shape)
        
        local_features = self.local_mlp(gaussians)
        # print(local_features.shape)
        
        global_features = self.pooling(local_features)
        # print(global_features.shape)
        
        global_features = global_features.unsqueeze(2).repeat(1, 1, local_features.size(2))
        # print(global_features.shape)
        
        combined_features = torch.cat([local_features, global_features], dim=1)
        # combined_features = combined_features + t_emb  # add time embedding
        # print(combined_features.shape)
        
        output = self.output_mlp(combined_features)
        # print(output.shape)
        
        return output

if __name__ == "__main__":
    batch_size = 32
    num_splats = 70
    feature_dim = 7
    hidden_dim = 8
    output_dim = 70
    time_emb_dim = 8
    t = torch.randn(batch_size)

    model = GaussianSplatProcessor(input_dim=num_splats, hidden_dim=hidden_dim, output_dim=output_dim, time_emb_dim=time_emb_dim, feature_dim=feature_dim)
    gaussian_inputs = torch.randn(batch_size, num_splats, feature_dim)
    predicted_noise = model(gaussian_inputs, t)
    print("Predcited noise shape: ", predicted_noise.shape)