import torch
from transformer_model import GaussianTransformer
from torchinfo import summary  # Import torchinfo summary

def count_parameters(model):
    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Trainable and non-trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params

def sprites_model_param_counts(device):
    # Sprites model configuration
    T = 200
    num_gaussians = 500
    feature_dim = 9
    time_emb_dim = 32
    num_blocks = 16
    num_heads = 64
    num_timestamps = T

    model = GaussianTransformer(
        input_dim=num_gaussians,
        time_emb_dim=time_emb_dim,
        feature_dim=feature_dim,
        num_timestamps=num_timestamps,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
    ).to(device)
    
    # Print detailed model summary using torchinfo
    print("Sprites Model Summary:")
    print(summary(model, input_size=(1, num_timestamps, num_gaussians), verbose=2))
    
    total, trainable, non_trainable = count_parameters(model)
    print("Sprites Model Parameter Count:")
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")
    print(f"Non-trainable parameters: {non_trainable}\n")

def mnist_model_param_counts(device):
    # MNIST model configuration
    T = 200
    num_gaussians = 70
    feature_dim = 7
    time_emb_dim = 512
    num_blocks = 32
    num_heads = 256
    num_timestamps = T

    model = GaussianTransformer(
        input_dim=num_gaussians,
        time_emb_dim=time_emb_dim,
        feature_dim=feature_dim,
        num_timestamps=num_timestamps,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
    ).to(device)
    
    # Print detailed model summary using torchinfo
    print("MNIST Model Summary:")
    print(summary(model, input_size=(1, num_timestamps, num_gaussians), verbose=2))
    
    total, trainable, non_trainable = count_parameters(model)
    print("MNIST Model Parameter Count:")
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")
    print(f"Non-trainable parameters: {non_trainable}\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sprites_model_param_counts(device)
    mnist_model_param_counts(device)

if __name__ == "__main__":
    main()
