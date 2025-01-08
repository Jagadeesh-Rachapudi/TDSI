import torch
import numpy as np
from src.allModels.models import AudioSealWM
from src.allModels.SEANet import SEANetDecoder, SEANetEncoderKeepDimension

# Configuration
audio_length = 8000  # 0.5 seconds
batch_size = 1       # Batch size
latent_dim = 128     # Latent space dimensionality
num_bits_to_embed = 33  # Number of bits to embed

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize SEANet encoder and decoder
encoder = SEANetEncoderKeepDimension(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
    output_dim=latent_dim,
).to(device)

decoder = SEANetDecoder(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
).to(device)

# Initialize watermarking model (generator for reconstruction)
wm_model = AudioSealWM(
    encoder=encoder,
    decoder=decoder,
).to(device)

# Generate random input audio
audio = torch.randn(batch_size, 1, audio_length).to(device)  # Random audio input

# Generate random 33-bit binary message
message = torch.randint(0, 2, (num_bits_to_embed,)).float().to(device)  # Random binary message

# Perform reconstruction with embedded message
with torch.no_grad():
    reconstructed_audio = wm_model(audio, message)

# Compute Mean Squared Error (MSE) Loss
mse_loss = torch.mean((audio - reconstructed_audio) ** 2)

# Display results
print("\nOriginal Audio:")
print(audio.cpu().numpy())
print("\nReconstructed Audio:")
print(reconstructed_audio.cpu().numpy())
print("\nShape of Reconstructed Audio:")
print(reconstructed_audio.shape)
print("\nRandom 33-bit Message Embedded:")
print(message.cpu().numpy())
print("\nMean Squared Error (MSE) Loss:")
print(mse_loss.item())
