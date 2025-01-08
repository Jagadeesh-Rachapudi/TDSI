import torch
import torchaudio
from src.makingDecEncoder.test_modelArchitecure import AudioSealWM , Detector
from src.allModels.SEANet import SEANetDecoder, SEANetEncoderKeepDimension
from src.utils.utility_functions import find_least_important_components

# Configuration
audio_length = 16000  # Length of the audio in samples (1 second at 16kHz)
latent_dim = 128
nbits = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate synthetic audio
waveform = torch.randn(1, 1, audio_length).to(device)  # Shape: (batch_size=1, channels=1, time_steps)
waveform = waveform / waveform.abs().max()  # Normalize to the range [-1, 1]

# Initialize Generator
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

generator = AudioSealWM(encoder=encoder, decoder=decoder).to(device)

# Initialize Detector Encoder
# Initialize Detector using the Detector class from models.py
detector = Detector(
    encoder=SEANetEncoderKeepDimension(
        channels=1,
        dimension=latent_dim,
        n_filters=32,
        n_residual_layers=3,
        ratios=[8, 5, 4, 2],
        output_dim=latent_dim,
    ).to(device)
).to(device)
# Generate random 32-bit message
message = torch.randint(0, 2, (1, nbits)).float().to(device)

# Forward pass through Generator
with torch.no_grad():
    watermarked_audio, embedded_ls, original_ls = generator(waveform, message)

    # Compute MSE loss for audio reconstruction
    mse_loss_audio = torch.nn.functional.mse_loss(waveform, watermarked_audio)

# Forward pass through Detector
with torch.no_grad():
    probable_embedded_ls = detector.encoder(watermarked_audio)

    # Compute the least important positions for embedding
    original_ls_np = original_ls.cpu().numpy().reshape(-1, latent_dim)
    bit_positions, _ = find_least_important_components(original_ls_np, nbits)
    bit_positions_tensor = torch.tensor(bit_positions, device=device)

    # Compute losses
    embedded_loss = torch.nn.functional.mse_loss(
        embedded_ls[:, bit_positions_tensor, :],
        probable_embedded_ls[:, bit_positions_tensor, :]
    )
    latent_loss = torch.nn.functional.mse_loss(embedded_ls, probable_embedded_ls)

    # Assign weights to embedded bit positions
    weights = torch.ones_like(original_ls)
    weights[:, bit_positions_tensor, :] *= 10  # Assign higher weight to embedded bits
    weighted_loss = torch.mean(weights * (embedded_ls - probable_embedded_ls).pow(2))

    # Print all losses
    print(f"MSE Loss for reconstructed audio: {mse_loss_audio.item():.6f}")
    print(f"Weighted Loss: {weighted_loss.item():.6f}")
    print(f"Embedded Loss: {embedded_loss.item():.6f}")
    print(f"Latent Loss: {latent_loss.item():.6f}")
