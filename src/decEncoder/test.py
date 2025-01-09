import torch
import torchaudio
from src.decEncoder.models import AudioSealWM, Detector
from src.allModels.SEANet import SEANetDecoder, SEANetEncoderKeepDimension
from src.utils.utility_functions import find_least_important_components

# Configuration
latent_dim = 128
nbits = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model_path = r"/content/BestGenEncoder.pth"  # Path to the pretrained model

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
detector_encoder = SEANetEncoderKeepDimension(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
    output_dim=latent_dim,
).to(device)

detector = Detector(encoder=detector_encoder).to(device)

# Load Pretrained Model Weights
checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
generator.encoder.load_state_dict(checkpoint["encoder_state_dict"])
generator.decoder.load_state_dict(checkpoint["decoder_state_dict"])
detector.encoder.load_state_dict(checkpoint["encoder_state_dict"])

print("Loaded pretrained weights from BestGenEncoder.pth")

# Create a dummy audio sample (16 kHz, 1 second)
sample_rate = 16000
waveform = torch.randn(1, 1, sample_rate).to(device)  # 1-second audio with random noise

# Generate random 32-bit message
message = torch.randint(0, 2, (1, nbits)).float().to(device)

# Forward pass through Generator
with torch.no_grad():
    watermarked_audio, embedded_ls, original_ls = generator(waveform, message)

    # Ensure the output size matches the input size
    if watermarked_audio.shape[-1] != waveform.shape[-1]:
        watermarked_audio = watermarked_audio[..., :waveform.shape[-1]]

    # Compute MSE loss for audio reconstruction
    mse_loss_audio = torch.nn.functional.mse_loss(waveform, watermarked_audio)

# Forward pass through Detector
with torch.no_grad():
    probable_embedded_ls = detector.encoder(watermarked_audio)

    # Compute the least important positions for embedding
    original_ls_np = original_ls.cpu().numpy().reshape(-1, latent_dim)
    bit_positions, _ = find_least_important_components(original_ls_np, nbits)
    bit_positions_tensor = torch.tensor(bit_positions, device=device)

# Extract probable embedded bits from the detector's output at the identified bit positions
probable_embedded_bits = probable_embedded_ls[:, bit_positions_tensor, 0]  # Extract specific bit positions and remove extra dimensions
extracted_bits_sigmoid = torch.sigmoid(probable_embedded_bits)  # Apply sigmoid to map values to [0, 1]

# Compute bit-wise loss
bit_loss = torch.abs(extracted_bits_sigmoid - message)  # Now shapes match: (batch_size, nbits)
total_bit_loss = bit_loss.mean()  # Mean loss across all bits and batch

# Print extracted bits and original message
print("\nExtracted Probable Embedded Bits at Bit Positions (Sigmoid):")
print(extracted_bits_sigmoid)

print("\nOriginal Message:")
print(message)

# Print all losses
print(f"\nMSE Loss for reconstructed audio: {mse_loss_audio.item():.6f}")
print(f"Total Bit Loss: {total_bit_loss.item():.6f}")
