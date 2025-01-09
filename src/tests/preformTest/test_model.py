import torch
import numpy as np
from src.decEncoder.models import AudioSealWM, Detector
from src.allModels.SEANet import SEANetDecoder, SEANetEncoderKeepDimension
from src.utils.utility_functions import find_least_important_components

# Configuration
audio_length = 8000
batch_size = 1
latent_dim = 128
num_bits_to_embed = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize generator encoder and decoder
generator_encoder = SEANetEncoderKeepDimension(
    channels=1, dimension=latent_dim, n_filters=32, n_residual_layers=3, ratios=[8, 5, 4, 2], output_dim=latent_dim
).to(device)

generator_decoder = SEANetDecoder(
    channels=1, dimension=latent_dim, n_filters=32, n_residual_layers=3, ratios=[8, 5, 4, 2]
).to(device)

# Initialize generator
generator = AudioSealWM(encoder=generator_encoder, decoder=generator_decoder).to(device)

# Load pre-trained generator weights
generator_checkpoint_path = r"D:\trainedModels\TDSI\Generator\BestGenEncoder.pth"
generator_checkpoint = torch.load(generator_checkpoint_path, map_location=device ,weights_only=True)

# Load encoder and decoder weights into the generator
generator.encoder.load_state_dict(generator_checkpoint["encoder_state_dict"])
generator.decoder.load_state_dict(generator_checkpoint["decoder_state_dict"])
print("Generator weights loaded successfully.")

# Initialize detector encoder
detector_encoder = SEANetEncoderKeepDimension(
    channels=1, dimension=latent_dim, n_filters=32, n_residual_layers=3, ratios=[8, 5, 4, 2], output_dim=latent_dim
).to(device)

# Load pre-trained detector encoder weights
detector_checkpoint_path = r"D:\trainedModels\TDSI\Detector\BestDecEncoder.pth"
detector_checkpoint = torch.load(detector_checkpoint_path, map_location=device ,weights_only=True)

# Load encoder weights into the detector
detector_encoder.load_state_dict(detector_checkpoint["detector_state_dict"])
print("Detector's encoder weights loaded successfully.")

# Initialize detector
detector = Detector(encoder=detector_encoder).to(device)

# Generate random input audio and message
audio = torch.randn(batch_size, 1, audio_length).to(device)
message = torch.randint(0, 2, (batch_size, num_bits_to_embed)).float().to(device)

# Forward through generator to get watermarked audio
with torch.no_grad():
    watermarked_audio, embedded_ls, original_ls = generator(audio, message)

# Perform PCA to find least important components
latent_space_np = generator.encoder(audio).detach().cpu().numpy().reshape(-1, latent_dim)
bit_positions, _ = find_least_important_components(latent_space_np, num_bits_to_embed)
bit_positions_tensor = torch.tensor(bit_positions, dtype=torch.long).to(device)

# Forward through detector to extract the message probabilities
with torch.no_grad():
    # Detector's encoder processes the watermarked audio
    probable_embedded_ls = detector_encoder(watermarked_audio)
    
    # Extract probable embedded bits for the selected positions
    extracted_bits = torch.sigmoid(probable_embedded_ls[:, bit_positions_tensor, :]).mean(dim=-1)  # Shape: (batch_size, num_bits_to_embed)

# Reshape the original message to match the extracted bits' shape for comparison
reshaped_message = message.view_as(extracted_bits)

# Compute bitwise absolute difference (error per bit)
bitwise_diff = torch.abs(extracted_bits - reshaped_message)

# Compute Detector Bit Loss as the mean of the bitwise differences
detector_bit_loss = bitwise_diff.mean().item()

# Compare probabilities with the original message
correct_bits = (extracted_bits.round() == reshaped_message).sum().item()  # Count correct bits
accuracy = (correct_bits / num_bits_to_embed) * 100

# Print results
print("\nOriginal Message (Ground Truth Bits):")
print(message)

print("\nExtracted Bit Probabilities (0-1 scale):")
print(extracted_bits)  # Displays probabilities

print("\nBitwise Absolute Difference (Error per Bit):")
print(bitwise_diff)  # Displays absolute differences

print(f"\nDetector Bit Loss: {detector_bit_loss:.4f}")  # Displays the mean bitwise error as Detector Bit Loss
print(f"\nCorrect Bits: {correct_bits}/{num_bits_to_embed} ({accuracy:.2f}%)")
