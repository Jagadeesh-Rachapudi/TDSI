import torch
import numpy as np
from src.allModels.models import AudioSealWM, Detector
from src.allModels.SEANet import SEANetDecoder, SEANetEncoderKeepDimension
from src.utils.utility_functions import find_least_important_components

# Configuration
audio_length = 8000
batch_size = 1
latent_dim = 128
num_bits_to_embed = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize encoder and decoder
encoder = SEANetEncoderKeepDimension(
    channels=1, dimension=latent_dim, n_filters=32, n_residual_layers=3, ratios=[8, 5, 4, 2], output_dim=latent_dim
).to(device)

decoder = SEANetDecoder(
    channels=1, dimension=latent_dim, n_filters=32, n_residual_layers=3, ratios=[8, 5, 4, 2]
).to(device)

# Initialize generator
generator = AudioSealWM(encoder=encoder, decoder=decoder).to(device)

# Load pre-trained generator weights
checkpoint_path = r"C:\Users\HP\TDSI\trainedModels\TDSI\Generator\BestModel.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load encoder and decoder weights into the generator
generator.encoder.load_state_dict(checkpoint["encoder_state_dict"])
generator.decoder.load_state_dict(checkpoint["decoder_state_dict"])
print("Generator weights loaded successfully.")

# Initialize detector with the same encoder
detector_encoder = SEANetEncoderKeepDimension(
    channels=1, dimension=latent_dim, n_filters=32, n_residual_layers=3, ratios=[8, 5, 4, 2], output_dim=latent_dim
).to(device)
detector_encoder.load_state_dict(checkpoint["encoder_state_dict"])
print("Detector's encoder weights loaded successfully.")

# Initialize detector
detector = Detector(encoder=detector_encoder, latent_dim=latent_dim, msg_size=num_bits_to_embed).to(device)

# Generate random input audio and message
audio = torch.randn(batch_size, 1, audio_length).to(device)
message = torch.randint(0, 2, (batch_size, num_bits_to_embed)).float().to(device)

# Forward through generator to get watermarked audio
watermarked_audio = generator(audio, message)

# Perform PCA to find least important components
latent_space_np = generator.encoder(audio).detach().cpu().numpy().reshape(-1, latent_dim)
bit_positions, _ = find_least_important_components(latent_space_np, num_bits_to_embed)
bit_positions = torch.tensor(bit_positions, dtype=torch.long).to(device)

# Forward through detector to extract the message
extracted_message = detector(watermarked_audio, bit_positions)

# Compare original and extracted message
correct_bits = (extracted_message == message).sum().item()
accuracy = (correct_bits / num_bits_to_embed) * 100

# Print results
print(f"Original Message: {message}")
print(f"Extracted Message: {extracted_message}")
print(f"Correct Bits: {correct_bits}/{num_bits_to_embed} ({accuracy:.2f}%)")
