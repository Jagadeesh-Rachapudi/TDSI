import torch
import numpy as np
from src.allModels.models import AudioSealWM, Detector
from src.allModels.SEANet import SEANetDecoder, SEANetEncoderKeepDimension
from src.utils.utility_functions import find_least_important_components

# Configuration
audio_length = 8000  # 0.5 seconds
batch_size = 1
latent_dim = 128
num_bits_to_embed = 32

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize encoder and decoder
encoder = SEANetEncoderKeepDimension(
    channels=1, dimension=latent_dim, n_filters=32, n_residual_layers=3, ratios=[8, 5, 4, 2], output_dim=latent_dim
).to(device)

decoder = SEANetDecoder(
    channels=1, dimension=latent_dim, n_filters=32, n_residual_layers=3, ratios=[8, 5, 4, 2]
).to(device)

# Initialize watermarking model and detector
wm_model = AudioSealWM(encoder=encoder, decoder=decoder).to(device)

# Load pre-trained model weights
checkpoint_path = r"D:\trainedModels\TDSI\Generator\BestGenerator.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
wm_model.load_state_dict(checkpoint["generator_state_dict"])
print("Pre-trained model loaded successfully.")

# Initialize detector with the same encoder
detector_encoder = SEANetEncoderKeepDimension(
    channels=1, dimension=latent_dim, n_filters=32, n_residual_layers=3, ratios=[8, 5, 4, 2], output_dim=latent_dim
).to(device)
detector_encoder.load_state_dict(checkpoint["encoder_state_dict"])
detector = Detector(encoder=detector_encoder, latent_dim=latent_dim, msg_size=num_bits_to_embed).to(device)

# Generate random input audio and message
audio = torch.randn(batch_size, 1, audio_length).to(device)
message = torch.randint(0, 2, (batch_size, num_bits_to_embed)).float().to(device)

# Capture latent spaces and positions
with torch.no_grad():
    # Latent space before embedding
    latent_space_before = wm_model.encoder(audio)
    
    # Find least important positions before embedding
    latent_space_before_np = latent_space_before.cpu().numpy().reshape(-1, latent_dim)
    positions_before_embedding, _ = find_least_important_components(latent_space_before_np, num_bits_to_embed)

    # Latent space after embedding
    latent_space_after = wm_model.embed_bits_in_latent_space(latent_space_before, message)

    # Find least important positions after embedding
    latent_space_after_np = latent_space_after.cpu().numpy().reshape(-1, latent_dim)
    positions_after_embedding, _ = find_least_important_components(latent_space_after_np, num_bits_to_embed)

    # Latent space after using detector's encoder
    latent_space_detector = detector.encoder(audio)

    # Find least important positions after using detector's encoder
    latent_space_detector_np = latent_space_detector.cpu().numpy().reshape(-1, latent_dim)
    positions_after_detector, _ = find_least_important_components(latent_space_detector_np, num_bits_to_embed)

# Function to calculate percentage difference
def calculate_percentage_difference(tensor1, tensor2):
    diff = torch.abs(tensor1 - tensor2)
    total_values = torch.numel(tensor1)
    percentage_difference = (torch.sum(diff) / total_values) * 100
    return percentage_difference.item()

# Calculate differences for latent spaces
diff_ls_before_after = calculate_percentage_difference(latent_space_before, latent_space_after)
diff_ls_before_detector = calculate_percentage_difference(latent_space_before, latent_space_detector)
diff_ls_after_detector = calculate_percentage_difference(latent_space_after, latent_space_detector)

# Calculate differences for positions
matching_positions_before_after = len(set(positions_before_embedding).intersection(positions_after_embedding))
matching_positions_before_detector = len(set(positions_before_embedding).intersection(positions_after_detector))
matching_positions_after_detector = len(set(positions_after_embedding).intersection(positions_after_detector))

# Print results
print("\nLatent Space Comparisons:")
print(f"Percentage Difference (Before vs After Embedding): {diff_ls_before_after:.2f}%")
print(f"Percentage Difference (Before vs Detector Encoder): {diff_ls_before_detector:.2f}%")
print(f"Percentage Difference (After vs Detector Encoder): {diff_ls_after_detector:.2f}%")

print("\nPosition Comparisons:")
print(f"Matching Positions (Before vs After Embedding): {matching_positions_before_after}/{num_bits_to_embed}")
print(f"Matching Positions (Before vs Detector Encoder): {matching_positions_before_detector}/{num_bits_to_embed}")
print(f"Matching Positions (After vs Detector Encoder): {matching_positions_after_detector}/{num_bits_to_embed}")
