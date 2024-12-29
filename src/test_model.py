import torch
from models.models import AudioSealWM, AudioSealDetector
from models.SEANet import SEANetEncoderKeepDimension, SEANetDecoder


# Configuration
audio_length = 8000  # 0.5 seconds
batch_size = 2       # Batch size
nbits = 32           # Number of bits in the watermark message
latent_dim = 128     # Latent space dimensionality

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

# Initialize watermarking model (generator)
wm_model = AudioSealWM(
    encoder=encoder,
    decoder=decoder,
    msg_processor=None
).to(device)

# Initialize detector
detector = AudioSealDetector(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
    output_dim=latent_dim,
    nbits=nbits  # Set to 32 bits
).to(device)

# Generate random input audio and 32-bit message
audio = torch.randn(batch_size, 1, audio_length).to(device)  # Random audio input
message = torch.randint(0, 2, (batch_size, nbits)).to(device)  # Random 32-bit binary message

# Generate watermarked audio
with torch.no_grad():
    watermarked_audio = wm_model(audio, message=message)

# Detect watermark and decode message
with torch.no_grad():
    detection_score, decoded_message_logits = detector(watermarked_audio)

# Extract the bits from the logits
decoded_message = (decoded_message_logits > 0.5).int()  # Convert logits to binary bits

# Calculate number of correctly identified bits
correct_bits = (decoded_message == message).sum(dim=1)  # Sum correct bits per sample
total_bits = message.size(1)  # Total number of bits per sample

# Calculate percentage of correctly decoded bits
correct_bits_percentage = (correct_bits / total_bits) * 100

# Determine if watermark is detected
# Detection score is assumed to classify watermarked audio if score > 0.5
watermark_detected = (detection_score[:, 1, :] > 0.5).float().mean(dim=1)  # Average over time
watermark_detected_binary = (watermark_detected > 0.5).int()  # Binary classification

# Print the results
print("Number of Correct Bits Per Sample:", correct_bits.cpu().numpy())
print("Percentage of Correct Bits Per Sample:", correct_bits_percentage.cpu().numpy())
print("Watermark Detected (Yes/No, Binary):", watermark_detected_binary.cpu().numpy())

# Print the input and output for manual inspection
print("\nInput Audio for Detector:")
print(watermarked_audio)

print("\nDetection Score (Watermark Presence):")
print(detection_score)

print("\nOriginal Message:")
print(message)

print("\nDecoded Message:")
print(decoded_message)

# Check if all bits match for each sample
for i in range(batch_size):
    is_correct = correct_bits[i] == total_bits
    print(f"Sample {i + 1}: {'Correctly Decoded' if is_correct else 'Decoding Error'}")
