import torch
from models import AudioSealWM, AudioSealDetector
from SEANet import SEANetEncoderKeepDimension, SEANetDecoder

# Configuration
audio_length = 16000  # Length of the audio signal
batch_size = 4        # Batch size for testing
nbits = 16            # Number of bits in the watermark message
latent_dim = 128      # Latent space dimensionality

# Initialize the SEANet encoder and decoder
encoder = SEANetEncoderKeepDimension(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
    output_dim=latent_dim,  # Match latent dimensions
)

decoder = SEANetDecoder(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
)

# Initialize the generator (watermarking model)
wm_model = AudioSealWM(
    encoder=encoder,
    decoder=decoder,
    msg_processor=None  # The MsgProcessor will be included automatically in the generator
)

# Initialize the detector
detector = AudioSealDetector(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
    output_dim=latent_dim,
    nbits=nbits
)

# Generate random input audio and message
audio = torch.randn(batch_size, 1, audio_length)  # Random audio input
message = torch.randint(0, 2, (batch_size, nbits))  # Random binary message

# Generate watermarked audio
watermarked_audio = wm_model(audio, message=message)

# # Detect watermark and decode message
# detection_score, decoded_message = detector(watermarked_audio)

# Output results
# print("Input Audio Shape:", audio.shape)
# print("Watermarked Audio Shape:", watermarked_audio.shape)
# print("Detection Score (Watermark Presence):", detection_score)
# print("Decoded Message:", decoded_message)

print("hello world")
