import torch
from models import AudioSealWM, AudioSealDetector
from SEANet import SEANetEncoderKeepDimension, SEANetDecoder

# Configuration
audio_length = 8000  # Reduced audio length to 0.5 seconds
batch_size = 2       # Reduced batch size
nbits = 16           # Number of bits in the watermark message
latent_dim = 64      # Reduced latent space dimensionality

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the SEANet encoder and decoder
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

# Initialize the generator (watermarking model)
wm_model = AudioSealWM(
    encoder=encoder,
    decoder=decoder,
    msg_processor=None
).to(device)

# Initialize the detector
detector = AudioSealDetector(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
    output_dim=latent_dim,
    nbits=nbits
).to(device)

# Generate random input audio and message
audio = torch.randn(batch_size, 1, audio_length).to(device)  # Random audio input
message = torch.randint(0, 2, (batch_size, nbits)).to(device)  # Random binary message

# Generate watermarked audio
with torch.no_grad():  # Disable gradients for testing
    watermarked_audio = wm_model(audio, message=message)

print("Input Audio Shape:", audio.shape)
print("Watermarked Audio Shape:", watermarked_audio.shape)
