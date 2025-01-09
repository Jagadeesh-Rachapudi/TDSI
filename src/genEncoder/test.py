import torch
from src.genEncoder.models import AudioSealWM, Detector
from src.allModels.SEANet import SEANetEncoderKeepDimension, SEANetDecoder

# Path to the pretrained model
best_model_path = r"D:\trainedModels\TDSI\Generator\BestGenEncoder.pth"

# Load the checkpoint
checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))

# Print keys in the checkpoint
print("Keys in checkpoint:")
for key in checkpoint.keys():
    print(f"- {key}")

# Initialize encoder and decoder
latent_dim = 128  # Adjust as per your configuration
encoder = SEANetEncoderKeepDimension(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
    output_dim=latent_dim,
)

decoder = SEANetDecoder(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
)

# Initialize generator and detector
generator = AudioSealWM(encoder=encoder, decoder=decoder)

detector_encoder = SEANetEncoderKeepDimension(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
    output_dim=latent_dim,  # Ensure output_dim is passed here
)
detector = Detector(encoder=detector_encoder)

# Load weights into generator and detector
generator.encoder.load_state_dict(checkpoint["encoder_state_dict"])
generator.decoder.load_state_dict(checkpoint["decoder_state_dict"])

# If detector.encoder has the same architecture as generator.encoder
detector.encoder.load_state_dict(checkpoint["encoder_state_dict"])

print("Weights loaded into Generator and Detector.")

# Print training hyperparameters if available
if "training_hyperparams" in checkpoint:
    print("Training Hyperparameters:", checkpoint["training_hyperparams"])
else:
    print("No training hyperparameters found in the checkpoint.")

# Generate a random synthetic audio
batch_size = 1
sample_rate = 16000  # Assuming audio sample rate is 16kHz
audio_length = sample_rate * 5  # 5 seconds of audio
synthetic_audio = torch.randn(batch_size, 1, audio_length)  # Generate random synthetic audio

# Generate random message for embedding
nbits = 32
message = torch.randint(0, 2, (batch_size, nbits)).float()

# Test the generator
generator.eval()
with torch.no_grad():
    watermarked_audio, original_ls, embedded_ls, valid_indices = generator(synthetic_audio, message)

# Compute MSE loss for the reconstructed audio
mse_loss_audio = torch.nn.functional.mse_loss(synthetic_audio, watermarked_audio)
print(f"MSE Loss for reconstructed audio: {mse_loss_audio.item():.6f}")
