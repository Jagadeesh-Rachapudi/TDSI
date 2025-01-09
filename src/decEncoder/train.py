import torchaudio
torchaudio.set_audio_backend("ffmpeg")

import torch
from torch.optim import Adam
from pathlib import Path
from src.decEncoder.models import AudioSealWM, Detector
from src.allModels.SEANet import SEANetDecoder, SEANetEncoderKeepDimension
from src.utils.data_prcocessing import get_dataloader
from src.losses.loss import compute_perceptual_loss
from src.decEncoder.trainee import train

# Configuration
num_epochs = 100
batch_size = 1
audio_length = 8000
learning_rate = 5e-3
nbits = 32
latent_dim = 128

# Paths
train_data_dir = Path("/content/TDSI/data/train").resolve()
test_data_dir = Path("/content/TDSI/data/validate").resolve()
validate_data_dir = Path("/content/TDSI/data/validate").resolve()
best_genEncoder = Path("/content/BestGenEncoder.pth").resolve()
best_decEncoder = Path("/content/BestDecEncoder.pth").resolve()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
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

    # Initialize generator and detector
    generator = AudioSealWM(
        encoder=encoder,
        decoder=decoder,
    ).to(device)

    detector_encoder = SEANetEncoderKeepDimension(
        channels=1,
        dimension=latent_dim,
        n_filters=32,
        n_residual_layers=3,
        ratios=[8, 5, 4, 2],
        output_dim=latent_dim,
    ).to(device)
    detector = Detector(
        encoder=detector_encoder,
    ).to(device)

    # Load pre-trained weights for generator and detector
    if best_genEncoder.exists():
        print(f"Loading generator weights from {best_genEncoder}")
        gen_checkpoint = torch.load(best_genEncoder, map_location=device)
        generator.encoder.load_state_dict(gen_checkpoint["encoder_state_dict"])
        generator.decoder.load_state_dict(gen_checkpoint["decoder_state_dict"])
    else:
        print(f"Warning: Pre-trained generator weights not found at {best_genEncoder}")

    if best_decEncoder.exists():
        print(f"Loading detector weights from {best_decEncoder}")
        dec_checkpoint = torch.load(best_decEncoder, map_location=device)
        detector.encoder.load_state_dict(dec_checkpoint["detector_state_dict"])
    else:
        print(f"Warning: Pre-trained detector weights not found at {best_decEncoder}")

    # Optimizers
    optimizer_g = Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer_d = Adam(detector.parameters(), lr=learning_rate, weight_decay=1e-4)

    # DataLoaders
    try:
        train_loader = get_dataloader(
            data_dir=train_data_dir,
            batch_size=batch_size,
            sample_rate=audio_length,
            shuffle=True,
            num_workers=0,
        )

        test_loader = get_dataloader(
            data_dir=test_data_dir,
            batch_size=batch_size,
            sample_rate=audio_length,
            shuffle=False,
            num_workers=0,
        )

        validate_loader = get_dataloader(
            data_dir=validate_data_dir,
            batch_size=batch_size,
            sample_rate=audio_length,
            shuffle=False,
            num_workers=0,
        )
    except FileNotFoundError as e:
        print(f"Error initializing DataLoaders: {e}")
        exit(1)

    # Dataset size check
    if len(train_loader.dataset) == 0 or len(validate_loader.dataset) == 0:
        print("Error: Empty datasets.")
        exit(1)

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(validate_loader.dataset)}")

    # Training process
    try:
        train(
            generator=generator,
            detector_encoder=detector.encoder,
            train_loader=train_loader,
            val_loader=validate_loader,
            lr_g=learning_rate,
            device=device,
            num_epochs=num_epochs,
            checkpoint_path="./checkpoints",
            log_path="./logs/losses.csv",
        )
    except Exception as e:
        print(f"Training error: {e}")
        exit(1)
