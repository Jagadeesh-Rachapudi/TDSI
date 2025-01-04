import torchaudio
torchaudio.set_audio_backend("ffmpeg")

import torch
from torch.optim import Adam
from pathlib import Path
from src.allModels.models import AudioSealDetector, AudioSealWM, MsgProcessor
from src.allModels.SEANet import SEANetDecoder, SEANetEncoderKeepDimension
from src.utils.data_prcocessing import get_dataloader
from src.losses.loss import compute_perceptual_loss
from src.utils.utility_functions import update_csv, initialize_csv
from src.tests.testLoop import train
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

# Configuration
num_epochs = 100
batch_size = 1
audio_length = 8000
learning_rate = 5e-3  # Updated learning rate
nbits = 32
latent_dim = 128

# Data paths
train_data_dir = Path("/content/TDSI/data/train").resolve()
test_data_dir = Path("/content/TDSI/data/validate").resolve()
validate_data_dir = Path("/content/TDSI/data/validate").resolve()

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

    msg_processor = MsgProcessor(
        nbits=32,  # Number of bits for the watermark message
        hidden_size=latent_dim,
    ).to(device)

    # Initialize generator and detector
    generator = AudioSealWM(
        encoder=encoder,
        decoder=decoder,
        msg_processor=msg_processor,
    ).to(device)

    detector = AudioSealDetector(
        channels=1,
        dimension=latent_dim,
        n_filters=32,
        n_residual_layers=3,
        ratios=[8, 5, 4, 2],
        output_dim=latent_dim,
        nbits=nbits,
    ).to(device)

    # Optimizers and scheduler
    optimizer_g = Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer_d = Adam(detector.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5, verbose=True)

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
            detector=detector,
            train_loader=train_loader,
            val_loader=validate_loader,
            lr_g=learning_rate,
            lr_d=learning_rate,
            device=device,
            num_epochs=num_epochs,
            compute_perceptual_loss=compute_perceptual_loss,
            checkpoint_path="./checkpoints",
            log_path="./logs/losses.csv",
            update_csv=update_csv,
            initialize_csv=initialize_csv,
            temperature=1.3,
            scheduler=scheduler,
        )
    except Exception as e:
        print(f"Training error: {e}")
        exit(1)
