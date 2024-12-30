import torch
from torch.optim import Adam
from pathlib import Path
from trainFolder.train import train
from models.models import AudioSealDetector, AudioSealWM
from models.SEANet import SEANetDecoder, SEANetEncoderKeepDimension
from utils.data_prcocessing import get_dataloader

# Configuration
num_epochs = 600
batch_size = 32
audio_length = 16000
learning_rate = 1e-4
nbits = 32
latent_dim = 128
loss_weights = {
    "l1": 0.1,
    "mspec": 2.0,
    "adv": 4.0,
    "loud": 10.0,
    "loc": 10.0,
    "dec": 1.0,
}

# Data paths
train_data_dir = Path(r"D:\TDSI\data\data\train").resolve()
validate_data_dir = Path(r"D:\TDSI\data\data\test").resolve()

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

    # Initialize generator (AudioSealWM)
    generator = AudioSealWM(
        encoder=encoder,
        decoder=decoder,
        msg_processor=None  # Custom message processor can be added if required
    ).to(device)

    # Initialize detector (AudioSealDetector)
    detector = AudioSealDetector(
        channels=1,
        dimension=latent_dim,
        n_filters=32,
        n_residual_layers=3,
        ratios=[8, 5, 4, 2],
        output_dim=latent_dim,
        nbits=nbits
    ).to(device)

    # Initialize optimizers
    optimizer_g = Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer_d = Adam(detector.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Define perceptual loss functions
    perceptual_loss_fns = {
        "l1": torch.nn.L1Loss(),
        "mspec": torch.nn.MSELoss(),  # Placeholder for Multi-Scale Mel-Spectrogram Loss
        "adv": torch.nn.MSELoss(),    # Placeholder for Adversarial Loss
        "loud": torch.nn.MSELoss(),   # Placeholder for Loudness Loss
    }

    # Define the localization and watermarking loss functions
    localization_loss_fn = torch.nn.BCELoss()
    wm_loss_fn = torch.nn.MSELoss()

    # Initialize DataLoaders
    try:
        train_loader = get_dataloader(
            data_dir=train_data_dir,
            batch_size=batch_size,
            sample_rate=audio_length,  # Pass total length for sampling
            window_size=1.0,          # 1-second window
            stride=0.5,               # 50% overlap
            shuffle=True,
            num_workers=4,
        )

        validate_loader = get_dataloader(
            data_dir=validate_data_dir,
            batch_size=batch_size,
            sample_rate=audio_length,
            window_size=1.0,
            stride=0.5,
            shuffle=False,
            num_workers=4,
        )
    except FileNotFoundError as e:
        print(f"Error initializing DataLoaders: {e}")
        exit(1)

    # Check dataset sizes
    if len(train_loader.dataset) == 0:
        print("Error: Train dataset is empty.")
        exit(1)

    if len(validate_loader.dataset) == 0:
        print("Error: Validation dataset is empty.")
        exit(1)

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(validate_loader.dataset)}")

    # Start training
    try:
        train(
            generator=generator,
            detector=detector,
            train_loader=train_loader,
            val_loader=validate_loader,
            perceptual_loss_fns=perceptual_loss_fns,
            localization_loss_fn=localization_loss_fn,
            wm_loss_fn=wm_loss_fn,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            num_epochs=num_epochs,
            lambda_1=loss_weights["l1"],
            lambda_mspec=loss_weights["mspec"],
            lambda_adv=loss_weights["adv"],
            lambda_loud=loss_weights["loud"],
            lambda_loc=loss_weights["loc"],
            lambda_dec=loss_weights["dec"],
            checkpoint_path="D:\\TDSI\\checkpoints",
            log_interval=10,
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        exit(1)
