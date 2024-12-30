import torch
from torch.optim import Adam
from pathlib import Path
from src.trainFolder.trainLoop import train
from models.models import AudioSealDetector, AudioSealWM , MsgProcessor
from models.SEANet import SEANetDecoder, SEANetEncoderKeepDimension
from utils.data_prcocessing import get_dataloader
from losses import compute_detection_loss,compute_decoding_loss ,compute_perceptual_loss

# Configuration
num_epochs = 50
batch_size = 1
audio_length = 4000
learning_rate = 1e-4
nbits = 32
latent_dim = 128


# Data paths
train_data_dir = Path(r"/content/semi-Voxpopuli/data/train").resolve()
validate_data_dir = Path(r"/content/semi-Voxpopuli/data/train").resolve()
# validate_data_dir = Path(r"D:\TDSI\data\data\test").resolve()

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
    nbits=32,       # Number of bits for the watermark message
    hidden_size=128 # Must match the encoder's latent dimension
    ).to(device)

    # Initialize generator (AudioSealWM)
    generator = AudioSealWM(
        encoder=encoder,
        decoder=decoder,
        msg_processor=msg_processor  # Custom message processor can be added if required
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

    # Initialize DataLoaders
    try:
        train_loader = get_dataloader(
            data_dir=train_data_dir,
            batch_size=batch_size,
            sample_rate=audio_length,  # Pass total length for sampling
            shuffle=True,
            num_workers=4,
        )

        validate_loader = get_dataloader(
            data_dir=validate_data_dir,
            batch_size=batch_size,
            sample_rate=audio_length,
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
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            num_epochs=num_epochs,
            compute_detection_loss=compute_detection_loss,
            compute_decoding_loss=compute_decoding_loss,
            compute_perceptual_loss=compute_perceptual_loss,
            checkpoint_path=".\checkpoints",
            log_interval=10,
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        exit(1)
