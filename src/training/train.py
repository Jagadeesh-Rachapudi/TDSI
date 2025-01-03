import torchaudio
torchaudio.set_audio_backend("ffmpeg")

import sys
# print("\n".join(sys.path))

import torch
from torch.optim import Adam
from pathlib import Path
from src.training.trainLoop import train
from src.allModels.models import AudioSealDetector, AudioSealWM, MsgProcessor
from src.allModels.SEANet import SEANetDecoder, SEANetEncoderKeepDimension
from src.utils.data_prcocessing import get_dataloader
from src.losses.loss import compute_detection_loss, compute_decoding_loss, compute_perceptual_loss
from src.utils.utility_functions import masker,update_csv
from src.utils.utility_functions import initialize_csv
from src.tests.testLoop import trainTest

import torch

def print_audio_shapes(data_loader, data_name):
    """
    Print the shape of each audio chunk in the given data loader and log unexpected sizes.

    Args:
        data_loader: DataLoader object.
        data_name: Name of the dataset (for logging purposes).
    """
    expected_shape = torch.Size([1, 8000])  # Hardcoded expected shape
    print(f"Printing audio shapes for {data_name} dataset with expected shape {expected_shape}:")

    total_audio = 0  # Counter for the total number of audio files
    unexpected_size_count = 0  # Counter for unexpected sizes

    for batch_idx, (audios, file_paths) in enumerate(data_loader):
        for audio_idx, (audio, file_path) in enumerate(zip(audios, file_paths)):
            # Check if audio is None
            if audio is None:
                print(f"Audio {total_audio} is None. Skipping...")
                unexpected_size_count += 1
                total_audio += 1
                continue

            audio_shape = audio.shape
            print(f"Audio index {total_audio} shape: {audio_shape}")

            # Check for unexpected shape
            if audio_shape != expected_shape:
                print(f"Unexpected shape found for audio {total_audio} at path: {file_path}")
                unexpected_size_count += 1

            # Increment the counter
            total_audio += 1

    print(f"Total audios checked: {total_audio}")
    print(f"Total unexpected size audios: {unexpected_size_count}")





# Configuration
num_epochs = 100
batch_size = 4
audio_length = 4000
learning_rate = 1e-4
nbits = 32
latent_dim = 128

# Data paths
train_data_dir = Path(r"/content/TDSI/data/test").resolve()
# train_csv = Path(r"../../data/train/train.csv").resolve()
test_data_dir = Path(r"/content/TDSI/data/test").resolve()
# test_csv = Path(r"../../data/test/test.csv").resolve()
validate_data_dir = Path(r"/content/TDSI/data/test").resolve()
# validate_csv = Path(r"../../data/validate/validate.csv").resolve()

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
        hidden_size=128,  # Must match the encoder's latent dimension
    ).to(device)

    # Initialize generator (AudioSealWM)
    generator = AudioSealWM(
        encoder=encoder,
        decoder=decoder,
        msg_processor=msg_processor,  # Custom message processor can be added if required
    ).to(device)

    # Initialize detector (AudioSealDetector)
    detector = AudioSealDetector(
        channels=1,
        dimension=latent_dim,
        n_filters=32,
        n_residual_layers=3,
        ratios=[8, 5, 4, 2],
        output_dim=latent_dim,
        nbits=nbits,
    ).to(device)

    # Initialize optimizers
    optimizer_g = Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer_d = Adam(detector.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Initialize DataLoaders
    try:
        train_loader = get_dataloader(
            data_dir=train_data_dir,
            # csv_file=train_csv,
            batch_size=batch_size,
            sample_rate=audio_length,
            shuffle=True,
            num_workers=4,
        )

        test_loader = get_dataloader(
            data_dir=test_data_dir,
            # csv_file=test_csv,
            batch_size=batch_size,
            sample_rate=audio_length,
            shuffle=False,
            num_workers=4,
        )

        validate_loader = get_dataloader(
            data_dir=validate_data_dir,
            # csv_file=validate_csv,
            batch_size=batch_size,
            sample_rate=audio_length,
            shuffle=False,
            num_workers=4,
        )

        print_audio_shapes(test_loader, data_name="Test")  

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
    # try:
        # train(
        #     generator=generator,
        #     detector=detector,
        #     train_loader=train_loader,
        #     val_loader=validate_loader,
        #     optimizer_g=optimizer_g,
        #     optimizer_d=optimizer_d,
        #     device=device,
        #     num_epochs=num_epochs,
        #     compute_detection_loss=compute_detection_loss,
        #     compute_decoding_loss=compute_decoding_loss,
        #     compute_perceptual_loss=compute_perceptual_loss,
        #     checkpoint_path="./checkpoints",
        #     log_path="./logs/losses.csv",
        #     masker=masker,
        #     update_csv=update_csv,
        #     initialize_csv=initialize_csv
        # )
        # trainTest(
        #     generator=generator,
        #     detector=detector,
        #     train_loader=train_loader,
        #     val_loader=validate_loader,
        #     optimizer_g=optimizer_g,
        #     optimizer_d=optimizer_d,
        #     device=device,
        #     num_epochs=num_epochs,
        #     compute_detection_loss=compute_detection_loss,
        #     compute_decoding_loss=compute_decoding_loss,
        #     compute_perceptual_loss=compute_perceptual_loss,
        #     checkpoint_path="./checkpoints",
        #     log_path="./logs/losses.csv",
        #     masker=masker,
        #     update_csv=update_csv,
        #     initialize_csv=initialize_csv
        # )
    # except Exception as e:
        # print(f"An error occurred during training: {e}")
        # exit(1)
