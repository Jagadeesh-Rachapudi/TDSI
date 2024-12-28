import sys
from pathlib import Path
import gc
import os
import random
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from torch.optim import Adam
from src.models.models import AudioSealWM, Encoder, Decoder
from src.train.train_loop import train_model
from src.utils.data_prcocessing import get_dataloader
from src.utils.utils import clear_gpu_memory, save_loss_data, plot_losses

sys.path.append(str(Path(__file__).resolve().parents[2]))


# Define the loss function
def loss_function(reconstructed, original):
    l1_loss = nn.L1Loss()(reconstructed, original)
    mse_loss = nn.MSELoss()(reconstructed, original)
    return l1_loss + mse_loss


# Hyperparameters
num_epochs = 50
learning_rate = 0.001
nbits = 32  # Number of bits for watermarking
patience = 5  # Early stopping patience
batch_size = 2
sample_rate = 16000
window_size = 2.0  # seconds
stride = 1.0  # seconds
num_workers = 2

# Model parameters
input_dim = 1
hidden_dim = 128
kernel_size = 7
num_layers = 4

# Initialize model
encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers)
decoder = Decoder(hidden_dim=hidden_dim, output_dim=input_dim, kernel_size=kernel_size, num_layers=num_layers)
model = AudioSealWM(encoder, decoder, nbits=nbits, hidden_dim=hidden_dim)

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Initialize DataLoaders
train_data_dir = Path("D:\\myCode\\data\\train").resolve()
validate_data_dir = Path("D:\\myCode\\data\\validate").resolve()

print(f"Train data directory: {train_data_dir}")
print(f"Validate data directory: {validate_data_dir}")

try:
    train_loader = get_dataloader(
        data_dir=train_data_dir,
        batch_size=batch_size,
        sample_rate=sample_rate,
        window_size=window_size,
        stride=stride,
        shuffle=True,
        num_workers=num_workers
    )

    validate_loader = get_dataloader(
        data_dir=validate_data_dir,
        batch_size=batch_size,
        sample_rate=sample_rate,
        window_size=window_size,
        stride=stride,
        shuffle=False,
        num_workers=num_workers
    )
except FileNotFoundError as e:
    print(f"Error initializing DataLoaders: {e}")
    sys.exit(1)

# Check dataset size
if len(train_loader.dataset) == 0:
    print("Warning: Train dataset is empty.")
    sys.exit(1)
else:
    print(f"Train dataset size: {len(train_loader.dataset)}")

if len(validate_loader.dataset) == 0:
    print("Warning: Validation dataset is empty.")
    sys.exit(1)
else:
    print(f"Validation dataset size: {len(validate_loader.dataset)}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear GPU memory before starting training
clear_gpu_memory()

# Call the training function
try:
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        loss_function=loss_function,
        num_epochs=num_epochs,
        validation_loader=validate_loader,
        patience=patience,
        nbits=nbits
    )
except Exception as e:
    print(f"An error occurred during training: {e}")
    clear_gpu_memory()
    sys.exit(1)
