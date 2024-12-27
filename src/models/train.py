import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import gc
import os
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
from torch.optim import Adam

from models.models import AudioSealWM, Encoder, Decoder
from models.train_loop import train_model
from utils.data_prcocessing import get_dataloader
from utils.utils import clear_gpu_memory, save_loss_data, plot_losses

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

# Initialize model
input_dim = 1
hidden_dim = 128
kernel_size = 7
num_layers = 4

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

train_loader = get_dataloader(
    data_dir=train_data_dir,  # Use absolute path
    batch_size=2,
    sample_rate=16000,
    window_size=2.0,
    stride=1.0,
    shuffle=True,
    num_workers=2
)

validate_loader = get_dataloader(
    data_dir=validate_data_dir,  # Use absolute path
    batch_size=2,
    sample_rate=16000,
    window_size=2.0,
    stride=1.0,
    shuffle=False,
    num_workers=2
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Call the training function
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