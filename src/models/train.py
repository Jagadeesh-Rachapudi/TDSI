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
import models from src.models.AudioSealWM


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

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

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