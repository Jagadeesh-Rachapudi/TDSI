import gc
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import csv
from pathlib import Path
import torch
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA


def get_white_noise(chs: int = 1, num_frames: int = 1):
    wav = torch.randn(chs, num_frames)
    return wav



def find_least_important_components(latent_space, num_bits):
    """
    Perform PCA on the latent space and find the components with the least variance.
    
    Args:
        latent_space (np.ndarray): The latent space matrix (samples x features).
        num_bits (int): Number of least important components to find.
        
    Returns:
        least_important_indices (list): Indices of the components with the least variance.
    """
    n_samples, n_features = latent_space.shape
    max_components = min(n_samples, n_features)  # Max allowable PCA components
    
    # Perform PCA with valid n_components
    pca = PCA(n_components=max_components)
    pca.fit(latent_space)
    
    # Get explained variance and their indices
    explained_variance = pca.explained_variance_ratio_
    sorted_indices = np.argsort(explained_variance)  # Sort indices by least variance
    
    # Select the indices of the least important components
    least_important_indices = sorted_indices[:num_bits]
    
    return least_important_indices, explained_variance



def masker(reference_chunk, chunk, P_mask, P_size, P_type):
    """
    Applies a masking operation on the input chunk based on the given probabilities.

    Args:
        reference_chunk (torch.Tensor): The last chunk of the entire batch used for reference.
        chunk (torch.Tensor): The chunk to be masked.
        P_mask (float): Probability score for applying the mask (P_mask > 0.5 to mask).
        P_size (float): Probability score (0.1 <= P_size <= 0.4) for determining the size of the mask.
        P_type (float): Probability score for the type of mask to apply.
            - P_type < 0.2: No mask.
            - 0.2 <= P_type < 0.4: Silence mask.
            - 0.4 <= P_type < 0.6: Replace with random bits from reference chunk.
            - P_type >= 0.6: Revert the chunk to the original reference chunk.
    
    Returns:
        masked_chunk (torch.Tensor): The masked version of the input chunk.
        full_mask (torch.Tensor): A binary mask (same shape as `chunk`) indicating masked regions.
    """
    # Ensure P_size is within valid range
    assert 0.1 <= P_size <= 0.4, "P_size must be between 0.1 and 0.4."

    # Check and align shapes of reference_chunk and chunk
    if chunk.shape != reference_chunk.shape:
        max_length = max(chunk.shape[-1], reference_chunk.shape[-1])

        # Pad both tensors to match the maximum length
        if chunk.shape[-1] < max_length:
            pad_length = max_length - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_length))
        if reference_chunk.shape[-1] < max_length:
            pad_length = max_length - reference_chunk.shape[-1]
            reference_chunk = torch.nn.functional.pad(reference_chunk, (0, pad_length))

    # If P_mask <= 0.5, no masking is applied
    if P_mask <= 0.5:
        return chunk.clone(), torch.zeros_like(chunk)

    # Determine the size of the mask
    chunk_length = chunk.size(-1)  # Assuming last dimension is time
    mask_size = int(P_size * chunk_length)

    # Generate a mask with random starting position
    start_idx = torch.randint(0, chunk_length - mask_size + 1, (1,)).item()
    mask = torch.zeros_like(chunk)
    mask[..., start_idx:start_idx + mask_size] = 1

    # Apply masking based on P_type
    masked_chunk = chunk.clone()
    if P_type < 0.2:
        # No mask applied
        pass
    elif 0.2 <= P_type < 0.4:
        # Silence mask: Replace with zeros
        masked_chunk[..., start_idx:start_idx + mask_size] = 0
    elif 0.4 <= P_type < 0.6:
        # Replace with random bits from the reference chunk
        random_start = torch.randint(0, chunk_length - mask_size + 1, (1,)).item()
        masked_chunk[..., start_idx:start_idx + mask_size] = reference_chunk[..., random_start:random_start + mask_size]
    else:  # P_type >= 0.6
        # Revert to the corresponding segment from the reference chunk
        masked_chunk[..., start_idx:start_idx + mask_size] = reference_chunk[..., start_idx:start_idx + mask_size]

    return masked_chunk, mask

def initialize_csv(log_path):
    """
    Initialize the CSV file if it doesn't exist.
    Adds a header row to the file.

    Args:
        log_path (str): Path to the CSV file.
    """
    if not Path(log_path).exists():
        with open(log_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Date", "Time", "Epoch", "Train_Bit_Recovery", 
                "Train_Audio_Reconstruction", "Train_decoding_Loss", 
                "Val_Bit_Recovery", "Val_Audio_Reconstruction", "Val_decoding_Loss"
            ])


def update_csv(
    log_path, 
    epoch, 
    train_bit_recovery, 
    train_audio_reconstruction, 
    train_decoding_loss, 
    val_bit_recovery, 
    val_audio_reconstruction, 
    val_decoding_loss, 
    date=None, 
    time=None
):
    """
    Update the CSV file with a new row of data.

    Args:
        log_path (str): Path to the CSV file.
        epoch (int): The epoch number.
        train_bit_recovery (float): Train bit recovery metric.
        train_audio_reconstruction (float): Train audio reconstruction metric.
        train_decoding_loss (float): Train decoding loss.
        val_bit_recovery (float): Validation bit recovery metric.
        val_audio_reconstruction (float): Validation audio reconstruction metric.
        val_decoding_loss (float): Validation decoding loss.
        date (str, optional): Date in "YYYY-MM-DD" format. Defaults to today's date if not provided.
        time (str, optional): Time in "HH:MM:SS" format. Defaults to current time if not provided.
    """
    # Use the current date and time if not provided
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    if time is None:
        time = datetime.now().strftime("%H:%M:%S")
    
    # Append the data to the CSV
    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            date, time, epoch, train_bit_recovery, train_audio_reconstruction, 
            train_decoding_loss, val_bit_recovery, val_audio_reconstruction, 
            val_decoding_loss
        ])



def custom_collate_fn(batch):
    """
    Filters out None values from the batch and collates remaining items.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# Export statements
__all__ = [
    "custom_collate_fn",
    "get_dataloader",
    "compute_sdr",
    "loss_function",
    "initialize_csv",
    "update_csv",
    "masker",
    "get_white_noise",
]
