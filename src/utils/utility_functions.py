import gc
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from utils.data_prcocessing import LazyAudioDataset
import csv
from pathlib import Path
import torch


def get_white_noise(chs: int = 1, num_frames: int = 1):
    wav = torch.randn(chs, num_frames)
    return wav


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
    assert 0.1 <= P_size <= 0.4, "P_size must be between 0.1 and 0.4."
    assert chunk.shape == reference_chunk.shape, "Chunks must have the same shape."

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
    Initialize the losses.csv file if it doesn't exist.
    Adds a header row to the file.

    Args:
        log_path (str): Path to the losses.csv file.
    """
    if not Path(log_path).exists():
        with open(log_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Start Date", "Start Time", "Epoch", "Training Accuracy (%)",
                "Validation Accuracy (%)", "Training Perceptual Loss",
                "Training Detection Loss", "Training Decoding Loss",
                "Validation Perceptual Loss", "Validation Detection Loss",
                "Validation Decoding Loss"
            ])


def update_csv(log_path, start_date, start_time, epoch, train_accuracy, val_accuracy,
               train_perceptual_loss, train_detection_loss, train_decoding_loss,
               val_perceptual_loss, val_detection_loss, val_decoding_loss):
    """
    Update the losses.csv file with the results of the current epoch.

    Args:
        log_path (str): Path to the losses.csv file.
        start_date (str): Training start date.
        start_time (str): Training start time.
        epoch (int): Current epoch number.
        train_accuracy (float): Training accuracy as a percentage.
        val_accuracy (float): Validation accuracy as a percentage.
        train_perceptual_loss (float): Training perceptual loss.
        train_detection_loss (float): Training detection loss.
        train_decoding_loss (float): Training decoding loss.
        val_perceptual_loss (float): Validation perceptual loss.
        val_detection_loss (float): Validation detection loss.
        val_decoding_loss (float): Validation decoding loss.
    """
    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            start_date, start_time, epoch, train_accuracy, val_accuracy,
            train_perceptual_loss, train_detection_loss, train_decoding_loss,
            val_perceptual_loss, val_detection_loss, val_decoding_loss
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
