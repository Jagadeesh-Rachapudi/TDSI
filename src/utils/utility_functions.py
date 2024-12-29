import gc
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.data_prcocessing import AudioSegmentDataset
def get_white_noise(chs: int = 1, num_frames: int = 1):
    wav = torch.randn(chs, num_frames)
    return wav

def custom_collate_fn(batch):
    """
    Filters out None values from the batch and collates remaining items.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataloader(data_dir, batch_size, sample_rate=16000, window_size=4.0, stride=4.0, shuffle=True, num_workers=0):
    """
    Creates a DataLoader for the dataset with specified parameters.
    """
    dataset = AudioSegmentDataset(
        data_dir=data_dir,
        sample_rate=sample_rate,
        window_size=window_size,
        stride=stride
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

def compute_sdr(original: torch.Tensor, watermarked: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the Signal-to-Distortion Ratio (SDR) between original and watermarked signals.

    Args:
        original (torch.Tensor): Original carrier signal (shape: [batch_size, num_samples]).
        watermarked (torch.Tensor): Watermarked carrier signal (shape: [batch_size, num_samples]).
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: SDR value in dB for each signal in the batch.
    """
    # Compute distortion (difference between signals)
    distortion = watermarked - original

    # Signal power: ||C||^2
    signal_power = torch.sum(original**2, dim=-1)

    # Distortion power: ||C' - C||^2
    distortion_power = torch.sum(distortion**2, dim=-1) + eps  # Add eps for numerical stability

    # SDR: 10 * log10(signal_power / distortion_power)
    sdr = 10 * torch.log10(signal_power / distortion_power)
    return sdr

def loss_function(watermarked_audio, original_audio, sdr_target=30.0):
    """
    Generator loss combining SDR-based loss and L1 loss for imperceptibility.

    Args:
        watermarked_audio (torch.Tensor): Watermarked audio signal.
        original_audio (torch.Tensor): Original audio signal.
        sdr_target (float): Target SDR value in dB.

    Returns:
        torch.Tensor: Combined loss value.
    """
    sdr = compute_sdr(original_audio, watermarked_audio)
    sdr_loss = torch.mean(torch.relu(sdr_target - sdr))  # Penalize SDR below target
    l1_loss = torch.mean(torch.abs(watermarked_audio - original_audio))
    return l1_loss + sdr_loss

# Export statements
__all__ = [
    "custom_collate_fn",
    "get_dataloader",
    "compute_sdr",
    "loss_function"
]
