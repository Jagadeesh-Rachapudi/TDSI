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

# Export statements
__all__ = [
    "custom_collate_fn",
    "get_dataloader",
    "compute_sdr",
    "loss_function"
]
