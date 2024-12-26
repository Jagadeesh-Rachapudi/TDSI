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

class AudioSegmentDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, window_size=2.0, stride=2.0):
        """
        Args:
            data_dir (str): Path to the directory containing audio files.
            sample_rate (int): Sampling rate of the audio.
            window_size (float): Size of the audio window in seconds.
            stride (float): Stride size in seconds for segmentation.
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.window_size = int(window_size * sample_rate)
        self.stride = int(stride * sample_rate)
        self.files = list(self.data_dir.glob("*.wav"))

        if not self.files:
            raise FileNotFoundError(f"No .wav files found in directory: {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        audio = torch.load(file_path)  # Assuming audio files are stored as tensors

        # Skip samples shorter than the window size
        if audio.shape[0] < self.window_size:
            return None

        start_idx = random.randint(0, len(audio) - self.window_size)
        audio_segment = audio[start_idx : start_idx + self.window_size]
        return audio_segment, 0  # Return audio segment and a dummy label

# Custom collate function to handle invalid data points
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# DataLoader function
def get_dataloader(data_dir, batch_size, sample_rate=16000, window_size=2.0, stride=2.0, shuffle=True, num_workers=0):
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

# Define directories for train, validate, and test
train_data_dir = r"C:\Users\Ritik\Jagadeesh\codes\data\train"
validate_data_dir = r"C:\Users\Ritik\Jagadeesh\codes\data\validate"
test_data_dir = r"C:\Users\Ritik\Jagadeesh\codes\data\test"

# Parameters for DataLoader
batch_size = 2
sample_rate = 16000
window_size = 2.0  # seconds
stride = 1.0       # seconds
num_workers = 2

# Initialize DataLoaders
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

# Test DataLoader
test_loader = get_dataloader(
    data_dir=test_data_dir,
    batch_size=batch_size,
    sample_rate=sample_rate,
    window_size=window_size,
    stride=stride,
    shuffle=False,
    num_workers=num_workers
)

# Check DataLoader sizes
print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Validation dataset size: {len(validate_loader.dataset)}")
print(f"Test dataset size: {len(test_loader.dataset)}")
