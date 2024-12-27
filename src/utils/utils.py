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


# Function to clear GPU memory
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Function to save losses as a JSON file
def save_loss_data(loss_data, filename="loss_data.json"):
    with open(filename, "w") as json_file:
        json.dump(loss_data, json_file, indent=4)
    print(f"Loss data saved to {filename}")

# Function to plot training and validation losses
def plot_losses(loss_data):
    if "train_loss" in loss_data and "validation_loss" in loss_data:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_data["train_loss"], label="Train Loss")
        plt.plot(loss_data["validation_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()
    else:
        print("Loss data does not contain 'train_loss' or 'validation_loss'.")

# Custom collate function for the DataLoader
def custom_collate_fn(batch):
    # Handles cases where some data points might be None
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Dataset class for audio segmentation
class AudioSegmentDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, window_size=2.0, stride=2.0):
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
        if audio.shape[0] < self.window_size:
            return None  # Skip samples shorter than window size
        start_idx = random.randint(0, len(audio) - self.window_size)
        audio_segment = audio[start_idx:start_idx + self.window_size]
        return audio_segment, 0  # Assuming a dummy label

# Function to load data with DataLoader
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
