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
import torchaudio  # Add this import for loading audio files
import librosa

class AudioSegmentDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, window_size=2.0, stride=2.0, file_extension="wav"):
            """
            Initialize the dataset loader.

            Args:
                data_dir (str): Path to the directory containing audio files.
                sample_rate (int): Sampling rate of the audio files.
                window_size (float): Window size in seconds.
                stride (float): Stride in seconds.
                file_extension (str): Audio file extension to look for. Default is "wav".
            """
            self.data_dir = Path(data_dir).resolve()
            self.sample_rate = sample_rate
            self.window_size = int(window_size * sample_rate)
            self.stride = int(stride * sample_rate)
            self.file_extension = file_extension
            print(vars(self))
            # Load and process the audio files
            self.audio_files = self.load_audio_files()

    def load_audio_files(self):
            # Assuming you're using librosa or any other audio library to load the files
            audio_files_paths = sorted([file for file in self.data_dir.rglob(f"*.{self.file_extension.lower()}") if file.suffix.lower() == f".{self.file_extension.lower()}"])

            audio_files = []
            for file_path in audio_files_paths:
                # Load the audio file using librosa or any other audio library
                audio_data, _ = librosa.load(file_path, sr=self.sample_rate)
                audio_files.append(audio_data)
            return audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        audio, sample_rate = torchaudio.load(file_path)  # Load audio file using torchaudio

        # Skip samples shorter than the window size
        if audio.shape[1] < self.window_size:  # Adjust for torchaudio's output shape
            return None

        start_idx = random.randint(0, audio.shape[1] - self.window_size)
        audio_segment = audio[:, start_idx: start_idx + self.window_size]  # Adjust for torchaudio's output shape
        return audio_segment, 0  # Return audio segment and a dummy label

# Custom collate function to handle invalid data points
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])  # Return empty tensors if batch is empty
    return torch.utils.data.dataloader.default_collate(batch)

# DataLoader function
def get_dataloader(data_dir, batch_size, sample_rate=16000, window_size=4.0, stride=4.0, shuffle=True, num_workers=0):
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

# Define directories for train, validate, and test (**Ensure these paths are correct!**)
train_data_dir = r"D:\myCode\data\train"  # Update this path to the correct training data directory
# validate_data_dir = r"D:\myCode\data\validate"
# test_data_dir = r"D:\myCode\data\test"  # Test data directory might be different from validation data directory

# Parameters for DataLoader
batch_size = 2
sample_rate = 16000
window_size = 4.0  # seconds
stride = 4.0  # seconds
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

# validate_loader = get_dataloader(
#     data_dir=validate_data_dir,
#     batch_size=batch_size,
#     sample_rate=sample_rate,
#     window_size=window_size,
#     stride=stride,
#     shuffle=False,
#     num_workers=num_workers
# )

# test_loader = get_dataloader(
#     data_dir=test_data_dir,
#     batch_size=batch_size,
#     sample_rate=sample_rate,
#     window_size=window_size,
#     stride=stride,
#     shuffle=False,
#     num_workers=num_workers
# )

# Check DataLoader sizes
if len(train_loader.dataset) == 0:
    print("Warning: Train dataset is empty.")
else:
    print(f"Train dataset size: {len(train_loader.dataset)}")

# if len(validate_loader.dataset) == 0:
    # print("Warning: Validation dataset is empty.")
# else:
    # print(f"Validation dataset size: {len(validate_loader.dataset)}")

# if len(test_loader.dataset) == 0:
    # print("Warning: Test dataset is empty.")
# else:
    # print(f"Test dataset size: {len(test_loader.dataset)}")