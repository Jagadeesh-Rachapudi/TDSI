import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from utils.utility_functions import custom_collate_fn

class AudioSegmentDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, window_size=4.0, stride=4.0, file_extension="wav"):
        """
        Args:
            data_dir: Directory containing audio files.
            sample_rate: Target sampling rate for audio.
            window_size: Window size in seconds for audio chunks.
            stride: Stride in seconds for sliding the window.
            file_extension: Extension of audio files (e.g., 'wav').
        """
        self.data_dir = Path(data_dir).resolve()
        self.sample_rate = sample_rate
        self.window_size = int(window_size * sample_rate)  # Convert seconds to samples
        self.stride = int(stride * sample_rate)  # Convert seconds to samples
        self.file_extension = file_extension
        self.audio_files = self.load_audio_files()

    def load_audio_files(self):
        """
        Loads all audio file paths in the directory matching the file extension.
        """
        return sorted([file for file in self.data_dir.rglob(f"*.{self.file_extension.lower()}")])

    def __len__(self):
        """
        Returns the total number of audio files.
        """
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Extracts a random chunk from the audio file specified by idx.
        """
        file_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample if the audio sampling rate doesn't match the target
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Ensure mono audio by averaging channels if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Handle edge cases where the audio file is shorter than the window size
        if waveform.shape[1] < self.window_size:
            padding = self.window_size - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Randomly select a chunk from the audio file
        max_start_idx = max(0, waveform.shape[1] - self.window_size)
        start_idx = random.randint(0, max_start_idx)
        audio_segment = waveform[:, start_idx : start_idx + self.window_size]

        return audio_segment, 0  # Returning a dummy label for now


def get_dataloader(data_dir, batch_size, sample_rate=16000, window_size=4.0, stride=4.0, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size.
        sample_rate (int): Target sampling rate.
        window_size (float): Duration of each audio segment in seconds.
        stride (float): Stride in seconds for segmenting audio.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes for loading data.

    Returns:
        DataLoader: Torch DataLoader for the dataset.
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

# Export statements
__all__ = ["AudioSegmentDataset", "get_dataloader"]
