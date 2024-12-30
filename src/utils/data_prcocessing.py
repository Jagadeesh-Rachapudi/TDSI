import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class AudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, file_extension="wav"):
        """
        Args:
            data_dir: Directory containing audio files.
            sample_rate: Target sampling rate for audio.
            file_extension: Extension of audio files (e.g., 'wav').
        """
        self.data_dir = Path(data_dir).resolve()
        self.sample_rate = sample_rate
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
        Loads the entire audio file specified by idx.
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

        return waveform, 0  # Returning a dummy label for now


def custom_collate_fn(batch):
    # Dummy implementation of a custom collate function (replace if needed)
    waveforms, labels = zip(*batch)
    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)  # Pad waveforms to the same length
    labels = torch.tensor(labels)
    return waveforms, labels


def get_dataloader(data_dir, batch_size, sample_rate=16000, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size.
        sample_rate (int): Target sampling rate.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes for loading data.

    Returns:
        DataLoader: Torch DataLoader for the dataset.
    """
    dataset = AudioDataset(
        data_dir=data_dir,
        sample_rate=sample_rate
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
__all__ = ["AudioDataset", "get_dataloader"]
