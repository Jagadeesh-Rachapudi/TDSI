import os
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

def print_audio_shapes(data_loader, data_name):
    """
    Print the shape of each audio chunk in the given data loader and log unexpected sizes.

    Args:
        data_loader: DataLoader object.
        data_name: Name of the dataset (for logging purposes).
    """
    expected_shape = torch.Size([1, 8000])  # Hardcoded expected shape
    print(f"Printing audio shapes for {data_name} dataset with expected shape {expected_shape}:")

    total_audio = 0  # Counter for the total number of audio files
    unexpected_size_count = 0  # Counter for unexpected sizes

    for batch_idx, (audios, file_paths) in enumerate(data_loader):
        for audio_idx, (audio, file_path) in enumerate(zip(audios, file_paths)):
            # Check if audio is None
            if audio is None:
                print(f"Audio {total_audio} is None. Skipping...")
                unexpected_size_count += 1
                total_audio += 1
                continue

            audio_shape = audio.shape
            print(f"Audio index {total_audio} shape: {audio_shape}")

            # Check for unexpected shape
            if audio_shape != expected_shape:
                print(f"Unexpected shape found for audio {total_audio} at path: {file_path}")
                unexpected_size_count += 1

            # Increment the counter
            total_audio += 1

    print(f"Total audios checked: {total_audio}")
    print(f"Total unexpected size audios: {unexpected_size_count}")


class LazyAudioDataset(Dataset):
    """
    Dataset for lazy loading and processing audio files without chunking.
    """

    def __init__(self, data_dir, sample_rate=16000):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.audio_files = self._get_audio_files()

        if not self.audio_files:
            raise FileNotFoundError(f"No .wav files found in {self.data_dir}")

        print(f"Found {len(self.audio_files)} audio files in {self.data_dir}.")

    def _get_audio_files(self):
        """
        Collect all .wav files in the given directory and subdirectories and print their shapes.
        """
        audio_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    try:
                        # Load the audio file to get its shape
                        waveform, sample_rate = torchaudio.load(file_path)
                        print(f"Audio path: {file_path}, Shape: {waveform.shape}, Sample rate: {sample_rate}")
                        audio_files.append(file_path)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        return audio_files

    def __len__(self):
        """
        Total number of audio files.
        """
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Return the entire audio file at the given index.
        """
        file_path = self.audio_files[idx]
        try:
            # Load the entire audio file
            waveform, sample_rate = torchaudio.load(file_path)

            # Ensure waveform matches the expected sample rate
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            print(f"Audio loaded: Path: {file_path}, Shape: {waveform.shape}")
            return waveform, file_path

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of (audio, file_path).
    """
    audios = []
    file_paths = []

    for item in batch:
        if item is not None:  # Skip None items
            audio, file_path = item
            audios.append(audio)
            file_paths.append(file_path)

    # Stack audio tensors if all have the same shape
    if len(audios) > 0:
        stacked_audios = torch.stack(audios)
    else:
        raise ValueError("All elements in the batch are None.")

    return stacked_audios, file_paths


def get_dataloader(data_dir, batch_size, sample_rate=16000, shuffle=True, num_workers=4):
    """
    Create a DataLoader for LazyAudioDataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size.
        sample_rate (int): Target sampling rate.
        shuffle (bool): Shuffle the dataset.
        num_workers (int): Number of DataLoader workers.

    Returns:
        DataLoader: Torch DataLoader for the dataset.
    """
    dataset = LazyAudioDataset(data_dir=data_dir, sample_rate=sample_rate)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

# Export the module for external use
__all__ = ["LazyAudioDataset", "get_dataloader"]
