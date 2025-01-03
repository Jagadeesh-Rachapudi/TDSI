import os
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


class LazyAudioDataset(Dataset):
    """
    Dataset for lazy loading and processing audio chunks from .wav files.
    """

    def __init__(self, data_dir, sample_rate=16000, chunk_duration=0.5):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(chunk_duration * sample_rate)
        self.audio_files = self._get_audio_files()

        if not self.audio_files:
            raise FileNotFoundError(f"No .wav files found in {self.data_dir}")

        print(f"Found {len(self.audio_files)} audio files in {self.data_dir}.")

    def _get_audio_files(self):
        """
        Collect all .wav files in the given directory and subdirectories.
        """
        audio_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".wav"):
                    audio_files.append(os.path.join(root, file))
        return audio_files

    def __len__(self):
        """
        Total number of chunks across all files.
        """
        total_chunks = 0
        for file_path in self.audio_files:
            try:
                metadata = torchaudio.info(file_path)
                num_chunks = metadata.num_frames // self.chunk_size
                total_chunks += num_chunks
                print("total chunks loaded is",total_chunks)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        return total_chunks

    def __getitem__(self, idx):
        """
        Return the entire audio file at the given index.
        """
        file_path = self.audio_files[idx]
        print("this is the index",idx)

        try:
            # Load the entire audio file
            # Load the audio file
            waveform, sample_rate = torchaudio.load(file_path)

            # Print the shape of the loaded audio
            print("The just now loaded audio is", waveform.shape)

            # Assign a label (if required, replace this with actual logic)
            label = random.randint(0, 1)  # Placeholder for label logic

            return waveform, label

        except Exception as e:
            print("This is the index value creating the proble",idx)
            print(f"Error loading file {file_path}: {e}")
            return None


def custom_collate_fn(batch):
    """
    Collate function to handle batches with variable-length or missing chunks.
    """
    batch = [item for item in batch if item is not None]  # Filter out None
    if len(batch) == 0:
        raise ValueError("All elements in the batch are None.")

    waveforms, labels = zip(*batch)
    waveforms = torch.stack(waveforms)  # Stack into batch tensor
    labels = torch.tensor(labels, dtype=torch.long)
    return waveforms, labels


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
