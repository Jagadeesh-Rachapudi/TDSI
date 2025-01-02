import csv
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import csv
import torch
import random

class LazyAudioDataset(Dataset):
    def __init__(self, data_dir, csv_file, sample_rate=16000, file_extension="wav", chunk_duration=0.5):
        """
        Args:
            data_dir: Directory containing audio files.
            csv_file: Path to the CSV file with labels for each audio file.
            sample_rate: Target sampling rate for audio.
            file_extension: Extension of audio files (e.g., 'wav').
            chunk_duration: Duration of each chunk in seconds.
        """
        self.data_dir = Path(data_dir).resolve()
        self.csv_file = Path(csv_file).resolve()
        self.sample_rate = sample_rate
        self.file_extension = file_extension
        self.chunk_duration = chunk_duration
        self.audio_files, self.labels = self.load_audio_files_and_labels()

    def load_audio_files_and_labels(self):
        """
        Loads audio file paths and their corresponding labels from the CSV file.
        """
        audio_files = []
        labels = {}

        with open(self.csv_file, mode="r") as f:
            reader = csv.reader(f)
            for row in reader:
                filename, label = row
                labels[filename] = int(label)

        for file in self.data_dir.rglob(f"*.{self.file_extension.lower()}"):
            filename = file.name
            if filename in labels:
                audio_files.append((file, labels[filename]))

        return audio_files, labels

    def __len__(self):
        """
        Returns the total number of chunks across all audio files.
        """
        total_chunks = 0
        for file_path, _ in self.audio_files:
            metadata = torchaudio.info(file_path)
            num_chunks = int(metadata.num_frames / (self.chunk_duration * self.sample_rate))
            total_chunks += num_chunks
        return total_chunks

    def __getitem__(self, idx):
        """
        Lazily loads the audio file, extracts the corresponding chunk, and retrieves its label.
        """
        chunk_size = int(self.chunk_duration * self.sample_rate)
        file_idx = 0
        current_chunk = idx

        # Identify which file and chunk the index corresponds to
        while current_chunk >= 0:
            metadata = torchaudio.info(self.audio_files[file_idx][0])
            num_chunks = int(metadata.num_frames / chunk_size)
            if current_chunk < num_chunks:
                break
            current_chunk -= num_chunks
            file_idx += 1

        file_path, _ = self.audio_files[file_idx]
        start_frame = current_chunk * chunk_size

        # Lazily load only the required chunk
        waveform, sample_rate = torchaudio.load(
            file_path, frame_offset=start_frame, num_frames=chunk_size
        )

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Generate a random 32-bit label
        label = random.getrandbits(32)

        return waveform, label



def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length audio files.
    """
    waveforms, labels = zip(*batch)

    # Pad all waveforms to the length of the longest waveform in the batch
    max_length = max([waveform.shape[-1] for waveform in waveforms])
    padded_waveforms = torch.zeros((len(waveforms), 1, max_length))

    for i, waveform in enumerate(waveforms):
        padded_waveforms[i, :, :waveform.shape[-1]] = waveform

    labels = torch.tensor(labels, dtype=torch.long)
    return padded_waveforms, labels


def get_dataloader(data_dir, csv_file, batch_size, sample_rate=16000, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        csv_file (str): Path to the CSV file with labels.
        batch_size (int): Batch size.
        sample_rate (int): Target sampling rate.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes for loading data.

    Returns:
        DataLoader: Torch DataLoader for the dataset.
    """
    dataset = LazyAudioDataset(
        data_dir=data_dir,
        csv_file=csv_file,
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
__all__ = ["LazyAudioDataset", "get_dataloader"]
