import os
import random
import wave
import contextlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from pydub import AudioSegment
import torchaudio

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset


class LazyAudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, chunk_duration=0.5):
        """
        Args:
            data_dir: Directory containing audio files.
            sample_rate: Target sampling rate for audio.
            chunk_duration: Duration of each chunk in seconds.
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.audio_files = self._find_audio_files()

        if not self.audio_files:
            raise FileNotFoundError(f"No MP3 files found in {self.data_dir}")

        print(f"Found {len(self.audio_files)} MP3 files.")

    def _find_audio_files(self):
        return [
            os.path.join(root, file)
            for root, _, files in os.walk(self.data_dir)
            for file in files
            if file.endswith(".mp3")
        ]

    def __len__(self):
        total_chunks = 0
        chunk_size = int(self.chunk_duration * self.sample_rate)
        for file_path in self.audio_files:
            metadata = torchaudio.info(file_path)
            num_chunks = int(metadata.num_frames / chunk_size)
            total_chunks += num_chunks
        return total_chunks

    def __getitem__(self, idx):
        chunk_size = int(self.chunk_duration * self.sample_rate)
        file_idx = 0
        current_chunk = idx

        while current_chunk >= 0:
            metadata = torchaudio.info(self.audio_files[file_idx])
            num_chunks = int(metadata.num_frames / chunk_size)
            if current_chunk < num_chunks:
                break
            current_chunk -= num_chunks
            file_idx += 1

        file_path = self.audio_files[file_idx]
        start_frame = current_chunk * chunk_size

        waveform, sample_rate = torchaudio.load(
            file_path, frame_offset=start_frame, num_frames=chunk_size
        )

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:  # Convert to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad shorter chunks to the desired length
        if waveform.shape[1] < chunk_size:
            pad_length = chunk_size - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # print(f"Waveform shape: {waveform.shape}")

        label = random.randint(0, 1)  # Assign a random label for simplicity
        return waveform, label



def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length or multiple chunks.
    """
    waveforms, labels = zip(*batch)

    # Determine the maximum length in the batch
    max_length = max([waveform.shape[-1] for waveform in waveforms])
    padded_waveforms = torch.zeros((len(waveforms), 1, max_length))

    # Pad each waveform to the maximum length
    for i, waveform in enumerate(waveforms):
        padded_waveforms[i, :, :waveform.shape[-1]] = waveform

    labels = torch.tensor(labels, dtype=torch.long)
    return padded_waveforms, labels


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
    dataset = LazyAudioDataset(
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
__all__ = ["LazyAudioDataset", "get_dataloader"]
