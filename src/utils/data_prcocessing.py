import os
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio


class LazyAudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, chunk_duration=0.5):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(chunk_duration * sample_rate)
        self.audio_files, self.file_chunk_counts = self._prepare_audio_files()

        if not self.audio_files:
            raise FileNotFoundError(f"No .wav files found in {self.data_dir}")

        print(f"Found {len(self.audio_files)} .wav files.")

    def _prepare_audio_files(self):
        audio_files = []
        file_chunk_counts = []

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    metadata = torchaudio.info(file_path)
                    num_chunks = metadata.num_frames // self.chunk_size
                    if num_chunks > 0:
                        audio_files.append(file_path)
                        file_chunk_counts.append(num_chunks)

        return audio_files, file_chunk_counts

    def __len__(self):
        return sum(self.file_chunk_counts)

    def __getitem__(self, idx):
        file_idx = 0
        current_chunk = idx

        # Identify the file and chunk
        while current_chunk >= self.file_chunk_counts[file_idx]:
            current_chunk -= self.file_chunk_counts[file_idx]
            file_idx += 1

        file_path = self.audio_files[file_idx]
        start_frame = current_chunk * self.chunk_size

        # Load the chunk
        waveform, sample_rate = torchaudio.load(file_path, frame_offset=start_frame, num_frames=self.chunk_size)

        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Pad to ensure uniform size
        if waveform.shape[1] < self.chunk_size:
            print("Found a other sized chunk", waveform.shape )
            pad_length = self.chunk_size - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        label = random.randint(0, 1)
        return waveform, label

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length or multiple chunks.
    """
    waveforms, labels = zip(*batch)

    # Stack waveforms directly since all have the same length
    stacked_waveforms = torch.stack(waveforms)

    labels = torch.tensor(labels, dtype=torch.long)
    return stacked_waveforms, labels


def get_dataloader(data_dir, batch_size, sample_rate=16000, shuffle=True, num_workers=4):
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
