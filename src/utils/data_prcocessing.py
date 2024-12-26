import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


class AudioSegmentDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing audio segments.
    """

    def __init__(self, data_dir: str, sample_rate: int = 16000, transform=None, normalize: bool = True):
        """
        Args:
            data_dir (str): Directory containing processed audio segments.
            sample_rate (int): Target sample rate for the audio.
            transform: Optional transformation to apply on the audio data.
            normalize (bool): Whether to normalize audio to the range [-1, 1].
        """
        self.data_dir = Path(data_dir).resolve()  # Ensure absolute path
        self.sample_rate = sample_rate
        self.audio_files = list(self.data_dir.glob("*.wav"))  # Extend this to handle other formats if needed
        self.transform = transform
        self.normalize = normalize

        if not self.audio_files:
            raise ValueError(f"No audio files found in {data_dir}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        try:
            # Load audio file
            audio, sr = torchaudio.load(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {audio_path}: {e}")

        # Resample if necessary
        if sr != self.sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = resample_transform(audio)

        # Normalize audio to range [-1, 1] if needed
        if self.normalize:
            audio = audio / torch.max(torch.abs(audio))

        # Apply optional transformations
        if self.transform:
            audio = self.transform(audio)

        return audio, str(audio_path)


def get_data_loader(
    data_dir: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    sample_rate: int = 16000,
    transform=None,
    normalize: bool = True,
):
    """
    Create a DataLoader for the processed audio dataset.

    Args:
        data_dir (str): Directory containing processed audio segments.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for data loading.
        sample_rate (int): Target sample rate for the audio.
        transform: Optional transformation to apply on the audio data.
        normalize (bool): Whether to normalize audio to the range [-1, 1].

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    dataset = AudioSegmentDataset(data_dir=data_dir, sample_rate=sample_rate, transform=transform, normalize=normalize)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader
