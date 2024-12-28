import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
from pathlib import Path
import random

class AudioSegmentDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, window_size=2.0, stride=2.0, file_extension="wav"):
        """
        Dataset for loading and segmenting audio files.

        Args:
            data_dir (str): Path to the directory containing audio files.
            sample_rate (int): Target sampling rate of the audio files.
            window_size (float): Duration of each audio segment in seconds.
            stride (float): Stride in seconds for segmenting audio.
            file_extension (str): File extension of the audio files (default: "wav").
        """
        self.data_dir = Path(data_dir).resolve()
        self.sample_rate = sample_rate
        self.window_size = int(window_size * sample_rate)
        self.stride = int(stride * sample_rate)
        self.file_extension = file_extension
        self.audio_files = self.load_audio_files()

    def load_audio_files(self):
        """
        Load audio file paths from the directory.

        Returns:
            list: List of file paths to audio files.
        """
        audio_files_paths = sorted(
            [file for file in self.data_dir.rglob(f"*.{self.file_extension.lower()}")]
        )
        return audio_files_paths

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Retrieve an audio segment from a file.

        Args:
            idx (int): Index of the file.

        Returns:
            tuple: Audio segment and dummy label (0).
        """
        file_path = self.audio_files[idx]
        try:
            audio, _ = torchaudio.load(file_path)  # Load audio using torchaudio
            audio = audio.squeeze(0)  # Remove channel dimension for mono audio

            # Check if the audio length is sufficient for the window size
            if len(audio) < self.window_size:
                return None

            # Randomly select a start index for the segment
            start_idx = random.randint(0, len(audio) - self.window_size)
            audio_segment = audio[start_idx : start_idx + self.window_size]
            return audio_segment, 0  # Return the segment and dummy label
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

def custom_collate_fn(batch):
    """
    Custom collate function to handle None values in batches.

    Args:
        batch (list): Batch of data.

    Returns:
        tuple: Collated batch or empty tensors if batch is empty.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

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
__all__ = ["AudioSegmentDataset", "get_dataloader", "custom_collate_fn"]
