import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random



import random
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate):
        """
        Args:
            data_dir (str or Path): Directory containing audio files.
            sample_rate (int): Expected sample rate for the audio data.
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.audio_files = list(self.data_dir.glob("*.wav"))

        if not self.audio_files:
            raise FileNotFoundError(f"No audio files found in directory {self.data_dir}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Load audio and generate a random label.
        Args:
            idx (int): Index of the audio file.
        Returns:
            Tensor: Raw audio tensor.
            int: Randomly generated 32-bit integer label.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.audio_files[idx]

        try:
            # Load the audio file
            waveform, sample_rate = torchaudio.load(audio_path)

            # Generate a random 32-bit label
            label = random.randint(0, 2**32 - 1)

            return waveform, label

        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None, None






def get_dataloader(data_dir, batch_size, sample_rate, shuffle, num_workers):
    dataset = AudioDataset(data_dir=data_dir, sample_rate=sample_rate)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: list(zip(*[d for d in x if d[0] is not None])),
    )
    return data_loader




    # Export the module for external use
__all__ = ["AudioDataset", "get_dataloader"]
