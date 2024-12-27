import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
import librosa
from pathlib import Path
import random

class AudioSegmentDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, window_size=2.0, stride=2.0, file_extension="wav"):
        self.data_dir = Path(data_dir).resolve()
        self.sample_rate = sample_rate
        self.window_size = int(window_size * sample_rate)
        self.stride = int(stride * sample_rate)
        self.file_extension = file_extension
        self.audio_files = self.load_audio_files()

    def load_audio_files(self):
        audio_files_paths = sorted(
            [file for file in self.data_dir.rglob(f"*.{self.file_extension.lower()}")]
        )
        audio_files = []
        for file_path in audio_files_paths:
            audio_data, _ = librosa.load(file_path, sr=self.sample_rate)
            audio_files.append(audio_data)
        return audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        audio, _ = torchaudio.load(file_path)
        if audio.shape[1] < self.window_size:
            return None
        start_idx = random.randint(0, audio.shape[1] - self.window_size)
        audio_segment = audio[:, start_idx : start_idx + self.window_size]
        return audio_segment, 0

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

def get_dataloader(data_dir, batch_size, sample_rate=16000, window_size=4.0, stride=4.0, shuffle=True, num_workers=0):
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
