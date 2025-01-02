import os
import random
import wave
import contextlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from pydub import AudioSegment

class LazyAudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, chunk_duration=0.5, file_extension="mp3"):
        """
        Args:
            data_dir: Directory containing audio files.
            sample_rate: Target sampling rate for audio.
            chunk_duration: Duration of each chunk in seconds.
            file_extension: Extension of input audio files (e.g., 'mp3').
        """
        self.data_dir = Path(data_dir).resolve()
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.file_extension = file_extension
        self.audio_files = list(self.data_dir.rglob(f"*.{self.file_extension.lower()}"))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Lazily loads and processes an audio file.
        """
        file_path = self.audio_files[idx]

        # Convert to .wav and ensure 1 channel
        wav_file_path = self.convert_to_wav(file_path)

        # Check the duration of the audio
        with contextlib.closing(wave.open(wav_file_path, 'r')) as wav_file:
            frame_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            duration = wav_file.getnframes() / float(frame_rate)

        if frame_rate != self.sample_rate or num_channels != 1:
            raise ValueError(f"Audio file {wav_file_path} does not meet requirements.")

        if duration > self.chunk_duration:
            # Chop audio into 0.5-second chunks
            chunks = self.chop_audio(wav_file_path)
            return [self.load_audio(chunk) for chunk in chunks]
        elif duration == self.chunk_duration:
            return self.load_audio(wav_file_path)
        else:
            raise ValueError(f"Audio file {wav_file_path} is less than {self.chunk_duration} seconds.")

    def convert_to_wav(self, file_path):
        """
        Converts an audio file to .wav format with a single channel and 16kHz sampling rate.
        """
        wav_file_path = file_path.with_suffix('.wav')
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
        audio.export(wav_file_path, format="wav")
        return wav_file_path

    def chop_audio(self, file_path):
        """
        Splits an audio file into 0.5-second chunks.
        """
        audio = AudioSegment.from_file(file_path)
        chunk_length_ms = int(self.chunk_duration * 1000)
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        chunk_paths = []

        for idx, chunk in enumerate(chunks):
            if len(chunk) == chunk_length_ms:  # Ensure chunk is exactly 0.5 seconds
                chunk_path = file_path.with_name(f"{file_path.stem}_chunk{idx}.wav")
                chunk.export(chunk_path, format="wav")
                chunk_paths.append(chunk_path)

        return chunk_paths

    def load_audio(self, file_path):
        """
        Loads an audio file and generates a random 32-bit label.
        """
        waveform, sample_rate = torchaudio.load(file_path)
        label = random.getrandbits(32)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        return waveform, label


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length or multiple chunks.
    """
    waveforms = []
    labels = []

    for item in batch:
        if isinstance(item, list):
            # Flatten nested lists (e.g., from chunks)
            waveforms.extend([i[0] for i in item])
            labels.extend([i[1] for i in item])
        else:
            waveforms.append(item[0])
            labels.append(item[1])

    max_length = max([waveform.shape[-1] for waveform in waveforms])
    padded_waveforms = torch.zeros((len(waveforms), 1, max_length))

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
