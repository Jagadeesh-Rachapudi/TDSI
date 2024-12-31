import torch
import torchaudio
import torch.nn as nn
from pathlib import Path

# Perceptual loss function
def compute_perceptual_loss(audio1, audio2):
    """
    Compute perceptual loss based on similarity of two audio inputs.

    Args:
        audio1: First audio tensor [batch_size, time_steps].
        audio2: Second audio tensor [batch_size, time_steps].

    Returns:
        Perceptual loss value.
    """
    assert audio1.shape == audio2.shape, "Audio inputs must have the same shape."

    # Compute mean squared error as a basic perceptual loss
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(audio1, audio2)

    # Add optional perceptual distance metric (e.g., cosine similarity)
    cosine_sim = torch.nn.functional.cosine_similarity(audio1, audio2, dim=-1)
    perceptual_loss = mse_loss * (1 - cosine_sim.mean())

    return perceptual_loss


# Function to load and preprocess audio
def load_audio(file_path, sample_rate=16000):
    """
    Load and preprocess audio.

    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Desired sample rate for the audio.

    Returns:
        Tensor: Preprocessed audio waveform.
    """
    waveform, original_sample_rate = torchaudio.load(file_path)

    # Resample if needed
    if original_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Ensure mono audio
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform


# Main code
if __name__ == "__main__":
    # Paths to audio files
    file_path1 = Path(r"../../data/test/20190115-0900-PLENARY_en.wav").resolve()
    file_path2 = Path(r"../../data/test/20190115-0900-PLENARY_en.wav").resolve()  # Same file for demonstration

    # Load audio files
    audio1 = load_audio(file_path1)
    audio2 = load_audio(file_path2)

    # Ensure the same length by padding/truncating
    max_length = min(audio1.shape[-1], audio2.shape[-1])
    audio1 = audio1[..., :max_length]
    audio2 = audio2[..., :max_length]

    # Call perceptual loss
    perceptual_loss = compute_perceptual_loss(audio1, audio2)
    print(f"Perceptual Loss: {perceptual_loss.item()}")
