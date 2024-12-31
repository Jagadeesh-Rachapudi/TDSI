# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# import random
# import torch
# from Losses import (
#     MelSpectrogramL1Loss,
#     MultiScaleMelSpectrogramLoss,
#     MRSTFTLoss,
#     SISNR,
#     STFTLoss,
# )
# from Losses.loudnessloss import TFLoudnessRatio
# from Losses.wmloss import WMMbLoss
# from utils.utility_functions import get_white_noise


# def test_mel_l1_loss():
#     print("Testing MelSpectrogramL1Loss...")
#     N, C, T = 2, 1, random.randrange(1000, 20_000)  # Single channel
#     t1 = torch.randn(N, C, T)
#     t2 = torch.randn(N, C, T)

#     mel_l1 = MelSpectrogramL1Loss(sample_rate=22_050)
#     loss = mel_l1(t1, t2)
#     loss_same = mel_l1(t1, t1)

#     print(f"MelSpectrogramL1Loss: Loss={loss.item():.4f}, Loss Same={loss_same.item():.4f}")
#     assert isinstance(loss, torch.Tensor)
#     assert loss_same.item() == 0.0


# def test_msspec_loss():
#     print("Testing MultiScaleMelSpectrogramLoss...")
#     N, C, T = 2, 1, random.randrange(1000, 20_000)  # Single channel
#     t1 = torch.randn(N, C, T)
#     t2 = torch.randn(N, C, T)

#     msspec = MultiScaleMelSpectrogramLoss(sample_rate=22_050)
#     loss = msspec(t1, t2)
#     loss_same = msspec(t1, t1)

#     print(f"MultiScaleMelSpectrogramLoss: Loss={loss.item():.4f}, Loss Same={loss_same.item():.4f}")
#     assert isinstance(loss, torch.Tensor)
#     assert loss_same.item() == 0.0


# def test_mrstft_loss():
#     print("Testing MRSTFTLoss...")
#     N, C, T = 2, 1, random.randrange(1000, 20_000)  # Single channel
#     t1 = torch.randn(N, C, T)
#     t2 = torch.randn(N, C, T)

#     mrstft = MRSTFTLoss()
#     loss = mrstft(t1, t2)

#     print(f"MRSTFTLoss: Loss={loss.item():.4f}")
#     assert isinstance(loss, torch.Tensor)


# def test_sisnr_loss():
#     print("Testing SISNR...")
#     N, C, T = 2, 1, random.randrange(1000, 20_000)  # Single channel
#     t1 = torch.randn(N, C, T)
#     t2 = torch.randn(N, C, T)

#     sisnr = SISNR()
#     loss = sisnr(t1, t2)

#     print(f"SISNR: Loss={loss.item():.4f}")
#     assert isinstance(loss, torch.Tensor)


# def test_stft_loss():
#     print("Testing STFTLoss...")
#     N, C, T = 2, 1, random.randrange(1000, 20_000)  # Single channel
#     t1 = torch.randn(N, C, T)
#     t2 = torch.randn(N, C, T)

#     stft = STFTLoss()
#     loss = stft(t1, t2)

#     print(f"STFTLoss: Loss={loss.item():.4f}")
#     assert isinstance(loss, torch.Tensor)


# def test_wm_loss():
#     print("Testing WMMbLoss...")
#     N, nbits, T = 2, 16, random.randrange(1000, 20_000)
#     positive = torch.randn(N, 2 + nbits, T)
#     mask = torch.ones(N, 1, T)
#     message = torch.randint(0, 2, (N, nbits)).float()

#     wm_loss = WMMbLoss(temperature=0.3, loss_type="mse")
#     loss = wm_loss(positive, None, mask, message)

#     print(f"WMMbLoss: Loss={loss.item():.4f}")
#     assert isinstance(loss, torch.Tensor)


# def test_loudness_loss():
#     print("Testing TFLoudnessRatio...")
#     sr = 16_000
#     duration = 1.0
#     wav = get_white_noise(1, int(sr * duration)).unsqueeze(0)
#     tfloss = TFLoudnessRatio(sample_rate=sr, n_bands=1)

#     loss = tfloss(wav, wav)
#     print(f"TFLoudnessRatio: Loss={loss.item():.4f}")
#     assert isinstance(loss, torch.Tensor)


# if __name__ == "__main__":
#     print("Running all tests...")
#     test_mel_l1_loss()
#     test_msspec_loss()
#     test_mrstft_loss()
#     test_sisnr_loss()
#     test_stft_loss()
#     test_wm_loss()
#     test_loudness_loss()
#     print("All tests completed successfully.")

import os
import torch
import torchaudio
from Losses.combinedloss import CombinedLoss
from Losses.balancer import Balancer

# Constants
SAMPLE_RATE = 16000
NBITS = 32
DATA_DIR = "data/test"

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize CombinedLoss and Balancer
combined_loss = CombinedLoss(sample_rate=SAMPLE_RATE, nbits=NBITS)
balancer = Balancer(
    weights={
        "l1": 0.1,
        "mspec": 2.0,
        "adv": 4.0,
        "loud": 10.0,
        "loc": 10.0,
        "dec": 1.0,
    },
    balance_grads=True,
    total_norm=1.0,
    monitor=True,
)

def generate_random_audio(file_path, duration=5, sample_rate=SAMPLE_RATE):
    """
    Generate random audio and save it as a .wav file.
    
    Args:
        file_path (str): Path to save the generated audio file.
        duration (int): Duration of the audio in seconds.
        sample_rate (int): Sampling rate of the audio.
    """
    num_samples = duration * sample_rate
    random_waveform = torch.rand(1, num_samples) * 2 - 1  # Random values between -1 and 1
    torchaudio.save(file_path, random_waveform, sample_rate)
    print(f"Random audio file saved at {file_path}")

def load_audio(file_path, sample_rate=SAMPLE_RATE):
    """Loads an audio file and converts it to a tensor."""
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    return waveform.unsqueeze(0)  # Add batch dimension

def test_loss(original_audio_path, watermarked_audio_path, original_message, predicted_message):
    """Tests the loss functions and balancer using given audio and message inputs."""
    # Load audios
    original_audio = load_audio(original_audio_path)
    watermarked_audio = load_audio(watermarked_audio_path)

    # Ensure inputs have the right shape
    original_audio = original_audio.to(torch.float32)
    watermarked_audio = watermarked_audio.to(torch.float32)
    original_message = torch.tensor(original_message).unsqueeze(0).unsqueeze(-1).to(torch.float32)  # Add batch and time dims
    predicted_message = torch.tensor(predicted_message).unsqueeze(0).unsqueeze(-1).to(torch.float32)  # Add batch and time dims

    # Create a dummy mask (Assume all samples are watermarked for this test)
    mask = torch.ones_like(original_message).unsqueeze(-1)  # Add time dimension

    # Compute individual losses
    loss_dict = {
        "perceptual_loss": combined_loss.loudness_loss(watermarked_audio, original_audio),
        "spectrogram_loss": combined_loss.spectrogram_loss(watermarked_audio, original_audio),
        "stft_loss": combined_loss.stft_loss(watermarked_audio, original_audio),
        "detection_loss": combined_loss.detection_loss(watermarked_audio, original_audio, mask),
        "decoding_loss": combined_loss.decoding_loss(predicted_message, original_message, mask),
        "sisnr_loss": combined_loss.sisnr_loss(watermarked_audio, original_audio),
    }

    # Use balancer to compute effective loss
    dummy_input = torch.zeros_like(watermarked_audio)  # Placeholder for gradient computation
    effective_loss = balancer.backward(loss_dict, dummy_input)

    # Log results
    print(f"Effective Loss: {effective_loss.item()}")
    for loss_name, value in loss_dict.items():
        print(f"{loss_name}: {value.item()}")


# Example usage
if __name__ == "__main__":
    # Generate random audio for testing
    original_audio_path = os.path.join(DATA_DIR, "original_audio.wav")
    watermarked_audio_path = os.path.join(DATA_DIR, "watermarked_audio.wav")
    generate_random_audio(original_audio_path, duration=5, sample_rate=SAMPLE_RATE)
    generate_random_audio(watermarked_audio_path, duration=5, sample_rate=SAMPLE_RATE)

    # Define example messages
    original_message = [1, 0, 1, 0, 1]  # Example message
    predicted_message = [1, 0, 1, 1, 0]  # Example predicted message

    # Run the test
    test_loss(original_audio_path, watermarked_audio_path, original_message, predicted_message)
