import typing as tp
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram
from models.conv import pad_for_conv1d

import warnings
warnings.filterwarnings("ignore", message="At least one mel filterbank has all zero values.")

class MelSpectrogramWrapper(nn.Module):
    """
    Wrapper around MelSpectrogram torchaudio transform providing proper padding
    and additional post-processing including log scaling.
    """
    def __init__(self, n_fft: int, hop_length: int, win_length: int,
                 n_mels: int, sample_rate: int, f_min: float = 0.0, f_max: float = None,
                 log: bool = True, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.floor_level = floor_level
        self.log = log
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=min(n_mels, n_fft // 2 + 1),  # Ensure n_mels <= n_fft / 2 + 1
            f_min=f_min,
            f_max=f_max,
            normalized=normalized,
            window_fn=torch.hann_window,
            center=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = int((self.n_fft - self.hop_length) // 2)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (p, p), mode="reflect")
        x = pad_for_conv1d(x, self.n_fft, self.hop_length)
        self.mel_transform.to(x.device)
        mel_spec = self.mel_transform(x)
        if self.log:
            mel_spec = torch.log10(self.floor_level + mel_spec)
        return mel_spec


class MelSpectrogramL1Loss(nn.Module):
    """
    L1 Loss on MelSpectrogram.
    This is used to measure the perceptual loss between `original_audio` and `watermarked_audio`.
    """
    def __init__(self, sample_rate: int, n_fft: int = 512, hop_length: int = 128, win_length: int = 512,
                 n_mels: int = 32, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.melspec = MelSpectrogramWrapper(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            log=log,
            normalized=normalized,
            floor_level=floor_level
        )

    def forward(self, original_audio: torch.Tensor, watermarked_audio: torch.Tensor) -> torch.Tensor:
        self.melspec.to(original_audio.device)
        s_original = self.melspec(original_audio)
        s_watermarked = self.melspec(watermarked_audio)
        return self.l1(s_watermarked, s_original)


class MultiScaleMelSpectrogramLoss(nn.Module):
    """
    Multi-Scale spectrogram loss (msspec).
    This is a more advanced perceptual loss that operates at multiple resolutions.
    """
    def __init__(self, sample_rate: int, range_start: int = 6, range_end: int = 10,
                 n_mels: int = 32, f_min: float = 0.0, f_max: float = None,
                 normalized: bool = False, alphas: bool = True, floor_level: float = 1e-5):
        super().__init__()
        self.l1s = nn.ModuleList()
        self.l2s = nn.ModuleList()
        self.alphas = []
        self.total = 0
        self.normalized = normalized

        for i in range(range_start, range_end):
            fft_size = 2 ** i
            mel_bands = min(n_mels, fft_size // 2 + 1)
            self.l1s.append(
                MelSpectrogramWrapper(
                    n_fft=fft_size,
                    hop_length=fft_size // 4,
                    win_length=fft_size,
                    n_mels=mel_bands,
                    sample_rate=sample_rate,
                    f_min=f_min,
                    f_max=f_max,
                    log=False,
                    normalized=normalized,
                    floor_level=floor_level
                )
            )
            self.l2s.append(
                MelSpectrogramWrapper(
                    n_fft=fft_size,
                    hop_length=fft_size // 4,
                    win_length=fft_size,
                    n_mels=mel_bands,
                    sample_rate=sample_rate,
                    f_min=f_min,
                    f_max=f_max,
                    log=True,
                    normalized=normalized,
                    floor_level=floor_level
                )
            )
            alpha = np.sqrt(fft_size - 1) if alphas else 1.0
            self.alphas.append(alpha)
            self.total += alpha + 1

    def forward(self, original_audio: torch.Tensor, watermarked_audio: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for i, alpha in enumerate(self.alphas):
            s_original_1 = self.l1s[i](original_audio)
            s_watermarked_1 = self.l1s[i](watermarked_audio)
            s_original_2 = self.l2s[i](original_audio)
            s_watermarked_2 = self.l2s[i](watermarked_audio)
            loss += F.l1_loss(s_watermarked_1, s_original_1) + alpha * F.mse_loss(s_watermarked_2, s_original_2)
        if self.normalized:
            loss /= self.total
        return loss
