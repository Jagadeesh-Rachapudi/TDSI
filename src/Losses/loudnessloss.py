import math
import typing as tp
import torch
import torchaudio
import julius
from torch import nn
from torch.nn import functional as F
from torchaudio.functional.filtering import highpass_biquad, treble_biquad


def basic_loudness(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Compute a basic loudness metric for an audio waveform.

    Args:
        waveform (torch.Tensor): Audio tensor of shape (..., channels, time).
        sample_rate (int): Sampling rate of the audio.

    Returns:
        torch.Tensor: Scalar loudness value.
    """
    if waveform.size(-2) > 5:
        raise ValueError("Only up to 5 channels are supported.")
    eps = torch.finfo(torch.float32).eps
    gate_duration = 0.4
    overlap = 0.75
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))

    # Apply K-weighting filters
    waveform = treble_biquad(waveform, sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
    waveform = highpass_biquad(waveform, sample_rate, 38.0, 0.5)

    # Compute energy in blocks
    energy = torch.square(waveform).unfold(-1, gate_samples, step)
    energy = torch.mean(energy, dim=-1)

    # Weighted summation across channels
    g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)
    g = g[: energy.size(-2)]
    energy_weighted = torch.sum(g.unsqueeze(-1) * energy, dim=-2)

    # Convert energy to loudness
    loudness = -0.691 + 10 * torch.log10(energy_weighted + eps)
    return loudness


def _unfold(a: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Extract overlapping frames from the input tensor.

    Args:
        a (torch.Tensor): Input tensor.
        kernel_size (int): Length of each frame.
        stride (int): Stride between consecutive frames.

    Returns:
        torch.Tensor: Unfolded tensor with frames.
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, "Data should be contiguous."
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


class TFLoudnessRatio(nn.Module):
    """
    Time-Frequency Loudness Ratio Loss.

    Args:
        sample_rate (int): Sample rate of the audio.
        segment (float): Duration of each audio segment (in seconds).
        overlap (float): Overlap between consecutive segments (0.0 to 1.0).
        n_bands (int): Number of frequency bands for filtering.
        temperature (float): Temperature parameter for softmax.
        clip_min (float): Minimum value for clipping loudness.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        segment: float = 0.5,
        overlap: float = 0.5,
        n_bands: int = 0,
        clip_min: float = -100.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.clip_min = clip_min
        self.temperature = temperature

        if n_bands > 0:
            self.n_bands = n_bands
            self.filter = julius.SplitBands(sample_rate=sample_rate, n_bands=n_bands)
        else:
            self.filter = None

    def forward(self, watermarked_audio: torch.Tensor, original_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute the Time-Frequency Loudness Ratio Loss.

        Args:
            watermarked_audio (torch.Tensor): Watermarked audio tensor (B, C, T).
            original_audio (torch.Tensor): Original audio tensor (B, C, T).

        Returns:
            torch.Tensor: Loudness ratio loss.
        """
        B, C, T = original_audio.shape
        assert original_audio.shape == watermarked_audio.shape, "Shape mismatch between original and watermarked audio."
        assert C == 1, "Only single-channel audio is supported."

        if self.filter is not None:
            bands_ref = self.filter(original_audio).view(B * self.n_bands, 1, -1)
            bands_out = self.filter(watermarked_audio).view(B * self.n_bands, 1, -1)
        else:
            bands_ref = original_audio.view(B, 1, -1)
            bands_out = watermarked_audio.view(B, 1, -1)

        # Frame-wise processing
        frame = int(self.segment * self.sample_rate)
        stride = int(frame * (1 - self.overlap))
        gt = _unfold(bands_ref, frame, stride).squeeze(1).contiguous().view(-1, 1, frame)
        est = _unfold(bands_out, frame, stride).squeeze(1).contiguous().view(-1, 1, frame)

        # Loudness calculations
        l_noise = basic_loudness(est - gt, sample_rate=self.sample_rate)
        l_ref = basic_loudness(gt, sample_rate=self.sample_rate)
        l_ratio = (l_noise - l_ref).view(-1, B)

        # Weighted loss with temperature scaling
        loss = torch.nn.functional.softmax(l_ratio / self.temperature, dim=0) * l_ratio
        return loss.mean()
