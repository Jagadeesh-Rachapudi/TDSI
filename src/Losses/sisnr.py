import math
import typing as tp
import torch
from torch import nn
from torch.nn import functional as F


def _unfold(a: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Extract frames from the audio tensor using the specified kernel size and stride.

    Args:
        a (torch.Tensor): Input audio tensor.
        kernel_size (int): Size of the frame.
        stride (int): Step size between frames.

    Returns:
        torch.Tensor: Unfolded tensor of shape [..., frames, kernel_size].
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, "Input data should be contiguous."
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


def _center(x: torch.Tensor) -> torch.Tensor:
    """
    Center the input tensor by subtracting its mean along the last dimension.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Centered tensor.
    """
    return x - x.mean(-1, keepdim=True)


def _norm2(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared norm along the last dimension.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Squared norm of the input tensor.
    """
    return x.pow(2).sum(-1, keepdim=True)


class SISNR(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss for audio evaluation.

    Args:
        sample_rate (int): Sample rate of the audio signals.
        segment (float or None): Duration (in seconds) of each evaluation segment.
        overlap (float): Overlap between consecutive segments (0.0 to 1.0).
        epsilon (float): Small value for numerical stability.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        segment: tp.Optional[float] = 0.5,
        overlap: float = 0.5,
        epsilon: float = torch.finfo(torch.float32).eps,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.epsilon = epsilon

    def forward(self, original_audio: torch.Tensor, watermarked_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute the SI-SNR loss between original and watermarked audio signals.

        Args:
            original_audio (torch.Tensor): Ground truth audio signal (B, C, T).
            watermarked_audio (torch.Tensor): Predicted (watermarked) audio signal (B, C, T).

        Returns:
            torch.Tensor: SI-SNR loss (scalar).
        """
        B, C, T = original_audio.shape
        assert original_audio.shape == watermarked_audio.shape, "Input shapes must match."

        # Determine the frame size and stride based on segment length and overlap.
        if self.segment is None:
            frame = T
            stride = T
        else:
            frame = int(self.segment * self.sample_rate)
            stride = int(frame * (1 - self.overlap))

        epsilon = self.epsilon * frame  # Scale epsilon by frame size.

        # Unfold the audio into overlapping frames.
        gt = _unfold(original_audio, frame, stride)
        est = _unfold(watermarked_audio, frame, stride)

        # Center the audio frames.
        gt = _center(gt)
        est = _center(est)

        # Compute the projection of estimated signal onto ground truth.
        dot = torch.einsum("bcft,bcft->bcf", gt, est)
        proj = dot[:, :, :, None] * gt / (epsilon + _norm2(gt))

        # Compute the noise component.
        noise = est - proj

        # Compute SI-SNR.
        sisnr = 10 * (
            torch.log10(epsilon + _norm2(proj)) - torch.log10(epsilon + _norm2(noise))
        )

        # Return the negative SI-SNR as a loss value (lower is better).
        return -1 * sisnr[..., 0].mean()
