import typing as tp
import torch
from torch import nn
from torch.nn import functional as F

def _stft(x: torch.Tensor, fft_size: int, hop_length: int, win_length: int,
          window: tp.Optional[torch.Tensor], normalized: bool) -> torch.Tensor:
    """
    Perform Short-Time Fourier Transform (STFT) and convert to magnitude spectrogram.
    """
    B, C, T = x.shape
    x_stft = torch.stft(
        x.view(-1, T), fft_size, hop_length, win_length, window,
        normalized=normalized, return_complex=True,
    )
    x_stft = x_stft.view(B, C, *x_stft.shape[1:])
    real = x_stft.real
    imag = x_stft.imag
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)

class SpectralConvergenceLoss(nn.Module):
    """Compute Spectral Convergence Loss."""
    def __init__(self, epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor):
        """
        Args:
            x_mag: Predicted spectrogram.
            y_mag: Target spectrogram.
        Returns:
            Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + self.epsilon)

class LogSTFTMagnitudeLoss(nn.Module):
    """Compute Log STFT Magnitude Loss."""
    def __init__(self, epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor):
        """
        Args:
            x_mag: Predicted spectrogram.
            y_mag: Target spectrogram.
        Returns:
            Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(self.epsilon + y_mag), torch.log(self.epsilon + x_mag))

class STFTLosses(nn.Module):
    """STFT losses."""
    def __init__(self, n_fft: int, hop_length: int, win_length: int, window: str,
                 normalized: bool, epsilon: float):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergenceLoss(epsilon)
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(epsilon)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Predicted signal.
            y: Target signal.
        Returns:
            Spectral convergence loss and log magnitude loss.
        """
        x_mag = _stft(x, self.n_fft, self.hop_length, self.win_length, self.window, self.normalized)
        y_mag = _stft(y, self.n_fft, self.hop_length, self.win_length, self.window, self.normalized)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss

class STFTLoss(nn.Module):
    """Single Resolution STFT Loss."""
    def __init__(self, n_fft: int = 1024, hop_length: int = 120, win_length: int = 600,
                 window: str = "hann_window", normalized: bool = False,
                 factor_sc: float = 0.1, factor_mag: float = 0.1,
                 epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.loss = STFTLosses(n_fft, hop_length, win_length, window, normalized, epsilon)
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Predicted audio signal.
            y: Ground truth audio signal.
        Returns:
            Combined STFT loss.
        """
        sc_loss, mag_loss = self.loss(x, y)
        return self.factor_sc * sc_loss + self.factor_mag * mag_loss

class MRSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss."""
    def __init__(self, n_ffts: tp.Sequence[int] = [1024, 2048, 512], hop_lengths: tp.Sequence[int] = [120, 240, 50],
                 win_lengths: tp.Sequence[int] = [600, 1200, 240], window: str = "hann_window",
                 factor_sc: float = 0.1, factor_mag: float = 0.1,
                 normalized: bool = False, epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.stft_losses = nn.ModuleList([
            STFTLosses(n_fft, hop, win, window, normalized, epsilon)
            for n_fft, hop, win in zip(n_ffts, hop_lengths, win_lengths)
        ])
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Predicted audio signal.
            y: Ground truth audio signal.
        Returns:
            Combined Multi-Resolution STFT loss.
        """
        sc_loss = torch.tensor(0.0, device=x.device)
        mag_loss = torch.tensor(0.0, device=x.device)
        for stft_loss in self.stft_losses:
            sc_l, mag_l = stft_loss(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        return self.factor_sc * sc_loss + self.factor_mag * mag_loss
