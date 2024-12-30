"""
Loss-related classes and functions.

This module provides various loss functions required for training and evaluating the AudioSeal model.
- `Balancer`: Balances gradients when combining multiple losses.
- `TFLoudnessRatio`: Computes time-frequency loudness loss.
- `SISNR`: Computes Scale-Invariant Signal-to-Noise Ratio loss.
- Spectrogram Losses:
  - `MelSpectrogramL1Loss`: L1 loss on Mel-spectrograms.
  - `MultiScaleMelSpectrogramLoss`: Multi-scale spectrogram loss.
- STFT Losses:
  - `LogSTFTMagnitudeLoss`: Log magnitude loss for STFT.
  - `MRSTFTLoss`: Multi-resolution STFT loss.
  - `SpectralConvergenceLoss`: Measures the spectral convergence loss.
  - `STFTLoss`: Single-resolution STFT loss.
- Watermark Losses:
  - `WMDetectionLoss`: Loss for watermark detection.
  - `WMMbLoss`: Loss for decoding multi-bit watermarks.
"""


# flake8: noqa
from .balancer import Balancer
from .loudnessloss import TFLoudnessRatio
from .sisnr import SISNR
from .specloss import MelSpectrogramL1Loss, MultiScaleMelSpectrogramLoss
from .stftloss import (LogSTFTMagnitudeLoss, MRSTFTLoss,
                       SpectralConvergenceLoss, STFTLoss)
from .wmloss import WMDetectionLoss, WMMbLoss
