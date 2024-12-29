import torch
import torch.nn as nn
from losses.loudnessloss import TFLoudnessRatio
from losses.wmloss import WMDetectionLoss, WMMbLoss
from losses.specloss import MultiScaleMelSpectrogramLoss
from losses.stftloss import MRSTFTLoss
from losses.sisnr import SISNR
from losses.balancer import Balancer  # Import Balancer

class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions for AudioSeal training with gradient balancing.
    Includes Loudness Loss, Detection Loss, Spectrogram Loss, STFT Loss, and SI-SNR Loss.
    """
    def __init__(self, sample_rate=16000, nbits=16):
        super().__init__()
        # Instantiate individual loss functions
        self.loudness_loss = TFLoudnessRatio(sample_rate=sample_rate)
        self.detection_loss = WMDetectionLoss(p_weight=10.0, n_weight=10.0)
        self.decoding_loss = WMMbLoss(temperature=1.0, loss_type="bce")
        self.spectrogram_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
        self.stft_loss = MRSTFTLoss()
        self.sisnr_loss = SISNR(sample_rate=sample_rate)

        # Define weights for each loss
        self.weights = {
            "loudness_loss": 10.0,
            "detection_loss": 10.0,
            "decoding_loss": 1.0,
            "spectrogram_loss": 2.0,
            "stft_loss": 1.0,
            "sisnr_loss": 1.0,
        }

        # Initialize the balancer
        self.balancer = Balancer(weights=self.weights, balance_grads=True, total_norm=1.0)

    def forward(self, watermarked_audio, original_audio, message):
        """
        Compute the combined loss with gradient balancing.

        Args:
            watermarked_audio (torch.Tensor): Watermarked audio signal.
            original_audio (torch.Tensor): Original audio signal.
            message (torch.Tensor): Ground truth watermark message.

        Returns:
            torch.Tensor: Combined loss value.
        """
        # Compute individual losses
        losses = {
            "loudness_loss": self.loudness_loss(watermarked_audio, original_audio),
            "detection_loss": self.detection_loss(watermarked_audio, message),
            "decoding_loss": self.decoding_loss(watermarked_audio, message),
            "spectrogram_loss": self.spectrogram_loss(watermarked_audio, original_audio),
            "stft_loss": self.stft_loss(watermarked_audio, original_audio),
            "sisnr_loss": self.sisnr_loss(watermarked_audio, original_audio),
        }

        # Use the balancer to compute the combined loss
        combined_loss = self.balancer.backward(losses, watermarked_audio)

        return combined_loss
