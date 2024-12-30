import torch
import torch.nn as nn
from Losses.loudnessloss import TFLoudnessRatio
from Losses.specloss import MultiScaleMelSpectrogramLoss
from Losses.stftloss import MRSTFTLoss
from Losses.wmloss import WMDetectionLoss, WMMbLoss
from Losses.sisnr import SISNR


class CombinedLoss(nn.Module):
    def __init__(self, sample_rate: int, nbits: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.nbits = nbits

        # Initialize individual loss components
        self.loudness_loss = TFLoudnessRatio(sample_rate=sample_rate, segment=0.5, overlap=0.5, n_bands=8)
        self.spectrogram_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
        self.stft_loss = MRSTFTLoss()
        self.detection_loss = WMDetectionLoss()  # Removed p_weight and n_weight
        self.decoding_loss = WMMbLoss(temperature=1.0, loss_type="bce")
        self.sisnr_loss = SISNR(sample_rate=sample_rate)

        # Weights for each loss
        self.weights = {
            "l1": 0.1,
            "mspec": 2.0,
            "adv": 4.0,
            "loud": 10.0,
            "loc": 10.0,
            "dec": 1.0,
        }

    def forward(self, watermarked_audio, original_audio, message, mask):
        # Compute individual losses
        loudness_loss = self.loudness_loss(watermarked_audio, original_audio)
        spectrogram_loss = self.spectrogram_loss(watermarked_audio, original_audio)
        stft_loss = self.stft_loss(watermarked_audio, original_audio)
        detection_loss = self.detection_loss(watermarked_audio, original_audio, mask)  # Assuming default behavior
        decoding_loss = self.decoding_loss(watermarked_audio, original_audio, mask, message)
        sisnr_loss = self.sisnr_loss(watermarked_audio, original_audio)

        # Weighted combination of losses
        perceptual_loss = (
            self.weights["l1"] * loudness_loss +
            self.weights["mspec"] * spectrogram_loss +
            self.weights["adv"] * stft_loss +
            self.weights["loud"] * sisnr_loss
        )

        localization_loss = self.weights["loc"] * detection_loss
        watermarking_loss = self.weights["dec"] * decoding_loss

        # Return the losses in a structured format
        return {
            "perceptual_loss": perceptual_loss,
            "localization_loss": localization_loss,
            "watermarking_loss": watermarking_loss,
        }
