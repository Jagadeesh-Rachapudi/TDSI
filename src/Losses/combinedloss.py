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
        self.detection_loss = WMDetectionLoss(p_weight=10.0, n_weight=10.0)
        self.decoding_loss = WMMbLoss(temperature=1.0, loss_type="bce")
        self.sisnr_loss = SISNR(sample_rate=sample_rate)

        # Weights for each loss
        self.weights = {
            "loudness": 0.1,
            "spectrogram": 1.0,
            "stft": 1.0,
            "detection": 10.0,
            "decoding": 1.0,
            "sisnr": 0.1,
        }

    def forward(self, watermarked_audio, original_audio, message, mask):
        # Compute individual losses
        loudness_loss = self.loudness_loss(watermarked_audio, original_audio)
        spectrogram_loss = self.spectrogram_loss(watermarked_audio, original_audio)
        stft_loss = self.stft_loss(watermarked_audio, original_audio)
        detection_loss = self.detection_loss(watermarked_audio, original_audio, mask)
        print("decoding is about to call")
        decoding_loss = self.decoding_loss(watermarked_audio, original_audio, mask, message)
        print("decoding is about to call")
        sisnr_loss = self.sisnr_loss(watermarked_audio, original_audio)

        # Weighted combination of losses
        total_loss = (
            self.weights["loudness"] * loudness_loss +
            self.weights["spectrogram"] * spectrogram_loss +
            self.weights["stft"] * stft_loss +
            self.weights["detection"] * detection_loss +
            self.weights["decoding"] * decoding_loss +
            self.weights["sisnr"] * sisnr_loss
        )

        return total_loss
