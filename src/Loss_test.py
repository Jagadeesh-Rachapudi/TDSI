import torch
from Losses.combinedloss import CombinedLoss  # Import your combined loss function
from Losses.loudnessloss import TFLoudnessRatio
from Losses.wmloss import WMDetectionLoss, WMMbLoss
from Losses.specloss import MultiScaleMelSpectrogramLoss
from Losses.stftloss import MRSTFTLoss
from Losses.sisnr import SISNR

def test_losses():
    # Configuration
    sample_rate = 16000  # Sample rate of the audio
    nbits = 16  # Number of bits for watermarking
    batch_size = 4  # Number of samples in a batch
    audio_length = sample_rate * 1  # 1 second of audio (16,000 samples)

    # Generate random audio (batch_size, 1, audio_length)
    original_audio = torch.randn(batch_size, 1, audio_length)
    watermarked_audio = original_audio + 0.01 * torch.randn_like(original_audio)  # Add small noise to simulate watermark
    message = torch.randint(0, 2, (batch_size, nbits))  # Random binary message

    # Initialize individual loss functions
    loudness_loss = TFLoudnessRatio(sample_rate=sample_rate)
    detection_loss = WMDetectionLoss(p_weight=10.0, n_weight=10.0)
    decoding_loss = WMMbLoss(temperature=1.0, loss_type="bce")
    spectrogram_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
    stft_loss = MRSTFTLoss()
    sisnr_loss = SISNR(sample_rate=sample_rate)

    # Test individual losses
    print("Testing Individual Loss Functions...")
    loud_loss = loudness_loss(watermarked_audio, original_audio)
    print(f"Loudness Loss: {loud_loss.item():.4f}")

    spec_loss = spectrogram_loss(watermarked_audio, original_audio)
    print(f"Spectrogram Loss: {spec_loss.item():.4f}")

    stft_loss_value = stft_loss(watermarked_audio, original_audio)
    print(f"STFT Loss: {stft_loss_value.item():.4f}")

    detect_loss = detection_loss(watermarked_audio, message)
    print(f"Detection Loss: {detect_loss.item():.4f}")

    decode_loss = decoding_loss(watermarked_audio, message)
    print(f"Decoding Loss: {decode_loss.item():.4f}")

    sisnr_loss_value = sisnr_loss(watermarked_audio, original_audio)
    print(f"SI-SNR Loss: {sisnr_loss_value.item():.4f}")

    # Initialize the CombinedLoss
    print("\nTesting Combined Loss Function...")
    combined_loss_fn = CombinedLoss(sample_rate=sample_rate, nbits=nbits, use_balancer=True)
    combined_loss_value = combined_loss_fn(watermarked_audio, original_audio, message)
    print(f"Combined Loss: {combined_loss_value.item():.4f}")

if __name__ == "__main__":
    test_losses()
