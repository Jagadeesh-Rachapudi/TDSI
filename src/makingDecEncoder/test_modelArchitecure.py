from typing import Tuple
import torch
import torch.nn as nn
from src.utils.utility_functions import find_least_important_components

class AudioSealWM(nn.Module):
    """
    Generator model: Embeds the message into the audio and returns watermarked audio,
    original latent space, and the embedded latent space.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, scale_factor: float = 1.5):
        super(AudioSealWM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.scale_factor = scale_factor  # Scaling factor for embedding bits

    def embed_bits_in_latent_space(
        self, latent_space: torch.Tensor, message: torch.Tensor, num_bits: int = 32
    ) -> torch.Tensor:
        """
        Embeds a binary message into the latent space by modifying the least important components.

        Args:
            latent_space: Latent representation of the audio (batch x latent_dim x time_steps).
            message: Binary message to embed (batch x num_bits).
            num_bits: Number of bits to embed (default=32).

        Returns:
            embedded_latent_space: Modified latent space with the message embedded.
        """
        # Scale the binary message into a range suitable for embedding
        mean_variance = latent_space.var().item() ** 0.5
        scaled_message = torch.where(
            message == 0,
            torch.tensor(-mean_variance * self.scale_factor, device=latent_space.device),  # Class 0
            torch.tensor(mean_variance * self.scale_factor, device=latent_space.device)    # Class 1
        )

        # Find least important components
        latent_space_np = latent_space.detach().cpu().numpy().reshape(-1, latent_space.shape[1])
        least_important_indices, _ = find_least_important_components(latent_space_np, num_bits)

        # Validate bit positions
        max_index = latent_space.size(1)
        valid_indices = [idx for idx in least_important_indices if idx < max_index]
        if len(valid_indices) < num_bits:
            raise ValueError(
                f"Not enough valid indices to embed {num_bits} bits. "
                f"Only {len(valid_indices)} indices available."
            )

        # Initialize a new latent space to avoid in-place modifications
        embedded_latent_space = latent_space.clone()

        # Modify the latent space with the scaled message
        for i, idx in enumerate(valid_indices):
            updated_values = embedded_latent_space[:, idx, :] + scaled_message[:, i].unsqueeze(-1)
            # Using slicing and concatenation, avoid in-place operations
            embedded_latent_space = torch.cat(
                (embedded_latent_space[:, :idx, :], updated_values.unsqueeze(1), embedded_latent_space[:, idx + 1 :, :]),
                dim=1,
            )

        return embedded_latent_space

    def forward(self, x: torch.Tensor, message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the generator.

        Args:
            x: Input audio signal.
            message: Binary message to embed.

        Returns:
            watermarked_audio: Watermarked audio after embedding the message.
            original_latent_space: Latent space of the original audio.
            embedded_latent_space: Latent space with the message embedded.
        """
        # Pass audio through encoder
        original_latent_space = self.encoder(x)
        # Embed the binary message into the latent space
        embedded_latent_space = self.embed_bits_in_latent_space(original_latent_space, message)
        # Decode the embedded latent space to obtain the watermarked audio
        watermarked_audio = self.decoder(embedded_latent_space)

        # Ensure the output dimensions match the input dimensions
        if watermarked_audio.shape[-1] > x.shape[-1]:
            watermarked_audio = watermarked_audio[..., :x.shape[-1]]  # Crop to match input size
        elif watermarked_audio.shape[-1] < x.shape[-1]:
            padding = x.shape[-1] - watermarked_audio.shape[-1]
            watermarked_audio = torch.nn.functional.pad(watermarked_audio, (0, padding))  # Pad to match input size

        return watermarked_audio, original_latent_space, embedded_latent_space


class Detector(nn.Module):
    """
    Detector model: Encodes the audio and returns the latent space.
    """
    def __init__(self, encoder: nn.Module):
        super(Detector, self).__init__()
        self.encoder = encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Input audio signal.

        Returns:
            latent_space: Latent space representation of the audio.
        """
        latent_space = self.encoder(audio)
        return latent_space



class Detector(nn.Module):
    """
    Detector model: Encodes the audio and returns the latent space.
    """
    def __init__(self, encoder: nn.Module):
        super(Detector, self).__init__()
        self.encoder = encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Input audio signal.

        Returns:
            latent_space: Latent space representation of the audio.
        """
        latent_space = self.encoder(audio)
        return latent_space
