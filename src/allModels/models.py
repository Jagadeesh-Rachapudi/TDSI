from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.utils.utility_functions import find_least_important_components


class Latent2Msg(nn.Module):
    """
    Latent2Msg model: Extracts bits from the latent space based on the provided bit positions.
    """
    def __init__(self, latent_dim: int, msg_size: int = 32):
        super(Latent2Msg, self).__init__()
        self.latent_dim = latent_dim
        self.msg_size = msg_size

    def forward(self, latent_space: torch.Tensor, bit_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_space: Latent space representation (batch x latent_dim x time_steps).
            bit_positions: Indices of the positions to extract bits from (msg_size).

        Returns:
            extracted_bits: Extracted values from the latent space (batch x msg_size).
        """
        # Select values at bit positions
        extracted_bits = latent_space[:, bit_positions, 0]  # Select values directly at bit positions
        return extracted_bits


class AudioSealWM(nn.Module):
    """
    Generator model: Embeds the message into the audio and returns watermarked audio.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(AudioSealWM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

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
            updated_latent_space: Modified latent space with the message embedded.
        """
        # Scale the binary message into a range suitable for embedding
        scaled_message = (message * 2) - 1  # Convert {0, 1} to {-1, 1}

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

        # Modify the latent space with the scaled message
        updated_latent_space = latent_space.clone()
        for i, idx in enumerate(valid_indices):
            updated_latent_space[:, idx, :] = scaled_message[:, i].unsqueeze(-1)
        
        return updated_latent_space

    def forward(self, x: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        latent_space = self.encoder(x)
        updated_latent_space = self.embed_bits_in_latent_space(latent_space, message)
        watermarked_audio = self.decoder(updated_latent_space)
        return watermarked_audio


class Detector(nn.Module):
    """
    Detector model: Extracts the message from the watermarked audio.
    """
    def __init__(self, encoder: nn.Module, latent_dim: int, msg_size: int = 32):
        super(Detector, self).__init__()
        self.encoder = encoder
        self.latent2msg = Latent2Msg(latent_dim, msg_size)

    def forward(self, audio: torch.Tensor, bit_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Input audio signal.
            bit_positions: Indices of the least important components.

        Returns:
            extracted_message: Extracted message from the audio.
        """
        latent_space = self.encoder(audio)
        msg_logits = self.latent2msg(latent_space, bit_positions)

        # Convert extracted values back to binary {0, 1}
        extracted_message = (msg_logits > 0).float()  # Threshold at 0 since {-1, 1} is used for embedding
        return extracted_message
