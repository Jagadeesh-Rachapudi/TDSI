from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.utils.utility_functions import find_least_important_components

class Latent2Msg(nn.Module):
    """
    Latent2Msg model dynamically extracts bits from the latent space
    based on the provided bit positions.
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
            msg_logits: Logits corresponding to the extracted bits (batch x msg_size).
        """
        selected_latent_space = latent_space[:, bit_positions, :]  # Shape: (batch x msg_size x time_steps)
        msg_logits = torch.mean(selected_latent_space, dim=-1)  # Shape: (batch x msg_size)
        return msg_logits


class AudioSealWM(nn.Module):
    """
    Generator model: Embed the message into the audio and return watermarked audio.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def embed_bits_in_latent_space(
        self, latent_space: torch.Tensor, message: torch.Tensor, num_bits: int = 32
    ) -> torch.Tensor:
        # Perform PCA to find least important components
        latent_space_np = latent_space.detach().cpu().numpy().reshape(-1, latent_space.shape[1])
        least_important_indices, _ = find_least_important_components(latent_space_np, num_bits)

        max_index = latent_space.size(1)
        valid_indices = [idx for idx in least_important_indices if idx < max_index]

        if len(valid_indices) < num_bits:
            raise ValueError(
                f"Not enough valid indices to embed {num_bits} bits. "
                f"Only {len(valid_indices)} indices available."
            )

        updated_latent_space = latent_space.clone()
        for i, idx in enumerate(valid_indices):
            updated_latent_space[:, idx, :] = message[:, i].unsqueeze(-1)

        return updated_latent_space

    def forward(self, x: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        latent_space = self.encoder(x)
        updated_latent_space = self.embed_bits_in_latent_space(latent_space, message)
        watermarked_audio = self.decoder(updated_latent_space)
        return watermarked_audio


class Detector(nn.Module):
    """
    Detector model: Extract the message from the watermarked audio.
    """
    def __init__(self, encoder: nn.Module, latent_dim: int, msg_size: int = 32):
        super(Detector, self).__init__()
        self.encoder = encoder
        self.latent2msg = Latent2Msg(latent_dim, msg_size)

    def forward(self, audio: torch.Tensor, bit_positions: torch.Tensor) -> torch.Tensor:
        latent_space = self.encoder(audio)
        msg_logits = self.latent2msg(latent_space, bit_positions)
        extracted_message = (msg_logits > 0.5).float()
        return extracted_message
