from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.utils.utility_functions import find_least_important_components

class Latent2Msg(nn.Module):
    """
    Extracts bits from the latent space based on the provided bit positions.
    """
    def __init__(self, latent_dim: int, msg_size: int = 32):
        super(Latent2Msg, self).__init__()
        self.latent_dim = latent_dim
        self.msg_size = msg_size

    def forward(self, latent_space: torch.Tensor, bit_positions: torch.Tensor) -> torch.Tensor:
        """
        Extracts bits from the latent space without aggregation.

        Args:
            latent_space: Latent space tensor (batch x latent_dim x time_steps).
            bit_positions: Indices of the positions to extract bits from (msg_size).

        Returns:
            extracted_bits: Tensor containing the extracted bits (batch x msg_size).
        """
        extracted_bits = latent_space[:, bit_positions, 0]  # Select direct values
        print("Extracted Bits (Direct Values):", extracted_bits)
        return extracted_bits


class AudioSealWM(nn.Module):
    """
    Generator model: Embed the message into the audio and return watermarked audio.
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

        # Print latent space values at bit positions BEFORE embedding
        print("Latent Space Values at Bit Positions (Before Embedding):")
        print(latent_space[:, valid_indices, 0])

        # Modify the latent space with the message
        updated_latent_space = latent_space.clone()
        for i, idx in enumerate(valid_indices):
            updated_latent_space[:, idx, :] = message[:, i].unsqueeze(-1)

        # Print latent space values at bit positions AFTER embedding
        print("Latent Space Values at Bit Positions (After Embedding):")
        print(updated_latent_space[:, valid_indices, 0])

        # Save the latent space for later use
        # torch.save(updated_latent_space, "saved_latent_space.pt")
        # print("Latent space after embedding saved to 'saved_latent_space.pt'")
        return updated_latent_space


    def forward(self, x: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Forward method to embed the message and produce the watermarked audio.

        Args:
            x: Input audio tensor (batch x time_steps).
            message: Binary message to embed (batch x num_bits).

        Returns:
            watermarked_audio: Watermarked audio tensor (batch x time_steps).
        """
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
        Forward method to extract the message from watermarked audio.

        Args:
            audio: Watermarked audio tensor (batch x time_steps).
            bit_positions: Indices of the positions to extract bits from (msg_size).

        Returns:
            extracted_message: Extracted binary message (batch x msg_size).
        """
        latent_space = self.encoder(audio)
        msg_logits = self.latent2msg(latent_space, bit_positions)
        extracted_message = (msg_logits > 0.5).float()
        return extracted_message
