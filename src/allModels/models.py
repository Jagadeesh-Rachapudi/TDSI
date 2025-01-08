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
        extracted_bits = latent_space[:, bit_positions, 0]  # Select values directly at bit positions
        print("\n[Detector] Extracted Bits at Bit Positions:", extracted_bits)
        return extracted_bits


class AudioSealWM(nn.Module):
    """
    Generator model: Embed the message into the audio and return watermarked audio.
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
            updated_latent_space: Modified latent space with the message embedded.
        """
        # Scale the binary message into a range suitable for embedding
        mean_variance = latent_space.var().item() ** 0.5
        scaled_message = torch.where(
            message == 0, 
            torch.tensor(-mean_variance * self.scale_factor, device=latent_space.device),  # Class 0
            torch.tensor(mean_variance * self.scale_factor, device=latent_space.device)   # Class 1
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

        # Print latent space values at bit positions before embedding
        bits_before = latent_space[:, valid_indices, 0]
        print("\n[Generator] Latent Space Values at Bit Positions (Before Embedding):", bits_before)

        # Modify the latent space with the scaled message
        updated_latent_space = latent_space.clone()
        for i, idx in enumerate(valid_indices):
            updated_latent_space[:, idx, :] = scaled_message[:, i].unsqueeze(-1)

        # Normalize latent space variance
        updated_latent_space = self._normalize_variance(latent_space, updated_latent_space)

        # Print latent space values at bit positions after embedding
        bits_after = updated_latent_space[:, valid_indices, 0]
        print("\n[Generator] Latent Space Values at Bit Positions (After Embedding):", bits_after)

        return updated_latent_space

    def _normalize_variance(self, original_latent_space, updated_latent_space):
        """
        Normalize the variance of the latent space after embedding.
        """
        original_variance = original_latent_space.var(dim=1, keepdim=True)
        updated_variance = updated_latent_space.var(dim=1, keepdim=True)
        scale = torch.sqrt(original_variance / updated_variance + 1e-6)
        return updated_latent_space * scale

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
        extracted_message = (msg_logits > 0).float()  # Threshold at 0 since scaled values are used for embedding
        return extracted_message
