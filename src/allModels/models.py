from typing import Optional, Tuple

import julius
import torch

from src.allModels.SEANet import SEANetEncoderKeepDimension
from src.utils.utility_functions import find_least_important_components

class MsgProcessor(torch.nn.Module):
    """
    Apply the secret message to the encoder output.
    Args:
        nbits: Number of bits used to generate the message. Must be non-zero
        hidden_size: Dimension of the encoder output
    """

    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        assert nbits > 0, "MsgProcessor should not be built in 0bit watermarking"
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_processor = torch.nn.Embedding(2 * nbits, hidden_size)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """
        Build the embedding map: 2 x k -> k x h, then sum on the first dim
        Args:
            hidden: The encoder output, size: batch x hidden x frames
            msg: The secret message, size: batch x k
        """
        # create indices to take from embedding layer
        indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)  # k: 0 2 4 ... 2k
        indices = indices.repeat(msg.shape[0], 1)  # b x k
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)  # b x k -> b x k x h
        msg_aux = msg_aux.sum(dim=-2)  # b x k x h -> b x h
        msg_aux = msg_aux.unsqueeze(-1).repeat(
            1, 1, hidden.shape[2]
        )  # b x h -> b x h x t/f
        hidden = hidden + msg_aux  # -> b x h x t/f
        return hidden



from typing import Optional, Tuple
import torch
import julius
from src.utils.utility_functions import find_least_important_components


class AudioSealWM(torch.nn.Module):
    """
    Perform audio reconstruction and message embedding.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def embed_bits_in_latent_space(
        self,
        latent_space: torch.Tensor,
        message: torch.Tensor,
        num_bits: int = 33,
    ) -> torch.Tensor:
        """
        Embed a message into the latent space by replacing values at the least important components.

        Args:
            latent_space: Latent representation of the audio, size: batch x features x time_steps
            message: Binary message to embed, size: num_bits
            num_bits: Number of bits to embed (default: 33)

        Returns:
            updated_latent_space: Latent space with the message embedded
        """
        # Convert latent space to numpy for PCA analysis
        latent_space_np = latent_space.detach().cpu().numpy().reshape(-1, latent_space.shape[1])

        # Find least important components for embedding
        least_important_indices, _ = find_least_important_components(latent_space_np, num_bits)

        # Embed the message in the latent space
        updated_latent_space = latent_space.clone()
        for i, idx in enumerate(least_important_indices):
            updated_latent_space[:, idx, :] = message[i].item()

        return updated_latent_space

    def reconstruct_with_message(
        self,
        x: torch.Tensor,
        message: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Embed a message into the latent space and reconstruct the audio.

        Args:
            x: Input audio signal, size: batch x frames
            message: Binary message to embed, size: num_bits
            sample_rate: The sample rate of the input audio (default 16kHz)

        Returns:
            reconstructed_audio: Reconstructed audio signal with the message embedded
        """
        length = x.size(-1)
        if sample_rate is None:
            sample_rate = 16_000
        assert sample_rate
        if sample_rate != 16000:
            x = julius.resample_frac(x, old_sr=sample_rate, new_sr=16000)
        
        # Step 1: Encode the audio into the latent space
        latent_space = self.encoder(x)

        # Step 2: Embed the message into the latent space
        updated_latent_space = self.embed_bits_in_latent_space(latent_space, message)

        # Step 3: Decode the updated latent space to reconstruct the audio
        reconstructed_audio = self.decoder(updated_latent_space)

        if sample_rate != 16000:
            reconstructed_audio = julius.resample_frac(
                reconstructed_audio, old_sr=16000, new_sr=sample_rate
            )

        return reconstructed_audio[..., :length]  # Trim output to input length

    def forward(
        self,
        x: torch.Tensor,
        message: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Combine audio and message, and reproduce the audio.

        Args:
            x: Input audio signal, size: batch x frames
            message: Binary message to embed, size: num_bits
            sample_rate: Sample rate of the input audio

        Returns:
            reconstructed_audio: Reconstructed audio signal with the message embedded
        """
        return self.reconstruct_with_message(x, message, sample_rate=sample_rate)

    

class AudioSealDetector(torch.nn.Module):
    """
    Detect the watermarking from an audio signal
    Args:
        SEANetEncoderKeepDimension (_type_): _description_
        nbits (int): The number of bits in the secret message. The result will have size
            of 2 + nbits, where the first two items indicate the possibilities of the
            audio being watermarked (positive / negative scores), he rest is used to decode
            the secret message. In 0bit watermarking (no secret message), the detector just
            returns 2 values.
    """

    def __init__(self, *args, nbits: int = 0, **kwargs):
        super().__init__()
        encoder = SEANetEncoderKeepDimension(*args, **kwargs)
        last_layer = torch.nn.Conv1d(encoder.output_dim, 2 + nbits, 1)
        self.detector = torch.nn.Sequential(encoder, last_layer)
        self.nbits = nbits

    def detect_watermark(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
        message_threshold: float = 0.5,
    ) -> Tuple[float, torch.Tensor]:
        """
        A convenience function that returns a probability of an audio being watermarked,
        together with its message in n-bits (binary) format. If the audio is not watermarked,
        the message will be random.
        Args:
            x: Audio signal, size: batch x frames
            sample_rate: The sample rate of the input audio
            message_threshold: threshold used to convert the watermark output (probability
                of each bits being 0 or 1) into the binary n-bit message.
        """
        if sample_rate is None:
            sample_rate = 16_000
        result, message = self.forward(x, sample_rate=sample_rate)  # b x 2+nbits
        detected = (
            torch.count_nonzero(torch.gt(result[:, 1, :], 0.5)) / result.shape[-1]
        )
        detect_prob = detected.cpu().item()  # type: ignore
        message = torch.gt(message, message_threshold).int()
        return detect_prob, message

    def decode_message(self, result: torch.Tensor) -> torch.Tensor:
        """
        Decode the message from the watermark result (batch x nbits x frames)
        Args:
            result: watermark result (batch x nbits x frames)
        Returns:
            The message of size batch x nbits, indicating probability of 1 for each bit
        """
        assert (result.dim() > 2 and result.shape[1] == self.nbits) or (
            self.dim() == 2 and result.shape[0] == self.nbits
        ), f"Expect message of size [,{self.nbits}, frames] (get {result.size()})"
        decoded_message = result.mean(dim=-1)
        return torch.sigmoid(decoded_message)

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect the watermarks from the audio signal
        Args:
            x: Audio signal, size batch x frames
            sample_rate: The sample rate of the input audio
        """
        if sample_rate is None:
            sample_rate = 16_000
        assert sample_rate
        if sample_rate != 16000:
            x = julius.resample_frac(x, old_sr=sample_rate, new_sr=16000)
        result = self.detector(x)  # b x 2+nbits
        # hardcode softmax on 2 first units used for detection
        result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)
        message = self.decode_message(result[:, 2:, :])
        return result[:, :2, :], message