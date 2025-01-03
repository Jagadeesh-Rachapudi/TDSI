import torch
import torch.nn as nn

import torch
import torch.nn as nn

def compute_detection_loss(positive, negative, mask=None, p_weight=1.0, n_weight=1.0):
    """
    Compute detection loss with optional mask.

    Args:
        positive: Model predictions for watermarked samples [bsz, 2+nbits, time_steps].
        negative: Model predictions for non-watermarked samples [bsz, 2+nbits, time_steps].
        mask: Optional mask for valid regions [bsz, 1, time_steps]. If None, considers all regions valid.
        p_weight: Weight for positive loss.
        n_weight: Weight for negative loss.

    Returns:
        Combined loss value.
    """
    # Define the loss function
    criterion = nn.NLLLoss()

    # Extract the first two channels for watermarked vs non-watermarked classification
    positive = positive[:, :2, :]  # Shape: [bsz, 2, time_steps]
    negative = negative[:, :2, :]  # Shape: [bsz, 2, time_steps]

    # Ground-truth labels
    classes_shape = positive[:, 0, :]  # Shape: [bsz, time_steps]
    pos_correct_classes = torch.ones_like(classes_shape, dtype=int)  # Class 1 for watermarked
    neg_correct_classes = torch.zeros_like(classes_shape, dtype=int)  # Class 0 for non-watermarked

    # Apply logarithmic transformation for NLLLoss
    positive = torch.log(positive)
    negative = torch.log(negative)

    # Check if a mask is provided
    if mask is not None:
        # Ensure mask has the correct shape
        assert mask.shape == (positive.shape[0], 1, positive.shape[2]), "Mask shape must be [bsz, 1, time_steps]."
        mask = mask[:, 0, :]  # Remove the singleton dimension for broadcasting

        # Apply mask to the ground-truth classes
        pos_correct_classes = pos_correct_classes * mask.to(dtype=torch.int32)
        neg_correct_classes = neg_correct_classes * mask.to(dtype=torch.int32)

        # Compute positive loss with masked regions
        loss_p = p_weight * criterion(positive.transpose(1, 2), pos_correct_classes)
        return loss_p  # Only positive loss is computed when mask is applied
    else:
        # Compute both positive and negative losses without masking
        loss_p = p_weight * criterion(positive.transpose(1, 2), pos_correct_classes)
        loss_n = n_weight * criterion(negative.transpose(1, 2), neg_correct_classes)
        return loss_p + loss_n


def compute_decoding_loss(decoded_logits, message, temperature=1.0, loss_type="bce"):
    """
    Compute decoding loss for predicted message logits and ground-truth message.
    
    Args:
        decoded_logits: Predicted message logits [bsz, nbits] or [bsz, nbits, time_steps].
        message: Ground-truth binary message [bsz, nbits].
        temperature: Temperature for logits scaling.
        loss_type: Type of loss to compute ("bce" or "mse").
    
    Returns:
        Loss value.
    """
    if message is None or message.size(0) == 0:
        return torch.tensor(0.0)

    # Ensure decoded_logits has three dimensions
    if len(decoded_logits.shape) == 2:  # [bsz, nbits]
        decoded_logits = decoded_logits.unsqueeze(-1)  # Add time_steps dimension -> [bsz, nbits, 1]

    # Expand ground-truth message to match time_steps in decoded_logits
    message = message.unsqueeze(-1).repeat(1, 1, decoded_logits.shape[2])  # [bsz, nbits, time_steps]

    # Scale logits with temperature
    decoded_logits = decoded_logits / temperature

    # Compute loss
    if loss_type == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(decoded_logits, message.float())
    elif loss_type == "mse":
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(decoded_logits, message.float())
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return loss


def compute_perceptual_loss(audio1, audio2, mask=None):
    """
    Compute perceptual loss based on similarity of two audio inputs.

    Args:
        audio1: First audio tensor [bsz, time_steps].
        audio2: Second audio tensor [bsz, time_steps].
        mask: Optional mask tensor [bsz, 1, time_steps] to select valid regions.

    Returns:
        Perceptual loss value.
    """
    assert audio1.shape == audio2.shape, "Audio inputs must have the same shape."
    
    # Apply mask if provided
    if mask is not None:
        assert mask.shape == audio1.unsqueeze(1).shape, "Mask shape must match [bsz, 1, time_steps]."
        audio1 = torch.masked_select(audio1.unsqueeze(1), mask == 1).reshape(audio1.shape)
        audio2 = torch.masked_select(audio2.unsqueeze(1), mask == 1).reshape(audio2.shape)
    
    # Compute mean squared error as a basic perceptual loss
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(audio1, audio2)
    
    # Add optional perceptual distance metric (e.g., cosine similarity)
    cosine_sim = torch.nn.functional.cosine_similarity(audio1, audio2, dim=-1)
    perceptual_loss = mse_loss * (1 - cosine_sim.mean().clamp(0, 1))
    
    return perceptual_loss
