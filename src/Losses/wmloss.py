import torch
import torch.nn as nn

def compute_detection_loss(positive, negative, mask, p_weight=1.0, n_weight=1.0):
    """
    Compute detection loss.

    Args:
        positive: Model predictions for watermarked samples [bsz, 2+nbits, time_steps].
        negative: Model predictions for non-watermarked samples [bsz, 2+nbits, time_steps].
        mask: Mask for valid regions [bsz, 1, time_steps].
        p_weight: Weight for positive loss.
        n_weight: Weight for negative loss.

    Returns:
        Loss value.
    """
    criterion = nn.NLLLoss()
    positive = positive[:, :2, :]  # b 2+nbits t -> b 2 t
    negative = negative[:, :2, :]  # b 2+nbits t -> b 2 t

    classes_shape = positive[:, 0, :]  # shape [bsz, time_steps]
    pos_correct_classes = torch.ones_like(classes_shape, dtype=int)
    neg_correct_classes = torch.zeros_like(classes_shape, dtype=int)

    positive = torch.log(positive)
    negative = torch.log(negative)

    if not torch.all(mask == 1):
        pos_correct_classes = pos_correct_classes * mask[:, 0, :].to(int)
        loss_p = p_weight * criterion(positive, pos_correct_classes)
        return loss_p
    else:
        loss_p = p_weight * criterion(positive, pos_correct_classes)
        loss_n = n_weight * criterion(negative, neg_correct_classes)
        return loss_p + loss_n


def compute_decoding_loss(positive, mask, message, temperature, loss_type="bce"):
    """
    Compute decoding loss.

    Args:
        positive: Predictions for watermarked samples [bsz, 2+nbits, time_steps].
        mask: Mask for valid regions [bsz, 1, time_steps].
        message: Original watermark message [bsz, nbits].
        temperature: Temperature for loss computation.
        loss_type: Type of loss ("bce" or "mse").

    Returns:
        Loss value.
    """
    if message.size(0) == 0:
        return torch.tensor(0.0)

    positive = positive[:, 2:, :]  # b 2+nbits t -> b nbits t
    assert (
        positive.shape[1] == message.shape[1]
    ), "In decoding loss: enc and dec don't share nbits, are you using multi-bit?"

    new_shape = [positive.shape[0], positive.shape[1], -1]  # b nbits -1
    positive = torch.masked_select(positive, mask == 1).reshape(new_shape)

    message = message.unsqueeze(-1).repeat(1, 1, positive.shape[2])  # b nbits -> b nbits t

    if loss_type == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(positive / temperature, message.float())
    elif loss_type == "mse":
        loss_fn = nn.MSELoss()
        loss = loss_fn(positive / temperature, message.float())

    return loss


def compute_perceptual_loss(audio1, audio2):
    """
    Compute perceptual loss based on similarity of two audio inputs.

    Args:
        audio1: First audio tensor [bsz, time_steps].
        audio2: Second audio tensor [bsz, time_steps].

    Returns:
        Perceptual loss value.
    """
    assert audio1.shape == audio2.shape, "Audio inputs must have the same shape."
    
    # Compute mean squared error as a basic perceptual loss
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(audio1, audio2)
    
    # Add optional perceptual distance metric (e.g., cosine similarity)
    cosine_sim = torch.nn.functional.cosine_similarity(audio1, audio2, dim=-1)
    perceptual_loss = mse_loss * (1 - cosine_sim.mean())
    
    return perceptual_loss
