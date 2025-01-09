import os
import time
import torch
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from src.utils.utility_functions import find_least_important_components


def penalty_based_bit_loss(predicted, ground_truth, threshold=0.5):
    """
    Computes penalty-based bit-wise loss:
    - Heavy penalty for incorrect predictions far from the threshold.
    - Less penalty for correct predictions near the threshold.

    Args:
        predicted (Tensor): Probabilities for predicted bits (values between 0 and 1).
        ground_truth (Tensor): Ground truth bits (0 or 1).
        threshold (float): The threshold for classification (default=0.5).

    Returns:
        loss (Tensor): The computed penalty-based loss.
    """
    # Heavy penalty when predictions are incorrect
    heavy_penalty = torch.abs(predicted - ground_truth) * 10.0

    # Less penalty when predictions are correct
    less_penalty = torch.abs(predicted - ground_truth) * 1.0

    # Condition for applying heavy penalty
    condition = (predicted > threshold) & (ground_truth == 0) | (predicted <= threshold) & (ground_truth == 1)

    # Apply penalty
    return torch.where(condition, heavy_penalty, less_penalty).mean()


def train(
    generator,
    detector_encoder,
    train_loader,
    val_loader,
    lr_g=1e-4,
    lr_d=1e-4,
    device="cuda",
    num_epochs=100,
    checkpoint_path="./checkpoints",
    log_path="./logs/losses.csv",
):
    # Ensure checkpoint and log directories exist
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, weight_decay=1e-4)
    optimizer_d = torch.optim.Adam(detector_encoder.parameters(), lr=lr_d, weight_decay=1e-4)

    # Track the best losses for saving the best model
    lowest_val_detector_loss = float('inf')

    print("Starting training...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        generator.train()
        detector_encoder.train()

        train_detector_loss = 0
        total_train_batches = len(train_loader)

        # Training loop
        for batch_idx, (audio, labels) in enumerate(train_loader):
            # Move audio and labels to device
            audio = audio.to(device).unsqueeze(1)  # Add channel dimension if needed
            labels = labels.to(device)

            # Convert labels to 32-bit binary messages
            message = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).float().to(device)

            # Forward pass through the generator
            watermarked_audio, embedded_ls, original_ls = generator(audio, message)

            # Forward pass through the detector
            probable_embedded_ls = detector_encoder(watermarked_audio)

            # Find bit positions using PCA
            original_ls_np = original_ls.detach().cpu().numpy().reshape(-1, embedded_ls.shape[1])
            bit_positions, _ = find_least_important_components(original_ls_np, message.size(1))
            bit_positions_tensor = torch.tensor(bit_positions, device=device)

            # Extract probable embedded bits
            extracted_bits = torch.sigmoid(probable_embedded_ls[:, bit_positions_tensor, :]).mean(dim=-1)

            # Compute penalty-based bit-wise loss
            bit_loss = penalty_based_bit_loss(extracted_bits, message)

            # Backward pass for the detector
            optimizer_d.zero_grad()
            bit_loss.backward()
            clip_grad_norm_(detector_encoder.parameters(), max_norm=5)
            optimizer_d.step()

            # Accumulate losses
            train_detector_loss += bit_loss.item()

        # Compute average training loss
        avg_train_detector_loss = train_detector_loss / total_train_batches
        epoch_duration = time.time() - epoch_start_time

        print(
            f"Epoch {epoch + 1} Summary: "
            f"Detector Loss: {avg_train_detector_loss:.4f}, "
            f"Duration: {epoch_duration:.2f}s"
        )

        # Validation loop
        generator.eval()
        detector_encoder.eval()
        val_detector_loss = 0
        total_val_batches = len(val_loader)

        with torch.no_grad():
            for val_audio, val_labels in val_loader:
                val_audio = val_audio.to(device).unsqueeze(1)
                val_labels = val_labels.to(device)

                # Convert labels to 32-bit binary messages
                val_message = torch.stack([(val_labels >> i) & 1 for i in range(32)], dim=-1).float()

                # Forward pass through generator
                val_watermarked_audio, val_embedded_ls, val_original_ls = generator(val_audio, val_message)

                # Forward pass through detector
                val_probable_embedded_ls = detector_encoder(val_watermarked_audio)

                # Find bit positions using PCA
                val_original_ls_np = val_original_ls.detach().cpu().numpy().reshape(-1, val_embedded_ls.shape[1])
                bit_positions, _ = find_least_important_components(val_original_ls_np, val_message.size(1))
                bit_positions_tensor = torch.tensor(bit_positions, device=device)

                # Extract probable embedded bits
                val_extracted_bits = torch.sigmoid(
                    val_probable_embedded_ls[:, bit_positions_tensor, :].clone()
                ).mean(dim=-1)

                # Compute penalty-based bit-wise loss
                val_bit_loss = penalty_based_bit_loss(val_extracted_bits, val_message)

                # Accumulate validation losses
                val_detector_loss += val_bit_loss.item()

        avg_val_detector_loss = val_detector_loss / total_val_batches

        print(
            f"Validation Loss: Detector Loss: {avg_val_detector_loss:.4f}"
        )

        # Save the best model based on validation detector loss
        if avg_val_detector_loss < lowest_val_detector_loss:
            best_model_file = f"{checkpoint_path}/BestDecEncoder.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "detector_state_dict": detector_encoder.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                },
                best_model_file,
            )
            print(f"\n[INFO] Best model saved as: {best_model_file}")
            lowest_val_detector_loss = avg_val_detector_loss

        # Clear GPU memory
        torch.cuda.empty_cache()

    print("Training completed.")
