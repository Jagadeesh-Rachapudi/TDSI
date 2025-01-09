import os
import time
import torch
from pathlib import Path
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from src.utils.utility_functions import find_least_important_components


def train(
    generator,
    detector_encoder,
    train_loader,
    val_loader,
    lr_g=1e-4,
    lr_d=1e-4,
    device=None,
    num_epochs=100,
    compute_perceptual_loss=None,
    checkpoint_path="./checkpoints",
    log_path="./logs/losses.csv",
    update_csv=None,
    initialize_csv=None,
    scheduler=None,
):
    # Ensure checkpoint and log directories exist
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    initialize_csv(log_path)

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, weight_decay=1e-4)
    optimizer_d = torch.optim.Adam(detector_encoder.parameters(), lr=lr_d, weight_decay=1e-4)

    print("Starting training...")

    # Initialize variables to track the lowest training and validation loss
    lowest_training_loss = float("inf")
    lowest_validation_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        generator.train()
        detector_encoder.train()

        train_loss_g, train_loss_d, train_embedded_loss, train_latent_loss = 0, 0, 0, 0
        total_train_batches = len(train_loader)

        # Training loop
        for batch_idx, (audio, labels) in enumerate(train_loader):
            # Move audio and labels to device
            audio = audio.to(device).unsqueeze(1)  # Add channel dimension
            labels = labels.to(device)

            # Convert labels to 32-bit binary messages
            message = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).float()

            # Forward pass: Generate watermarked audio and latent spaces
            watermarked_audio, embedded_ls, original_ls = generator(audio, message)

            # Compute generator loss (MSE between watermarked and original audio)
            gen_audio_loss = (
                compute_perceptual_loss(audio, watermarked_audio)
                if compute_perceptual_loss
                else torch.nn.functional.mse_loss(audio, watermarked_audio)
            )

            # Backward pass for generator
            optimizer_g.zero_grad()
            gen_audio_loss.backward()
            clip_grad_norm_(generator.parameters(), max_norm=5)
            optimizer_g.step()

            # Forward pass for detector
            probable_embedded_ls = detector_encoder(watermarked_audio)

            # Find least important components in the original latent space
            original_ls_np = original_ls.detach().cpu().numpy().reshape(-1, original_ls.shape[1])
            bit_positions, _ = find_least_important_components(original_ls_np, num_bits=32)

            # Compute detector loss
            weights = torch.ones_like(original_ls, device=device)  # Start with uniform weights
            weights[:, bit_positions, :] *= 10  # Increase weight for embedded bits

            # Split losses for embedded and latent regions
            embedded_loss = torch.mean(
                weights[:, bit_positions, :] * (embedded_ls[:, bit_positions, :] - probable_embedded_ls[:, bit_positions, :]).pow(2)
            )
            latent_loss = torch.mean(
                (embedded_ls - probable_embedded_ls).pow(2)
            )

            detector_loss = embedded_loss + latent_loss

            # Backward pass for detector
            optimizer_d.zero_grad()
            detector_loss.backward()
            clip_grad_norm_(detector_encoder.parameters(), max_norm=5)
            optimizer_d.step()

            # Accumulate losses
            train_loss_g += gen_audio_loss.item() / len(audio)  # Divide by batch size
            train_loss_d += detector_loss.item() / len(audio)  # Divide by batch size
            train_embedded_loss += embedded_loss.item() / len(audio)
            train_latent_loss += latent_loss.item() / len(audio)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Batch {batch_idx + 1}/{total_train_batches} - "
                    f"Generator Loss: {gen_audio_loss.item():.4f} - "
                    f"Detector Loss: {detector_loss.item():.4f} - "
                    f"Embedded Loss: {embedded_loss.item():.4f} - "
                    f"Latent Loss: {latent_loss.item():.4f}"
                )

        # Compute average training loss
        avg_train_loss_g = train_loss_g / total_train_batches
        avg_train_loss_d = train_loss_d / total_train_batches
        avg_train_embedded_loss = train_embedded_loss / total_train_batches
        avg_train_latent_loss = train_latent_loss / total_train_batches
        epoch_duration = time.time() - epoch_start_time

        print(
            f"Epoch {epoch + 1} Summary: "
            f"Generator Loss: {avg_train_loss_g:.4f}, "
            f"Detector Loss: {avg_train_loss_d:.4f}, "
            f"Embedded Loss: {avg_train_embedded_loss:.4f}, "
            f"Latent Loss: {avg_train_latent_loss:.4f}, "
            f"Duration: {epoch_duration:.2f}s"
        )

        # Validation loop
        generator.eval()
        detector_encoder.eval()
        val_loss_g, val_loss_d, val_embedded_loss, val_latent_loss = 0, 0, 0, 0
        total_val_batches = len(val_loader)

        with torch.no_grad():
            for val_audio, val_labels in val_loader:
                val_audio = val_audio.to(device).unsqueeze(1)  # Add channel dimension
                val_labels = val_labels.to(device)

                # Convert labels to 32-bit binary messages
                val_message = torch.stack([(val_labels >> i) & 1 for i in range(32)], dim=-1).float()

                # Forward pass: Generate watermarked audio and latent spaces
                val_watermarked_audio, val_embedded_ls, val_original_ls = generator(val_audio, val_message)
                val_probable_embedded_ls = detector_encoder(val_watermarked_audio)

                # Compute validation generator loss
                val_gen_audio_loss = (
                    compute_perceptual_loss(val_audio, val_watermarked_audio)
                    if compute_perceptual_loss
                    else torch.nn.functional.mse_loss(val_audio, val_watermarked_audio)
                )
                val_loss_g += val_gen_audio_loss.item() / len(val_audio)

                # Find least important components in the original latent space
                val_original_ls_np = val_original_ls.detach().cpu().numpy().reshape(-1, val_original_ls.shape[1])
                val_bit_positions, _ = find_least_important_components(val_original_ls_np, num_bits=32)

                # Split validation losses
                val_embedded_loss += torch.mean(
                    (val_embedded_ls[:, val_bit_positions, :] - val_probable_embedded_ls[:, val_bit_positions, :]).pow(2)
                ).item() / len(val_audio)

                val_latent_loss += torch.mean(
                    (val_embedded_ls - val_probable_embedded_ls).pow(2)
                ).item() / len(val_audio)

                val_loss_d += (val_embedded_loss + val_latent_loss)

        avg_val_loss_g = val_loss_g / total_val_batches
        avg_val_loss_d = val_loss_d / total_val_batches
        avg_val_embedded_loss = val_embedded_loss / total_val_batches
        avg_val_latent_loss = val_latent_loss / total_val_batches

        print(
            f"Validation Loss - Generator: {avg_val_loss_g:.4f}, "
            f"Detector: {avg_val_loss_d:.4f}, "
            f"Embedded Loss: {avg_val_embedded_loss:.4f}, "
            f"Latent Loss: {avg_val_latent_loss:.4f}"
        )

        # Scheduler step
        if scheduler is not None:
            scheduler.step(avg_val_loss_g)

        # Save best detector encoder if conditions are met
        if avg_train_loss_d < lowest_training_loss and avg_val_loss_d < lowest_validation_loss:
            best_model_file = f"{checkpoint_path}/BestDecEncoder.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "detector_encoder_state_dict": detector_encoder.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                },
                best_model_file,
            )
            print(f"Best model saved as: {best_model_file}")
            lowest_training_loss = avg_train_loss_d
            lowest_validation_loss = avg_val_loss_d

    print("Training complete.")
