import os
import time
import torch
from pathlib import Path
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from src.utils.utility_functions import find_least_important_components

def train(
    generator,
    detector,
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
    scheduler_g=None,
    scheduler_d=None,
):
    # Ensure checkpoint and log directories exist
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    initialize_csv(log_path)

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, weight_decay=1e-4)
    optimizer_d = torch.optim.Adam(detector.parameters(), lr=lr_d, weight_decay=1e-4)

    # Track the best losses for saving the best model
    lowest_train_detector_loss = float('inf')
    lowest_val_detector_loss = float('inf')

    print("Starting training...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        generator.train()
        detector.train()

        train_gen_loss, train_detector_loss = 0, 0
        total_train_batches = len(train_loader)

        # Training loop
        for batch_idx, (audio, labels) in enumerate(train_loader):
            # Move audio and labels to device
            audio = audio.to(device).unsqueeze(1)  # Add channel dimension
            labels = labels.to(device)

            # Convert labels to 32-bit binary messages
            message = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).float().to(device)

            # Forward pass through the generator
            watermarked_audio, embedded_ls, original_ls = generator(audio, message)

            # Compute reconstruction loss for the generator
            gen_recons_loss = (
                compute_perceptual_loss(audio, watermarked_audio)
                if compute_perceptual_loss
                else torch.nn.functional.mse_loss(audio, watermarked_audio)
            )

            # Forward pass through the detector
            probable_embedded_ls = detector.encoder(watermarked_audio)

            # Find bit positions using PCA
            original_ls_np = original_ls.cpu().numpy().reshape(-1, embedded_ls.shape[1])
            bit_positions, _ = find_least_important_components(original_ls_np, message.size(1))
            bit_positions_tensor = torch.tensor(bit_positions, device=device)

            # Extract probable embedded bits
            extracted_bits = torch.sigmoid(probable_embedded_ls[:, bit_positions_tensor, :]).squeeze(-1)

            # Compute bit-wise loss for the detector
            bit_loss = torch.abs(extracted_bits - message).mean()

            # Backward pass for generator
            optimizer_g.zero_grad()
            gen_recons_loss.backward()
            clip_grad_norm_(generator.parameters(), max_norm=5)
            optimizer_g.step()

            # Backward pass for detector
            optimizer_d.zero_grad()
            bit_loss.backward()
            clip_grad_norm_(detector.parameters(), max_norm=5)
            optimizer_d.step()

            # Accumulate losses
            train_gen_loss += gen_recons_loss.item()
            train_detector_loss += bit_loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Batch {batch_idx + 1}/{total_train_batches} - "
                    f"Generator Reconstruction Loss: {gen_recons_loss.item():.4f}, "
                    f"Detector Bit Loss: {bit_loss.item():.4f}"
                )

        # Compute average training loss
        avg_train_gen_loss = train_gen_loss / total_train_batches
        avg_train_detector_loss = train_detector_loss / total_train_batches
        epoch_duration = time.time() - epoch_start_time

        print(
            f"Epoch {epoch + 1} Summary: "
            f"Generator Loss: {avg_train_gen_loss:.4f}, "
            f"Detector Loss: {avg_train_detector_loss:.4f}, "
            f"Duration: {epoch_duration:.2f}s"
        )

        # Validation loop
        generator.eval()
        detector.eval()
        val_gen_loss, val_detector_loss = 0, 0
        total_val_batches = len(val_loader)

        with torch.no_grad():
            for val_audio, val_labels in val_loader:
                val_audio = val_audio.to(device).unsqueeze(1)  # Add channel dimension
                val_labels = val_labels.to(device)

                # Convert labels to 32-bit binary messages
                val_message = torch.stack([(val_labels >> i) & 1 for i in range(32)], dim=-1).float()

                # Forward pass through generator
                val_watermarked_audio, val_embedded_ls, val_original_ls = generator(val_audio, val_message)

                # Compute reconstruction loss for the generator
                val_gen_recons_loss = (
                    compute_perceptual_loss(val_audio, val_watermarked_audio)
                    if compute_perceptual_loss
                    else torch.nn.functional.mse_loss(val_audio, val_watermarked_audio)
                )

                # Forward pass through detector
                val_probable_embedded_ls = detector.encoder(val_watermarked_audio)

                # Extract probable embedded bits
                val_extracted_bits = torch.sigmoid(
                    val_probable_embedded_ls[:, bit_positions_tensor, :]
                ).squeeze(-1)

                # Compute bit-wise loss for the detector
                val_bit_loss = torch.abs(val_extracted_bits - val_message).mean()

                # Accumulate validation losses
                val_gen_loss += val_gen_recons_loss.item()
                val_detector_loss += val_bit_loss.item()

        avg_val_gen_loss = val_gen_loss / total_val_batches
        avg_val_detector_loss = val_detector_loss / total_val_batches

        print(
            f"Validation Losses: Generator Loss: {avg_val_gen_loss:.4f}, "
            f"Detector Loss: {avg_val_detector_loss:.4f}"
        )

        # Scheduler step
        if scheduler_g is not None:
            scheduler_g.step(avg_val_gen_loss)
        if scheduler_d is not None:
            scheduler_d.step(avg_val_detector_loss)

        # Save the best model based on validation detector loss
        if avg_train_detector_loss < lowest_train_detector_loss and avg_val_detector_loss < lowest_val_detector_loss:
            best_model_file = f"{checkpoint_path}/BestDecEncoder.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "detector_state_dict": detector.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "scheduler_g_state_dict": scheduler_g.state_dict() if scheduler_g else None,
                    "scheduler_d_state_dict": scheduler_d.state_dict() if scheduler_d else None,
                    "training_hyperparams": {
                        "learning_rate_g": lr_g,
                        "learning_rate_d": lr_d,
                        "batch_size": train_loader.batch_size,
                        "num_epochs": num_epochs,
                        "latent_dim": generator.encoder.dimension,
                        "nbits": 32,
                    },
                    "loss_metrics": {
                        "train_gen_loss": avg_train_gen_loss,
                        "train_detector_loss": avg_train_detector_loss,
                        "val_gen_loss": avg_val_gen_loss,
                        "val_detector_loss": avg_val_detector_loss,
                    },
                },
                best_model_file,
            )
            print(f"\n[INFO] Best model saved as: {best_model_file}")
            print(f"Training Generator Loss: {avg_train_gen_loss:.6f}")
            print(f"Training Detector Loss: {avg_train_detector_loss:.6f}")
            print(f"Validation Generator Loss: {avg_val_gen_loss:.6f}")
            print(f"Validation Detector Loss: {avg_val_detector_loss:.6f}")
            lowest_train_detector_loss = avg_train_detector_loss
            lowest_val_detector_loss = avg_val_detector_loss

    print("Training completed.")
