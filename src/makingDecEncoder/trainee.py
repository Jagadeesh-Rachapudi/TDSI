import os
import time
import torch
from pathlib import Path
from datetime import datetime
from torch.nn.utils import clip_grad_norm_

def train(
    generator,
    train_loader,
    val_loader,
    lr_g=1e-4,
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

    # Optimizer
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, weight_decay=1e-4)

    print("Starting training...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        generator.train()

        train_loss = 0
        total_train_batches = len(train_loader)

        # Training loop
        for batch_idx, (audio, labels) in enumerate(train_loader):
            # Move audio and labels to device
            audio = audio.to(device).unsqueeze(1)  # Add channel dimension
            labels = labels.to(device)

            # Convert labels to 32-bit binary messages
            message = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).float()

            # Forward pass: Generate watermarked audio
            watermarked_audio = generator(audio, message)

            # Compute perceptual loss or MSE loss
            gen_audio_loss = (
                compute_perceptual_loss(audio, watermarked_audio)
                if compute_perceptual_loss
                else torch.nn.functional.mse_loss(audio, watermarked_audio)
            )

            # Backward pass
            optimizer_g.zero_grad()
            gen_audio_loss.backward()

            # Gradient clipping
            clip_grad_norm_(generator.parameters(), max_norm=5)

            optimizer_g.step()

            train_loss += gen_audio_loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Batch {batch_idx + 1}/{total_train_batches} - "
                    f"Generator Loss: {gen_audio_loss.item():.4f}"
                )

        # Compute average training loss
        avg_train_loss = train_loss / total_train_batches
        epoch_duration = time.time() - epoch_start_time

        print(
            f"Epoch {epoch + 1} Summary: "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Duration: {epoch_duration:.2f}s"
        )

        # Validation loop
        generator.eval()
        val_loss = 0
        total_val_batches = len(val_loader)

        with torch.no_grad():
            for val_audio, val_labels in val_loader:
                val_audio = val_audio.to(device).unsqueeze(1)  # Add channel dimension
                val_labels = val_labels.to(device)

                # Convert labels to 32-bit binary messages
                val_message = torch.stack([(val_labels >> i) & 1 for i in range(32)], dim=-1).float()

                # Forward pass: Generate watermarked audio
                val_watermarked_audio = generator(val_audio, val_message)

                # Compute validation loss
                val_gen_audio_loss = (
                    compute_perceptual_loss(val_audio, val_watermarked_audio)
                    if compute_perceptual_loss
                    else torch.nn.functional.mse_loss(val_audio, val_watermarked_audio)
                )

                val_loss += val_gen_audio_loss.item()

        avg_val_loss = val_loss / total_val_batches

        print(
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        # Scheduler step
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Save checkpoint
        checkpoint_file = f"{checkpoint_path}/epoch_{epoch + 1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
            },
            checkpoint_file,
        )
        print(f"Checkpoint saved: {checkpoint_file}")

        # Log training and validation metrics to CSV
        if update_csv is not None:
            update_csv(
                log_path=log_path,
                epoch=epoch + 1,
                train_audio_reconstruction=avg_train_loss,
                val_audio_reconstruction=avg_val_loss,
            )

    print("Training completed.")
