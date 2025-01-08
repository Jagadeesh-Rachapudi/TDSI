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

        train_loss, train_gen_loss = 0, 0
        total_train_batches = len(train_loader)

        # Training loop
        for batch_idx, (audio_tensors, labels) in enumerate(train_loader):
            # Ensure audio_tensors is a list of tensors
            if isinstance(audio_tensors, tuple):
                audio_tensors = list(audio_tensors)

            # Concatenate audio tensors
            audio = torch.cat(audio_tensors, dim=0).to(device)

            # Process labels
            labels = torch.tensor(labels, dtype=torch.int32).to(device)
            labels_binary = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).to(device)

            # Add channel dimension to audio
            audio = audio.unsqueeze(1)

            # Generate random 33-bit messages
            message = torch.randint(0, 2, (audio.size(0), 33)).float().to(device)

            # Forward pass: Generate watermarked audio
            watermarked_audio = generator(audio, message)

            # Compute perceptual loss or MSE loss
            gen_audio_loss = (
                compute_perceptual_loss(audio, watermarked_audio)
                if compute_perceptual_loss
                else torch.nn.functional.mse_loss(audio, watermarked_audio)
            )

            optimizer_g.zero_grad()
            gen_audio_loss.backward()

            # Gradient clipping
            clip_grad_norm_(generator.parameters(), max_norm=5)

            optimizer_g.step()

            train_loss += gen_audio_loss.item()
            train_gen_loss += gen_audio_loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Batch {batch_idx + 1}/{total_train_batches} - "
                    f"Generator Loss: {gen_audio_loss.item():.4f}"
                )

        # Compute average training loss
        avg_train_loss = train_loss / total_train_batches
        epoch_duration = time.time() - epoch_start_time

        # Print training epoch summary
        print(
            f"\nEpoch {epoch + 1} Summary: "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Duration: {epoch_duration:.2f}s"
        )

        # Validation loop
        generator.eval()
        val_loss = 0
        total_val_batches = len(val_loader)

        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                # Unpack the batch (handling nested structures)
                if isinstance(batch, list) and len(batch) == 2:
                    val_audio_tensors, val_labels = batch

                    # Handle nested audio tensor structure
                    if isinstance(val_audio_tensors, tuple) and len(val_audio_tensors) == 1:
                        val_audio_tensors = val_audio_tensors[0]  # Extract the actual tensor

                    # Handle nested label structure
                    if isinstance(val_labels, tuple) and len(val_labels) == 1:
                        val_labels = val_labels[0]  # Extract the actual label
                else:
                    raise ValueError(f"Unexpected batch format: {batch}")

                # Concatenate audio tensors
                val_audio = torch.cat([val_audio_tensors], dim=0).to(device)
                val_audio = val_audio.unsqueeze(1)

                # Reshape labels properly
                val_labels = torch.tensor(val_labels, dtype=torch.int32).to(device).view(-1)

                # Process labels for 33-bit message embedding
                val_message = torch.stack([(val_labels >> i) & 1 for i in range(33)], dim=-1).float()

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

        # Print validation epoch summary
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

        # # Log training and validation metrics to CSV
        # update_csv(
        #     log_path=log_path,
        #     epoch=epoch + 1,
        #     train_audio_reconstruction=avg_train_loss,
        #     val_audio_reconstruction=avg_val_loss,
        # )

    print("Training completed.")
