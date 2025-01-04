import time
import sys
import torch
from pathlib import Path
from datetime import datetime
from torch.nn.utils import clip_grad_norm_

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
    temperature=1.0,
    scheduler=None,
):
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    initialize_csv(log_path)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, weight_decay=1e-4)
    optimizer_d = torch.optim.Adam(detector.parameters(), lr=lr_d, weight_decay=1e-4)

    print("Starting training...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        generator.train()
        detector.train()

        train_loss = 0
        train_gen_loss = 0
        train_label_loss = 0
        total_bits_train = 0
        total_bits_correct_train = 0

        for batch_idx, (audio_tensors, labels) in enumerate(train_loader):
            audio = torch.cat(audio_tensors, dim=0).to(device)
            labels = torch.tensor(labels, dtype=torch.int32).to(device)
            labels_binary = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).to(device)

            audio = audio.unsqueeze(1)

            # Forward pass
            watermarked_audio = generator(audio, sample_rate=16000, message=labels_binary, alpha=1.0)
            detection_score, decoded_message_logits = detector(watermarked_audio)

            gen_audio_loss = (
                compute_perceptual_loss(audio, watermarked_audio)
                if compute_perceptual_loss
                else torch.nn.functional.mse_loss(audio, watermarked_audio)
            )

            scaled_logits = decoded_message_logits / temperature
            label_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scaled_logits, labels_binary.float()
            )

            total_loss = 0.5 * gen_audio_loss + 1.5 * label_loss

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            total_loss.backward()

            # Gradient clipping
            clip_grad_norm_(generator.parameters(), max_norm=5)
            clip_grad_norm_(detector.parameters(), max_norm=5)

            optimizer_g.step()
            optimizer_d.step()

            predictions = (decoded_message_logits > 0).int()
            correct_bits = (predictions == labels_binary).sum().item()
            total_bits = labels_binary.numel()

            train_loss += total_loss.item()
            train_gen_loss += gen_audio_loss.item()
            train_label_loss += label_loss.item()
            total_bits_train += total_bits
            total_bits_correct_train += correct_bits

            if batch_idx % 10 == 0:
                sys.stdout.write(
                    f"\rBatch {batch_idx}/{len(train_loader)} - "
                    f"Total Loss: {total_loss.item():.4f}, "
                    f"Gen Loss: {gen_audio_loss.item():.4f}, "
                    f"Label Loss: {label_loss.item():.4f}, "
                    f"Batch Accuracy: {(correct_bits / total_bits) * 100:.2f}%"
                )
                sys.stdout.flush()

        if scheduler:
            scheduler.step(train_loss)

        train_bit_accuracy = (total_bits_correct_train / total_bits_train) * 100
        epoch_duration = time.time() - epoch_start_time

        print(
            f"\nEpoch {epoch + 1} Summary: "
            f"Train Loss: {train_loss:.4f}, Gen Loss: {train_gen_loss:.4f}, Label Loss: {train_label_loss:.4f}, "
            f"Train Accuracy: {train_bit_accuracy:.2f}%, Duration: {epoch_duration:.2f}s"
        )

        # Save checkpoint
        checkpoint_file = f"{checkpoint_path}/epoch_{epoch + 1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "detector_state_dict": detector.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
            },
            checkpoint_file,
        )
        print(f"Checkpoint saved: {checkpoint_file}")

        # Validation step
        generator.eval()
        detector.eval()
        with torch.no_grad():
            val_loss = 0
            val_total_bits = 0
            val_correct_bits = 0
            for val_audio_tensors, val_labels in val_loader:
                val_audio = torch.cat(val_audio_tensors, dim=0).to(device)
                val_labels = torch.tensor(val_labels, dtype=torch.int32).to(device)
                val_labels_binary = torch.stack([(val_labels >> i) & 1 for i in range(32)], dim=-1).to(device)
                val_audio = val_audio.unsqueeze(1)

                val_watermarked_audio = generator(val_audio, sample_rate=16000, message=val_labels_binary, alpha=1.0)
                _, val_decoded_message_logits = detector(val_watermarked_audio)

                val_scaled_logits = val_decoded_message_logits / temperature
                val_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                    val_scaled_logits, val_labels_binary.float()
                ).item()

                val_predictions = (val_decoded_message_logits > 0).int()
                val_correct_bits += (val_predictions == val_labels_binary).sum().item()
                val_total_bits += val_labels_binary.numel()

            val_bit_accuracy = (val_correct_bits / val_total_bits) * 100
            print(
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_bit_accuracy:.2f}%"
            )

        # Log training and validation metrics to CSV
        update_csv(
            log_path,
            epoch + 1,
            train_loss,
            train_bit_accuracy,
            train_gen_loss,
            train_label_loss,
            val_loss,
            val_bit_accuracy,
        )
