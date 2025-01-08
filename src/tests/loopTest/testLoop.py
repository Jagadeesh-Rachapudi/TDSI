import os
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
    # Ensure checkpoint and log directories exist
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path("/content/TDSI/logs").mkdir(parents=True, exist_ok=True)
    initialize_csv(log_path)

    # # Redirect only stdout to a log file
    # log_file_path = "/content/TDSI/logs/consolelogs.txt"
    # log_file = open(log_file_path, "w")  # Open in write mode to overwrite previous logs
    # sys.stdout = log_file  # Redirect stdout to the log file

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, weight_decay=1e-4)
    optimizer_d = torch.optim.Adam(detector.parameters(), lr=lr_d, weight_decay=1e-4)

    print("Starting training...")

    for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            generator.train()
            detector.train()

            train_loss, train_gen_loss, train_label_loss = 0, 0, 0
            total_bits_train, total_bits_correct_train = 0, 0

            # Number of batches for proper averaging
            num_train_batches = len(train_loader)

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

                noise_penalty = torch.mean(torch.abs(watermarked_audio - audio))
                total_loss = 0.8 * gen_audio_loss + 0.2 * label_loss + 0.1 * noise_penalty

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

                if (batch_idx + 1) % 10 == 0:
                    batch_accuracy = (correct_bits / total_bits) * 100
                    print(
                        f"Batch {batch_idx + 1}/{len(train_loader)} - "
                        f"Total Loss: {total_loss.item():.4f}, Gen Loss: {gen_audio_loss.item():.4f}, "
                        f"Label Loss: {label_loss.item():.4f}, Batch Accuracy: {batch_accuracy:.2f}%"
                    )

            num_batches = len(train_loader)  # Total number of batches in the training epoch
            # Compute average losses
            avg_train_loss = train_loss / num_batches
            avg_train_gen_loss = train_gen_loss / num_batches
            avg_train_label_loss = train_label_loss / num_batches

            train_bit_accuracy = (total_bits_correct_train / total_bits_train) * 100
            epoch_duration = time.time() - epoch_start_time

            # Print the averaged metrics in the summary
            print(
                f"\nEpoch {epoch + 1} Summary: "
                f"Train Loss: {avg_train_loss:.4f}, Gen Loss: {avg_train_gen_loss:.4f}, "
                f"Label Loss: {avg_train_label_loss:.4f}, Train Accuracy: {train_bit_accuracy:.2f}%, "
                f"Duration: {epoch_duration:.2f}s"
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

        #     # Validation step
        #     generator.eval()
        #     detector.eval()
        #     with torch.no_grad():
        #         val_loss_g = 0  # Validation perceptual loss (generator)
        #         val_loss_d = 0  # Validation label loss (detector)
        #         val_total_bits = 0
        #         val_correct_bits = 0
        #         num_val_batches = len(val_loader)

        #         for val_audio_tensors, val_labels in val_loader:
        #             val_audio = torch.cat(val_audio_tensors, dim=0).to(device)
        #             val_labels = torch.tensor(val_labels, dtype=torch.int32).to(device)
        #             val_labels_binary = torch.stack([(val_labels >> i) & 1 for i in range(32)], dim=-1).to(device)
        #             val_audio = val_audio.unsqueeze(1)

        #             # Generator forward pass
        #             val_watermarked_audio = generator(val_audio, sample_rate=16000, message=val_labels_binary, alpha=1.0)

        #             # Compute perceptual loss for the generator
        #             if compute_perceptual_loss:
        #                 val_loss_g += compute_perceptual_loss(val_audio, val_watermarked_audio).item()
        #             else:
        #                 val_loss_g += torch.nn.functional.mse_loss(val_audio, val_watermarked_audio).item()

        #             # Detector forward pass
        #             _, val_decoded_message_logits = detector(val_watermarked_audio)

        #             # Compute label loss for the detector
        #             val_scaled_logits = val_decoded_message_logits / temperature
        #             val_loss_d += torch.nn.functional.binary_cross_entropy_with_logits(
        #                 val_scaled_logits, val_labels_binary.float()
        #             ).item()

        #             # Compute bit-level accuracy
        #             val_predictions = (val_decoded_message_logits > 0).int()
        #             val_correct_bits += (val_predictions == val_labels_binary).sum().item()
        #             val_total_bits += val_labels_binary.numel()

        #         # Compute average validation losses and accuracy
        #         avg_val_loss_g = val_loss_g / num_val_batches if num_val_batches > 0 else 0.0
        #         avg_val_loss_d = val_loss_d / num_val_batches if num_val_batches > 0 else 0.0
        #         val_bit_accuracy = (val_correct_bits / val_total_bits) * 100 if val_total_bits > 0 else 0.0

        #         print(
        #             f"Validation Loss - Generator: {avg_val_loss_g:.4f}, Detector: {avg_val_loss_d:.4f}, "
        #             f"Validation Accuracy: {val_bit_accuracy:.2f}%"
        #         )

        #    # Log training and validation metrics to CSV
        #     print("Before saving into CSV")
        #     update_csv(
        #         log_path=log_path,
        #         epoch=epoch + 1,
        #         train_bit_recovery=train_bit_accuracy,
        #         train_audio_reconstruction=train_gen_loss,
        #         train_decoding_loss=train_label_loss,  # Assuming this is the decoding loss during training
        #         val_bit_recovery=val_bit_accuracy,
        #         val_audio_reconstruction=val_loss_g,  # Assuming this is the perceptual loss during validation
        #         val_decoding_loss=val_loss_d  # Assuming this is the decoding loss during validation
        #     )
        #     print("Metrics successfully saved into CSV")

    # Restore original stdout
    # sys.stdout = sys.__stdout__
    # log_file.close()
    # print(f"Logs saved to {log_file_path}")
