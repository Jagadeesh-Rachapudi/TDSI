import torch
from pathlib import Path
import csv
from datetime import datetime

def train(
    generator,
    detector,
    train_loader,
    val_loader,
    optimizer_g,
    optimizer_d,
    device,
    num_epochs=100,
    compute_perceptual_loss=None,
    checkpoint_path="./checkpoints",
    log_path="./logs/losses.csv",
    masker=None,
    update_csv=None,
    initialize_csv=None,
):
    """
    Training and validation function for generator and detector models.

    Args:
        generator (torch.nn.Module): The watermark generator model.
        detector (torch.nn.Module): The watermark detector model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_d (torch.optim.Optimizer): Optimizer for the detector.
        device (torch.device): Device to run the training on (CPU/GPU).
        num_epochs (int): Number of training epochs.
        compute_perceptual_loss: Function to compute perceptual loss.
        checkpoint_path (str): Path to save the best model.
        log_path (str): Path to save the CSV log.
        masker: Optional masking function.
        update_csv: Function to update the CSV log.
        initialize_csv: Function to initialize the CSV log.
    """
    # Create checkpoint directory if it doesn't exist
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Initialize CSV logging
    initialize_csv(log_path)

    # Record start date and time
    start_date = datetime.now().strftime("%Y-%m-%d")
    start_time = datetime.now().strftime("%H:%M:%S")

    # Training variables
    best_val_loss = float("inf")
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        # Training loop
        generator.train()
        detector.train()

        train_loss = 0
        total_bits_train = 0
        total_bits_correct_train = 0

        for batch_idx, batch in enumerate(train_loader):
            audio_tensors, labels = batch
            audio = torch.cat(audio_tensors, dim=0).to(device)
            labels = torch.tensor(labels, dtype=torch.int32).to(device)

            # Convert labels to binary
            labels_binary = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).to(device)

            # Add channel dimension to audio
            audio = audio.unsqueeze(1)

            # Forward pass through generator
            watermarked_audio = generator(audio, sample_rate=16000, message=labels_binary, alpha=1.0)

            # Forward pass through detector
            output = detector(watermarked_audio)
            if isinstance(output, tuple) and len(output) == 2:
                detection_score, decoded_message_logits = output
            elif isinstance(output, torch.Tensor):
                detection_score, decoded_message_logits = output[:, :2, :], output[:, 2:, :]
            else:
                raise ValueError(f"Unexpected detector output: {output}")

            # Compute perceptual loss
            if compute_perceptual_loss:
                gen_audio_loss = compute_perceptual_loss(audio, watermarked_audio)
            else:
                gen_audio_loss = torch.nn.functional.mse_loss(audio, watermarked_audio)

            # Compute label loss (BCE between labels and decoded message)
            label_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                decoded_message_logits, labels_binary.float()
            )

            # Combine losses
            total_loss = gen_audio_loss + label_loss

            # Backpropagation and optimization
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            total_loss.backward()
            optimizer_g.step()
            optimizer_d.step()

            # Update training metrics
            train_loss += total_loss.item()
            correct_bits_train = (decoded_message_logits.argmax(dim=1) == labels_binary).sum().item()
            total_bits_train += labels_binary.numel()
            total_bits_correct_train += correct_bits_train

            # Print batch progress and accuracy every 100 batches
            if (batch_idx + 1) % 5 == 0:
                batch_bit_accuracy = (total_bits_correct_train / total_bits_train) * 100
                print(f"Batch {batch_idx + 1}/{len(train_loader)}: Loss={total_loss:.4f}, Bit Accuracy={batch_bit_accuracy:.2f}%")

        # Calculate epoch training accuracy
        train_bit_accuracy = (total_bits_correct_train / total_bits_train) * 100
        print(f"Epoch {epoch + 1}: Training Loss={train_loss:.4f}, Training Bit Accuracy={train_bit_accuracy:.2f}%")

        # Validation loop
        generator.eval()
        detector.eval()

        val_loss_g = 0
        val_loss_d = 0
        total_val_bits = 0
        total_correct_val_bits = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                audio_tensors, labels = batch
                audio = torch.cat(audio_tensors, dim=0).to(device)
                labels = torch.tensor(labels, dtype=torch.int32).to(device)

                # Convert labels to binary
                labels_binary = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).to(device)

                # Add channel dimension to audio
                audio = audio.unsqueeze(1)

                # Forward pass through generator
                watermarked_audio = generator(audio, sample_rate=16000, message=labels_binary, alpha=1.0)

                # Forward pass through detector
                output = detector(watermarked_audio)
                if isinstance(output, tuple) and len(output) == 2:
                    detection_score, decoded_message_logits = output
                elif isinstance(output, torch.Tensor):
                    detection_score, decoded_message_logits = output[:, :2, :], output[:, 2:, :]
                else:
                    raise ValueError(f"Unexpected detector output: {output}")

                # Compute perceptual loss
                if compute_perceptual_loss:
                    val_loss_g += compute_perceptual_loss(audio, watermarked_audio).item()
                else:
                    val_loss_g += torch.nn.functional.mse_loss(audio, watermarked_audio).item()

                # Compute label loss
                scaled_logits = decoded_message_logits / 1.0  # e.g., temperature=1.0
                val_loss_d += torch.nn.functional.binary_cross_entropy_with_logits(
                    scaled_logits, labels_binary.float()
                ).item()

                # Track bit-level accuracy
                correct_bits = (decoded_message_logits.argmax(dim=1) == labels_binary).sum().item()
                total_val_bits += labels_binary.numel()
                total_correct_val_bits += correct_bits

        # Combine validation losses
        val_loss_total = val_loss_g + val_loss_d
        
        val_bit_accuracy = (total_correct_val_bits / total_val_bits) * 100

        print(f"Epoch {epoch + 1}: Validation Loss: {val_loss_total:.4f}, Validation Bit Accuracy: {val_bit_accuracy:.2f}%")

        # Save the best model
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "detector": detector.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "epoch": epoch + 1,
                    "Bit accuracy": train_bit_accuracy,
                },
                f"{checkpoint_path}/best_model.pth",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # Update CSV
        update_csv(
            log_path,
            start_date,
            start_time,
            epoch + 1,
            train_bit_accuracy,
            val_bit_accuracy,
            val_loss_total,
            val_loss_g,
            val_loss_d,
            val_loss_total,  # Placeholder for val decoding loss
        )
