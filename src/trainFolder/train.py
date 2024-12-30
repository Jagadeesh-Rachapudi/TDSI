import torch
from tqdm import tqdm
import random
import json
from pathlib import Path


def train(
    generator,                    # Watermark generator model
    detector,                     # Watermark detector model
    train_loader,                 # DataLoader for training dataset
    val_loader,                   # DataLoader for validation dataset
    optimizer_g,                  # Optimizer for generator
    optimizer_d,                  # Optimizer for detector
    compute_detection_loss,       # Function to compute detection loss
    compute_decoding_loss,        # Function to compute decoding loss
    compute_perceptual_loss,      # Function to compute perceptual loss
    window_size=0.5,              # Window size in seconds for chunking
    stride=0.25,                  # Stride in seconds for chunking
    batch_size=32,                # Batch size for processing
    device="cuda",                # Device for training (default: "cuda")
    num_epochs=50,                # Number of epochs
    patience=5,                   # Early stopping patience
    checkpoint_path=None,         # Path to save model checkpoints
    log_interval=10,              # Interval for logging
):
    """
    Training function with early stopping, validation, periodic test accuracy, and logging to JSON.
    """
    generator.to(device)
    detector.to(device)

    # Path for saving logs
    log_path = Path("logs.json")
    logs = {"training": [], "validation": []}

    if log_path.exists():
        with open(log_path, "r") as f:
            logs = json.load(f)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss_g = 0.0
        train_loss_d = 0.0
        batch_accuracies = []

        generator.train()
        detector.train()

        for batch_idx, full_audio in enumerate(tqdm(train_loader)):
            full_audio = full_audio.to(device)

            # Step 1: Generate 32-bit messages
            messages = torch.randint(0, 2, (batch_size, 32), dtype=torch.float32, device=device)

            # Step 2: Chunk full audio into two equal sets: watermarked and non-watermarked
            chunk_size = int(window_size * 16000)  # Convert window size to samples
            stride_size = int(stride * 16000)      # Convert stride size to samples
            num_chunks = batch_size * 2

            chunks = []
            for i in range(full_audio.size(0)):  # Iterate over batch
                for j in range(0, full_audio.size(2) - chunk_size + 1, stride_size):
                    chunks.append(full_audio[i:i+1, :, j:j+chunk_size])
                    if len(chunks) == num_chunks:  # Stop once we have enough chunks
                        break
                if len(chunks) == num_chunks:
                    break
            chunks = torch.cat(chunks, dim=0)  # Combine into a single batch

            watermarked_chunks = chunks[:batch_size]  # First half for watermarked
            non_watermarked_chunks = chunks[batch_size:]  # Second half for non-watermarked

            # Step 3: Pass half of each set to the generator
            half_watermarked = watermarked_chunks[:batch_size // 2]
            half_non_watermarked = non_watermarked_chunks[:batch_size // 2]

            # Process through the generator
            watermarked_audio = generator(half_watermarked, message=messages[:batch_size // 2])

            # Step 4: Concatenate watermarked and non-watermarked audio
            set1 = torch.cat((watermarked_audio, half_watermarked), dim=0)
            set2 = torch.cat((half_non_watermarked, non_watermarked_chunks[batch_size // 2:]), dim=0)
            final_input_to_detector = torch.cat((set1, set2), dim=0)

            # Step 5: Pass the concatenated set through the detector
            decoded_message, detection_scores = detector(final_input_to_detector)

            # Step 6: Split detector output into positive and negative cases
            positive_detection_scores = detection_scores[:batch_size]
            negative_detection_scores = detection_scores[batch_size:]

            positive_decoded_message = decoded_message[:batch_size]
            negative_decoded_message = decoded_message[batch_size:]

            # Step 7: Compute perceptual loss
            perceptual_loss = compute_perceptual_loss(half_watermarked, watermarked_audio)

            # Step 8: Compute detection loss
            detection_loss = compute_detection_loss(
                positive=positive_detection_scores,
                negative=negative_detection_scores,
                mask=torch.ones_like(positive_detection_scores),  # Example mask
                p_weight=1.0,
                n_weight=1.0
            )

            # Step 9: Compute decoding loss
            decoding_loss = compute_decoding_loss(
                positive=positive_decoded_message,
                mask=torch.ones_like(positive_decoded_message),  # Example mask
                message=messages[:batch_size // 2],
                temperature=0.1,
                loss_type="bce"
            )

            # Generator loss
            loss_g = perceptual_loss

            # Detector loss
            loss_d = detection_loss + decoding_loss

            # Backpropagation for generator
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            # Backpropagation for detector
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            train_loss_g += loss_g.item()
            train_loss_d += loss_d.item()

            # Periodic test accuracy
            if (batch_idx + 1) % 100 == 0:
                generator.eval()
                detector.eval()

                correct = 0
                total = 0
                with torch.no_grad():
                    for val_audio in val_loader:
                        val_audio = val_audio.to(device)
                        decoded_message, detection_scores = detector(val_audio)
                        predictions = (detection_scores.argmax(dim=1) == 1)
                        correct += predictions.sum().item()
                        total += predictions.numel()

                accuracy = 100 * correct / total
                batch_accuracies.append(accuracy)
                print(f"Test Accuracy after {batch_idx + 1} batches: {accuracy:.2f}%")

                generator.train()
                detector.train()

        # Validation loop
        val_loss_g = 0.0
        val_loss_d = 0.0
        generator.eval()
        detector.eval()

        with torch.no_grad():
            for val_audio in val_loader:
                val_audio = val_audio.to(device)
                decoded_message, detection_scores = detector(val_audio)
                val_loss_g += compute_perceptual_loss(val_audio, generator(val_audio))
                val_loss_d += compute_detection_loss(
                    positive=detection_scores,
                    negative=torch.zeros_like(detection_scores),  # Dummy for validation
                    mask=torch.ones_like(detection_scores),
                    p_weight=1.0,
                    n_weight=1.0
                )

        val_loss_g /= len(val_loader)
        val_loss_d /= len(val_loader)
        val_loss_total = val_loss_g + val_loss_d
        print(f"Validation: Generator Loss = {val_loss_g:.4f}, Detector Loss = {val_loss_d:.4f}")

        # Save metrics to logs
        logs["training"].append({
            "epoch": epoch + 1,
            "generator_loss": train_loss_g / len(train_loader),
            "detector_loss": train_loss_d / len(train_loader),
            "batch_accuracies": batch_accuracies
        })

        logs["validation"].append({
            "epoch": epoch + 1,
            "generator_loss": val_loss_g,
            "detector_loss": val_loss_d
        })

        with open(log_path, "w") as f:
            json.dump(logs, f, indent=4)

        # Early Stopping Check
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            epochs_without_improvement = 0
            # Save best model
            if checkpoint_path:
                torch.save({
                    "generator": generator.state_dict(),
                    "detector": detector.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "epoch": epoch + 1,
                }, f"{checkpoint_path}/best_model.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
