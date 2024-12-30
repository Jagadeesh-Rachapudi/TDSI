import torch
from tqdm import tqdm
import random
import json
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence


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
    window_size=0.25,             # Reduced window size in seconds for chunking
    stride=0.125,                 # Reduced stride in seconds for chunking
    batch_size=16,                # Reduced batch size for processing
    device="cuda",                # Device for training (default: "cuda")
    num_epochs=5,                 # Reduced number of epochs for demonstration
    patience=5,                   # Early stopping patience
    checkpoint_path="./checkpoints",  # Path to save model checkpoints
    log_interval=10,              # Interval for logging
):
    generator.to(device)
    detector.to(device)

    # Create checkpoint directory if it doesn't exist
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Initialize logging
    log_path = Path("logs.json")
    logs = {"training": [], "validation": []}
    if log_path.exists():
        with open(log_path, "r") as f:
            logs = json.load(f)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Calculate chunk parameters
    chunk_size = int(window_size * 16000)  # Convert window size to samples
    stride_size = int(stride * 16000)      # Convert stride size to samples

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss_g = 0.0
        train_loss_d = 0.0

        generator.train()
        detector.train()

        for batch_idx, full_audio in enumerate(tqdm(train_loader)):
            if isinstance(full_audio, list) and len(full_audio) == 0:
                continue

            if isinstance(full_audio, list):
                full_audio = pad_sequence(full_audio, batch_first=True)

            full_audio = full_audio.to(device)

            # Pad short audio to meet chunk_size
            if full_audio.size(2) < chunk_size:
                pad_length = chunk_size - full_audio.size(2)
                full_audio = torch.nn.functional.pad(full_audio, (0, pad_length))

            # Chunk full audio
            chunks = []
            for i in range(full_audio.size(0)):
                for j in range(0, full_audio.size(2) - chunk_size + 1, stride_size):
                    chunks.append(full_audio[i:i+1, :, j:j+chunk_size])

            if len(chunks) == 0:
                continue

            chunks = torch.cat(chunks, dim=0)

            watermarked_chunks = chunks[:batch_size]
            non_watermarked_chunks = chunks[batch_size:]

            # Process through the generator
            half_watermarked = watermarked_chunks[:batch_size // 2]
            watermarked_audio = generator(half_watermarked, message=torch.randint(0, 2, (batch_size // 2, 32), device=device))

            # Concatenate watermarked and non-watermarked audio
            set1 = torch.cat((watermarked_audio, half_watermarked), dim=0)
            set2 = torch.cat((non_watermarked_chunks[:batch_size // 2], non_watermarked_chunks[batch_size // 2:]), dim=0)
            final_input_to_detector = torch.cat((set1, set2), dim=0)

            # Pass through the detector
            decoded_message, detection_scores = detector(final_input_to_detector)

            # Compute losses
            perceptual_loss = compute_perceptual_loss(half_watermarked, watermarked_audio)
            detection_loss = compute_detection_loss(
                positive=detection_scores[:batch_size],
                negative=detection_scores[batch_size:],
                mask=torch.ones_like(detection_scores[:batch_size]),
                p_weight=1.0,
                n_weight=1.0,
            )
            decoding_loss = compute_decoding_loss(
                positive=decoded_message[:batch_size],
                mask=torch.ones_like(decoded_message[:batch_size]),
                message=torch.randint(0, 2, (batch_size // 2, 32), device=device),
                temperature=0.1,
                loss_type="bce",
            )

            # Generator and detector losses
            loss_g = perceptual_loss
            loss_d = detection_loss + decoding_loss

            # Backpropagation
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            train_loss_g += loss_g.item()
            train_loss_d += loss_d.item()

        # Validation loop
        val_loss_g = 0.0
        val_loss_d = 0.0
        generator.eval()
        detector.eval()

        with torch.no_grad():
            for val_batch_idx, val_audio in enumerate(val_loader):
                if isinstance(val_audio, list):
                    if len(val_audio) == 0:
                        continue
                    val_audio = pad_sequence(val_audio, batch_first=True)

                val_audio = val_audio.to(device)

                # Pad short audio for validation
                if val_audio.size(2) < chunk_size:
                    pad_length = chunk_size - val_audio.size(2)
                    val_audio = torch.nn.functional.pad(val_audio, (0, pad_length))

                chunks = []
                for i in range(val_audio.size(0)):
                    for j in range(0, val_audio.size(2) - chunk_size + 1, stride_size):
                        chunks.append(val_audio[i:i+1, :, j:j+chunk_size])

                if len(chunks) == 0:
                    continue

                chunks = torch.cat(chunks, dim=0)
                watermarked_chunks = chunks[:batch_size]
                non_watermarked_chunks = chunks[batch_size:]

                half_watermarked = watermarked_chunks[:batch_size // 2]
                watermarked_audio = generator(half_watermarked, message=torch.randint(0, 2, (batch_size // 2, 32), device=device))

                set1 = torch.cat((watermarked_audio, half_watermarked), dim=0)
                set2 = torch.cat((non_watermarked_chunks[:batch_size // 2], non_watermarked_chunks[batch_size // 2:]), dim=0)
                final_input_to_detector = torch.cat((set1, set2), dim=0)

                decoded_message, detection_scores = detector(final_input_to_detector)

                perceptual_loss = compute_perceptual_loss(half_watermarked, watermarked_audio)
                detection_loss = compute_detection_loss(
                    positive=detection_scores[:batch_size],
                    negative=detection_scores[batch_size:],
                    mask=torch.ones_like(detection_scores[:batch_size]),
                    p_weight=1.0,
                    n_weight=1.0,
                )

                val_loss_g += perceptual_loss.item()
                val_loss_d += detection_loss.item()

        val_loss_g /= len(val_loader)
        val_loss_d /= len(val_loader)
        val_loss_total = val_loss_g + val_loss_d

        print(f"Validation: Generator Loss = {val_loss_g:.4f}, Detector Loss = {val_loss_d:.4f}")

        # Save the best model
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
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
