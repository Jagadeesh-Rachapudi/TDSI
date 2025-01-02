import torch
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import random


def train(
    generator,
    detector,
    train_loader,
    val_loader,
    optimizer_g,
    optimizer_d,
    compute_detection_loss,
    compute_decoding_loss,
    compute_perceptual_loss,
    masker,
    initialize_csv,
    update_csv,
    batch_size=32,
    device="cuda",
    num_epochs=50,
    patience=10,
    checkpoint_path="../../checkpoints",
    log_path="../../logs/losses.csv",
    
):
    # Move models to device
    generator.to(device)
    detector.to(device)

    # Create checkpoint directory if it doesn't exist
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Initialize CSV logging
    initialize_csv(log_path)

    # Record start date and time
    start_date = datetime.now().strftime("%Y-%m-%d")
    start_time = datetime.now().strftime("%H:%M:%S")

    # Training metadata
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss_g, train_loss_d = 0.0, 0.0
        val_loss_g, val_loss_d = 0.0, 0.0

        total_bits_correct_train = 0
        total_bits_train = 0

        generator.train()
        detector.train()

        # Training loop
        for batch_idx, (audio_chunks, labels) in enumerate(tqdm(train_loader)):
            # Ensure batch size is fixed at 32
            # assert audio_chunks.size(0) == batch_size, "Batch size must be 32."

            audio_chunks, labels = audio_chunks.to(device), labels.to(device)

            # Split the batch into SetA and SetB
            setA = audio_chunks.clone()
            setB = audio_chunks.clone()

            # Process first half of SetA through watermark generator
            first_half_setA = setA[:batch_size // 2]
            first_half_labels = labels[:batch_size // 2]
            watermarked_chunks = generator(first_half_setA, message=first_half_labels)

            # Delete the second half of SetA
            setA = watermarked_chunks

            # Compute perceptual loss and backpropagate for generator
            perceptual_loss = compute_perceptual_loss(first_half_setA, watermarked_chunks)
            optimizer_g.zero_grad()
            perceptual_loss.backward()
            optimizer_g.step()

            # Process first half of SetB (bypass watermarking)
            setB = setB[batch_size // 2:]  # Retain only the second half

            # Mask first half of SetA
            last_chunk_setB = setB[-1:]  # Reference chunk from SetB
            masked_chunks = []
            for i in range(setA.size(0)):
                P_mask = random.uniform(0, 1)
                P_size = random.uniform(0.1, 0.4)
                P_type = random.uniform(0, 1)
                masked_chunk, _ = masker(last_chunk_setB, setA[i], P_mask, P_size, P_type)
                masked_chunks.append(masked_chunk)
            setA = torch.stack(masked_chunks)

            # Concatenate masked SetA with SetB
            concatenated_set = torch.cat((setA, setB), dim=0)

            # Send concatenated set to the detector
            decoded_output, detection_scores = detector(concatenated_set)

            # Compute the number of bits predicted correctly in training
            predicted_bits_train = (decoded_output[:batch_size // 2] > 0.5).long()
            correct_bits_train = (predicted_bits_train == labels[:batch_size // 2].unsqueeze(-1)).sum().item()
            total_bits_correct_train += correct_bits_train
            total_bits_train += batch_size // 2 * 32  # Total bits for the batch

            if (batch_idx + 1) % 75 == 0:
                batch_bit_accuracy = (correct_bits_train / (batch_size // 2 * 32)) * 100
                print(f"Batch {batch_idx + 1}: Correct bits: {correct_bits_train}/{batch_size // 2 * 32} ({batch_bit_accuracy:.2f}%)")

            # Compute losses for the detector
            detection_loss = compute_detection_loss(
                positive=detection_scores[:batch_size // 2],
                negative=detection_scores[batch_size // 2:],
                mask=torch.ones_like(detection_scores[:batch_size // 2]),
                p_weight=1.0,
                n_weight=1.0,
            )
            decoding_loss = compute_decoding_loss(
                positive=decoded_output[:batch_size // 2],
                mask=torch.ones_like(decoded_output[:batch_size // 2]),
                message=labels[:batch_size // 2],
                temperature=0.1,
                loss_type="bce",
            )

            loss_d = detection_loss + decoding_loss

            # Backpropagate for detector
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # Track losses
            train_loss_g += perceptual_loss.item()
            train_loss_d += loss_d.item()

        train_bit_accuracy = (total_bits_correct_train / total_bits_train) * 100
        print(f"Epoch {epoch + 1}: Training bit accuracy: {train_bit_accuracy:.2f}%")


        # Validation loop
        generator.eval()
        detector.eval()
        total_bits_correct_val = 0
        total_bits_val = 0
        with torch.no_grad():
            for val_batch_idx, (val_chunks, val_labels) in enumerate(val_loader):
                val_chunks, val_labels = val_chunks.to(device), val_labels.to(device)

                # Split the batch into SetA and SetB
                setA = val_chunks.clone()
                setB = val_chunks.clone()

                # Process SetA and SetB
                watermarked_chunks = generator(setA[:batch_size // 2], message=val_labels[:batch_size // 2])
                setA = watermarked_chunks
                setB = setB[batch_size // 2:]

                # Compute perceptual loss for validation
                val_perceptual_loss = compute_perceptual_loss(setA[:batch_size // 2], watermarked_chunks)

                # Mask SetA
                last_chunk_setB = setB[-1:]
                masked_chunks = []
                for i in range(setA.size(0)):
                    P_mask = random.uniform(0, 1)
                    P_size = random.uniform(0.1, 0.4)
                    P_type = random.uniform(0, 1)
                    masked_chunk, _ = masker(last_chunk_setB, setA[i], P_mask, P_size, P_type)
                    masked_chunks.append(masked_chunk)
                setA = torch.stack(masked_chunks)

                # Concatenate and send to detector
                concatenated_set = torch.cat((setA, setB), dim=0)
                decoded_output, detection_scores = detector(concatenated_set)

                # Compute the number of bits predicted correctly in validation
                predicted_bits_val = (decoded_output[:batch_size // 2] > 0.5).long()
                correct_bits_val = (predicted_bits_val == val_labels[:batch_size // 2].unsqueeze(-1)).sum().item()
                total_bits_correct_val += correct_bits_val
                total_bits_val += batch_size // 2 * 32  # Total bits for the batch

                # Compute losses
                detection_loss = compute_detection_loss(
                    positive=detection_scores[:batch_size // 2],
                    negative=detection_scores[batch_size // 2:],
                    mask=torch.ones_like(detection_scores[:batch_size // 2]),
                    p_weight=1.0,
                    n_weight=1.0,
                )
                val_loss_g += val_perceptual_loss.item()
                val_loss_d += detection_loss.item()

        # Compute averages and log results
        train_loss_g /= len(train_loader)
        train_loss_d /= len(train_loader)
        val_loss_g /= len(val_loader)
        val_loss_d /= len(val_loader)

        train_accuracy = (1 - train_loss_g) * 100  # Hypothetical metric
        val_accuracy = (1 - val_loss_g) * 100  # Hypothetical metric

        val_bit_accuracy = (total_bits_correct_val / total_bits_val) * 100
        print(f"Epoch {epoch + 1}: Validation bit accuracy: {val_bit_accuracy:.2f}%")
        # Update CSV file
        update_csv(
            log_path, start_date, start_time, epoch + 1,
            train_accuracy, val_accuracy,
            train_loss_g, train_loss_d, decoding_loss.item(),
            val_loss_g, val_loss_d, detection_loss.item(),
            train_bit_accuracy, val_bit_accuracy
        )

        # Save best model
        val_loss_total = val_loss_g + val_loss_d
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            torch.save({
                "generator": generator.state_dict(),
                "detector": detector.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "epoch": epoch + 1,
                "Bit accuracy": train_bit_accuracy
            }, f"{checkpoint_path}/best_model.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
