import torch
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import random


def trainTest(
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
            print(f"\nBatch {batch_idx + 1} - Starting")

            # Debugging: Input shape
            print(f"Audio chunks shape: {audio_chunks.shape}, Labels shape: {labels.shape}")

            try:
                audio_chunks, labels = audio_chunks.to(device), labels.to(device)

                # Split the batch into SetA and SetB
                setA = audio_chunks.clone()
                setB = audio_chunks.clone()
                if len(setB) == 0:
                  print(f"Empty SetB in batch {batch_idx}. Input data: {input_data_info}")


                # Debugging: SetA and SetB shapes
                print(f"SetA shape: {setA.shape}, SetB shape: {setB.shape}")

                # Process first half of SetA through watermark generator
                first_half_setA = setA[:batch_size // 2]
                first_half_labels = labels[:batch_size // 2]

                # Debugging: First half shapes
                print(f"First half SetA shape: {first_half_setA.shape}, First half labels shape: {first_half_labels.shape}")

                watermarked_chunks = generator(first_half_setA, message=first_half_labels)

                # Debugging: Watermarked chunks shape
                print(f"Watermarked chunks shape: {watermarked_chunks.shape}")

                # Delete the second half of SetA
                setA = watermarked_chunks

                # Compute perceptual loss and backpropagate for generator
                perceptual_loss = compute_perceptual_loss(first_half_setA, watermarked_chunks)
                optimizer_g.zero_grad()
                perceptual_loss.backward()
                optimizer_g.step()

                # Debug SetB before slicing
                setB = setB[batch_size // 2:]
                if setB.size(0) == 0:
                    print(f"SetB is empty in batch {batch_idx}. Skipping.")
                    continue

                # Debug Masker
                masked_chunks = []
                for i in range(setA.size(0)):
                    P_mask = random.uniform(0, 1)
                    P_size = random.uniform(0.1, 0.4)
                    P_type = random.uniform(0, 1)
                    masked_chunk, _ = masker(setB[-1:], setA[i], P_mask, P_size, P_type)

                    # Debugging
                    print(f"Masked chunk {i} shape: {masked_chunk.shape}")
                    masked_chunks.append(masked_chunk)

                # Ensure masked chunks are valid
                setA = torch.stack(masked_chunks)

                # Debug Concatenation
                print(f"SetA shape: {setA.shape}, SetB shape: {setB.shape}")
                concatenated_set = torch.cat((setA, setB), dim=0)

                # Debug Detector Output
                decoded_output, detection_scores = detector(concatenated_set)
                print(f"Decoded output shape: {decoded_output.shape}, Detection scores shape: {detection_scores.shape}")


                # Compute the number of bits predicted correctly in training
                predicted_bits_train = (decoded_output[:batch_size // 2] > 0.5).long()
                correct_bits_train = (predicted_bits_train == labels[:batch_size // 2].unsqueeze(-1)).sum().item()

                # Debugging: Bit accuracy
                print(f"Correct bits in batch: {correct_bits_train}")

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

            except Exception as e:
                print(f"Error in batch {batch_idx + 1}: {e}")
                continue

        # Training summary
        train_bit_accuracy = (total_bits_correct_train / total_bits_train) * 100
        print(f"Epoch {epoch + 1}: Training bit accuracy: {train_bit_accuracy:.2f}%")

        # Add similar debugging for the validation loop if needed
