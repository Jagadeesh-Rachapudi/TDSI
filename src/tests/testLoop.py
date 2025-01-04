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

    best_val_loss = float("inf")
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        generator.train()
        detector.train()

        train_loss = 0
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

            if compute_perceptual_loss:
                gen_audio_loss = compute_perceptual_loss(audio, watermarked_audio)
            else:
                gen_audio_loss = torch.nn.functional.mse_loss(audio, watermarked_audio)

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
            total_bits_train += total_bits
            total_bits_correct_train += correct_bits

        if scheduler:
            scheduler.step(train_loss)

        train_bit_accuracy = (total_bits_correct_train / total_bits_train) * 100
        print(f"Epoch {epoch + 1}: Train Accuracy: {train_bit_accuracy:.2f}%")

        # Validation loop and model checkpointing logic...
        # (Continue from previous implementation, adding dynamic LR adjustments)
