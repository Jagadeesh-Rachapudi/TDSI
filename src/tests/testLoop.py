import torch
from tqdm import tqdm
from src.losses.loss import compute_detection_loss, compute_decoding_loss, compute_perceptual_loss


def trainTest(
    generator,
    detector,
    train_loader,
    optimizer_g,
    optimizer_d,
    device,
    num_epochs=100,
):
    # Loss functions
    def loss_generated_audio(original_audio, generated_audio):
        return torch.nn.functional.mse_loss(original_audio, generated_audio)

    def loss_labels(predicted_labels, actual_labels):
        return torch.nn.functional.binary_cross_entropy_with_logits(predicted_labels, actual_labels.float())

    generator.train()
    detector.train()

    from tqdm import tqdm

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Unpack the batch
            audio_tensors, labels = batch
            audio = torch.cat(audio_tensors, dim=0).to(device)
            labels = torch.tensor(labels, dtype=torch.int32).to(device)

            # Convert labels to binary
            labels_binary = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).to(device)

            # Add channel dimension to audio
            audio = audio.unsqueeze(1)

            # Forward pass through generator (no gradients)
            # with torch.no_grad():
            watermarked_audio = generator(audio, sample_rate=16000, message=labels_binary, alpha=1.0)

            # Detector forward pass
            output = detector(watermarked_audio)
            if isinstance(output, tuple) and len(output) == 2:
                detection_score, decoded_message_logits = output
            elif isinstance(output, torch.Tensor):
                detection_score, decoded_message_logits = output[:, :2, :], output[:, 2:, :]
            else:
                raise ValueError(f"Unexpected detector output: {output}")

            # Compute losses
            # Compute perceptual loss using compute_perceptual_loss
            if compute_perceptual_loss:
                gen_audio_loss = compute_perceptual_loss(audio, watermarked_audio)
            else:
                gen_audio_loss = torch.tensor(1.1, device=device, requires_grad=True)  # Default placeholder loss

            label_loss = loss_labels(decoded_message_logits, labels_binary)
            # print("The losses are", gen_audio_loss, label_loss)

            # Compute decoding loss using compute_decoding_loss
            if compute_decoding_loss:
                # Compute decoding loss
                label_loss = compute_decoding_loss(
                    decoded_logits=decoded_message_logits,
                    message=labels_binary,
                    temperature=1.0,
                    loss_type="bce"
                )
            else:
                label_loss = torch.tensor(1.1, device=device, requires_grad=True)

            # Combine and optimize
            total_loss = gen_audio_loss + label_loss
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            total_loss.backward()
            optimizer_g.step()
            optimizer_d.step()

            # Track training loss
            train_loss += total_loss.item()
        
        # print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

