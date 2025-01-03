import torch
from tqdm import tqdm

def trainTest(
    generator,
    detector,
    train_loader,
    optimizer_g,
    optimizer_d,
    device,
    num_epochs=100,
):
    """
    Function to train the generator and detector models.

    Args:
        generator (torch.nn.Module): The watermark generator model.
        detector (torch.nn.Module): The watermark detector model.
        train_loader (DataLoader): DataLoader for training data.
        optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_d (torch.optim.Optimizer): Optimizer for the detector.
        device (torch.device): Device to run the training on (CPU/GPU).
        num_epochs (int): Number of training epochs.
    """
    # Loss functions
    def loss_generated_audio(original_audio, generated_audio):
        return torch.nn.functional.mse_loss(original_audio, generated_audio)

    def loss_labels(predicted_labels, actual_labels):
        return torch.nn.functional.binary_cross_entropy_with_logits(predicted_labels, actual_labels.float())

    generator.train()
    detector.train()

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_loss = 0  # Initialize epoch loss

        for batch in tqdm(train_loader, desc="Training"):
            # Unpack the batch
            audio_tensors, labels = batch
            audio = torch.cat(audio_tensors, dim=0).to(device)  # Combine batch tensors
            labels = torch.tensor(labels, dtype=torch.int32).to(device)  # Convert labels to int32

            # Convert labels to binary representation
            labels_binary = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).to(device)

            # Add channel dimension to audio
            audio = audio.unsqueeze(1)  # Shape: [batch_size, 1, frames]

            # Forward pass through generator
            watermarked_audio = generator(audio, sample_rate=16000, message=labels_binary, alpha=1.0)

            # Forward pass through detector
            output = detector(watermarked_audio)
            if isinstance(output, tuple) and len(output) == 2:
                detection_score, decoded_message_logits = output
            elif isinstance(output, torch.Tensor):  # Handle single tensor output
                detection_score, decoded_message_logits = output[:, :2, :], output[:, 2:, :]
            else:
                raise ValueError(f"Unexpected detector output type: {type(output)}")

            # Compute losses
            gen_audio_loss = loss_generated_audio(audio, watermarked_audio)
            label_loss = loss_labels(decoded_message_logits, labels_binary)
            print(f"The losses are: Generator Loss={gen_audio_loss}, Label Loss={label_loss}")

            # Combine and backpropagate total loss
            total_loss = gen_audio_loss + label_loss
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            total_loss.backward()
            optimizer_g.step()
            optimizer_d.step()

            # Accumulate training loss
            train_loss += total_loss.item()

        print(f"Epoch {epoch + 1} - Average Train Loss: {train_loss / len(train_loader):.4f}")
