import torch
from tqdm import tqdm

def trainTest(
    generator,
    detector,
    train_loader,
    optimizer_g,
    optimizer_d,
    device,
    num_epochs,
):
    # Loss functions
    def loss_generated_audio(original_audio, generated_audio):
        return torch.nn.functional.mse_loss(original_audio, generated_audio)

    def loss_labels(predicted_labels, actual_labels):
        return torch.nn.functional.binary_cross_entropy_with_logits(predicted_labels, actual_labels.float())

    generator.train()
    detector.train()

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # print(f"Loaded batch: {batch}")
            audio_tensors, labels = batch
            audio = torch.cat(audio_tensors, dim=0).to(device)
            # labels = torch.tensor(labels, dtype=torch.int64).to(device)

            labels = torch.tensor(labels, dtype=torch.int32).to(device) 
            # labels_binary = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).to(device) # Ensure labels are int32
            labels_binary = torch.stack([(labels >> i) & 1 for i in range(32)], dim=-1).to(device)  # Convert to binary
            print(f"Label values: {labels}")
            audio = audio.unsqueeze(1)  # Shape becomes [batch_size, 1, frames]
            # Forward pass through generator
            # Forward pass through generator
            watermarked_audio = generator(audio, sample_rate=16000, message=labels_binary, alpha=1.0)

            
            # # Handle detector output
            # output = detector(watermarked_audio)
            # print(f"Detector output type: {type(output)}")
            # if isinstance(output, tuple) and len(output) == 2:
            #     detection_score, decoded_message_logits = output
            # elif isinstance(output, torch.Tensor):  # Single tensor output
            #     detection_score, decoded_message_logits = output[:, :2, :], output[:, 2:, :]
            # else:
            #     raise ValueError(f"Unexpected detector output: {output}")

            # # Compute losses
            # gen_audio_loss = loss_generated_audio(audio, watermarked_audio)
            # label_loss = loss_labels(decoded_message_logits, labels)

            # # Optimize generator
            # loss_g = gen_audio_loss + label_loss
            # optimizer_g.zero_grad()
            # loss_g.backward()
            # optimizer_g.step()

            # # Optimize detector
            # loss_d = label_loss
            # optimizer_d.zero_grad()
            # loss_d.backward()
            # optimizer_d.step()

            # # Track training loss
            # train_loss += loss_g.item()



        # Average training loss for the epoch
        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
