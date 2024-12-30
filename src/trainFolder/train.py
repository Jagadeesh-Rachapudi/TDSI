import torch
import torch.nn as nn
from tqdm import tqdm

def train(
    generator,                    # Watermark generator model
    detector,                     # Watermark detector model
    train_loader,                 # DataLoader for training dataset
    val_loader,                   # DataLoader for validation dataset
    perceptual_loss_fns,          # List of perceptual loss functions
    localization_loss_fn,         # Localization loss function
    wm_loss_fn,                   # Watermarking loss function
    optimizer_g,                  # Optimizer for generator
    optimizer_d,                  # Optimizer for detector
    device="cuda",                # Device for training (default: "cuda")
    num_epochs=50,                # Number of epochs
    lambda_1=0.1,                 # Weight for L1 perceptual loss
    lambda_mspec=2.0,             # Weight for MelSpectrogram perceptual loss
    lambda_adv=4.0,               # Weight for adversarial loss
    lambda_loud=10.0,             # Weight for loudness loss
    lambda_loc=10.0,              # Weight for localization loss
    lambda_dec=1.0,               # Weight for watermarking loss
    checkpoint_path=None,         # Path to save model checkpoints
    log_interval=10,              # Interval for logging
):
    """
    Function to train the watermark generator and detector jointly.

    Args:
        generator (torch.nn.Module): Watermark generator model.
        detector (torch.nn.Module): Watermark detector model.
        train_loader (DataLoader): DataLoader for training.
        val_loader (DataLoader): DataLoader for validation.
        perceptual_loss_fns (list): List of perceptual loss functions.
        localization_loss_fn (torch.nn.Module): Localization loss function.
        wm_loss_fn (torch.nn.Module): Watermarking loss function.
        optimizer_g (torch.optim.Optimizer): Optimizer for generator.
        optimizer_d (torch.optim.Optimizer): Optimizer for detector.
        device (str): Device for training (e.g., "cuda" or "cpu").
        num_epochs (int): Number of training epochs.
        lambda_1 (float): Weight for L1 perceptual loss.
        lambda_mspec (float): Weight for MelSpectrogram perceptual loss.
        lambda_adv (float): Weight for adversarial perceptual loss.
        lambda_loud (float): Weight for loudness loss.
        lambda_loc (float): Weight for localization loss.
        lambda_dec (float): Weight for watermarking loss.
        checkpoint_path (str): Path to save checkpoints.
        log_interval (int): Interval for logging training progress.
    """
    generator.to(device)
    detector.to(device)

    generator.train()
    detector.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss_g = 0.0
        train_loss_d = 0.0

        # Training Loop
        for batch_idx, (original_audio, labels, messages) in enumerate(tqdm(train_loader)):
            original_audio = original_audio.to(device)  # Original audio
            labels = labels.to(device)  # Ground truth labels for watermark presence
            messages = messages.to(device)  # Watermark binary messages

            # Step 1: Generate watermarked audio
            watermarked_audio = generator(original_audio, message=messages)

            # Step 2: Compute perceptual losses
            perceptual_loss = (
                lambda_1 * perceptual_loss_fns["l1"](original_audio, watermarked_audio) +
                lambda_mspec * perceptual_loss_fns["mspec"](original_audio, watermarked_audio) +
                lambda_adv * perceptual_loss_fns["adv"](original_audio, watermarked_audio) +
                lambda_loud * perceptual_loss_fns["loud"](original_audio, watermarked_audio)
            )

            # Step 3: Detector outputs
            logits_watermarked = detector(watermarked_audio)
            logits_original = detector(original_audio)

            # Step 4: Compute localization loss
            localization_loss = lambda_loc * localization_loss_fn(logits_watermarked, labels)

            # Step 5: Compute watermarking loss
            watermarking_loss = lambda_dec * wm_loss_fn(logits_watermarked, None, labels, messages)

            # Step 6: Total generator loss
            loss_g = perceptual_loss + localization_loss + watermarking_loss

            # Optimize generator
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            # Step 7: Detector loss (only localization loss is used for detector)
            loss_d = localization_loss
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            train_loss_g += loss_g.item()
            train_loss_d += loss_d.item()

            # Logging
            if batch_idx % log_interval == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}")
                print(f"Generator Loss: {loss_g.item():.4f}, Detector Loss: {loss_d.item():.4f}")

        # Average training loss
        train_loss_g /= len(train_loader)
        train_loss_d /= len(train_loader)
        print(f"Epoch {epoch+1}: Generator Loss = {train_loss_g:.4f}, Detector Loss = {train_loss_d:.4f}")

        # Validation Loop
        val_loss_g = 0.0
        val_loss_d = 0.0
        generator.eval()
        detector.eval()
        with torch.no_grad():
            for original_audio, labels, messages in val_loader:
                original_audio = original_audio.to(device)
                labels = labels.to(device)
                messages = messages.to(device)

                # Generate watermarked audio
                watermarked_audio = generator(original_audio, message=messages)

                # Compute perceptual losses
                perceptual_loss = (
                    lambda_1 * perceptual_loss_fns["l1"](original_audio, watermarked_audio) +
                    lambda_mspec * perceptual_loss_fns["mspec"](original_audio, watermarked_audio) +
                    lambda_adv * perceptual_loss_fns["adv"](original_audio, watermarked_audio) +
                    lambda_loud * perceptual_loss_fns["loud"](original_audio, watermarked_audio)
                )

                # Detector outputs
                logits_watermarked = detector(watermarked_audio)
                logits_original = detector(original_audio)

                # Compute localization loss
                localization_loss = lambda_loc * localization_loss_fn(logits_watermarked, labels)

                # Compute watermarking loss
                watermarking_loss = lambda_dec * wm_loss_fn(logits_watermarked, None, labels, messages)

                # Validation generator loss
                val_loss_g += (perceptual_loss + localization_loss + watermarking_loss).item()

                # Validation detector loss
                val_loss_d += localization_loss.item()

        val_loss_g /= len(val_loader)
        val_loss_d /= len(val_loader)
        print(f"Validation: Generator Loss = {val_loss_g:.4f}, Detector Loss = {val_loss_d:.4f}")

        # Save checkpoint
        if checkpoint_path:
            torch.save({
                "generator": generator.state_dict(),
                "detector": detector.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "epoch": epoch + 1,
            }, f"{checkpoint_path}/checkpoint_epoch_{epoch+1}.pth")


