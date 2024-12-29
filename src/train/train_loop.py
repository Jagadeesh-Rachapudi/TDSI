import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import gc
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from torch.optim import Adam

from models import AudioSealWM  # Watermarking Model
from losses import CombinedLoss  # Custom loss function combining all losses

# Enable cuDNN benchmarking
cudnn.benchmark = True

def train_model(model, train_loader, optimizer, device, loss_function, num_epochs, validation_loader=None, patience=5, nbits=16):
    """
    Trains the given model and validates it.

    Args:
        model: PyTorch model to be trained.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for the model.
        device: Device to run the training (CPU or GPU).
        loss_function: Loss function to optimize.
        num_epochs: Number of training epochs.
        validation_loader: DataLoader for validation data (default: None).
        patience: Early stopping patience (default: 5).
        nbits: Number of bits for watermarking (default: 16).
    """
    model = model.to(device).float()  # Ensure model is in float32
    best_val_loss = float("inf")
    early_stopping_counter = 0

    # Dictionary to store losses
    loss_data = {"train_loss": [], "validation_loss": []}

    for epoch in range(num_epochs):
        print(f"{'=' * 50}\nEpoch [{epoch + 1}/{num_epochs}] started...")
        model.train()
        train_loss = 0.0

        for batch_idx, (audio, _) in enumerate(train_loader):
            if audio is None:
                print(f"Skipping batch {batch_idx} due to no valid data.")
                continue
            
            audio = audio.to(device, non_blocking=True)
            batch_size = audio.size(0)
            msg = torch.randint(0, 2, (batch_size, nbits), device=device)

            optimizer.zero_grad()
            with autocast():
                # Forward pass
                watermarked_audio = model(audio, message=msg)
                loss = loss_function(watermarked_audio, audio, msg)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"[Epoch {epoch + 1}] Batch [{batch_idx + 1}/{len(train_loader)}]: Loss = {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        loss_data["train_loss"].append(avg_train_loss)  # Save train loss
        print(f"\n[Epoch {epoch + 1}] Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        if validation_loader:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for val_audio, _ in validation_loader:
                    val_audio = val_audio.to(device, non_blocking=True)
                    batch_size = val_audio.size(0)
                    val_msg = torch.randint(0, 2, (batch_size, nbits), device=device)

                    with autocast():
                        # Validation forward pass
                        watermarked_audio = model(val_audio, message=val_msg)
                        val_loss += loss_function(watermarked_audio, val_audio, val_msg).item()

            avg_val_loss = val_loss / len(validation_loader)
            loss_data["validation_loss"].append(avg_val_loss)
            print(f"\n[Epoch {epoch + 1}] Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_model.pth")
                print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{patience}")
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

        # Save losses to JSON after each epoch
        with open("loss_data.json", "w") as json_file:
            json.dump(loss_data, json_file, indent=4)

    print("\nTraining complete.")

def evaluate_batch(model, batch, device, loss_function, nbits=16):
    """
    Evaluates a single batch.

    Args:
        model: PyTorch model to be evaluated.
        batch: A batch of data from DataLoader.
        device: Device to run the evaluation (CPU or GPU).
        loss_function: Loss function to calculate the loss.
        nbits: Number of bits for watermarking (default: 16).

    Returns:
        Loss value for the batch.
    """
    model.eval()
    with torch.no_grad():
        audio, _ = batch
        audio = audio.to(device)

        # Generate random watermark messages for evaluation
        batch_size = audio.size(0)
        msg = torch.randint(0, 2, (batch_size, nbits), device=device)

        with autocast():
            # Forward pass for evaluation
            watermarked_audio = model(audio, message=msg)
            loss = loss_function(watermarked_audio, audio, msg)
    return loss.item()

# Export statements
__all__ = ["train_model", "evaluate_batch"]
