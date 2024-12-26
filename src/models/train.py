import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json

# Dynamically add the `utils` directory to sys.path
current_dir = Path(__file__).resolve().parent
utils_dir = current_dir.parent / "utils"
sys.path.append(str(utils_dir))

# Import the DataLoader and model
from data_prcocessing import get_data_loader
from models import AudioRegenModel

# Training function
def train_model(model, train_loader, optimizer, device, loss_function, num_epochs, validation_loader=None, test_loader=None):
    best_val_loss = float("inf")  # Best validation loss to save the model
    log_data = {"train_loss": [], "validation_loss": [], "validation_accuracy": [], "test_loss": None, "test_accuracy": None}

    for epoch in range(num_epochs):
        # TRAINING
        model.train()
        train_loss = 0.0
        for batch_idx, (audio, _) in enumerate(train_loader):
            audio = audio.to(device)
            optimizer.zero_grad()

            # Forward pass
            reconstructed_audio = model(audio)

            # Loss calculation
            loss = loss_function(reconstructed_audio, audio)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        log_data["train_loss"].append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")

        # VALIDATION
        if validation_loader:
            val_loss, val_accuracy = evaluate_model(model, validation_loader, device, loss_function)
            log_data["validation_loss"].append(val_loss)
            log_data["validation_accuracy"].append(val_accuracy)
            print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            # Save model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_audio_regen_model_epoch_{epoch + 1}.pth")
                print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")

    # TESTING
    if test_loader:
        test_loss, test_accuracy = evaluate_model(model, test_loader, device, loss_function)
        log_data["test_loss"] = test_loss
        log_data["test_accuracy"] = test_accuracy
        print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Save logs to a JSON file
    with open("training_log.json", "w") as log_file:
        json.dump(log_data, log_file, indent=4)
    print("Training log saved to training_log.json")

# Evaluation function (validation/testing)
def evaluate_model(model, loader, device, loss_function):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (audio, _) in enumerate(loader):
            audio = audio.to(device)

            # Forward pass
            reconstructed_audio = model(audio)

            # Loss calculation
            loss = loss_function(reconstructed_audio, audio)
            total_loss += loss.item()

            # Accuracy calculation
            correct += torch.sum((torch.abs(reconstructed_audio - audio) < 0.1).float()).item()
            total += audio.numel()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * (correct / total)
    return avg_loss, accuracy

if __name__ == "__main__":
    # Parameters
    data_directory = "/scratch/rachapudij.cair.iitmandi/project/myCode/src/data/validate"
    validation_directory = "/scratch/rachapudij.cair.iitmandi/project/myCode/src/data/train"
    test_directory = "/scratch/rachapudij.cair.iitmandi/project/myCode/src/data/test"
    batch_size = 4
    input_dim = 64000  # 4 seconds of audio at 16kHz
    hidden_dim = 128
    num_epochs = 10
    learning_rate = 0.001

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    train_loader = get_data_loader(data_dir=data_directory, batch_size=batch_size)
    validation_loader = None
    test_loader = None

    if Path(validation_directory).exists():
        validation_loader = get_data_loader(data_dir=validation_directory, batch_size=batch_size, shuffle=False)

    if Path(test_directory).exists():
        test_loader = get_data_loader(data_dir=test_directory, batch_size=batch_size, shuffle=False)

    # Model and optimizer
    model = AudioRegenModel(input_dim=1, hidden_dim=hidden_dim, kernel_size=7, num_layers=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Loss function (Combined L1 and MSE Loss)
    def loss_function(reconstructed, original):
        l1_loss = nn.L1Loss()(reconstructed, original)
        mse_loss = nn.MSELoss()(reconstructed, original)
        return l1_loss + mse_loss

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        loss_function=loss_function,
        num_epochs=num_epochs,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )
