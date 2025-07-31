import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np

from data_loader import get_cifar10_dataloader
from models import BaselineResNet18

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluates the model on a given DataLoader.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        learning_rate=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        log_dir="./runs/baseline_resnet18",
        model_save_path="./models/baseline_resnet18.pth",
):
    """
    Trains the baseline ResNet-18 model on CIFAR-10 dataset.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        momentum (float): Momentum for the optimizer.
        log_dir (str): Directory to save TensorBoard logs.
        model_save_path (str): Path to save the trained model.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75], gamma=0.1)

    # TensorBoard logger
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to {log_dir}")

    # Training loop
    best_val_acc = 0.0

    # Creating directories if they do not exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("Starting training...")
    for epoch in range(num_epochs):
        # ---- Training Phase ----
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for batch_idx, (inputs, labels) in enumerate(train_loop):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / total_train
        epoch_train_accuracy = correct_train / total_train

        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_accuracy, epoch)

        # ---- Validation Phase ----
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

        scheduler.step()  # Step the learning rate scheduler

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save the model if validation accuracy improves
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    print("Training complete.")
    writer.close()


if __name__ == "__main__":
    print("Starting baseline ResNet-18 training on CIFAR-10...")

    # Define Hyperparameters
    BATCH_SIZE = 128
    NUM_EPOCHS = 2
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_cifar10_dataloader(
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        random_seed=42
    )

    # Initialize the model
    baseline_model = BaselineResNet18(num_classes=10)

    # Train the model
    train_model(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
        log_dir="./runs/baseline_resnet18",
        model_save_path="./models/baseline_resnet18.pth"
    )

    # Load the best model for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a new model instance to load the state_dict
    final_model = BaselineResNet18(num_classes=10)
    final_model.to(device)

    # Load the best saved weights
    model_path = "./models/baseline_resnet18.pth"

    if os.path.exists(model_path):
        final_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from {model_path}")

        criterion = nn.CrossEntropyLoss()
        test_loss, test_accuracy = evaluate_model(final_model, test_loader, criterion, device)
        print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")

    else:
        print(f"Model weights not found at {model_path}. Please check the path and try again.")
    
    
    print("Baseline ResNet-18 training script completed successfully.")