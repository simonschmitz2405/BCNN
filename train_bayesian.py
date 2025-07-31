import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np

from data_loader import get_cifar10_dataloader
from models import BayesianResNet18

def evaluate_bayesian_model(model, data_loader, device, num_samples=10):
    """
    Evaluates a Bayesian model by averaging over multiple forward passes.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_losses = []
    
    # We need a criterion to compute the NLL for the average loss
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Perform multiple forward passes to sample from the posterior
            mc_predictions = []
            for _ in range(num_samples):
                outputs = model(inputs)
                mc_predictions.append(outputs)
            
            # Stack the predictions to get a tensor of shape [num_samples, batch_size, num_classes]
            mc_predictions = torch.stack(mc_predictions)
            
            # Take the mean of the predictions to get the final class probabilities
            mean_predictions = mc_predictions.mean(dim=0)
            
            # Calculate average loss and accuracy
            loss = criterion(mean_predictions, labels)
            all_losses.append(loss.item())
            
            _, predicted = torch.max(mean_predictions, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = np.sum(all_losses) / len(data_loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy

def train_bayesian_model(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        learning_rate=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        log_dir='./runs/bayesian_resnet18',
        model_save_path='./models/bayesian_resnet18.pth',
):
    """
    Train a Bayesian ResNet-18 model on CIFAR-10 dataset using variational inference.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75], gamma=0.1)

    # TensorBoard logger
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to {log_dir}")

    # Training loop
    best_val_accuracy = 0.0
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("Starting Bayesian training...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_nll = 0.0
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # The loss for variational inference is Variational Free Energy
            nll_loss = criterion(outputs, labels)
            kl_loss = model.log_variational_posterior - model.log_prior

            kl_scaling_factor = 1.0 / len(train_loader.dataset)
            loss = nll_loss + kl_scaling_factor * kl_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_nll += nll_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = correct_train / total_train

        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/NLL_train', running_nll / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', epoch_train_accuracy, epoch)

        # --- Validation Phase ---
        val_loss, val_accuracy = evaluate_bayesian_model(model, val_loader, device, num_samples=10)

        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    print("Bayesian training complete.")
    writer.close()