import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from tqdm import tqdm

from data_loader import get_cifar10_dataloader
from models import BaselineResNet18, BayesianResNet18

def get_predictions_and_confidences(model, data_loader, device, num_samples=1):
    """
    Perform inference and return predictions, true labels, and confidences.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation set.
        device (torch.device): Device to run the evaluation on.
        num_samples (int): Number of forward passes for Bayesian model.

    Returns:
        tuple: (all_labels, all_preds, all_confidences)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_confidences = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels, = inputs.to(device), labels.to(device)

            # For Bayesian model perform multiple forward passes
            mc_predictions = []
            for _ in range(num_samples):
                outputs = model(inputs)
                mc_predictions.append(outputs)

            # Stack the predictions and take mean over samples
            mc_predictions = torch.stack(mc_predictions)
            mean_logits = mc_predictions.mean(dim=0)

            # Get predicted class and confidence
            probababilities = F.softmax(mean_logits, dim=1)
            confidences, predictions = torch.max(probababilities, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_confidences)

def calculate_ece(labels, confidences, preds, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    Args: 
        labels (np.array): True labels.
        confidences (np.array): Model confidence scores.
        preds (np.array): Model predictions.
        n_bins (int): Number of bins for calibration.

    Returns:
        float: The ECE value.
    """

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Filter predictions in the current bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)

        if np.any(in_bin):
            bin_accuracy = np.mean(labels[in_bin] == preds[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += np.abs(bin_accuracy - bin_confidence) * len(confidences[in_bin])

            ece /= len(labels)
            return ece
        
def plot_reliability_diagram(confidences, accuracies, ece, title, save_path):
    """
    Plot the reliability diagram.
    """

    plt.figure(figsize=(8,8))
    plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.plot(confidences, accuracies, marker='o', label='Model Calibration')
    plt.title(f"{title}/n ECE: {ece:.4f}", fontsize=16)
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reliability diagram saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    print("Starting uncertainty evaluation...")

    # Define hyperparameters
    BATCH_SIZE = 64
    NUM_SAMPLES = 10 # Number of forward passes for Bayesian model evaluation
    NUM_BINS = 10 # Number of bins for ECE calculation

    # Device and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = get_cifar10_dataloader(batch_size=BATCH_SIZE)

    # Load models
    baseline_model = BaselineResNet18(num_classes=10).to(device)
    bayesian_model = BayesianResNet18(num_classes=10).to(device)

    baseline_path = "./models/baseline_resnet18.pth"
    bayesian_path = "./models/bayesian_resnet18.pth"

    # Load state dicts
    if os.path.exists(baseline_path):
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
        print(f"Loaded baseline model from {baseline_path}")

    else:
        print(f"Baseline model weights not found at {baseline_path}. Skipping evaluation.")

    
    if os.path.exists(bayesian_path):
        bayesian_model.load_state_dict(torch.load(bayesian_path, map_location=device))
        print(f"Loaded Bayesian model from {bayesian_path}")
    else:
        print(f"Bayesian model weights not found at {bayesian_path}. Skipping evaluation.")

    # # Evaluate Baseline Model
    # print("Evaluating Baseline Model...")
    # labels_base, preds_base, confs_base = get_predictions_and_confidences(baseline_model, test_loader, device, num_samples=1)

    # accuracy_base = np.mean(labels_base == preds_base)
    # print(f"Baseline Model Accuracy: {accuracy_base:.4f}")

    # ece_base = calculate_ece(labels_base, confs_base, preds_base, n_bins=NUM_BINS)
    # print(f"Baseline Model ECE: {ece_base:.4f}")

    # # Generate data for reliability diagram
    # frac_of_positives_base, mean_predicted_value_base = calibration_curve(labels_base==preds_base, confs_base, n_bins=NUM_BINS)
    # plot_reliability_diagram(mean_predicted_value_base, frac_of_positives_base, ece_base, "Baseline ResNet-18", "./results/baseline_reliability_diagram.png")

    # Evaluate Bayesian model
    print("Evaluating Bayesian Model...")
    labels_bayesian, preds_bayesian, confs_bayesian = get_predictions_and_confidences(bayesian_model, test_loader, device, num_samples=NUM_SAMPLES)

    accracy_bayesian = np.mean(labels_bayesian == preds_bayesian)
    print(f"Bayesian Model Accuracy: {accracy_bayesian:.4f}")

    ece_bayesian = calculate_ece(labels_bayesian, confs_bayesian, preds_bayesian, n_bins=NUM_BINS)
    print(f"Bayesian Model ECE: {ece_bayesian:.4f}")

    # Generate data for reliability diagram
    frac_of_positives_bayesian, mean_predicted_value_bayesian = calibration_curve(labels_bayesian==preds_bayesian, confs_bayesian, n_bins=NUM_BINS)
    plot_reliability_diagram(mean_predicted_value_bayesian, frac_of_positives_bayesian, ece_bayesian, "Bayesian ResNet-18", "./results/bayesian_reliability_diagram.png")

    print("Uncertainty evaluation completed successfully.")