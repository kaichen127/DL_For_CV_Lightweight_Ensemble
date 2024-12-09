import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from timm import create_model
from tqdm import tqdm
import os
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# Define a function to load individual models
def load_model(model_name, checkpoint_path=None, num_classes=10, use_pretrained=True):
    if use_pretrained:
        print("Not using fine-tuned weights")
    model = create_model(model_name, pretrained=use_pretrained)
    model.reset_classifier(num_classes=num_classes)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

class EnsembleModel(nn.Module):
    def __init__(self, models, return_individual_accuracies=False):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.return_individual_accuracies = return_individual_accuracies

    def forward(self, x, labels=None):
        # Collect probabilities from all models
        outputs = [torch.softmax(model(x), dim=1) for model in self.models]

        # Average probabilities
        ensemble_output = sum(outputs) / len(outputs)

        if self.return_individual_accuracies and labels is not None:
            # Calculate individual model accuracies
            individual_accuracies = []
            for model_output in outputs:
                _, predicted = torch.max(model_output, 1)
                accuracy = (predicted == labels).float().mean().item()
                individual_accuracies.append(accuracy * 100)  # Convert to percentage
            return ensemble_output, individual_accuracies

        return ensemble_output

class WeightedAccuracyEnsembleModel(nn.Module):
    def __init__(self, models, accuracies, return_individual_accuracies=False):
        super(WeightedAccuracyEnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.return_individual_accuracies = return_individual_accuracies

        # Normalize accuracies to sum to 1
        self.weights = torch.tensor(accuracies, dtype=torch.float32)
        self.weights /= self.weights.sum()  # Normalize weights

    def forward(self, x, labels=None):
        outputs = [torch.softmax(model(x), dim=1) for model in self.models]
        ensemble_output = sum(w * o for w, o in zip(self.weights, outputs))

        if self.return_individual_accuracies and labels is not None:
            # Calculate individual model accuracies
            individual_accuracies = []
            for model_output in outputs:
                _, predicted = torch.max(model_output, 1)
                accuracy = (predicted == labels).float().mean().item()
                individual_accuracies.append(accuracy * 100)  # Convert to percentage
            return ensemble_output, individual_accuracies

        return ensemble_output

# Define the weighted ensemble model
class WeightedClassEnsembleModel(nn.Module):
    def __init__(self, models, weights, return_individual_accuracies=False):
        super(WeightedClassEnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights  # Shape: (num_models, num_classes)
        self.return_individual_accuracies = return_individual_accuracies

    def forward(self, x, labels=None):
        batch_size = x.size(0)
        num_classes = self.weights.size(1)
        outputs = torch.zeros(batch_size, num_classes, device=x.device)

        individual_accuracies = []

        # Collect predictions and weight them
        for i, model in enumerate(self.models):
            model_logits = model(x)  # Shape: (batch_size, num_classes)
            model_probs = torch.softmax(model_logits, dim=1)  # Convert logits to probabilities
            weighted_probs = model_probs * self.weights[i]  # Element-wise multiplication
            outputs += weighted_probs

            if self.return_individual_accuracies and labels is not None:
                # Calculate individual model accuracies
                _, predicted = torch.max(model_probs, 1)  # Use probabilities for predictions
                accuracy = (predicted == labels).float().mean().item()
                individual_accuracies.append(accuracy * 100)  # Convert to percentage

        # Normalize by the sum of weights for each class
        weights_sum = self.weights.sum(dim=0, keepdim=True)  # Shape: (1, num_classes)
        outputs /= weights_sum

        if self.return_individual_accuracies and labels is not None:
            return outputs, individual_accuracies

        return outputs


def compute_weights_from_confusion_matrices(models, test_loader, num_classes):
    """
    Compute weights matrix for weighted ensemble using confusion matrices.

    Args:
        models (list): List of models to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        num_classes (int): Number of classes in the dataset.

    Returns:
        torch.Tensor: Weights matrix with shape (num_models, num_classes).
    """
    device = next(models[0].parameters()).device  # Assume all models are on the same device
    weights = []

    for model in models:
        # Evaluate model and compute confusion matrix
        y_true, y_pred = [], []
        model.eval()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

        # Derive weights from confusion matrix (accuracy per class)
        class_accuracies = np.diag(cm) / cm.sum(axis=1)
        class_accuracies[np.isnan(class_accuracies)] = 0  # Handle divisions by zero

        # Invert accuracies to penalize high-error classes
        class_weights = 1 / (class_accuracies + 1e-6)  # Add epsilon to avoid division by zero
        weights.append(class_weights)

    # Convert weights to tensor and normalize
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    weights /= weights.sum(dim=0, keepdim=True)  # Normalize weights across models for each class

    return weights


# Set paths to model weights based on the dataset
def get_weight_paths(dataset_name):
    if dataset_name == "CIFAR-10":
        return {
            "efficientformer": "./EfficientFormer/efficientformer_finetuned.pth",
            "deit": "./Deit/deit_small_finetuned.pth",
            "crossvit": "./crossViT/crossvit_finetuned.pth",
        }
    elif dataset_name == "AID":
        return {
            "efficientformer": "./EfficientFormer/efficientformer_aid_finetuned.pth",
            "deit": "./Deit/deit_aid_finetuned.pth",
            "crossvit": "./crossViT/crossvit_aid_finetuned.pth",
        }
    else:
        raise ValueError("Unsupported dataset. Choose either 'CIFAR-10' or 'AID'.")

# Dynamic selection of dataset
def prepare_dataset(dataset_name):
    if dataset_name == "CIFAR-10":
        # Define transformations for CIFAR-10
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize CIFAR-10 images to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
        ])
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "AID":
        # Define transformations for AID
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize AID images to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Load AID dataset
        data_dir = "./AID"
        full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
        print("length of full dataset", len(full_dataset))
        test_size = int(0.2 * len(full_dataset))  # 20% for testing
        _, test_dataset = random_split(full_dataset, [len(full_dataset) - test_size, test_size])
        num_classes = len(full_dataset.classes)
        print("length of test dataset", len(test_dataset))
    else:
        raise ValueError("Unsupported dataset. Choose either 'CIFAR-10' or 'AID'.")

    # Create DataLoader
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    return test_loader, num_classes

def evaluate(model, data_loader, return_individual_accuracies=False):
    model.eval()
    correct = 0
    total = 0
    individual_accuracies_list = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            if return_individual_accuracies:
                outputs, individual_accuracies = model(images, labels)
                individual_accuracies_list.append(individual_accuracies)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    ensemble_accuracy = 100 * correct / total

    if return_individual_accuracies:
        # Compute average individual accuracies across batches
        individual_accuracies_avg = torch.tensor(individual_accuracies_list).mean(dim=0).tolist()
        return ensemble_accuracy, individual_accuracies_avg

    return ensemble_accuracy

if __name__ == "__main__":
    # Select dataset: "CIFAR-10" or "AID"
    dataset_name = "CIFAR-10"
    # dataset_name = "AID"

    # Toggle to use pretrained weights instead of fine-tuned weights
    use_pretrained = False

    # Prepare dataset and get test loader
    test_loader, num_classes = prepare_dataset(dataset_name)

    # Get weight paths for the chosen dataset
    weight_paths = get_weight_paths(dataset_name)

    # Load individual models
    models = [
        load_model('efficientformerv2_l', weight_paths["efficientformer"] if not use_pretrained else None, num_classes, use_pretrained),
        load_model('deit3_small_patch16_224', weight_paths["deit"] if not use_pretrained else None, num_classes, use_pretrained),
        load_model('crossvit_15_240', weight_paths["crossvit"] if not use_pretrained else None, num_classes, use_pretrained)
    ]

    # Choose ensemble type: standard, weighted class, or weighted accuracy
    use_weighted_ensemble = True  # Toggle to use class-weighted ensemble
    use_accuracy_weighted_ensemble = False  # Toggle to use accuracy-weighted ensemble
    return_individual_accuracies = True

    if use_accuracy_weighted_ensemble:
        print("Using accuracy-weighted ensemble")

        # First set of accuracy is for CIFAR-10 and second set is for AID
        accuracies = [94.79166412353516, 78.62103271484375, 84.22618865966797] if dataset_name == "CIFAR-10" else [94.79166412353516, 78.62103271484375, 84.22618865966797]

        # Create accuracy-weighted ensemble model
        ensemble_model = WeightedAccuracyEnsembleModel(models, accuracies, return_individual_accuracies)

    elif use_weighted_ensemble:
        print("Using class-weighted ensemble")

        # Compute weights matrix using confusion matrices
        weights = compute_weights_from_confusion_matrices(models, test_loader, num_classes)

        # Create class-weighted ensemble model
        ensemble_model = WeightedClassEnsembleModel(models, weights, return_individual_accuracies)

    else:
        print("Using unweighted ensemble")
        ensemble_model = EnsembleModel(models, return_individual_accuracies)

    # Evaluate the ensemble model
    if return_individual_accuracies:
        ensemble_accuracy, individual_accuracies = evaluate(ensemble_model, test_loader, return_individual_accuracies=True)
        print(f"Ensemble Model Accuracy on {dataset_name}: {ensemble_accuracy:.2f}%")
        print(f"Individual Model Accuracies: {individual_accuracies}")
    else:
        ensemble_accuracy = evaluate(ensemble_model, test_loader)
        print(f"Ensemble Model Accuracy on {dataset_name}: {ensemble_accuracy:.2f}%")