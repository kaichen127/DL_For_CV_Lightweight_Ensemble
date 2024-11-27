import torch
import torchvision
import torchvision.transforms as transforms
from timm import create_model
from torch.utils.data import DataLoader

# Step 1: Define Test Transformations for CrossViT and DeiT
# CrossViT Test Transform
transform_test_crossvit = transforms.Compose([
    transforms.Resize(240),               # Resize to 240x240 for CrossViT
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

# DeiT Test Transform
transform_test_deit = transforms.Compose([
    transforms.Resize(224),               # Resize to 224x224 for DeiT
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

# Step 2: Load CIFAR-10 Dataset
test_dataset_crossvit = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_crossvit)
test_dataset_deit = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_deit)

test_loader_crossvit = DataLoader(test_dataset_crossvit, batch_size=256, shuffle=False, num_workers=4)
test_loader_deit = DataLoader(test_dataset_deit, batch_size=256, shuffle=False, num_workers=4)

# Step 3: Load Pretrained Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CrossViT
crossvit = create_model('crossvit_15_240', pretrained=False, num_classes=10)
crossvit.load_state_dict(torch.load('crossvit_finetuned.pth'))
crossvit = crossvit.to(device)
crossvit.eval()

# Load DeiT
deit = create_model('deit_small_patch16_224', pretrained=False, num_classes=10)
deit.load_state_dict(torch.load('deit_small_finetuned.pth'))
deit = deit.to(device)
deit.eval()

# Step 4: Ensemble Prediction Function
def ensemble_predict(test_loader_crossvit, test_loader_deit, models, device, method='average'):
    """
    Generate predictions using an ensemble of models.
    Args:
        test_loader_crossvit: DataLoader for CrossViT test data.
        test_loader_deit: DataLoader for DeiT test data.
        models: List of trained models (CrossViT, DeiT).
        device: Device (CPU or GPU).
        method: Combination method ('average' or 'weighted').
    Returns:
        Accuracy of the ensemble model.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for (images_crossvit, labels), (images_deit, _) in zip(test_loader_crossvit, test_loader_deit):
            images_crossvit, images_deit, labels = images_crossvit.to(device), images_deit.to(device), labels.to(device)

            # Get outputs from both models
            outputs_crossvit = models[0](images_crossvit)
            outputs_deit = models[1](images_deit)

            # Combine outputs
            if method == 'average':
                ensemble_output = torch.mean(torch.stack([outputs_crossvit, outputs_deit]), dim=0)
            elif method == 'weighted':
                weights = [0.7, 0.3]  # Adjust weights if needed
                ensemble_output = weights[0] * outputs_crossvit + weights[1] * outputs_deit
            else:
                raise ValueError("Unsupported combination method: choose 'average' or 'weighted'.")

            # Get predictions
            _, predicted = torch.max(ensemble_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Ensemble Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Step 5: Evaluate the Ensemble
models = [crossvit, deit]
ensemble_predict(test_loader_crossvit, test_loader_deit, models, device, method='weighted')  # Use 'weighted' for weighted averaging