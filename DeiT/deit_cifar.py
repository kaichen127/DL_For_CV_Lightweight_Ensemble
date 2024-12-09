import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from timm import create_model
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 0: Set Random Seed for Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Step 1: Define Data Transformations for CIFAR-10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

# Step 2: Load CIFAR-10 Dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

num_workers = multiprocessing.cpu_count()
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=num_workers)

# Step 3: Load Pretrained DeiT Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('deit_small_patch16_224', pretrained=True)
model.reset_classifier(num_classes=10)
model = model.to(device)

# Step 4: Mixed Precision Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scaler = GradScaler()

# Step 5: Training and Validation Function with Epoch Progress
def train_and_validate(model, train_loader, test_loader, optimizer, criterion, device, scaler, epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Display batch progress
            progress = (batch_idx / batch_count) * 100
            print(f"\rEpoch [{epoch + 1}/{epochs}] - Batch [{batch_idx}/{batch_count}] - {progress:.2f}% complete", end="")

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)
        print(f"\nEpoch [{epoch + 1}/{epochs}] - Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {100 * correct / total:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(test_loader))
        val_accuracies.append(100 * correct / total)
        print(f"Epoch [{epoch + 1}/{epochs}] - Validation Loss: {val_loss / len(test_loader):.4f}, Validation Accuracy: {100 * correct / total:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies

# Step 6: Evaluation with Confusion Matrix
def evaluate_model_with_confusion_matrix(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

# Step 7: Plotting Curves
def plot_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Step 8: Train, Evaluate, and Visualize
train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(
    model, train_loader, test_loader, optimizer, criterion, device, scaler, epochs=10
)
plot_curves(train_losses, val_losses, train_accuracies, val_accuracies)
evaluate_model_with_confusion_matrix(model, test_loader, device)

# Save the trained weights
torch.save(model.state_dict(), "deit_small_finetuned.pth")
print("Model weights saved to 'deit_small_finetuned.pth'")