import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from timm import create_model
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
torch.cuda.empty_cache()
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Random crop and resize to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = "../AID"
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

train_size = int(0.8 * len(full_dataset))  # 80% for training
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

test_dataset.dataset.transform = transform_test
batch_size = 32

num_workers = os.cpu_count()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Print dataset and loader info
print(f"Number of classes: {len(full_dataset.classes)}")
print(f"Class names: {full_dataset.classes}")
print(f"Train size: {train_size}, Test size: {test_size}")

# Load Pretrained EfficientFormer Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = create_model('crossvit_15_240', pretrained=True)
model.reset_classifier(num_classes=len(full_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scaler = torch.amp.GradScaler(init_scale=2 ** 16, growth_factor=2, backoff_factor=0.5, growth_interval=2000, enabled=True)

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
            with torch.amp.autocast(device_type='cuda'):
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

def plot_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()

train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(
    model, train_loader, test_loader, optimizer, criterion, device, scaler, epochs=10
)
plot_curves(train_losses, val_losses, train_accuracies, val_accuracies)
evaluate_model_with_confusion_matrix(model, test_loader, device)

# Save the trained weights
torch.save(model.state_dict(), "crossvit_aid_finetuned.pth")
print("Model weights saved to 'crossvit_aid_finetuned.pth'")