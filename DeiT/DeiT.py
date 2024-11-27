import torch
import torchvision
import torchvision.transforms as transforms
from timm import create_model
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import multiprocessing

# Step 1: Define Data Transformations for CIFAR-10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),       # Random cropping for data augmentation
    transforms.RandomHorizontalFlip(),          # Horizontal flip for data augmentation
    transforms.Resize(224),                     # Resize to 224x224 for DeiT
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

transform_test = transforms.Compose([
    transforms.Resize(224),                     # Resize to 224x224 for DeiT
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

# Step 2: Load CIFAR-10 Dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Optimize DataLoader
num_workers = multiprocessing.cpu_count()
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=num_workers)

# Step 3: Load Pretrained DeiT Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('deit_small_patch16_224', pretrained=True)  # Load pretrained DeiT
model.reset_classifier(num_classes=10)  # Reset classifier for CIFAR-10 (10 classes)
model = model.to(device)

# Step 4: Mixed Precision Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scaler = GradScaler()

# Step 5: Training Function
def train_model(model, train_loader, optimizer, criterion, device, scaler, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():  # Mixed precision training
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress = (batch_idx / len(train_loader)) * 100
            print(f"\rEpoch [{epoch + 1}/{epochs}] - {progress:.2f}% complete", end='')

        print(f"\nEpoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f}")

# Step 6: Evaluation Function
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Step 7: Train and Evaluate
train_model(model, train_loader, optimizer, criterion, device, scaler, epochs=10)
evaluate_model(model, test_loader, device)

# Save the trained weights
torch.save(model.state_dict(), "deit_small_finetuned.pth")
print("Model weights saved to 'deit_small_finetuned.pth'")