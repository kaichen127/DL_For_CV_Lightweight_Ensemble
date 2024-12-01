import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        
        if isinstance(outputs, tuple): 
            outputs = outputs[0]  
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

if __name__ == '__main__':
    from model import efficientformerv2_s2

    # Path to the pretrained weights
    pretrained_weights_path = "efficientFormer_cifar10_50epoch.pth"

    # Instantiate the model
    model = efficientformerv2_s2(pretrained=False, num_classes=1000)

    # Modify the classifier and distillation head for CIFAR-10
    model.head = nn.Linear(model.head.in_features, 10)
    model.dist_head = nn.Linear(model.dist_head.in_features, 10)

    if pretrained_weights_path:
        checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
        checkpoint = {k: v for k, v in checkpoint.items() if "head" not in k and "dist_head" not in k}
        model.load_state_dict(checkpoint, strict=False)
        print("Pretrained weights loaded successfully!")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize CIFAR-10 (32x32) to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Training loop
    num_epochs = 50
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        
        # Train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        test_accuracy = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {test_accuracy:.2f}%")

        # Log the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")

        # Update best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), "efficientFormer_cifar10.pth")
            print("Best model updated.")
        
        # Step the scheduler
        scheduler.step()

    print("Fine-tuning complete.")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
