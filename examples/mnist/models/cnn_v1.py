"""
Implementation 1.1: Simple 2-layer CNN for MNIST
Goal: Achieve >99% test accuracy
Architecture: Conv(32) -> Conv(64) -> FC -> 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time


class SimpleCNN(nn.Module):
    """Simple 2-layer CNN for MNIST"""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: 1 -> 32 channels, 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv2: 32 -> 64 channels, 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_data_loaders(batch_size=64):
    """Get MNIST train and test data loaders"""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader, desc='Training', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(test_loader), 100. * correct / total


def main():
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 5  # MNIST converges fast

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print("Loading MNIST...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Model
    model = SimpleCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'models/cnn_v1_best.pth')
            print(f"New best accuracy: {best_acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Training completed in {elapsed:.1f} seconds")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Target: >99% | {'ACHIEVED' if best_acc > 99 else 'NOT YET'}")
    print(f"{'='*50}")

    return best_acc


if __name__ == '__main__':
    main()
