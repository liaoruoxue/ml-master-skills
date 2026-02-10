"""
Implementation 1.2: CNN with BatchNorm + Dropout for CIFAR-10
Goal: Improve on baseline (75.47%) with regularization
Architecture: Conv(32) -> BN -> Conv(64) -> BN -> Conv(128) -> BN -> Dropout -> FC
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import time


class RegularizedCNN(nn.Module):
    """CNN with BatchNorm and Dropout for better generalization"""

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()

        # CIFAR-10: 3x32x32
        self.features = nn.Sequential(
            # Conv block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # Conv block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            # Conv block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CIFAR10Dataset(Dataset):
    """Wrapper for HuggingFace CIFAR-10 dataset"""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['img']  # PIL Image

        if self.transform:
            image = self.transform(image)

        label = item['label']
        return image, label


def get_data_loaders(batch_size=128):
    """Get CIFAR-10 train and test data loaders using HuggingFace"""

    print("Loading CIFAR-10 from HuggingFace...")
    dataset = load_dataset("uoft-cs/cifar10")

    # Transforms - still no augmentation for fair comparison
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = CIFAR10Dataset(dataset['train'], transform=transform_train)
    test_dataset = CIFAR10Dataset(dataset['test'], transform=transform_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
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

        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
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
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 30
    DROPOUT_RATE = 0.5

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Model
    model = RegularizedCNN(num_classes=10, dropout_rate=DROPOUT_RATE).to(device)
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
            torch.save(model.state_dict(), 'models/cnn_v2_best.pth')
            print(f"✓ New best accuracy: {best_acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Training completed in {elapsed/60:.1f} minutes")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Baseline (v1): 75.47% | Improvement: {best_acc - 75.47:+.2f}%")
    print(f"{'='*50}")

    return best_acc


if __name__ == '__main__':
    main()
