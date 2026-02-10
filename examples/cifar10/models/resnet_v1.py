"""
Implementation 2.1: Mini-ResNet for CIFAR-10
Goal: Improve on RegularizedCNN (82.59%) with skip connections
Architecture: Simplified ResNet with 3 groups of residual blocks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import time


class ResidualBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class MiniResNet(nn.Module):
    """Simplified ResNet for CIFAR-10"""

    def __init__(self, num_classes=10, blocks_per_group=2):
        super().__init__()

        # Initial convolution (no downsampling since CIFAR-10 is 32x32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks: 3 groups
        # Group 1: 64 channels, 32x32
        self.layer1 = self._make_layer(64, 64, blocks_per_group, stride=1)
        # Group 2: 128 channels, 16x16
        self.layer2 = self._make_layer(64, 128, blocks_per_group, stride=2)
        # Group 3: 256 channels, 8x8
        self.layer3 = self._make_layer(128, 256, blocks_per_group, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block may downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

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
        image = item['img']

        if self.transform:
            image = self.transform(image)

        label = item['label']
        return image, label


def get_data_loaders(batch_size=128):
    """Get CIFAR-10 train and test data loaders"""

    print("Loading CIFAR-10 from HuggingFace...")
    dataset = load_dataset("uoft-cs/cifar10")

    # Transforms (no augmentation for fair comparison)
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
    LEARNING_RATE = 0.1  # Higher LR for ResNet with SGD
    EPOCHS = 30
    BLOCKS_PER_GROUP = 2

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Model
    model = MiniResNet(num_classes=10, blocks_per_group=BLOCKS_PER_GROUP).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer (SGD with momentum for ResNet)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

    # Training
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} (LR: {current_lr:.4f}) ===")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'models/resnet_v1_best.pth')
            print(f"✓ New best accuracy: {best_acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Training completed in {elapsed/60:.1f} minutes")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Previous best (CNN v2): 82.59% | Improvement: {best_acc - 82.59:+.2f}%")
    print(f"{'='*50}")

    return best_acc


if __name__ == '__main__':
    main()
