"""
Evaluate saved CIFAR-10 model
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm


class BasicCNN(nn.Module):
    """Simple 3-layer CNN"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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
        return image, item['label']


def main():
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = BasicCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load('models/cnn_v1_best.pth', map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully")

    # Load test data
    print("Loading test data...")
    dataset = load_dataset("uoft-cs/cifar10")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_dataset = CIFAR10Dataset(dataset['test'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f'\n{"="*50}')
    print(f'Best Model Test Accuracy: {accuracy:.2f}%')
    print(f'Total: {total}, Correct: {correct}')
    print(f'{"="*50}')

    return accuracy


if __name__ == '__main__':
    main()
