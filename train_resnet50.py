"""
Optimized training script for ImageNet-10 classification with ResNet-50 from scratch.
All-in-one: data loading, model definition, and training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import time
from datetime import datetime


# ============================================================================
# ResNet-50 Model (Built from Scratch)
# ============================================================================

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class ResNet50(nn.Module):
    """ResNet-50 architecture built from scratch."""

    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers (3, 4, 6, 3 blocks)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================================
# Dataset and Data Loading
# ============================================================================

class ImageNetDataset(Dataset):
    """ImageNet-10 dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Collect samples
        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = idx
                for img_path in class_dir.glob('*.JPEG'):
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(image_size=224, is_train=True):
    """
    Get optimized data transforms for ImageNet.

    Training augmentations based on best practices:
    - RandomResizedCrop with scale (0.08, 1.0) - standard ImageNet
    - RandomHorizontalFlip with p=0.5
    - Moderate ColorJitter for better generalization
    - RandAugment can be added for more aggressive augmentation
    - ToTensor and ImageNet normalization
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def create_dataloaders(data_dir, batch_size=128, num_workers=4):
    """Create train and validation dataloaders."""
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'

    train_dataset = ImageNetDataset(train_dir, get_transforms(is_train=True))
    val_dataset = ImageNetDataset(val_dir, get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    num_classes = len(train_dataset.class_to_idx)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")

    return train_loader, val_loader, num_classes


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch:02d} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Step scheduler per batch if OneCycle
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch:02d} [Val]  ')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    return running_loss / len(loader), 100. * correct / total


def train_resnet50(
    data_dir='data',
    epochs=30,
    batch_size=128,
    lr=0.001,
    max_lr=0.01,
    num_workers=4,
    output_dir='outputs',
    use_onecycle=True
):
    """
    Main training function with MPS support and OneCycle LR.

    Args:
        data_dir: Path to data directory
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Base learning rate (for OneCycle, this is the initial LR)
        max_lr: Max learning rate for OneCycle (default: 0.01)
        num_workers: Number of data loading workers
        output_dir: Output directory
        use_onecycle: Use OneCycle LR scheduler (recommended)
    """

    # Setup device - support CUDA, MPS (Apple Silicon), and CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        device_type = 'CUDA GPU'
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'Apple Silicon'
        device_type = 'MPS (Metal Performance Shaders)'
    else:
        device = torch.device('cpu')
        device_name = 'CPU'
        device_type = 'CPU'

    print(f"\n{'='*80}")
    print(f"ResNet-50 Training for ImageNet-10")
    print(f"{'='*80}")
    print(f"Device: {device_type}")
    print(f"Device name: {device_name}")

    # Load data
    print(f"\nLoading data from: {data_dir}")
    train_loader, val_loader, num_classes = create_dataloaders(
        data_dir, batch_size, num_workers
    )

    # Create model
    print(f"\nCreating ResNet-50 model from scratch...")
    model = ResNet50(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # OneCycle LR scheduler (recommended for faster convergence)
    if use_onecycle:
        steps_per_epoch = len(train_loader)
        total_steps = epochs * steps_per_epoch
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.3,  # Warm up for 30% of training
            anneal_strategy='cos',
            div_factor=25.0,  # initial_lr = max_lr / div_factor
            final_div_factor=1e4  # min_lr = initial_lr / final_div_factor
        )
        scheduler_type = f'OneCycle (max_lr={max_lr})'
    else:
        scheduler = None
        scheduler_type = 'None'

    print(f"Optimizer: SGD (momentum=0.9, weight_decay=1e-4)")
    print(f"Scheduler: {scheduler_type}")

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir) / f"resnet50_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        'model': 'ResNet50',
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'num_classes': num_classes,
        'total_params': total_params,
        'device': str(device)
    }
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"{'='*80}\n")

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Print summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, output_path / 'best_model.pth')
            print(f">>> Saved best model: {val_acc:.2f}% <<<")

        print(f"{'='*80}\n")

    # Save final model and history
    torch.save(model.state_dict(), output_path / 'final_model.pth')
    with open(output_path / 'history.json', 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\nTraining Complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {output_path}")
    print(f"{'='*80}\n")

    return output_path, best_val_acc


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train ResNet-50 on ImageNet-10 with MPS/CUDA/CPU support and OneCycle LR'
    )
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory containing train/val folders (default: data)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--max-lr', type=float, default=0.01,
                        help='Maximum learning rate for OneCycle (default: 0.01)')
    parser.add_argument('--use-onecycle', action='store_true', default=True,
                        help='Use OneCycle LR scheduler (default: True)')
    parser.add_argument('--no-onecycle', dest='use_onecycle', action='store_false',
                        help='Disable OneCycle LR scheduler')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for models and logs (default: outputs)')

    args = parser.parse_args()

    train_resnet50(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_lr=args.max_lr,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        use_onecycle=args.use_onecycle
    )
