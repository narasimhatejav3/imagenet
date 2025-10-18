import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

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


def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label


class TrivialAugmentWide:
    def get_transform(self, mode='trivial'):
        if mode == 'trivial':
            return transforms.TrivialAugmentWide()
        elif mode == 'rand':
            return transforms.RandAugment(num_ops=2, magnitude=9)
        elif mode == 'auto':
            return transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)
        else:
            raise ValueError(f"Unknown mode: {mode}")


def get_transforms(augmentation_mode='trivial', img_size=224):
    aug_helper = TrivialAugmentWide()
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        aug_helper.get_transform(augmentation_mode),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, batch, labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        mixed_batch = lam * batch + (1 - lam) * batch[index]
        labels_a, labels_b = labels, labels[index]
        return mixed_batch, labels_a, labels_b, lam


class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        _, _, h, w = batch.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        batch[:, :, bby1:bby2, bbx1:bbx2] = batch[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        labels_a, labels_b = labels, labels[index]
        return batch, labels_a, labels_b, lam


class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_state = model.state_dict()
        self.optimizer_state = optimizer.state_dict()

    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100, smooth_f=0.05, diverge_th=5):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        lrs = []
        losses = []
        best_loss = float('inf')
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr
        self.optimizer.param_groups[0]['lr'] = lr
        iterator = iter(train_loader)
        smoothed_loss = 0
        print("\n🔍 Running LR Finder...")
        progress_bar = tqdm(range(num_iter), desc="LR Range Test")
        for iteration in progress_bar:
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels = next(iterator)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            lrs.append(lr)
            lr *= lr_mult
            self.optimizer.param_groups[0]['lr'] = lr
            current_loss = loss.item()
            if iteration == 0:
                smoothed_loss = current_loss
            else:
                smoothed_loss = smooth_f * current_loss + (1 - smooth_f) * smoothed_loss
            losses.append(smoothed_loss)
            best_loss = min(best_loss, smoothed_loss)
            progress_bar.set_postfix({'lr': f'{lr:.2e}', 'loss': f'{smoothed_loss:.4f}'})
            if smoothed_loss > diverge_th * best_loss:
                print(f"\n⚠️  Stopping early - loss diverged at lr={lr:.2e}")
                break
        suggested_lr = self._find_steepest_descent(lrs, losses)
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        return lrs, losses, suggested_lr

    def _find_steepest_descent(self, lrs, losses):
        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients)
        suggested_lr = lrs[min_gradient_idx]
        min_loss_idx = np.argmin(losses)
        conservative_lr = lrs[min_loss_idx] / 10
        print(f"\n📊 LR Finder Results:")
        print(f"   Steepest descent LR: {suggested_lr:.2e}")
        print(f"   Conservative LR (min_loss/10): {conservative_lr:.2e}")
        print(f"   Recommended: Use LR between {conservative_lr:.2e} and {suggested_lr:.2e}")
        return suggested_lr

    def plot(self, lrs, losses, save_path='lr_finder_plot.png', skip_start=10, skip_end=5):
        if skip_start >= len(lrs):
            skip_start = 0
        if skip_end >= len(lrs):
            skip_end = 0
        lrs = lrs[skip_start:-skip_end] if skip_end > 0 else lrs[skip_start:]
        losses = losses[skip_start:-skip_end] if skip_end > 0 else losses[skip_start:]
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('LR Finder - Loss vs Learning Rate')
        plt.grid(True, alpha=0.3)
        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients)
        plt.axvline(x=lrs[min_gradient_idx], color='r', linestyle='--',
                   label=f'Steepest descent: {lrs[min_gradient_idx]:.2e}')
        min_loss_idx = np.argmin(losses)
        plt.axvline(x=lrs[min_loss_idx], color='g', linestyle='--',
                   label=f'Min loss: {lrs[min_loss_idx]:.2e}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📈 LR Finder plot saved to {save_path}")
        plt.close()


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, output_dir='outputs',
                 use_amp=True, mixup_alpha=0.2, cutmix_alpha=1.0, mixup_prob=0.5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        self.mixup = Mixup(alpha=mixup_alpha) if mixup_alpha > 0 else None
        self.cutmix = CutMix(alpha=cutmix_alpha) if cutmix_alpha > 0 else None
        self.mixup_prob = mixup_prob
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lrs': []}

    def mixup_criterion(self, pred, labels_a, labels_b, lam):
        return lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(pred, labels_b)

    def train_epoch(self, epoch, scheduler=None):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            use_mixup_cutmix = (self.mixup is not None or self.cutmix is not None)
            if use_mixup_cutmix and np.random.rand() < 0.5:
                if self.mixup and self.cutmix:
                    if np.random.rand() < self.mixup_prob:
                        inputs, labels_a, labels_b, lam = self.mixup(inputs, labels)
                    else:
                        inputs, labels_a, labels_b, lam = self.cutmix(inputs, labels)
                elif self.mixup:
                    inputs, labels_a, labels_b, lam = self.mixup(inputs, labels)
                else:
                    inputs, labels_a, labels_b, lam = self.cutmix(inputs, labels)
                use_mixed_loss = True
            else:
                use_mixed_loss = False
            self.optimizer.zero_grad()
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    if use_mixed_loss:
                        loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                if use_mixed_loss:
                    loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            if use_mixed_loss:
                correct += (lam * predicted.eq(labels_a).sum().item() + (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix({
                'loss': f'{running_loss/(progress_bar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix({
                'loss': f'{running_loss/(progress_bar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def fit(self, epochs, scheduler=None):
        best_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch, scheduler)
            val_loss, val_acc = self.validate(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lrs'].append(self.optimizer.param_groups[0]['lr'])
            print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.2e}\n")
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(epoch, val_acc, best=True)
                print(f"✅ New best model saved! Accuracy: {val_acc:.2f}%\n")
        print(f"\n🎉 Training completed! Best validation accuracy: {best_acc:.2f}%")
        self.plot_history()
        return self.history

    def save_checkpoint(self, epoch, accuracy, best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'history': self.history
        }
        if best:
            path = self.output_dir / 'best_model.pth'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, path)

    def plot_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(self.history['lrs'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule (OneCycleLR)')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.output_dir / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📈 Training history saved to {save_path}")
        plt.close()


def main():
    config = {
        # Data & Performance Settings
        'batch_size': 256,              
        'num_workers': 4,
        'img_size': 224,

        # Augmentation - RandAugment performs better than TrivialAugment
        'augmentation_mode': 'rand',    
        'num_classes': 100,

        # LR Finder (optional)
        'run_lr_finder': False,
        'lr_finder_start': 1e-7,
        'lr_finder_end': 1,
        'lr_finder_num_iter': 100,
        'lr_multiplier': 0.5,

        # Training Duration - Fast training for quick results
        'epochs': 30,                   

        # Regularization - Optimized based on research
        'weight_decay': 2e-4,           
        'label_smoothing': 0.1,
        'dropout': 0.0,

        # Mixup/CutMix - CRITICAL for accuracy boost!
        'mixup_alpha': 0.2,              
        'cutmix_alpha': 1.0,             
        'mixup_prob': 0.5,              

        # OneCycleLR Settings - Higher LR for faster convergence
        'max_lr': 0.3,                   
        'pct_start': 0.3,                
        'div_factor': 25,
        'final_div_factor': 10000,       

        # Performance Optimizations
        'use_amp': True,                 
        'pin_memory': True,
        'output_dir': 'outputs/imagenet100_t4_optimized',
    }

    print("=" * 80)
    print("🚀 ResNet50 Training on ImageNet-100 (T4 GPU Optimized)")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️  Device: CUDA")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🖥️  Device: MPS (Apple Silicon)")
        print(f"   Using Metal Performance Shaders\n")
    else:
        device = torch.device('cpu')
        print(f"🖥️  Device: CPU\n")

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("📦 Loading ImageNet-100 dataset from HuggingFace...")
    dataset = load_dataset("clane9/imagenet-100")
    print(f"   Train samples: {len(dataset['train'])}")
    print(f"   Val samples: {len(dataset['validation'])}\n")

    print(f"🎨 Setting up data augmentation: {config['augmentation_mode'].upper()}")
    train_transform, val_transform = get_transforms(
        augmentation_mode=config['augmentation_mode'],
        img_size=config['img_size']
    )

    train_dataset = ImageNetDataset(dataset['train'], transform=train_transform)
    val_dataset = ImageNetDataset(dataset['validation'], transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True if config['num_workers'] > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True if config['num_workers'] > 0 else False,
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}\n")

    print("🏗️  Building custom ResNet50 model from scratch...")
    model = resnet50(num_classes=config['num_classes'])

    if config['dropout'] > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=config['dropout']),
            nn.Linear(512 * 4, config['num_classes'])
        )
        print(f"   Custom ResNet50 with dropout: {config['dropout']}")
    else:
        print("   Custom ResNet50 (no dropout)")

    model = model.to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])

    if config['run_lr_finder']:
        print("=" * 80)
        print("🔍 Step 1: Finding Optimal Learning Rate")
        print("=" * 80)
        temp_optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr_finder_start'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
        lr_finder = LRFinder(model, temp_optimizer, criterion, device)
        lrs, losses, suggested_lr = lr_finder.range_test(
            train_loader,
            start_lr=config['lr_finder_start'],
            end_lr=config['lr_finder_end'],
            num_iter=config['lr_finder_num_iter']
        )
        lr_finder.plot(lrs, losses, save_path=str(output_dir / 'lr_finder.png'))
        config['max_lr'] = suggested_lr * config['lr_multiplier']
        print(f"\n✅ Found LR: {suggested_lr:.2e}")
        print(f"✅ Using max_lr = {config['max_lr']:.2e} for OneCycleLR (multiplier: {config['lr_multiplier']})\n")
    else:
        config['max_lr'] = 0.1
        print(f"⚠️  Skipping LR Finder, using default max_lr = {config['max_lr']}\n")

    print("=" * 80)
    print("🎯 Step 2: Training with OneCycleLR Policy")
    print("=" * 80)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config['max_lr'],
        momentum=0.9,
        weight_decay=config['weight_decay']
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['max_lr'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=config['pct_start'],
        div_factor=config['div_factor'],
        final_div_factor=config['final_div_factor'],
        anneal_strategy='cos'
    )

    print(f"\n📚 OneCycleLR Schedule:")
    print(f"   Max LR: {config['max_lr']:.2e}")
    print(f"   Initial LR: {config['max_lr'] / config['div_factor']:.2e}")
    print(f"   Final LR: {config['max_lr'] / config['final_div_factor']:.2e}")
    print(f"   Warmup: {config['pct_start']*100:.0f}% of training")
    print(f"   Total steps: {config['epochs'] * len(train_loader)}\n")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        use_amp=config['use_amp'],
        mixup_alpha=config['mixup_alpha'],
        cutmix_alpha=config['cutmix_alpha'],
        mixup_prob=config['mixup_prob']
    )

    history = trainer.fit(epochs=config['epochs'], scheduler=scheduler)

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("✨ Training Complete!")
    print("=" * 80)
    print(f"📁 Results saved to: {output_dir}")
    print(f"   - best_model.pth: Best model checkpoint")
    print(f"   - lr_finder.png: LR finder plot")
    print(f"   - training_history.png: Training curves")
    print(f"   - config.json: Configuration")
    print(f"   - history.json: Training history")
    print()


if __name__ == '__main__':
    main()

