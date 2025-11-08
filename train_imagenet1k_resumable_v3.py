#!/usr/bin/env python3
import os, sys, time, math, random, copy, glob
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torchvision import datasets, transforms
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image
from tqdm import tqdm

# ====================================================
# CONFIGURATION
# ====================================================
DATA_DIR = "/mnt/imagenet"
CKPT_DIR = os.path.join(DATA_DIR, "checkpoints")
LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

NUM_CLASSES = 1000
BATCH_SIZE = 128
NUM_WORKERS = 8
EPOCHS = 115
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_BATCH_INTERVAL = 1000
EMA_DECAY = 0.9999

# ====================================================
# HUGGING FACE LOGIN
# ====================================================
try:
    token = ""  # Replace with your actual HF token
    if token.startswith("hf_"):
        login(token=token, add_to_git_credential=False)
        print("✅ Logged into Hugging Face.")
    else:
        print("⚠️ Invalid HF token placeholder.")
except Exception as e:
    print(f"⚠️ HF login skipped: {e}")

# ====================================================
# LOGGING UTILS
# ====================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, "train.log"), "a") as f:
        f.write(line + "\n")

# ====================================================
# MODEL: ResNet50Custom
# ====================================================
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop_prob = drop_prob

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1 - self.drop_prob
            mask = torch.rand((out.size(0), 1, 1, 1), device=out.device) < keep_prob
            out = out * mask / keep_prob
        out += identity
        return self.relu(out)

class ResNet50Custom(nn.Module):
    def __init__(self, num_classes=1000, drop_prob=0.2):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 3, 1, drop_prob)
        self.layer2 = self._make_layer(128, 4, 2, drop_prob)
        self.layer3 = self._make_layer(256, 6, 2, drop_prob)
        self.layer4 = self._make_layer(512, 3, 2, drop_prob)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, planes, blocks, stride, drop_prob):
        downsample = None
        if stride != 1 or self.in_planes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * 4, 1, stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = []
        for i in range(blocks):
            dp = drop_prob * i / max(1, blocks - 1)
            s = stride if i == 0 else 1
            layers.append(Bottleneck(self.in_planes, planes, s, downsample, dp))
            self.in_planes = planes * 4
            downsample = None
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ====================================================
# TRANSFORMS & DATASETS
# ====================================================
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

log("📂 Loading local training data...")
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)

log("🌐 Loading validation data from Hugging Face (streaming=True)...")
imagenet_val_stream = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

class HFStreamingValDataset(IterableDataset):
    def __init__(self, hf_stream, transform=None):
        self.hf_stream = hf_stream
        self.transform = transform

    def __iter__(self):
        for sample in self.hf_stream:
            try:
                img = sample["image"]
                # Handle image type conversions robustly
                if isinstance(img, Image.Image):
                    img = img.convert("RGB")
                elif hasattr(img, "convert"):
                    img = img.convert("RGB")
                else:
                    # Some streaming backends provide numpy arrays
                    from torchvision.transforms.functional import to_pil_image
                    img = to_pil_image(img).convert("RGB")

                label = int(sample["label"])
                if self.transform:
                    img = self.transform(img)
                yield img, label

            except Exception as e:
                # Skip any broken or unreadable sample
                print(f"[⚠️ Skipped corrupted val sample: {e}]")
                continue


val_dataset = HFStreamingValDataset(imagenet_val_stream, transform=val_transforms)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, multiprocessing_context=None)

log("✅ Validation streaming dataset ready.")

# ====================================================
# MODEL, OPTIMIZER, SCALER, EMA
# ====================================================
model = ResNet50Custom(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.module = copy.deepcopy(model)
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
                ema_p.copy_(ema_p * self.decay + (1 - self.decay) * model_p.detach())

ema = ModelEMA(model, EMA_DECAY)

# ====================================================
# AUTO-RESUME CHECKPOINT
# ====================================================
def resume_latest():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "epoch_*.pth")), key=os.path.getmtime)
    if not ckpts:
        log("No checkpoint found — starting fresh training.")
        return 0
    latest = ckpts[-1]
    ckpt = torch.load(latest, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    ema.module.load_state_dict(ckpt["ema"])
    log(f"✅ Resumed from checkpoint: {latest}")
    return ckpt["epoch"] + 1

# ====================================================
# TRAINING LOOP
# ====================================================
def save_checkpoint(epoch, acc):
    path = os.path.join(CKPT_DIR, f"epoch_{epoch}_val{acc:.2f}.pth")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.module.state_dict(),
    }, path)
    log(f"💾 Saved checkpoint: {path}")

def train_and_validate(start_epoch=0):
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Random MixUp or CutMix
            if random.random() < 0.5:
                use_mixup, use_cutmix = True, False
            else:
                use_mixup, use_cutmix = False, True

            lam = 1.0
            if use_mixup:
                lam = random.betavariate(0.8, 0.8)
                index = torch.randperm(inputs.size(0)).to(DEVICE)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
                targets_a, targets_b = targets, targets[index]
            elif use_cutmix:
                lam = random.betavariate(1.0, 1.0)
                index = torch.randperm(inputs.size(0)).to(DEVICE)
                bbx1, bby1, bbx2, bby2 = 32, 32, 96, 96
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]
                mixed_inputs = inputs
                targets_a, targets_b = targets, targets[index]
            else:
                mixed_inputs, targets_a, targets_b = inputs, targets, targets

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % LOG_BATCH_INTERVAL == 0:
                acc = 100. * correct / total
                log(f"[Train] Epoch {epoch} Step {batch_idx+1}/{len(train_loader)} "
                    f"Loss {running_loss/(batch_idx+1):.4f} Acc {acc:.2f}%")

        scheduler.step()

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

                # Log every N steps instead of tqdm
                if (batch_idx + 1) % LOG_BATCH_INTERVAL == 0:
                    avg_loss = val_loss / (batch_idx + 1)
                    acc = 100.0 * correct / total
                    log(f"[Validation] Epoch {epoch} Step {batch_idx+1} "
                        f"Loss {avg_loss:.4f} Acc {acc:.2f}%")

        # Compute final epoch metrics
        val_loss /= max(1, (batch_idx + 1))
        val_acc = 100. * correct / total
        log(f"[Validation] Epoch {epoch} ValLoss {val_loss:.4f} ValAcc {val_acc:.2f}%")

        # Save checkpoint at end of epoch
        save_checkpoint(epoch, val_acc)
    log("✅ Training complete.")

if __name__ == "__main__":
    start_epoch = resume_latest()
    train_and_validate(start_epoch)
