#!/usr/bin/env python3
import os, sys, time, math, random, copy, glob, re
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torchvision import datasets, transforms
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image

# ====================================================
# CONFIGURATION (minimal edits from your original)
# ====================================================
DATA_DIR = "/mnt/imagenet"
CKPT_DIR = os.path.join(DATA_DIR, "checkpoints")
LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

NUM_CLASSES = 1000
BATCH_SIZE = 128
NUM_WORKERS = 8
EPOCHS = 130            # extended a bit to allow fine-tuning window
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_BATCH_INTERVAL = 1000
EMA_DECAY = 0.9999

# MixUp/CutMix config (very low probability when resuming)
DEFAULT_MIX_PROB = 0.05   # overall probability in mid/late training (very low)
EARLY_MIX_PROB = 0.2      # if starting from scratch (higher early)
MIXUP_ALPHA_EARLY = 0.8
MIXUP_ALPHA_FINE = 0.4
CUTMIX_ALPHA = 1.0

# ====================================================
# Hugging Face login (kept, harmless placeholder)
# ====================================================
try:
    token = ""
    if token.startswith("hf_"):
        login(token=token, add_to_git_credential=False)
        print("✅ Logged into Hugging Face.")
    else:
        print("⚠️ Invalid HF token placeholder.")
except Exception as e:
    print(f"⚠️ HF login skipped: {e}")

# ====================================================
# Logging
# ====================================================
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, "train.log"), "a") as f:
        f.write(line + "\n")

# ====================================================
# Model (UNCHANGED)
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
# Transforms & Datasets (augmentation strengthened safely)
# ====================================================
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25),
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
                if isinstance(img, Image.Image):
                    img = img.convert("RGB")
                elif hasattr(img, "convert"):
                    img = img.convert("RGB")
                else:
                    from torchvision.transforms.functional import to_pil_image
                    img = to_pil_image(img).convert("RGB")
                label = int(sample["label"])
                if self.transform:
                    img = self.transform(img)
                yield img, label
            except Exception as e:
                print(f"[⚠️ Skipped corrupted val sample: {e}]")
                continue

val_dataset = HFStreamingValDataset(imagenet_val_stream, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, multiprocessing_context=None)

log("✅ Validation streaming dataset ready.")

# ====================================================
# Model, optimizer, scaler, EMA
# ====================================================
model = ResNet50Custom(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing added
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# Keep same scheduler class for checkpoint compatibility
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

class ModelEMA:
    def __init__(self, model, decay=EMA_DECAY):
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
# Utilities: pick-best-checkpoint resume, rand_bbox, mixprob
# ====================================================
def parse_val_from_filename(fname):
    # Expect pattern like epoch_109_val70.85.pth
    try:
        m = re.search(r"_val([0-9]+\.?[0-9]*)", fname)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None

def find_best_checkpoint(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    if not ckpts:
        return None
    best = None
    best_val = -1.0
    for p in ckpts:
        v = parse_val_from_filename(os.path.basename(p))
        if v is None:
            # try loading header if filename doesn't contain val
            try:
                hdr = torch.load(p, map_location="cpu")
                v = float(hdr.get("val_acc", -1.0)) if "val_acc" in hdr else v
            except Exception:
                v = None
        if v is not None and v > best_val:
            best_val = v
            best = p
    return best

def rand_bbox(size, lam):
    # size: tensor.size() => (B, C, H, W) or (B, C, W, H) depending; our code uses H=size[2], W=size[3]
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def get_mix_prob(epoch, total_epochs):
    # If starting from scratch (epoch==0), allow EARLY_MIX_PROB; otherwise keep low
    if epoch == 0:
        return EARLY_MIX_PROB
    if epoch >= int(0.8 * total_epochs):
        return DEFAULT_MIX_PROB
    return max(DEFAULT_MIX_PROB, EARLY_MIX_PROB * (1 - epoch / (total_epochs * 0.8)))

# ====================================================
# Resume from best checkpoint (by ValAcc) and set fine-tune LR
# ====================================================
def resume_from_best():
    best_ckpt = find_best_checkpoint(CKPT_DIR)
    if not best_ckpt:
        log("No checkpoint found — starting fresh training.")
        return 0
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    # try safe loads (backward-compatible)
    try:
        model.load_state_dict(ckpt["model"])
    except Exception as e:
        log(f"[⚠️] model.load_state_dict failed, attempting non-strict load: {e}")
        model.load_state_dict(ckpt["model"], strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        log(f"[⚠️] optimizer.load_state_dict failed: {e}")
    try:
        scheduler.load_state_dict(ckpt["scheduler"])
    except Exception as e:
        log(f"[⚠️] scheduler.load_state_dict failed: {e}")
    try:
        scaler.load_state_dict(ckpt["scaler"])
    except Exception as e:
        log(f"[⚠️] scaler.load_state_dict failed: {e}")
    try:
        ema.module.load_state_dict(ckpt["ema"])
    except Exception as e:
        log(f"[⚠️] ema.load_state_dict failed: {e}")

    # detect last epoch
    last_epoch = None
    if "epoch" in ckpt:
        last_epoch = int(ckpt["epoch"])
    else:
        m = re.search(r"epoch_(\d+)_", os.path.basename(best_ckpt))
        last_epoch = int(m.group(1)) if m else 0

    # reduce LR for fine-tuning (10x smaller)
    for g in optimizer.param_groups:
        old = g.get("lr", LR)
        g["lr"] = max(1e-8, old * 0.1)
    log(f"✅ Resumed from BEST checkpoint: {best_ckpt} (epoch={last_epoch}) | Fine-tune LR set to {optimizer.param_groups[0]['lr']:.6f}")
    return last_epoch + 1

# ====================================================
# Save checkpoint helper (same keys as original)
# ====================================================
def save_checkpoint(epoch, val_acc):
    path = os.path.join(CKPT_DIR, f"epoch_{epoch}_val{val_acc:.2f}.pth")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.module.state_dict(),
        "val_acc": val_acc,
    }, path)
    log(f"💾 Saved checkpoint: {path}")

# ====================================================
# Evaluation (used for raw model and EMA)
# ====================================================
def evaluate_model(mdl):
    mdl.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                outputs = mdl(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            if (batch_idx + 1) % LOG_BATCH_INTERVAL == 0:
                avg_loss = val_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                log(f"[Validation] Step {batch_idx+1} Loss {avg_loss:.4f} Acc {acc:.2f}%")
    val_loss /= max(1, (batch_idx + 1))
    val_acc = 100. * correct / total
    return val_loss, val_acc

# ====================================================
# Training loop (with dynamic low-prob MixUp/CutMix and EMA updates)
# ====================================================
def train_and_validate(start_epoch=0):
    if start_epoch >= 100:
        log(f"🔁 Starting fine-tuning from epoch {start_epoch} with reduced LR {optimizer.param_groups[0]['lr']:.6f}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        mix_prob = get_mix_prob(epoch, EPOCHS)
        alpha_mix = MIXUP_ALPHA_FINE if epoch >= 100 else MIXUP_ALPHA_EARLY

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # decide mix application with very low probability late
            if random.random() < mix_prob:
                # prefer mixup most of the time
                if random.random() < 0.7:
                    use_mixup, use_cutmix = True, False
                else:
                    use_mixup, use_cutmix = False, True
            else:
                use_mixup = use_cutmix = False

            lam = 1.0
            if use_mixup:
                lam = float(np.random.beta(alpha_mix, alpha_mix))
                index = torch.randperm(inputs.size(0)).to(DEVICE)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
                targets_a, targets_b = targets, targets[index]
            elif use_cutmix:
                lam = float(np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA))
                index = torch.randperm(inputs.size(0)).to(DEVICE)
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
                mixed_inputs = inputs
                lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(2) * inputs.size(3)))
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

            # EMA update
            ema.update(model)

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % LOG_BATCH_INTERVAL == 0:
                acc = 100. * correct / total
                log(f"[Train] Epoch {epoch} Step {batch_idx+1}/{len(train_loader)} Loss {running_loss/(batch_idx+1):.4f} Acc {acc:.2f}%")

        # scheduler step (kept to original class for resume compatibility)
        scheduler.step()

        # Validation on raw model then EMA model
        val_loss, val_acc = evaluate_model(model)
        log(f"[Validation] Epoch {epoch} ValLoss {val_loss:.4f} ValAcc {val_acc:.2f}% (raw)")

        ema_val_loss, ema_val_acc = evaluate_model(ema.module)
        log(f"[Validation] Epoch {epoch} ValLoss {ema_val_loss:.4f} ValAcc {ema_val_acc:.2f}% (EMA)")

        # Save checkpoint (use best between raw and EMA for naming)
        save_checkpoint(epoch, max(val_acc, ema_val_acc))

    log("✅ Training complete.")

# ====================================================
# Main
# ====================================================
if __name__ == "__main__":
    start_epoch = resume_from_best()
    train_and_validate(start_epoch)
