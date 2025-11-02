#!/usr/bin/env python3
# train_imagenet1k_resumable_improved.py
# Single & Multi-GPU (torchrun) friendly, resumable, EMA, auto-LR-scaling, improved logging

import os
import argparse
import datetime
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, distributed
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast
import math

# ----------------------------
# Logging util (rank 0 only)
# ----------------------------
def log(msg, rank=0):
    if rank == 0:
        ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        print(f"{ts} {msg}", flush=True)

# ----------------------------
# EMA helper
# ----------------------------
class ModelEMA:
    def __init__(self, model, decay=0.9998):
        # copy state dict tensors (detached)
        msd = model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()
        self.ema = {k: v.clone().detach() for k, v in msd.items()}
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()
        for k, v in self.ema.items():
            v.copy_(v * self.decay + (1. - self.decay) * msd[k].detach())

# imagenet_val_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset

class ImageNetValDataset(Dataset):
    def __init__(self, val_dir, ground_truth_file, transform=None):
        self.val_dir = val_dir
        self.transform = transform

        # Load ground truth labels
        with open(ground_truth_file, "r") as f:
            self.labels = [int(line.strip()) - 1 for line in f.readlines()]  # make 0-based

        # List and sort JPEGs to align with ground_truth order
        self.images = sorted([f for f in os.listdir(val_dir) if f.lower().endswith(".jpeg")])
        assert len(self.images) == len(self.labels), \
            f"Mismatch: {len(self.images)} images vs {len(self.labels)} labels"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.val_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ----------------------------
# Main train function
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--log_interval", default=500, type=int)
    args = parser.parse_args()

    # --- DDP / torchrun environment ---
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))  # fallback

    # Set some NCCL envvars for robustness
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("NCCL_TIMEOUT", "1800")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    if world_size > 1:
        # bind to correct GPU for this local process BEFORE init_process_group
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ------------------ Data ------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_transform)
    val_dataset = ImageNetValDataset(
        val_dir=os.path.join(args.data_dir, "val"),
        ground_truth_file=os.path.join(args.data_dir, "ILSVRC2012_validation_ground_truth.txt"),
        transform=val_transform
    )
    if world_size > 1:
        train_sampler = distributed.DistributedSampler(train_dataset)
        val_sampler = distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            sampler=val_sampler,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)

    # ------------------ Model & optimizer ------------------
    model = models.resnet50(weights=None)
    model = model.to(device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    effective_lr = args.lr * max(1, world_size)  # simple LR scaling
    optimizer = optim.SGD(model.parameters(), lr=effective_lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=torch.cuda.is_available())
    ema = ModelEMA(model, decay=0.9998)

    start_epoch = 0

    # ------------------ Resume if available ------------------
    if args.resume and os.path.isfile(args.resume):
        if rank == 0:
            log(f"Loading checkpoint {args.resume}", rank)
        checkpoint = torch.load(args.resume, map_location="cpu")
        # load into underlying module if DDP
        if world_size > 1:
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        ema.ema = checkpoint.get("ema_state_dict", ema.ema)
        start_epoch = checkpoint.get("epoch", 0) + 1
        if rank == 0:
            log(f"Resumed from epoch {start_epoch}", rank)

    # ------------------ Training loop ------------------
    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        first_batch_debug_printed = False

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # debug print to confirm data loader and DDP are alive
            if not first_batch_debug_printed and rank == 0:
                log(f"[DEBUG] Starting first training batch of epoch {epoch}; train_loader_len={len(train_loader)}", rank)
                first_batch_debug_printed = True

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update EMA
            ema.update(model)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            running_correct += preds.eq(targets).sum().item()
            running_total += targets.size(0)

            if (step + 1) % args.log_interval == 0 and rank == 0:
                avg_loss = running_loss / running_total
                avg_acc = 100.0 * running_correct / running_total
                log(f"[Train] Epoch {epoch} Step {step+1}/{len(train_loader)} Loss {avg_loss:.4f} Acc {avg_acc:.2f}%", rank)

        # epoch summary (rank 0)
        if rank == 0:
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = 100.0 * running_correct / running_total if running_total > 0 else 0.0
            log(f"[Train] Epoch {epoch} Done | Loss {epoch_loss:.4f} Acc {epoch_acc:.2f}%", rank)

        # ------------------ Validation ------------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(targets).sum().item()
                val_total += targets.size(0)

        # compute and print val metrics (rank 0)
        if rank == 0:
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
            log(f"[Validation] Epoch {epoch} ValLoss {val_loss:.4f} ValAcc {val_acc:.2f}%", rank)

            # ------------------ Save checkpoint (always store underlying module state_dict)
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            state = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "ema_state_dict": ema.ema,
            }
            ckpt_path = os.path.join(args.checkpoint_dir, "last_checkpoint.pth")
            torch.save(state, ckpt_path)
            log(f"Saved checkpoint: {ckpt_path}", rank)

    # cleanup DDP
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
