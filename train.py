"""Train the ResNet34-U-Net on pre-extracted BraTS slices.

This is the source of truth for training; train.ipynb imports and calls
the same `train_model` function so the notebook and script never drift apart.
"""
import argparse
import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from constants import (
    BATCH_SIZE,
    BEST_CHECKPOINT_PATH,
    CHECKPOINT_DIR,
    DATA_PROCESSED_DIR,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    LR_PLATEAU_FACTOR,
    LR_PLATEAU_PATIENCE,
    MAX_EPOCHS,
    WEIGHT_DECAY,
)
from dataset import build_datasets
from model import build_loss, build_model, region_dice_scores


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    region_totals = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
    n_batches = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item()
        scores = region_dice_scores(logits.detach(), masks)
        for region, val in scores.items():
            region_totals[region] += val
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_region = {r: v / n_batches for r, v in region_totals.items()}
    return avg_loss, avg_region


def train_model(
    processed_dir=DATA_PROCESSED_DIR,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    max_epochs=MAX_EPOCHS,
    early_stop_patience=EARLY_STOP_PATIENCE,
    checkpoint_path=BEST_CHECKPOINT_PATH,
    num_workers=2,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds, val_ds, _ = build_datasets(processed_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    model = build_model().to(device)
    criterion = build_loss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                   factor=LR_PLATEAU_FACTOR, patience=LR_PLATEAU_PATIENCE)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_val_wt_dice = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_region = _run_epoch(model, train_loader, criterion, device,
                                                optimizer=optimizer, scaler=scaler)
        val_loss, val_region = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_region["WT"])

        elapsed = time.time() - t0
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_WT={val_region['WT']:.4f} val_TC={val_region['TC']:.4f} "
              f"val_ET={val_region['ET']:.4f} ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         **{f"val_{k}": v for k, v in val_region.items()}})

        if val_region["WT"] > best_val_wt_dice:
            best_val_wt_dice = val_region["WT"]
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_wt_dice": best_val_wt_dice,
            }, checkpoint_path)
            print(f"[epoch {epoch:03d}] new best val WT Dice={best_val_wt_dice:.4f}, saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"[epoch {epoch:03d}] early stopping (no val WT Dice improvement in "
                      f"{early_stop_patience} epochs)")
                break

    return history, best_val_wt_dice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", default=DATA_PROCESSED_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train_model(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
