import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scripts.inspect_datasets import build_metadata
from src.dataset import create_dataloaders
from src.model import build_model
from src.evaluation import Evaluator

def train_model(
    metadata_csv="data/metadata.csv",
    save_model_path="models/model.pt",
    phase1_epochs=10,
    phase2_epochs=15,
    batch_size=16,
    device=None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using compute device: {device}")
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Build metadata if missing
    if not os.path.exists(metadata_csv):
        print("Generating metadata CSV...")
        build_metadata(output_csv=metadata_csv)

    # 2. DataLoaders
    train_loader, val_loader, train_df, val_df = create_dataloaders(
        metadata_csv=metadata_csv, batch_size=batch_size
    )
    print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")

    # 3. Build Model
    model = build_model(num_classes=2, pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    # --- PHASE 1: Train FC Head Only ---
    print("\n--- PHASE 1: Training FC Head (Backbone Frozen) ---")
    optimizer_p1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler_p1 = ReduceLROnPlateau(optimizer_p1, mode='min', factor=0.5, patience=2)

    for epoch in range(phase1_epochs):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer_p1, device)
        val_loss, val_acc = _val_epoch(model, val_loader, criterion, device)
        scheduler_p1.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"P1 Epoch {epoch+1:02d}/{phase1_epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%", flush=True)

    # --- PHASE 2: Fine-tune Layer4 + FC Head ---
    print("\n--- PHASE 2: Fine-Tuning Last Residual Blocks (Unfrozen Layer4) ---")
    model.unfreeze_last_blocks(num_blocks=2)
    optimizer_p2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    scheduler_p2 = ReduceLROnPlateau(optimizer_p2, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')

    for epoch in range(phase2_epochs):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer_p2, device)
        val_loss, val_acc = _val_epoch(model, val_loader, criterion, device)
        scheduler_p2.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"P2 Epoch {epoch+1:02d}/{phase2_epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_model_path)
            print(f" Saved best checkpoint to '{save_model_path}' (Val Loss: {val_loss:.4f})")

    # Save training curves
    _plot_training_curves(history, output_path="results/training_curves.png")

    # Run Final Evaluation
    print("\n--- Running Final Evaluation on Validation Set ---")
    model.load_state_dict(torch.load(save_model_path, map_location=device))
    evaluator = Evaluator(model, device=device, output_dir="results")
    metrics = evaluator.evaluate_dataloader(val_loader)
    
    print("\nFinal Model Evaluation Metrics:")
    for k, v in metrics.items():
        if k != 'report_string':
            print(f"  {k}: {v}")
    print("\nClassification Report:\n", metrics['report_string'])

    return model, metrics


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        imgs = batch['image'].to(device)
        lbls = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == lbls).sum().item()
        total += imgs.size(0)

    epoch_loss = running_loss / (total + 1e-5)
    epoch_acc = (correct / (total + 1e-5)) * 100.0
    return epoch_loss, epoch_acc


def _val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)

            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == lbls).sum().item()
            total += imgs.size(0)

    epoch_loss = running_loss / (total + 1e-5)
    epoch_acc = (correct / (total + 1e-5)) * 100.0
    return epoch_loss, epoch_acc


def _plot_training_curves(history, output_path="results/training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_title("Cross-Entropy Loss Curve")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
    ax2.plot(history['val_acc'], label='Val Accuracy', color='green')
    ax2.set_title("Classification Accuracy Curve (%)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curves to '{output_path}'")


if __name__ == "__main__":
    train_model(phase1_epochs=3, phase2_epochs=5)
