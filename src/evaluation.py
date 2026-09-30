import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader

from src.dataset import ParkinsonDataset, LABEL_MAP, INV_LABEL_MAP
from src.augmentation import get_val_transforms, get_train_transforms
from src.model import build_model

class Evaluator:
    """
    Comprehensive Evaluation Suite for Early Parkinson's Disease Model.
    Computes Classification Report, Confusion Matrix, ROC-AUC, and Stratified K-Fold CV.
    """
    def __init__(self, model, device='cpu', output_dir='results'):
        self.model = model.to(device)
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_dataloader(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(images)
                
                probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of Parkinson's
                preds = torch.argmax(outputs, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = self.compute_metrics(all_labels, all_preds, all_probs)
        self.plot_confusion_matrix(all_labels, all_preds)
        self.plot_roc_curve(all_labels, all_probs)

        return metrics

    def compute_metrics(self, labels, preds, probs):
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.0

        report_str = classification_report(labels, preds, target_names=['Healthy', "Parkinson's"], zero_division=0)
        
        metrics = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(auc), 4),
            'report_string': report_str
        }

        # Save report text
        with open(os.path.join(self.output_dir, "classification_report.txt"), "w") as f:
            f.write(report_str)

        return metrics

    def plot_confusion_matrix(self, labels, preds, filename="confusion_matrix.png"):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', "Parkinson's"],
            yticklabels=['Healthy', "Parkinson's"]
        )
        plt.title("Confusion Matrix — Parkinson's Detection")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_roc_curve(self, labels, probs, filename="roc_curve.png"):
        try:
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = roc_auc_score(labels, probs)
            
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=300)
            plt.close()
        except Exception as e:
            print(f"Skipping ROC curve plot: {e}")


def run_kfold_cross_validation(metadata_csv="data/metadata.csv", n_splits=5, epochs=10, batch_size=16, device='cpu'):
    """
    Executes 5-fold Stratified Cross-Validation for statistical robustness reporting.
    """
    df = pd.read_csv(metadata_csv)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n--- Starting {n_splits}-Fold Stratified Cross-Validation ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n>>> Fold {fold+1}/{n_splits} <<<")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = ParkinsonDataset(train_df, transform=get_train_transforms(), is_train=True)
        val_dataset = ParkinsonDataset(val_df, transform=get_val_transforms(), is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model = build_model(pretrained=True, freeze_backbone=True).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                imgs = batch['image'].to(device)
                lbls = batch['label'].to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()

        evaluator = Evaluator(model, device=device, output_dir=f"results/fold_{fold+1}")
        metrics = evaluator.evaluate_dataloader(val_loader)
        print(f"Fold {fold+1} Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | AUC: {metrics['roc_auc']:.4f}")
        fold_results.append(metrics)

    avg_acc = np.mean([m['accuracy'] for m in fold_results])
    avg_f1 = np.mean([m['f1_score'] for m in fold_results])
    avg_auc = np.mean([m['roc_auc'] for m in fold_results])

    print("\n==========================================")
    print(f"5-Fold CV Mean Accuracy: {avg_acc:.4f}")
    print(f"5-Fold CV Mean F1-Score: {avg_f1:.4f}")
    print(f"5-Fold CV Mean ROC-AUC:  {avg_auc:.4f}")
    print("==========================================")

    return fold_results
