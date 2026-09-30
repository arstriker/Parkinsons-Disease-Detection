import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from src.preprocessing import ImagePreprocessor
from src.augmentation import get_train_transforms, get_val_transforms

LABEL_MAP = {'healthy': 0, 'parkinson': 1}
INV_LABEL_MAP = {0: 'healthy', 1: 'parkinson'}

class ParkinsonDataset(Dataset):
    """
    PyTorch Dataset for Early Parkinson's Disease stroke drawings.
    Integrates standardized preprocessing and on-the-fly Albumentations data augmentation.
    """
    def __init__(self, df, transform=None, is_train=True, target_size=(224, 224)):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train
        self.preprocessor = ImagePreprocessor(target_size=target_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        label_str = str(row['label']).lower()
        label = LABEL_MAP.get(label_str, 0)

        # Preprocess stroke drawing
        processed_rgb, tensor, stroke_metrics = self.preprocessor.preprocess_pipeline(img_path)

        # Apply augmentation if transform provided (Albumentations expect uint8 HWC)
        if self.transform is not None:
            augmented = self.transform(image=processed_rgb)
            tensor = augmented['image']

        return {
            'image': tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'filepath': img_path,
            'source_dataset': row.get('source_dataset', 'unknown'),
            'modality': row.get('modality', 'drawing')
        }


def create_dataloaders(metadata_csv="data/metadata.csv", batch_size=16, val_split=0.2, seed=42):
    """
    Loads metadata CSV, performs stratified train/val split, and returns PyTorch DataLoaders.
    """
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata file missing at '{metadata_csv}'. Run inspect_datasets.py first.")

    df = pd.read_csv(metadata_csv)
    if df.empty:
        raise ValueError("Metadata dataframe is empty.")

    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed, stratify=df['label']
    )

    train_dataset = ParkinsonDataset(train_df, transform=get_train_transforms(), is_train=True)
    val_dataset = ParkinsonDataset(val_df, transform=get_val_transforms(), is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_df, val_df
