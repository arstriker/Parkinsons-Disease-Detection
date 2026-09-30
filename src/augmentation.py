import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np

def get_train_transforms(image_size=224):
    """
    Data Augmentation pipeline for Parkinson's stroke drawings.
    Applies rotation, scale, elastic/affine transforms, and Gaussian noise.
    
    IMPORTANT DESIGN CHOICE:
    Horizontal and vertical flips are explicitly EXCLUDED to preserve
    clockwise vs. counter-clockwise stroke direction semantics.
    """
    return A.Compose([
        # Rotation (+/- 15 degrees)
        A.Rotate(limit=15, p=0.7, border_mode=0),
        
        # Slight scaling (0.9 to 1.1x) & Translation
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=0,
            border_mode=0,
            p=0.6
        ),
        
        # Elastic / Affine Distortion for natural hand tremor simulation
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=20, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
        ], p=0.4),
        
        # Gaussian Noise to simulate scanner / camera sensor noise
        A.GaussNoise(p=0.4),
        
        # Normalize with ImageNet mean and std
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transforms(image_size=224):
    """
    Validation / Inference transformation pipeline (only normalization).
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
