import torch
import torch.nn as nn
from torchvision import models

class ParkinsonResNet18(nn.Module):
    """
    ResNet-18 Architecture fine-tuned for Early Parkinson's Disease Detection.
    Lightweight backbone optimized for fast point-of-care inference.
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super(ParkinsonResNet18, self).__init__()
        
        # Load ResNet-18 backbone
        if hasattr(models, 'ResNet18_Weights'):
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.resnet = models.resnet18(weights=weights)
        else:
            self.resnet = models.resnet18(pretrained=pretrained)

        # Transfer Learning: Freeze early convolutional layers
        if freeze_backbone:
            # Freeze all layers except the final residual block (layer4) and FC head
            for param in list(self.resnet.parameters())[:-20]:
                param.requires_grad = False

        # Replace final Fully Connected layer for binary classification (0: Healthy, 1: Parkinson's)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def unfreeze_all(self):
        """
        Unfreezes all parameters for phase-2 fine-tuning.
        """
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_last_blocks(self, num_blocks=2):
        """
        Unfreezes the last N residual blocks for phase-2 fine-tuning.
        """
        # Unfreeze layer4 and FC
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True


def build_model(num_classes=2, pretrained=True, freeze_backbone=True):
    model = ParkinsonResNet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    return model
