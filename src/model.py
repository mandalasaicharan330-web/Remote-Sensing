"""
model.py
========
Deep learning architectures for Sentinel-2 multispectral land-cover classification.
Supports M1/M2/M3 band ablation experiments:
  - M1: 3 channels [B02, B03, B04] (Visible RGB)
  - M2: 4 channels [B02, B03, B04, B08] (Visible + NIR)
  - M3: 6 channels [B02, B03, B04, B08, B11, B12] (Visible + NIR + SWIR)

Outputs 7 independent unconstrained logits for multi-label classification via BCEWithLogitsLoss.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    Dataset = object


# Standard Band Ablation Configurations
BAND_CONFIGS: Dict[str, Dict[str, any]] = {
    "M1": {
        "name": "M1_Visible_RGB",
        "bands": ["B02", "B03", "B04"],
        "indices": [0, 1, 2],
        "in_channels": 3,
        "description": "Visible spectrum: Blue (490 nm), Green (560 nm), Red (665 nm)"
    },
    "M2": {
        "name": "M2_Visible_NIR",
        "bands": ["B02", "B03", "B04", "B08"],
        "indices": [0, 1, 2, 3],
        "in_channels": 4,
        "description": "Visible + Near-Infrared: RGB + NIR (842 nm)"
    },
    "M3": {
        "name": "M3_Full_Multispectral",
        "bands": ["B02", "B03", "B04", "B08", "B11", "B12"],
        "indices": [0, 1, 2, 3, 4, 5],
        "in_channels": 6,
        "description": "Full Multispectral: RGB + NIR + SWIR-1 (1610 nm) + SWIR-2 (2190 nm)"
    }
}


if TORCH_AVAILABLE:
    class MultispectralDataset(Dataset):
        """
        PyTorch Dataset for Sentinel-2 multispectral image patches.
        Slices selected channel indices on the fly with no data copying overhead.
        """
        def __init__(self, images: np.ndarray, labels: np.ndarray, channel_indices: List[int]):
            assert images.ndim == 4, f"Expected 4D array (N, C, H, W), got {images.shape}"
            assert labels.ndim == 2, f"Expected 2D array (N, num_classes), got {labels.shape}"
            self.images = torch.from_numpy(images[:, channel_indices, :, :]).float()
            self.labels = torch.from_numpy(labels).float()

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.images[idx], self.labels[idx]


    class MultispectralCNN(nn.Module):
        """
        Compact, high-efficiency Convolutional Neural Network designed for
        multispectral multi-label remote sensing classification on small-to-medium
        curated datasets (300-1000 patches).
        
        Features:
          - 4 convolutional blocks with Batch Normalization and ReLU activations
          - Max pooling for spatial downsampling (120x120 -> 60x60 -> 30x30 -> 15x15 -> 7x7)
          - Adaptive Global Average Pooling to 1x1
          - Dual Dropout regularization (p=0.4 and p=0.3) to prevent overfitting
          - Outputs 7 independent unconstrained logits (one per target class)
        """
        def __init__(self, in_channels: int, num_classes: int = 7):
            super().__init__()
            self.in_channels = in_channels
            self.num_classes = num_classes

            self.features = nn.Sequential(
                # Block 1: 120x120 -> 60x60
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                # Block 2: 60x60 -> 30x30
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                # Block 3: 30x30 -> 15x15
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                # Block 4: 15x15 -> 7x7
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

            self.pool = nn.AdaptiveAvgPool2d((1, 1))

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            x: (B, C, 120, 120)
            Returns: raw unconstrained logits of shape (B, num_classes)
            """
            feats = self.features(x)
            pooled = self.pool(feats)
            logits = self.classifier(pooled)
            return logits


    class ResNet18Adapted(nn.Module):
        """
        ResNet-18 adapted for arbitrary multispectral input channel counts (3, 4, or 6)
        and multi-label classification (7 unconstrained output logits).
        """
        def __init__(self, in_channels: int, num_classes: int = 7, pretrained: bool = False):
            super().__init__()
            try:
                import torchvision.models as models
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                base = models.resnet18(weights=weights)
            except Exception:
                import torchvision.models as models
                base = models.resnet18(pretrained=pretrained)

            # Replace first conv layer to accept in_channels
            if in_channels != 3 or not pretrained:
                self.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            else:
                self.conv1 = base.conv1

            self.bn1 = base.bn1
            self.relu = base.relu
            self.maxpool = base.maxpool
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.avgpool = base.avgpool
            self.fc = nn.Linear(512, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            logits = self.fc(x)
            return logits


def count_parameters(model: "nn.Module") -> Dict[str, int]:
    """Returns total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_model(
    model_type: str = "cnn",
    in_channels: int = 6,
    num_classes: int = 7
) -> "nn.Module":
    """
    Factory function to instantiate models.
    model_type: 'cnn' or 'resnet18'
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to build models.")
    if model_type.lower() == "cnn":
        return MultispectralCNN(in_channels=in_channels, num_classes=num_classes)
    elif model_type.lower() == "resnet18":
        return ResNet18Adapted(in_channels=in_channels, num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'cnn' or 'resnet18'.")
