"""
models.py — Model factory for all DEST experiments.

Models:
  SmallCNN  — 2-conv + FC, for MNIST / FashionMNIST (28x28 grayscale)
  ResNet9   — lightweight 9-layer residual, for CIFAR-10 (32x32)
  ResNet18  — standard torchvision ResNet-18, adapted for 32x32 (CIFAR-100)
              or 64x64 (TinyImageNet)

DeterministicDropout: shared deterministic / stochastic dropout implementation.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


# ──────────────────────── shared dropout module ──────────────────────────────
class DeterministicDropout(nn.Module):
    """
    Drop-in replacement for nn.Dropout that supports two modes:

    'stochastic' : standard random dropout (identical to nn.Dropout).
    'deterministic' : rotating binary mask — keeps exactly (1-p) fraction
                      of features, cycling the mask position each step.
                      This gives deterministic diversity without random noise.
    """

    def __init__(self, p: float = 0.5, mode: str = "stochastic"):
        super().__init__()
        self.p    = p
        self.mode = mode
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        if self.mode == "stochastic":
            mask = (torch.rand_like(x) >= self.p).float()
            return x * mask / (1.0 - self.p)

        elif self.mode == "deterministic":
            _, num_features = x.shape[0], x.shape[1]
            keep_n  = max(1, int(round((1.0 - self.p) * num_features)))
            base    = torch.zeros(num_features, device=x.device)
            base[:keep_n] = 1.0
            step    = int(self.step_counter.item())
            batch_size = x.shape[0]
            masks   = torch.stack(
                [torch.roll(base, (step + i) % num_features, 0) for i in range(batch_size)]
            )
            self.step_counter += 1
            scale = num_features / keep_n
            return x * masks * scale

        else:
            raise ValueError(f"Unknown dropout mode: {self.mode}")


# ──────────────────────── SmallCNN (MNIST / FashionMNIST) ───────────────────
class SmallCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout_mode: str = "stochastic",
        dropout_prob: float = 0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),          nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            DeterministicDropout(p=dropout_prob, mode=dropout_mode),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────── ResNet9 (CIFAR-10) ────────────────────────────────
def _conv_block(in_ch, out_ch, pool=False):
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        dropout_mode: str = "stochastic",
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.prep   = _conv_block(in_channels, 64)
        self.layer1 = nn.Sequential(_conv_block(64, 128, pool=True),
                                    nn.Sequential(_conv_block(128, 128), _conv_block(128, 128)))
        self.layer2 = _conv_block(128, 256, pool=True)
        self.layer3 = nn.Sequential(_conv_block(256, 512, pool=True),
                                    nn.Sequential(_conv_block(512, 512), _conv_block(512, 512)))
        self.head   = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            DeterministicDropout(p=dropout_prob, mode=dropout_mode),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self._fwd(x)

    def _fwd(self, x):
        x  = self.prep(x)
        x1 = self.layer1[0](x)
        x  = self.layer1[1](x1) + x1
        x  = self.layer2(x)
        x3 = self.layer3[0](x)
        x  = self.layer3[1](x3) + x3
        return self.head(x)


# ──────────────────────── ResNet18 (CIFAR-100 / TinyImageNet) ───────────────
def _build_resnet18(in_channels: int, num_classes: int, input_size: int) -> nn.Module:
    """
    Torchvision ResNet-18 adapted for the given spatial input size.
    - 32x32 (CIFAR-100): replace 7x7/stride-2 conv with 3x3/stride-1, remove maxpool.
    - 64x64 (TinyImageNet): keep standard first conv, remove maxpool.
    - Others: standard.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if input_size <= 32:
        model.conv1  = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif input_size <= 64:
        model.maxpool = nn.Identity()
    return model


# ──────────────────────── SimpleMLP (fallback) ──────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, dropout_mode="stochastic", dropout_prob=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256), nn.ReLU(),
            DeterministicDropout(p=dropout_prob, mode=dropout_mode),
            nn.Linear(256, 128), nn.ReLU(),
            DeterministicDropout(p=dropout_prob, mode=dropout_mode),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)


# ──────────────────────── factory ───────────────────────────────────────────
class ModelFactory:
    @staticmethod
    def get_model(
        model_name: str,
        input_shape: tuple,
        num_classes: int,
        dropout_mode: str = "stochastic",
    ) -> nn.Module:
        name       = model_name.lower()
        in_ch      = input_shape[0]
        input_size = input_shape[1]          # height (assume square)

        if name in ("smallcnn", "cnn"):
            return SmallCNN(in_channels=in_ch, num_classes=num_classes, dropout_mode=dropout_mode)

        elif name == "resnet9":
            return ResNet9(in_channels=in_ch, num_classes=num_classes, dropout_mode=dropout_mode)

        elif name in ("resnet18",):
            return _build_resnet18(in_ch, num_classes, input_size)

        elif name in ("simplemlp", "mlp"):
            in_features = int(in_ch * input_shape[1] * input_shape[2])
            return SimpleMLP(in_features=in_features, num_classes=num_classes, dropout_mode=dropout_mode)

        else:
            raise ValueError(f"Unknown model: {model_name}")
