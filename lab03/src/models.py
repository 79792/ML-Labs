"""Three Fashion-MNIST models: MLP, CNN, patch-attention."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class AttentionNet(nn.Module):
    """28x28 -> 16 patches of 7x7 -> Linear 49->64 -> 2 encoder layers -> mean pool -> 10."""

    def __init__(self, patch=7, d_model=64, nhead=4, ff=128, n_layers=2):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(patch * patch, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        b, p = x.shape[0], self.patch
        # (B,1,28,28) -> (B,1,4,4,7,7) -> (B,16,49)
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.reshape(b, -1, p * p)
        x = self.proj(x)          # (B,16,64)
        x = self.encoder(x)       # (B,16,64)
        x = x.mean(dim=1)         # mean pool over patches
        return self.head(x)


def build_models():
    return {"MLP": MLP(), "CNN": CNN(), "Attention": AttentionNet()}
