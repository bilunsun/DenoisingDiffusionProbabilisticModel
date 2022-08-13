import torch
import torch.nn.functional as F
from torch import nn


class TestModel(nn.Module):
    def __init__(self, width: int, height: int, n_channels: int) -> None:
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1) for _ in range(6)]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i != len(self.convs) - 1:
                x = F.relu(x)

        return x
